# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


# ============== Config =================
DAYMET_OUT   = Path(r"Data_by_Station")  
EXCEL_BASE   = Path(r"../LSTM-XAI/LSTM_Data_with_Area_and_prep")
OUTPUT_ROOT  = Path(r"../ConvLSTM-DayMet/data")  
REGIONS      = ["01","03","11","17"]         

VARS = ["prcp","srad","swe","tmax","tmin","vp"]
TIME_WINDOW   = 21           # config['sequence_length']
HORIZON       = 1            
TRAIN_SIZE_N  = 366 * 15
VALID_SIZE_N  = 366 * 4
QC_MISSING    = {"M", "m"}   
eps = 1e-15
max_workers = 1

def normalize_discharge(q_cfs, area_m2, precip_mm_day):
    return q_cfs * 0.0283168 * 86400 * 1000 / (area_m2 * precip_mm_day)
def transform_var(v):
    v = np.asarray(v, dtype=np.float32)
    return np.sqrt(np.clip(v, 0, None))
def inv_transform_var(vs):
    vs = np.asarray(vs, dtype=np.float32)
    return np.square(vs)

# ============================================================

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def _read_station_excel(excel_path: Path):
    df = pd.read_excel(excel_path, engine="openpyxl")
    cols = {c.lower(): c for c in df.columns}
    for need in ["date", "discharge", "qc_flag"]:
        if need not in cols:
            raise KeyError(f"{excel_path} lack columns：{need}")
    date_col = cols["date"]
    q_col    = cols["discharge"]
    flag_col = cols["qc_flag"]

    dates = pd.to_datetime(df[date_col]).dt.floor("D").values.astype("datetime64[D]")
    q     = pd.to_numeric(df[q_col], errors="coerce").values.astype(np.float32)
    flags = df[flag_col].astype(str).fillna("").values
    area   = float(df.loc[0, "Area"])            # m²
    precip = float(df.loc[0, "Precipitation"])   # mm/day
    return dates, q, flags, area, precip


def _read_var_npz(st_dir: Path, var: str):
    p = st_dir / f"{var}.npz"
    if not p.exists():
        raise FileNotFoundError(f"lack {p}")
    arr = np.load(p)
    data  = arr["data"].astype(np.float32)           # (T,H,W)
    dates = pd.to_datetime(arr["dates"]).floor("D").values.astype("datetime64[D]")
    return data, dates

def _align_by_dates(excel_dates, var_dates_list, var_data_list):
    common_dates = set(excel_dates.tolist())
    for d in var_dates_list:
        common_dates &= set(d.tolist())
    if not common_dates:
        raise RuntimeError("Excel and Daymet variable date do not intersect")

    common = np.array(sorted(list(common_dates))).astype("datetime64[D]")
    ex_map = {d:i for i,d in enumerate(excel_dates)}
    ex_idx = np.array([ex_map[d] for d in common], dtype=np.int64)

    data_aligned = []
    for vdates, vdata in zip(var_dates_list, var_data_list):
        vd_map = {d:i for i,d in enumerate(vdates)}
        vidx = np.array([vd_map[d] for d in common], dtype=np.int64)
        data_aligned.append(vdata[vidx])  # (T,H,W)
    data_4d = np.stack(data_aligned, axis=1)
    return common, ex_idx, data_4d  

def _fit_y_scaler_on_train(y_sqrt_all, flags_all, n_tr):

    y_sqrt_all = np.asarray(y_sqrt_all, dtype=np.float32)
    flags_all  = np.asarray(flags_all)
    mask = np.ones_like(y_sqrt_all, dtype=bool)
    mask[:n_tr] = np.array([f not in QC_MISSING for f in flags_all[:n_tr]])
    mask[n_tr:] = False
    ys = y_sqrt_all[mask]
    if ys.size == 0:
        raise RuntimeError("There are no samples available for fitting y standardization in the training stage (possibly all missing tests)")
    mu  = float(np.mean(ys))
    std = float(np.std(ys)+eps)  
    return mu, std

def _fit_x_scaler_on_train(data_4d, n_tr, basin_mask):

    T, C, H, W = data_4d.shape
    tr = data_4d[:n_tr]  # (n_tr, C, H, W)

    mask = np.asarray(basin_mask, dtype=bool)
    if mask.shape != (H, W):
        raise ValueError(f"basin_mask should be {(H, W)}, recieve {mask.shape}")
    if mask.sum() == 0:
        raise RuntimeError("basin_mask have no True point")

    x_mu  = np.zeros((C,), dtype=np.float32)
    x_std = np.zeros((C,), dtype=np.float32)
    for c in range(C):
        v = tr[:, c, :, :].astype(np.float64, copy=False)         # (n_tr, H, W)
        v_masked = v[:, mask]                                     # (n_tr, Nmask)
        mu  = np.nanmean(v_masked)
        std = np.nanstd(v_masked) + 1e-15
        x_mu[c]  = float(mu)
        x_std[c] = float(std)
    return x_mu, x_std


def _generate_samples(data_4d, q_all, flags_all, y_mu, y_std, x_mu, x_std):
    T, C, H, W = data_4d.shape
    samples, last_idxs = [], []
    q_sqrt = transform_var(q_all)  # (T,)

    for i in range(0, T - TIME_WINDOW - HORIZON + 1):
        last_idx = i + TIME_WINDOW + HORIZON - 1
        if flags_all[last_idx] in QC_MISSING:
            continue
        x_seq = data_4d[i:i+TIME_WINDOW].astype(np.float32, copy=False)  # (TW,C,H,W)
        x_seq = (x_seq - x_mu.reshape(1, C, 1, 1)) / x_std.reshape(1, C, 1, 1)

        y_stdzd = (q_sqrt[last_idx] - y_mu) / max(y_std, eps)
        y_hist_win = q_sqrt[last_idx - TIME_WINDOW:last_idx]           # (TW,)
        y_self_seq = (y_hist_win - y_mu) / max(y_std, eps)             # (TW,)
        y_self_seq = y_self_seq.astype(np.float32, copy=False)

        samples.append([last_idx, np.float32(y_stdzd), x_seq, y_self_seq])
        last_idxs.append(last_idx)

    return samples, np.asarray(last_idxs, dtype=np.int64)



def _split_and_save(samples, last_idxs, station_out_dir, meta_dict):
    station_out_dir.mkdir(parents=True, exist_ok=True)

    mask_tr  = last_idxs < TRAIN_SIZE_N
    mask_val = (last_idxs >= TRAIN_SIZE_N) & (last_idxs < TRAIN_SIZE_N + VALID_SIZE_N)
    mask_te  = ~(mask_tr | mask_val)

    def _save_subset(mask, fname):
        idx_sel = [i for i, keep in enumerate(mask) if keep]
        N = len(idx_sel)
        if N == 0:
            np.savez(station_out_dir / fname,
                     last_idx=np.empty((0,), np.int64),
                     y=np.empty((0,), np.float32),
                     X=np.empty((0, 0, 0, 0, 0), np.float32),
                     y_self=np.empty((0, 0), np.float32))
            return
        sample0 = samples[idx_sel[0]][2]
        TW, C, H, W = sample0.shape
        X      = np.empty((N, TW, C, H, W), dtype=np.float32)
        Y      = np.empty((N,), dtype=np.float32)
        L      = np.empty((N,), dtype=np.int64)
        YSELF  = np.empty((N, TW), dtype=np.float32)

        for k, j in enumerate(idx_sel):
            li, y, x, y_self = samples[j]
            L[k]     = li
            Y[k]     = y
            X[k]     = x.astype(np.float32, copy=False)
            YSELF[k] = y_self.astype(np.float32, copy=False)

        np.savez(station_out_dir / fname, last_idx=L, y=Y, X=X, y_self=YSELF)
        del X, Y, L, YSELF
        import gc; gc.collect()
    _save_subset(mask_tr,  "Train.npz")
    _save_subset(mask_val, "Val.npz")
    _save_subset(mask_te,  "Test.npz")

    with open(station_out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=2)

def process_one_station(region: str, gageid: str):
    st_dir = DAYMET_OUT / region / gageid
    if not st_dir.exists():
        print(f"[{region}:{gageid}] Missing Daymet site directory, skip")
        return

    excel = next((p for p in (EXCEL_BASE/region).glob(f"{gageid}.xlsx")), None)
    if excel is None:
        print(f"[{region}:{gageid}] do not find Excel ({EXCEL_BASE/region}/{gageid}.xlsx),skip")
        return
    
    mask_path = st_dir / "mask.npy"
    if not mask_path.exists():
        raise FileNotFoundError(f"lack {mask_path}")
    basin_mask = np.load(mask_path).astype(bool)

    ex_dates, q_raw, flags, area, precip = _read_station_excel(excel)
    q_norm = normalize_discharge(q_raw, area, precip)

    var_datas, var_dates = [], []
    for v in VARS:
        d, dt = _read_var_npz(st_dir, v)
        var_datas.append(d)   # (T,H,W)
        var_dates.append(dt)  # (T,)

    common_dates, ex_idx_map, data_4d = _align_by_dates(ex_dates, var_dates, var_datas)
    q_all     = q_norm[ex_idx_map]      # (T,)
    flags_all = flags[ex_idx_map]      # (T,)
    y_mu, y_std = _fit_y_scaler_on_train(transform_var(q_all), flags_all, TRAIN_SIZE_N)
    x_mu, x_std = _fit_x_scaler_on_train(data_4d, TRAIN_SIZE_N, basin_mask)    # (C,), (C,)

    samples, last_idxs = _generate_samples(data_4d, q_all, flags_all, y_mu, y_std, x_mu, x_std)
    if len(samples) == 0:
        print(f"[{region}:{gageid}] 无可用样本（多为缺测或数据过短），跳过")
        return
    # Metadata
    _, C, H, W = data_4d.shape
    meta = {
        "region": region,
        "gageid": gageid,
        "vars": VARS,
        "time_window": TIME_WINDOW,
        "horizon": HORIZON,
        "H": H, "W": W, "C": C,
        "train_size_n": TRAIN_SIZE_N,
        "valid_size_n": VALID_SIZE_N,
        "y_mean": y_mu,
        "y_std":  y_std,
        "transform": "transform_var= sqrt; inv_transform_var= square",
        "note": "label = (transform_var(discharge) - y_mean) / y_std；仅训练段拟合",
        "date_range": [str(common_dates[0]), str(common_dates[-1])],
        "x_mean_per_channel": x_mu.tolist(),
        "x_std_per_channel":  x_std.tolist()
    }

    out_dir = OUTPUT_ROOT / f"Seq{TIME_WINDOW}_Vars{C}_Mask" / region / gageid
    _split_and_save(samples, last_idxs, out_dir, meta)
    mask0 = basin_mask.astype(np.uint8)
    np.save(out_dir / "mask01.npy", mask0)
    print(f"[{region}:{gageid}] -> Save Complete：{out_dir}")
    
if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    all_tasks = []
    for rg in REGIONS:
        excels = sorted((EXCEL_BASE / rg).glob("*.xlsx"))
        if not excels:
            print(f"[{rg}] do not find Excel, skip")
            continue
        for xp in excels:
            gid = xp.stem 
            all_tasks.append((rg, gid))

    if not all_tasks:
        print("No processable sites were found.")
        raise SystemExit(0)

    print(f"Find {len(all_tasks)} Stations,Use {max_workers} processes for parallel processing.")
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_one_station, rg, gid): (rg, gid) for (rg, gid) in all_tasks}
        for fut in as_completed(futures):
            rg, gid = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[{rg}:{gid}] Processing failed: {e}")

