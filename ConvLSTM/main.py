import os
from encoder import Encoder
from model import ED
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
from earlystopping import EarlyStopping
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import json
import pandas as pd
from ConvRNN import CLSTM_cell  
from utils import make_layers  
from collections import OrderedDict
from decoder import Decoder 
from torch.cuda.amp import autocast, GradScaler

# ========== Configuration ==========
ARGS = {
    "convlstm": True,
    "convgru": False,
    "batch_size": 128,
    "lr_list": [4e-5,4e-4,4e-3,4e-2,1e-4,1e-3,1e-2,1e-1],  
    "epochs": 600,
    "data_root": "../data/Seq21_Vars6_Mask/",
    "save_root": "",
    "plot_root": "",
    "csv_save_root": "",
    # 新增 ↓
    "scheduler_factor": 0.7,
    "scheduler_patience": 5,
    "earlystop_patience": 30,
    "clip_grad": 10.0,
    "dropout": 0.1,   # dropout rate before final linear
    "pooling_mode": "mask_flatten",   # "mask_gap" or "mask_flatten"
}

torch.backends.cudnn.benchmark = False
print("GPUs visible:", torch.cuda.device_count())
USE_AMP = True  

def build_convlstm_encoder_params(C_in, H, W):
    subnets = [
        # in_channels=6 -> 8
        OrderedDict({"conv1_leaky_1": [C_in, 8, 3, 1, 1]}),
        OrderedDict({"conv2_leaky_1": [16, 16, 3, 1, 1]}),
        OrderedDict({"conv3_leaky_1": [32, 32, 3, 1, 1]}),
    ]
    rnns = [
        CLSTM_cell(shape=(H, W), input_channels=8, filter_size=3, num_features=16),
        CLSTM_cell(shape=(H, W), input_channels=16, filter_size=3, num_features=32),
        CLSTM_cell(shape=(H, W), input_channels=32, filter_size=3, num_features=64),
    ]
    return subnets, rnns

def build_convlstm_decoder_params(H, W, hidden_ch):
    subnets = [OrderedDict([])] 
    rnns = [CLSTM_cell(shape=(H, W), input_channels=1, filter_size=1, num_features=hidden_ch)]
    return subnets, rnns

# —— Data：Read Train.npz/Val.npz/Test.npz（ [idx, y, x_seq]）—————
class SimpleNPZDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.X = z["X"].astype(np.float32, copy=False)     # (N, T, C, H, W)
        self.y = z["y"].astype(np.float32, copy=False)     # (N,)
        self.L = z["last_idx"].astype(np.int64, copy=False)
        self.y_self = z["y_self"].astype(np.float32, copy=False)  # ✅ (N, Ty)

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).float()
        y = torch.from_numpy(self.y[i:i+1]).float()
        y_hist = torch.from_numpy(self.y_self[i]).float()   # ✅ (Ty,)
        return self.L[i], y, x, y_hist


def discover_stations(data_root):
    stations = []
    root = os.path.abspath(data_root)
    for dirpath, dirnames, filenames in os.walk(root):
        if "Train.npz" in filenames and "meta.json" in filenames and "Val.npz" in filenames and "Test.npz" in filenames:
            # station_name use {region}-{gageid} to represent
            rel = os.path.relpath(dirpath, root)
            parts = rel.split(os.sep)
            if len(parts) >= 2:
                station_name = f"{parts[-2]}-{parts[-1]}"
            else:
                station_name = parts[-1]
            stations.append((station_name, dirpath))
    return sorted(stations)

# —— inverse —— 
def inv_transform_var(vs):
    vs = np.asarray(vs, dtype=np.float32)
    return np.square(vs)
def inverse_normalize_y(y_stdzd, y_mean, y_std):
    return inv_transform_var(y_stdzd * y_std + y_mean)

def evaluate_on_loader(net, loader, meta, device, out_dir, tag):
    net.eval()
    y_mean, y_std = float(meta["y_mean"]), float(meta["y_std"])
    preds_all, trues_all = [], []
    with torch.no_grad():
        for _, (idx, y, x, y_hist) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1)
            y_hist = y_hist.to(device, non_blocking=True)
            with autocast(enabled=USE_AMP):
                pred = net(x, y_hist=y_hist).view(-1)
            preds_all.append(pred.float().cpu().numpy())
            trues_all.append(y.float().cpu().numpy())
    preds_z = np.concatenate(preds_all, 0)
    trues_z = np.concatenate(trues_all, 0)
    preds = inverse_normalize_y(preds_z, y_mean, y_std)
    trues = inverse_normalize_y(trues_z, y_mean, y_std)

    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    mae  = float(np.mean(np.abs(preds - trues)))
    nse  = float(1 - np.sum((trues - preds) ** 2) / (np.sum((trues - np.mean(trues)) ** 2) + 1e-12))

    os.makedirs(out_dir, exist_ok=True)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.plot(trues, label="True")
    plt.plot(preds, label="Pred")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    textstr = f"MAE={mae:.2f}\nNSE={nse:.2f}"
    ax = plt.gca()
    ax.text(0.95, 0.95, textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.savefig(os.path.join(out_dir, f"{tag}.png"), dpi=100)
    plt.close()
    return {"RMSE": rmse, "MAE": mae, "NSE": nse}

def train_and_eval_station(station_name, station_dir, current_lr, run_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_path = os.path.join(station_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    C, H, W = int(meta["C"]), int(meta["H"]), int(meta["W"])

    mask_np = np.load(os.path.join(station_dir, "mask01.npy")).astype(np.float32)  # (H,W) ∈ {0,1}
    mask_t  = torch.from_numpy(mask_np).to(device)                                  # (H,W)

    def make_loader(split):
        return torch.utils.data.DataLoader(
            SimpleNPZDataset(os.path.join(station_dir, f"{split}.npz")),
            batch_size=ARGS["batch_size"],
            shuffle=(split=="Train"),
            pin_memory=True,
            num_workers=4,
            persistent_workers=True)

    trainLoader = make_loader("Train")
    valLoader   = make_loader("Val")
    testLoader  = make_loader("Test")

    if ARGS["convlstm"]:
        subnets, rnns = build_convlstm_encoder_params(C, H, W)
    encoder = Encoder(subnets, rnns).to(device)
    hidden_ch = rnns[-1].num_features
    dec_subnets, dec_rnns = build_convlstm_decoder_params(H, W, hidden_ch)
    decoder = Decoder(dec_subnets, dec_rnns).to(device)

    net = ED(encoder, decoder, hidden_ch, H, W,
             dropout=ARGS["dropout"], use_gap=(ARGS["pooling_mode"] == "mask_gap")).to(device)  
    mask_hw_buf = torch.from_numpy(mask_np).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  
    net.register_buffer("mask_hw_buf", mask_hw_buf)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    optimizer = optim.Adam(net.parameters(), lr=current_lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=ARGS["scheduler_factor"],
        patience=ARGS["scheduler_patience"],
        verbose=True
    )
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=ARGS["earlystop_patience"], verbose=True)

    region, gageid = station_name.split("-", 1)
    save_dir = os.path.join(run_root, region, gageid, "save_root")
    plot_dir = os.path.join(run_root, region, gageid, "plot_root")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    best_ckpt = os.path.join(save_dir, "best_model.pth.tar")

    # Train
    scaler = GradScaler(enabled=USE_AMP)
    for epoch in range(ARGS["epochs"]):
        net.train()
        losses = []
        for _, (idx, y_true, x, y_hist) in enumerate(trainLoader):
            x = x.to(device, non_blocking=True)
            y_true = y_true.to(device, non_blocking=True).view(-1)
            y_hist = y_hist.to(device, non_blocking=True)
            with autocast(enabled=USE_AMP):
                # === mask ===
                pred     = net(x,     y_hist=y_hist).view(-1)
            loss = criterion(pred.float(), y_true)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), ARGS["clip_grad"])
            scaler.step(optimizer); scaler.update()
            losses.append(float(loss.item()))

        # ===== Validate =====
        val_losses = []
        net.eval()
        with torch.no_grad():
            for _, (idx, y_val, x_val, y_hist) in enumerate(valLoader):
                x_val = x_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True).view(-1)
                y_hist = y_hist.to(device, non_blocking=True)
                with autocast(enabled=USE_AMP):
                    pred_val = net(x_val, y_hist=y_hist).view(-1)
                vloss = criterion(pred_val.float(), y_val)
                val_losses.append(float(vloss.item()))

        train_loss, val_loss = np.mean(losses), np.mean(val_losses)
        scheduler.step(val_loss)
        early_stopping(val_loss, net, optimizer, epoch, save_dir)
        if early_stopping.early_stop:
            print(f"[{station_name}] Early stopping @ epoch {epoch}")
            break

    # —— Evaluate on Train/Val/Test sets ——
    #  Reload best model
    ckpt = torch.load(best_ckpt, map_location=device)
    net.load_state_dict(ckpt["state_dict"])
    results = {}
    for split, loader in [("Train",trainLoader),("Val",valLoader),("Test",testLoader)]:
        res = evaluate_on_loader(net, loader, meta, device, plot_dir, f"{split}")
        results[split] = res
    return results

if __name__ == "__main__":
    # Root：Results
    base_output = "Results_mask"
    for k in ["save_root", "plot_root", "csv_save_root"]:
        ARGS[k] = base_output
        os.makedirs(ARGS[k], exist_ok=True)

    #  "Seq30_Vars6_HW"
    base_tag = os.path.basename(os.path.normpath(ARGS["data_root"]))

    stations = discover_stations(ARGS["data_root"])
    print(f"discover {len(stations)} stations.")

    # multiple learning rates sweep
    for current_lr in ARGS["lr_list"]:
        # lr_tag，e.g. "lr1e-03"
        lr_tag = f"lr{current_lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        # Results/{base_tag}/{lr_tag}
        run_root = os.path.join(base_output, base_tag, lr_tag)
        os.makedirs(run_root, exist_ok=True)

        summary_dict = {}
        for (station_name, station_dir) in stations:
            print(f"[{lr_tag}] === Training {station_name} ===")
            res = train_and_eval_station(station_name, station_dir, current_lr, run_root)

            region, _ = station_name.split("-", 1)
            row = {
                "RMSE_train": res["Train"]["RMSE"],
                "MAE_train" : res["Train"]["MAE"],
                "NSE_train" : res["Train"]["NSE"],
                "RMSE_val"  : res["Val"]["RMSE"],
                "MAE_val"   : res["Val"]["MAE"],
                "NSE_val"   : res["Val"]["NSE"],
                "RMSE_test" : res["Test"]["RMSE"],
                "MAE_test"  : res["Test"]["MAE"],
                "NSE_test"  : res["Test"]["NSE"],
                "Region"    : region,
            }
            summary_dict[station_name] = row
            per_station_dir = os.path.join(run_root, "per_station")
            os.makedirs(per_station_dir, exist_ok=True)
            df_one = pd.DataFrame.from_dict({station_name: row}, orient="index")
            out_one = os.path.join(per_station_dir, f"{station_name}.csv")
            df_one.to_csv(out_one, encoding="utf-8-sig", index_label="station_name")  # 用 utf-8-sig 便于 Excel 直接打开
            print(f"[{lr_tag}] Save staion Results:{out_one}")

        df = pd.DataFrame.from_dict(summary_dict, orient="index")
        out_csv = os.path.join(run_root, "results.csv")
        df.to_csv(out_csv, encoding="utf-8-sig", index_label="station_name")
        print(f"[{lr_tag}] Save Results:", out_csv)
