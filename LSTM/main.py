# -*- coding: utf-8 -*-
import os, glob, copy
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd   
import seaborn as sns
from scipy.stats import kstest
from joblib import dump, load
import gc; gc.collect()

# 1) Configuration and Path
BASE_DIR = r'Y:/LSTM-Krazert/网格遍历实验' 
config = {
    'device'         : torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'feature_cols'   : ['prcp','srad','swe','tmax','tmin','vp'],
    'label_col'      : 'discharge',
    'qc_flag_col'    : 'qc_flag',
    'train_size_n'   : 366 * 14,
    'valid_size_n'   : 366 * 4,
    'batch_size'     : 256,         # Baseline
    'time_window'    : 21,
    'horizon'        : 1,
    'hidden_sizes'   : [20, 20],    # Baseline
    'dropout'        : 0.1,         # Baseline
    'learning_rate'  : 5e-3,        # Baseline
    'epochs'         : 250,
    'grad_clip'      : 10,
    'shuffle'        : True,
    # Early stopping & LR scheduler
    'patience'       : 24,
    'min_delta'      : 1e-9,
    'lr_patience'    : 4,
    'lr_factor'      : 0.7,
    'min_lr'         : 1e-7,
    'use_self'       : True,
    # 路径（★ 仅固定 processed_dir；其余三者在每个“组合”里动态生成）
    'processed_dir'  : os.path.join(BASE_DIR, 'processed_npz'),
    'input_data_dir' : 'LSTM_Data_with_Area_and_prep'
}

# -------------------------------------------------
# Grid Search settings
# -------------------------------------------------
param_grid = {
    'batch_size'   : [128,256,512,1024],
    'hidden_sizes' : [[64,64],[32,32],[20,20]],
    'dropout'      : [0.1,0.2],
    'learning_rate': [4e-5,4e-4,4e-3,4e-2,1e-4,1e-3,1e-2,1e-1]
}

os.makedirs(config['processed_dir'], exist_ok=True)

# 2) Tools
# -------------------------------------------------
def _fmt_float(v: float) -> str:
    s = f"{v:.6g}"  
    s = s.replace('E', 'e').replace('e-0', 'e-').replace('e+0', 'e+')
    return s.rstrip('.')  

def _hs_tag(hs_list) -> str:
    return 'x'.join(str(int(h)) for h in hs_list)

def build_exp_root(cfg, bs, hs, dp, lr):
    tag = f"sq_{cfg['time_window']}-bs_{bs}-hs_{_hs_tag(hs)}-dp_{_fmt_float(dp)}-lr_{_fmt_float(lr)}"
    return os.path.join(BASE_DIR, tag)

# 3) Data preparation function
def transform_var(v):
    return np.sqrt(v)

def inv_transform_var(vs):
    return np.square(vs)

def normalize_discharge(q_cfs, area_m2, precip_mm_day):
    return q_cfs * 0.0283168 * 86400 * 1000 / (area_m2 * precip_mm_day)

class FlexibleLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout):
        super().__init__()
        layers = []
        for i, h in enumerate(hidden_sizes):
            in_sz = input_size if i==0 else hidden_sizes[i-1]
            layers += [nn.LSTM(in_sz, h, batch_first=True), nn.Dropout(dropout)]
        self.layers = nn.ModuleList(layers)
        self.linear = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        out = x
        for m in self.layers:
            if isinstance(m, nn.LSTM):
                bs = out.size(0)
                h0 = torch.zeros(1, bs, m.hidden_size, device=out.device)
                c0 = torch.zeros(1, bs, m.hidden_size, device=out.device)
                out, _ = m(out, (h0, c0))
            else:
                out = m(out)
        return self.linear(out[:, -1, :])
    
def aggregate_epoch_stats(input_dir, output_path):
    all_dfs = []
    pattern = os.path.join(input_dir, '*_nse_by_epoch.xlsx')
    for fn in glob.glob(pattern):
        df = pd.read_excel(fn)
        all_dfs.append(df)

    long = pd.concat(all_dfs, ignore_index=True)
    stats = long.groupby('epoch')['NSE_val'].agg(
        NSE_median = 'median',
        NSE_mean   = 'mean'
    ).reset_index()
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if output_path.lower().endswith('.xlsx'):
        stats.to_excel(output_path, index=False, float_format="%.4f")
    else:
        stats.to_csv(output_path, index=False, float_format="%.4f")
    print(f"The summary table has been saved to {output_path}")
    
def nse_safe(pred, obs):
    num = np.sum((obs-pred)**2)
    den = np.sum((obs-obs.mean())**2)
    return np.nan if den==0 else 1 - num/den

# ----------------------------
# 4. EarlyStopping & LR Scheduler
# ----------------------------
def evaluate_loss(loader, model, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            preds = model(bx)
            losses.append(criterion(preds, by).item())
    return float(np.mean(losses))
# ----------------------------
# 6. Evaluation function
# ----------------------------
def eval_loader(loader, model, scaler_y, device):
    all_p, all_o = [], []
    model.eval()
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            p = model(bx).cpu().numpy().flatten()
            o = by.cpu().numpy().flatten()
            all_p.append(p); all_o.append(o)

    pred_star = scaler_y.inverse_transform(np.concatenate(all_p).reshape(-1,1)).flatten()
    obs_star  = scaler_y.inverse_transform(np.concatenate(all_o).reshape(-1,1)).flatten()
    pred = inv_transform_var(pred_star)
    obs  = inv_transform_var(obs_star)
    return obs, pred

# ----------------------------
# 7. Plot function
# ----------------------------
def plot_metrics(obs, pred, name, metrics, save_path):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(obs,  label='Observed')
    ax.plot(pred, label='Predicted')
    ax.set_title(f"{name} | MAE: {metrics['MAE']:.3f}, NSE: {metrics['NSE']:.3f}")
    ax.legend()
    ax.set_xlabel('Time step'); ax.set_ylabel('Discharge')
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# ----------------------------
# 3. Prepare data: Excel ->. npz cache ->Three dataset DataLoader
# ----------------------------
def prepare_data_from_excel(path, cfg):
    basin = os.path.basename(os.path.dirname(path))
    site_no = os.path.splitext(os.path.basename(path))[0]
    npz_path = os.path.join(cfg['processed_dir'], f"{basin}_{site_no}.npz")
    if os.path.exists(npz_path):
        arr = np.load(npz_path)
        train_X, train_y = arr['train_X'], arr['train_y']
        val_X,   val_y   = arr['val_X'],   arr['val_y']
        test_X,  test_y  = arr['test_X'],  arr['test_y']
        sx = StandardScaler(); sx.mean_, sx.scale_ = arr['x_mean'], arr['x_scale']
        sy = StandardScaler(); sy.mean_, sy.scale_ = arr['y_mean'], arr['y_scale']
    else:
        df = pd.read_excel(path, engine='openpyxl')
        area   = df.loc[0, 'Area']
        precip = df.loc[0, 'Precipitation']
        flags  = df[cfg['qc_flag_col']].astype(str).values
        raw_X  = df[cfg['feature_cols']].values.astype(float)
        raw_q  = df[cfg['label_col']].values.astype(float)

        q_norm = normalize_discharge(raw_q, area, precip)
        q_star = transform_var(q_norm).reshape(-1, 1)

        n_tr = cfg['train_size_n']
        n_val = cfg['valid_size_n']
        n_all = len(raw_X)
        tw, hor = cfg['time_window'], cfg['horizon']

        sx = StandardScaler().fit(raw_X[:(n_tr+tw)])
        valid_q = q_star[:(n_tr+tw)][flags[:(n_tr+tw)] != 'M']
        sy = StandardScaler().fit(valid_q)

        X_all_s = sx.transform(raw_X) if sx else np.empty((n_all,0))
        q_all_s = sy.transform(q_star)

        def gen_seq_full(X_s, q_s, flags_s):
            seqs, labs, last_idxs = [], [], []
            flat_q = q_s.flatten()
            L = len(X_s)
            for i in range(L - tw - hor + 1):
                idx = i + tw + hor - 1
                if flags_s[idx] == 'M': 
                    continue
                x_seq = X_s[i:i+tw]
                if cfg['use_self']:
                    self_seq = flat_q[i:i+tw].reshape(tw,1)
                    x_seq = np.concatenate([x_seq, self_seq], axis=1)
                seqs.append(x_seq)
                labs.append(q_s[idx])
                last_idxs.append(idx)
            return np.stack(seqs), np.stack(labs), np.array(last_idxs)

        all_X, all_y, all_idx = gen_seq_full(X_all_s, q_all_s, flags)

        train_mask = all_idx < n_tr
        valid_mask = (all_idx >= n_tr) & (all_idx < n_tr + n_val)
        test_mask  = all_idx >= n_tr + n_val

        train_X, train_y = all_X[train_mask], all_y[train_mask]
        val_X,   val_y   = all_X[valid_mask], all_y[valid_mask]
        test_X,  test_y  = all_X[test_mask],  all_y[test_mask]

        np.savez(npz_path,
                 train_X=train_X, train_y=train_y,
                 val_X=val_X,     val_y=val_y,
                 test_X=test_X,   test_y=test_y,
                 x_mean=sx.mean_ if sx else np.array([]),
                 x_scale=sx.scale_ if sx else np.array([]),
                 y_mean=sy.mean_, y_scale=sy.scale_)

    def to_loader(X, y, shuffle):
        if X.size == 0:
            raise ValueError(f"No valid sequences for {site_no}.")
        return DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y, dtype=torch.float32)),
            batch_size=cfg['batch_size'], shuffle=shuffle
        )

    train_loader = to_loader(train_X, train_y, cfg['shuffle'])
    valid_loader = to_loader(val_X,   val_y,   False)
    test_loader  = to_loader(test_X,  test_y,  False)
    return train_loader, valid_loader, test_loader, sy, basin, site_no

# ----------------------------
# 4. Train and save (including Early Stopping)
# ----------------------------
def train_on_loader(train_loader, valid_loader, cfg, best_model_path):
    model = FlexibleLSTM(
        input_size=len(cfg['feature_cols']) + (1 if cfg['use_self'] else 0),
        hidden_sizes=cfg['hidden_sizes'],
        output_size=1,
        dropout=cfg['dropout']
    ).to(cfg['device'])
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg['lr_factor'],
        patience=cfg['lr_patience'],
        min_lr=cfg['min_lr'])
    criterion = nn.MSELoss()
    loss_history = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, cfg['epochs']+1):
        # Training phase
        model.train()
        batch_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(cfg['device']), by.to(cfg['device'])
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            optimizer.step()
            batch_losses.append(loss.item())
        train_loss = float(np.mean(batch_losses))

        # Validation phase
        val_loss = evaluate_loss(valid_loader, model, criterion, cfg['device'])
        scheduler.step(val_loss)
        loss_history.append((train_loss, val_loss))

        # Early stopping
        if val_loss + cfg['min_delta'] < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save best model
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg['patience']:
                print(f"Early stopping at epoch {epoch}")
                break

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {lr:.2e}")

    # Load best model
    state = torch.load(best_model_path)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)
    return model, loss_history


# 6) run: outer layer=parameter combination; Inner layer=Multi site
# -------------------------------------------------
all_excels = glob.glob(os.path.join(config['input_data_dir'], '*', '*.xlsx'))

for bs, hs, dp, lr in product(
    param_grid['batch_size'],
    param_grid['hidden_sizes'],
    param_grid['dropout'],
    param_grid['learning_rate']
):
    cfg = copy.deepcopy(config)
    cfg['batch_size']    = int(bs)
    cfg['hidden_sizes']  = list(hs)
    cfg['dropout']       = float(dp)
    cfg['learning_rate'] = float(lr)

    # output paths
    exp_root       = build_exp_root(cfg, bs, hs, dp, lr)
    save_model_dir = os.path.join(exp_root, 'save_models')
    plot_dir_root  = os.path.join(exp_root, 'plots')
    stats_path     = os.path.join(exp_root, 'statistical_values.csv')

    # Root dirs
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(plot_dir_root, exist_ok=True)
    stats = []
    for excel in all_excels:
        try:
            tr_loader, val_loader, te_loader, scaler_y, basin, site_no = \
                prepare_data_from_excel(excel, cfg)
        except ValueError as e:
            print('Skipping', excel, 'due to:', e)
            continue

        model_dir = os.path.join(save_model_dir, basin, site_no)
        os.makedirs(model_dir, exist_ok=True)
        best_model_path = os.path.join(model_dir, 'best.pth')

        # 训练
        model, history = train_on_loader(tr_loader, val_loader, cfg, best_model_path)

        plot_dir = os.path.join(plot_dir_root, basin, site_no)
        os.makedirs(plot_dir, exist_ok=True)

        """
        fig, ax = plt.subplots(figsize=(8,4))
        train_losses, val_losses = zip(*history)
        ax.plot(train_losses, label='Train Loss')
        ax.plot(val_losses,   label='Val Loss')
        ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss'); ax.legend(); ax.grid(True)
        fig.savefig(os.path.join(plot_dir, 'loss_history.png'))
        plt.close(fig)
        """
        
        # Evaluation
        obs_tr, pred_tr = eval_loader(tr_loader, model, scaler_y, cfg['device'])
        obs_val, pred_val = eval_loader(val_loader, model, scaler_y, cfg['device'])
        obs_te, pred_te = eval_loader(te_loader, model, scaler_y, cfg['device'])

        def calc_metrics(obs, pred):
            return {
                'MAE': mean_absolute_error(obs, pred),
                'RMSE': np.sqrt(mean_squared_error(obs, pred)),
                'NSE': nse_safe(pred, obs)
            }
        m_tr  = calc_metrics(obs_tr,  pred_tr)
        m_val = calc_metrics(obs_val, pred_val)
        m_te  = calc_metrics(obs_te,  pred_te)

        plot_metrics(obs_tr, pred_tr, 'Train', m_tr, os.path.join(plot_dir, 'train.png'))
        plot_metrics(obs_val, pred_val, 'Valid', m_val, os.path.join(plot_dir, 'valid.png'))
        plot_metrics(obs_te, pred_te, 'Test',  m_te, os.path.join(plot_dir, 'test.png'))

        stats.append({
            'basin': basin, 'site_no': site_no,
            'MAE_Train': m_tr['MAE'], 'RMSE_Train': m_tr['RMSE'], 'NSE_Train': m_tr['NSE'],
            'MAE_Valid': m_val['MAE'], 'RMSE_Valid': m_val['RMSE'], 'NSE_Valid': m_val['NSE'],
            'MAE_Test':  m_te['MAE'],  'RMSE_Test':  m_te['RMSE'],  'NSE_Test':  m_te['NSE'],
            'time_window': cfg['time_window'],
            'batch_size' : cfg['batch_size'],
            'hidden_sizes': _hs_tag(cfg['hidden_sizes']),
            'dropout'    : cfg['dropout'],
            'learning_rate': _fmt_float(cfg['learning_rate'])
        })
    os.makedirs(os.path.dirname(stats_path) or '.', exist_ok=True)
    pd.DataFrame(stats).to_csv(stats_path, index=False)
    print('Statistics saved to', stats_path)
