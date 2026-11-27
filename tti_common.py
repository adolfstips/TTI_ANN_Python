# ==============================================================================
# Module: tti_common.py
# Title: Shared PyTorch GPU Training, NetCDF I/O & 2D/3D Shapley Engine
# Description: Core utilities for the TTI_ANN_Python reproducible workflow:
#                - NetCDF4 reader aligned with R terra::sds orientation & blocks
#                - 2D contemporaneous & 3D 7-pentad sequence builders
#                - MLP (128-64) & stacked LSTM (128-64) PyTorch architectures
#                - Full-VRAM GPU training loops (batch_size=2048 for Adam steps,
#                  val_batch_size=65536 for single-pass validation/inference)
#                - Pure-GPU 2D (MLP) & 3D trajectory-level (LSTM) Monte Carlo
#                  Shapley estimators (Strumbelj & Kononenko, 2014) plus exact
#                  2-block temporal inertia Shapley decomposition (Steps 1-6 vs 7)
# Reference: Stips et al. (2026) - Environmental Modelling & Software
# License: GPL-3.0
# ==============================================================================

import os
import copy
import pickle
import zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# NetCDF readers (netCDF4 / xarray) are imported lazily inside load_baltic_data()
# so post-processing scripts (06, 08, 09, 10) can import tti_common without NetCDF I/O.


VAR_COM = {
    "MSFD_Baseline":  ["Nit", "Pho", "Chla", "DO2_mg", "Secchi"],
    "Remote_Sensing": ["Temp", "Chla", "Depth", "Secchi"],
    "Comprehensive":  ["Nit", "Pho", "Chla", "DO2_mg", "Secchi", "Temp", "Depth"],
}


def init_torch_device(num_threads: int = 4, seed: int = 123) -> torch.device:
    """Restrict host OpenMP threads and initialize deterministic RNG seeds."""
    torch.set_num_threads(int(num_threads))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        return torch.device("cuda")
    return torch.device("cpu")
    # Inside init_torch_device():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def set_all_seeds(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def save_pickle(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_netcdf_unzipped(
    nc_path: str = "./GES_Baltic_5year_ALLE_Combined_4D.nc",
    zip_path: str = "./GES_Baltic_5year_ALLE_Combined_4D.nc.zip",
) -> str:
    if not os.path.exists(nc_path) and os.path.exists(zip_path):
        print(f"Unzipping {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(os.path.dirname(nc_path) or "./")
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")
    return nc_path


def _extract_var_3d(raw_arr: np.ndarray, nlat: int, nlon: int, ntime: int, flip_lat: bool) -> np.ndarray:
    """Standardize any NetCDF variable array to shape (ntime, nlat, nlon) with row 0 = North."""
    arr = np.asarray(raw_arr, dtype=np.float64)
    if np.ma.isMaskedArray(raw_arr):
        arr = np.ma.filled(raw_arr.astype(np.float64), np.nan)
    # Remove singleton dimensions (e.g., depth=1 in 4D NetCDF) except when ntime==1
    while arr.ndim > 3:
         squeezed = False
         for ax in range(arr.ndim):
             if arr.shape[ax] == 1 and arr.shape != (1, nlat, nlon):
                 arr = np.squeeze(arr, axis=ax)
                 squeezed = True
                 break
         if not squeezed:
             break

    if arr.ndim == 2 and arr.shape == (nlat, nlon):
        arr = np.repeat(arr[np.newaxis, :, :], ntime, axis=0)
    elif arr.ndim == 2 and arr.shape == (nlon, nlat):
        arr = np.repeat(arr.T[np.newaxis, :, :], ntime, axis=0)
    elif arr.ndim == 3:
        # Identify axes matching ntime, nlat, nlon
        shape = arr.shape
        if shape[1] == nlat and shape[2] == nlon:
            pass
        elif shape[0] == nlat and shape[1] == nlon:
            arr = np.moveaxis(arr, 2, 0)
        elif shape[0] == nlon and shape[1] == nlat:
            arr = np.transpose(arr, (2, 1, 0))
        elif shape[2] == nlat and shape[1] == nlon:
            arr = np.transpose(arr, (0, 2, 1))

        if arr.shape[0] == 1 and ntime > 1:
            arr = np.repeat(arr, ntime, axis=0)
        else:
            arr = arr[:ntime, :, :]
    else:
        raise ValueError(f"Unexpected variable shape {arr.shape} for grid ({nlat}, {nlon})")

    if flip_lat:
        arr = arr[:, ::-1, :]
    return arr


def load_baltic_data(
    nc_path: str = "./GES_Baltic_5year_ALLE_Combined_4D.nc",
    block_size: int = 10,
    ntime: int = 14,
) -> dict:
    """
    Load the 4D Baltic Sea NetCDF dataset and align wet-pixel ordering and
    10x10 (or 20x20) spatial blocks with R terra::sds (row 0 = North, row-major).
    """
    nc_path = ensure_netcdf_unzipped(nc_path)

    try:
        import netCDF4 as nc
        has_netcdf4 = True
    except ImportError:
        has_netcdf4 = False
        import xarray as xr

    raw_vars = {}
    if has_netcdf4:
        with nc.Dataset(nc_path, "r") as ds:
            lat_key = next((k for k in ds.variables if k.lower() in ("lat", "latitude", "y")), None)
            lon_key = next((k for k in ds.variables if k.lower() in ("lon", "longitude", "x")), None)
            if lat_key is None or lon_key is None:
                raise KeyError("Could not identify latitude/longitude coordinate variables in NetCDF.")
            lats = np.asarray(ds.variables[lat_key][:], dtype=np.float64).ravel()
            lons = np.asarray(ds.variables[lon_key][:], dtype=np.float64).ravel()

            coord_keys = {lat_key, lon_key, "time", "lev", "depth_coord", "z", "bnds", "time_bnds"}
            for k, v in ds.variables.items():
                if k in coord_keys or v.ndim < 2:
                    continue
                arr = v[:]
                fill_val = getattr(v, "_FillValue", None)
                if np.ma.isMaskedArray(arr):
                    arr = np.ma.filled(arr.astype(np.float64), np.nan)
                else:
                    arr = np.asarray(arr, dtype=np.float64)
                if fill_val is not None and not np.isnan(fill_val):
                    arr[np.isclose(arr, float(fill_val))] = np.nan
                arr[arr < -1e10] = np.nan
                arr[arr > 1e10] = np.nan
                raw_vars[k] = arr
    else:
        ds = xr.open_dataset(nc_path)
        lat_key = next((k for k in ds.coords if k.lower() in ("lat", "latitude", "y")), None)
        lon_key = next((k for k in ds.coords if k.lower() in ("lon", "longitude", "x")), None)
        lats = ds[lat_key].values.astype(np.float64).ravel()
        lons = ds[lon_key].values.astype(np.float64).ravel()
        for k in ds.data_vars:
            arr = ds[k].values.astype(np.float64)
            arr[np.abs(arr) > 1e10] = np.nan
            raw_vars[k] = arr
        ds.close()

    nlat, nlon = len(lats), len(lons)
    # Match R terra::sds top-to-bottom row indexing (row 0 = North)
    flip_lat = bool(lats[0] < lats[-1])
    if flip_lat:
        lats = lats[::-1].copy()

    var_names = list(raw_vars.keys())
    proc_vars = {
        k: _extract_var_3d(raw_vars[k], nlat=nlat, nlon=nlon, ntime=ntime, flip_lat=flip_lat)
        for k in var_names
    }

    depth_2d = proc_vars["Depth"][0]
    wet_mask = ~np.isnan(depth_2d)
    rows_0, cols_0 = np.where(wet_mask)  # row-major (C-order) matches terra cell index order
    ngoodp = len(rows_0)

    # 1-based row/col block IDs matching R: ceiling(rc[,1] / block_size) _ ceiling(rc[,2] / block_size)
    row_blk = ((rows_0 + 1 + block_size - 1) // block_size).astype(int)
    col_blk = ((cols_0 + 1 + block_size - 1) // block_size).astype(int)
    spatial_blocks = np.array([f"{r}_{c}" for r, c in zip(row_blk, col_blk)], dtype=object)

    master_3d = np.empty((ngoodp, ntime, len(var_names)), dtype=np.float64)
    for j, vname in enumerate(var_names):
        # proc_vars[vname] is (ntime, nlat, nlon) -> transpose wet pixels to (ngoodp, ntime)
        master_3d[:, :, j] = proc_vars[vname][:, rows_0, cols_0].T

    var_idx = {v: i for i, v in enumerate(var_names)}
    if "Chla" in var_idx:
        c_i = var_idx["Chla"]
        master_3d[:, :, c_i] = np.log10(np.maximum(master_3d[:, :, c_i] + 0.01, 1e-6))

    return {
        "master_3d":      master_3d,
        "var_names":      var_names,
        "var_idx":        var_idx,
        "ngoodp":         ngoodp,
        "grid_shape":     (nlat, nlon),
        "lats":           lats,
        "lons":           lons,
        "rows_0":         rows_0,
        "cols_0":         cols_0,
        "cell_lats":      lats[rows_0],
        "cell_lons":      lons[cols_0],
        "spatial_blocks": spatial_blocks,
    }


def assign_kfold_blocks(spatial_blocks: np.ndarray, n_splits: int = 10, seed: int = 123) -> np.ndarray:
    """Assign unique spatial blocks to 1..n_splits folds deterministically."""
    # Preserve first-appearance order of unique blocks (matching R's unique())
    _, first_idx = np.unique(spatial_blocks, return_index=True)
    unique_blocks = spatial_blocks[np.sort(first_idx)]
    n_blocks = len(unique_blocks)

    rng = np.random.default_rng(int(seed))
    base_folds = (np.arange(n_blocks) % n_splits) + 1
    fold_assignments = rng.permutation(base_folds)

    block_to_fold = dict(zip(unique_blocks, fold_assignments))
    return np.array([block_to_fold[b] for b in spatial_blocks], dtype=np.int64)


def assign_train_val_test_blocks(spatial_blocks: np.ndarray, train_frac: float = 0.60, val_frac: float = 0.20, seed: int = 123):
    """60% / 20% / 20% non-overlapping spatial block split for rolling-origin tasks."""
    _, first_idx = np.unique(spatial_blocks, return_index=True)
    unique_blocks = spatial_blocks[np.sort(first_idx)]
    n_blocks = len(unique_blocks)

    train_size = int(np.floor(n_blocks * train_frac))
    val_size = int(np.floor(n_blocks * val_frac))

    rng = np.random.default_rng(int(seed))
    perm_blocks = rng.permutation(unique_blocks)

    train_blocks = set(perm_blocks[:train_size])
    val_blocks = set(perm_blocks[train_size : train_size + val_size])
    test_blocks = set(perm_blocks[train_size + val_size :])

    train_idx = np.array([i for i, b in enumerate(spatial_blocks) if b in train_blocks], dtype=np.int64)
    val_idx = np.array([i for i, b in enumerate(spatial_blocks) if b in val_blocks], dtype=np.int64)
    test_idx = np.array([i for i, b in enumerate(spatial_blocks) if b in test_blocks], dtype=np.int64)
    return train_idx, val_idx, test_idx, len(train_blocks), len(test_blocks)


def flatten_2d(
    data_3d: np.ndarray,
    var_idx: dict,
    node_indices: np.ndarray,
    time_indices: np.ndarray,
    features: list,
    target_var: str = "TTI",
):
    """
    Flatten (nodes, times, features) in time-major / cell-minor order matching R:
    for each t in time_indices, rows are node_indices[0..N-1].
    """
    feat_cols = [var_idx[f] for f in features]
    targ_col = var_idx[target_var]

    # sub_x shape: (n_nodes, n_times, n_feat) -> transpose to (n_times, n_nodes, n_feat) -> reshape
    sub_x = data_3d[np.ix_(node_indices, time_indices, feat_cols)]
    n_nodes, n_times, n_feat = sub_x.shape
    X_mat = np.transpose(sub_x, (1, 0, 2)).reshape(n_nodes * n_times, n_feat).astype(np.float32)

    sub_y = data_3d[np.ix_(node_indices, time_indices, [targ_col])][:, :, 0]
    Y_vec = sub_y.T.reshape(n_nodes * n_times).astype(np.float32)
    return X_mat, Y_vec


def create_sequences_3d(
    data_3d: np.ndarray,
    var_idx: dict,
    node_indices: np.ndarray,
    target_times_0: list,
    features: list,
    target_var: str = "TTI",
    seq_len: int = 7,
):
    """
    Build 3D input tensors (N_samples, seq_len, n_features) and target vector (N_samples,).
    target_times_0 uses 0-based time indices (e.g. range(6, 14) for pentads 7..14 = 1985..2020).
    """
    feat_cols = [var_idx[f] for f in features]
    targ_col = var_idx[target_var]
    valid_t = [t for t in target_times_0 if t >= (seq_len - 1)]

    x_list, y_list = [], []
    for t_step in valid_t:
        t_win = list(range(t_step - seq_len + 1, t_step + 1))
        x_slice = data_3d[np.ix_(node_indices, t_win, feat_cols)]
        y_slice = data_3d[node_indices, t_step, targ_col]
        x_list.append(x_slice.astype(np.float32))
        y_list.append(y_slice.astype(np.float32))

    X_out = np.concatenate(x_list, axis=0)
    Y_out = np.concatenate(y_list, axis=0)
    return X_out, Y_out


def flatten_sequences_for_mlp_seq7(X_3d: np.ndarray) -> np.ndarray:
    """Flatten (N, seq_len, n_feat) -> (N, seq_len * n_feat) for controlled MLP_SEQ7_Windowed."""
    n_samples, seq_len, n_feat = X_3d.shape
    return X_3d.reshape(n_samples, seq_len * n_feat).astype(np.float32)


class MLPNet(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMNet(nn.Module):
    def __init__(self, nvari: int, seq_len: int = 7):
        super().__init__()
        self.seq_len = seq_len
        self.lstm1 = nn.LSTM(input_size=nvari, hidden_size=128, batch_first=True)
        self.drop1 = nn.Dropout(p=0.2)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.drop2 = nn.Dropout(p=0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, _ = self.lstm1(x)
        out1 = self.drop1(out1)
        out2, _ = self.lstm2(out1)
        last_step = self.drop2(out2[:, -1, :])
        return self.fc(last_step)


def train_model_vram(
    model: nn.Module,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
    val_batch_size: int = 65536,
    num_epochs: int = 150,
    patience: int = 20,
    lr: float = 0.002,
    fold_label: str = None,
) -> nn.Module:
    """
    Full-VRAM PyTorch trainer for both 2D MLP and 3D LSTM models.
    Preserves batch_size=2048 for Adam updates while evaluating validation MSE
    in fused 65,536-sample forward passes.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    x_tr = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.as_tensor(Y_train, dtype=torch.float32, device=device).unsqueeze(1)
    x_va = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    y_va = torch.as_tensor(Y_val, dtype=torch.float32, device=device).unsqueeze(1)

    n_train = x_tr.shape[0]
    n_val = x_va.shape[0]
    x_va_chunks = torch.split(x_va, val_batch_size, dim=0)
    y_va_chunks = torch.split(y_va, val_batch_size, dim=0)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_imp = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        x_batches = torch.split(x_tr[perm], batch_size, dim=0)
        y_batches = torch.split(y_tr[perm], batch_size, dim=0)

        for xb, yb in zip(x_batches, y_batches):
            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_sum = 0.0
        with torch.inference_mode():
            for xvb, yvb in zip(x_va_chunks, y_va_chunks):
                v_preds = model(xvb)
                v_loss = criterion(v_preds, yvb)
                val_loss_sum += float(v_loss.item()) * xvb.shape[0]

        avg_val_loss = val_loss_sum / n_val
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_imp = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_imp += 1

        if epochs_no_imp >= patience:
            if fold_label is not None:
                print(f"      {fold_label} early stopped at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model


def predict_batches(
    model: nn.Module,
    X_arr: np.ndarray,
    device: torch.device,
    batch_size: int = 65536,
) -> np.ndarray:
    model.eval()
    n_samples = X_arr.shape[0]
    preds_out = np.empty(n_samples, dtype=np.float64)
    with torch.inference_mode():
        for s_i in range(0, n_samples, batch_size):
            e_i = min(s_i + batch_size, n_samples)
            xb = torch.as_tensor(X_arr[s_i:e_i], dtype=torch.float32, device=device)
            preds_out[s_i:e_i] = model(xb).squeeze(-1).cpu().numpy()
    return preds_out


def calc_eval_metrics(true_tti: np.ndarray, prob_preds: np.ndarray, threshold: float = 0.71) -> dict:
    true_tti = np.asarray(true_tti, dtype=np.float64)
    prob_preds = np.asarray(prob_preds, dtype=np.float64)

    pred_ges = (prob_preds > threshold).astype(np.int32)
    true_ges = (true_tti > threshold).astype(np.int32)

    tp = int(np.sum((pred_ges == 1) & (true_ges == 1)))
    fp = int(np.sum((pred_ges == 1) & (true_ges == 0)))
    fn = int(np.sum((pred_ges == 0) & (true_ges == 1)))
    tn = int(np.sum((pred_ges == 0) & (true_ges == 0)))

    acc = (tp + tn) / max(len(true_ges), 1)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0

    rmse_val = float(np.sqrt(np.mean((true_tti - prob_preds) ** 2)))
    ss_res = float(np.sum((true_tti - prob_preds) ** 2))
    ss_tot = float(np.sum((true_tti - np.mean(true_tti)) ** 2))
    r2_val = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else np.nan

    return {
        "ACC": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "RMSE": rmse_val,
        "R2": r2_val,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "pred_ges": pred_ges,
        "true_ges": true_ges,
    }


def build_confusion_dict(true_ges_all: np.ndarray, pred_ges_all: np.ndarray) -> dict:
    t = np.asarray(true_ges_all, dtype=np.int32)
    p = np.asarray(pred_ges_all, dtype=np.int32)
    return {
        "TP": int(np.sum((p == 1) & (t == 1))),
        "FP": int(np.sum((p == 1) & (t == 0))),
        "FN": int(np.sum((p == 0) & (t == 1))),
        "TN": int(np.sum((p == 0) & (t == 0))),
        "Total": int(len(t)),
    }


def compute_2d_shap_mlp(
    model: nn.Module,
    X_bg_2d: np.ndarray,
    X_test_2d: np.ndarray,
    feature_names: list,
    device: torch.device,
    nsim: int = 50,
    seed: int = 123,
) -> dict:
    """
    Pure-GPU 2D Monte Carlo Shapley Estimator for MLP (Strumbelj & Kononenko, 2014).
    """
    model.eval()
    rng = np.random.default_rng(int(seed))
    n_sub, nvari = X_test_2d.shape
    n_bg = X_bg_2d.shape[0]

    x_bg_t = torch.as_tensor(X_bg_2d, dtype=torch.float32, device=device)
    x_te_t = torch.as_tensor(X_test_2d, dtype=torch.float32, device=device)
    shap_accum = torch.zeros((n_sub, nvari), dtype=torch.float64, device=device)
    row_arange = torch.arange(n_sub, device=device)

    with torch.inference_mode():
        for _ in range(nsim):
            bg_idx = torch.as_tensor(rng.integers(0, n_bg, size=n_sub), dtype=torch.long, device=device)
            x_curr = x_bg_t[bg_idx].clone()

            perms_np = np.argsort(rng.random((n_sub, nvari)), axis=1)
            perms = torch.as_tensor(perms_np, dtype=torch.long, device=device)

            prev_preds = model(x_curr).squeeze(-1).to(torch.float64)
            for s_step in range(nvari):
                feat_col = perms[:, s_step]
                x_curr[row_arange, feat_col] = x_te_t[row_arange, feat_col]
                new_preds = model(x_curr).squeeze(-1).to(torch.float64)
                diff = new_preds - prev_preds
                shap_accum[row_arange, feat_col] += diff
                prev_preds = new_preds

    shap_vals = (shap_accum / float(nsim)).cpu().numpy()
    feat_df = pd.DataFrame(X_test_2d, columns=feature_names)
    global_imp = {f: float(np.mean(np.abs(shap_vals[:, j]))) for j, f in enumerate(feature_names)}
    return {"shap_vals": shap_vals, "feature_data": feat_df, "global_imp": global_imp}


def compute_3d_shap_lstm(
    model: nn.Module,
    X_bg_3d: np.ndarray,
    X_test_3d: np.ndarray,
    feature_names: list,
    device: torch.device,
    nsim: int = 50,
    seed: int = 123,
) -> dict:
    """
    Pure-GPU 3D Trajectory-Level Monte Carlo Shapley Estimator for LSTM
    (Strumbelj & Kononenko, 2014; Reviewer 1, Revision 1, Comment 3).
    Swaps complete 7-pentad predictor trajectories X[i, 0:7, f] rather than flatlines.
    """
    model.eval()
    rng = np.random.default_rng(int(seed))
    n_sub, seq_len, nvari = X_test_3d.shape
    n_bg = X_bg_3d.shape[0]

    x_bg_t = torch.as_tensor(X_bg_3d, dtype=torch.float32, device=device)
    x_te_t = torch.as_tensor(X_test_3d, dtype=torch.float32, device=device)
    shap_accum = torch.zeros((n_sub, nvari), dtype=torch.float64, device=device)
    row_arange = torch.arange(n_sub, device=device)

    with torch.inference_mode():
        for _ in range(nsim):
            bg_idx = torch.as_tensor(rng.integers(0, n_bg, size=n_sub), dtype=torch.long, device=device)
            x_curr = x_bg_t[bg_idx].clone()

            perms_np = np.argsort(rng.random((n_sub, nvari)), axis=1)
            perms = torch.as_tensor(perms_np, dtype=torch.long, device=device)

            prev_preds = model(x_curr).squeeze(-1).to(torch.float64)
            for s_step in range(nvari):
                feat_col = perms[:, s_step]
                for f in range(nvari):
                    mask = feat_col == f
                    if torch.any(mask):
                        x_curr[mask, :, f] = x_te_t[mask, :, f]
                new_preds = model(x_curr).squeeze(-1).to(torch.float64)
                diff = new_preds - prev_preds
                shap_accum[row_arange, feat_col] += diff
                prev_preds = new_preds

    shap_vals = (shap_accum / float(nsim)).cpu().numpy()
    feat_df = pd.DataFrame(X_test_3d[:, seq_len - 1, :], columns=feature_names)
    global_imp = {f: float(np.mean(np.abs(shap_vals[:, j]))) for j, f in enumerate(feature_names)}
    return {"shap_vals": shap_vals, "feature_data": feat_df, "global_imp": global_imp}


def compute_temporal_block_shap(
    model: nn.Module,
    X_bg_3d: np.ndarray,
    X_test_3d: np.ndarray,
    feature_names: list,
    device: torch.device,
    nsim: int = 50,
    seed: int = 123,
) -> dict:
    """
    Exact 2-Player Temporal Block Shapley Decomposition:
    Antecedent History (Steps 1..6; 30 yr) vs Contemporaneous Target Pentad (Step 7; 5 yr).
    """
    model.eval()
    rng = np.random.default_rng(int(seed))
    n_sub, seq_len, nvari = X_test_3d.shape
    n_bg = X_bg_3d.shape[0]
    dyn_cols = [j for j, f in enumerate(feature_names) if f != "Depth"]
    depth_cols = [j for j, f in enumerate(feature_names) if f == "Depth"]

    x_bg_t = torch.as_tensor(X_bg_3d, dtype=torch.float32, device=device)
    x_te_t = torch.as_tensor(X_test_3d, dtype=torch.float32, device=device)

    phi_hist_all = torch.zeros(n_sub, dtype=torch.float64, device=device)
    phi_curr_all = torch.zeros(n_sub, dtype=torch.float64, device=device)
    phi_hist_dyn = torch.zeros(n_sub, dtype=torch.float64, device=device)
    phi_curr_dyn = torch.zeros(n_sub, dtype=torch.float64, device=device)

    with torch.inference_mode():
        v_11 = model(x_te_t).squeeze(-1).to(torch.float64)
        for _ in range(nsim):
            bg_idx = torch.as_tensor(rng.integers(0, n_bg, size=n_sub), dtype=torch.long, device=device)
            x_00 = x_bg_t[bg_idx]

            # A. Full sequence (Steps 0..5 vs Step 6)
            x_10 = x_00.clone()
            x_10[:, : seq_len - 1, :] = x_te_t[:, : seq_len - 1, :]
            x_01 = x_00.clone()
            x_01[:, seq_len - 1, :] = x_te_t[:, seq_len - 1, :]

            v_00 = model(x_00).squeeze(-1).to(torch.float64)
            v_10 = model(x_10).squeeze(-1).to(torch.float64)
            v_01 = model(x_01).squeeze(-1).to(torch.float64)

            phi_hist_all += 0.5 * ((v_10 - v_00) + (v_11 - v_01))
            phi_curr_all += 0.5 * ((v_01 - v_00) + (v_11 - v_10))

            # B. Dynamic predictors only (holding static Depth fixed at test cell)
            x_d00 = x_00.clone()
            for dc in depth_cols:
                x_d00[:, :, dc] = x_te_t[:, :, dc]
            x_d10 = x_d00.clone()
            x_d10[:, : seq_len - 1, dyn_cols] = x_te_t[:, : seq_len - 1, dyn_cols]
            x_d01 = x_d00.clone()
            x_d01[:, seq_len - 1, dyn_cols] = x_te_t[:, seq_len - 1, dyn_cols]

            vd_00 = model(x_d00).squeeze(-1).to(torch.float64)
            vd_10 = model(x_d10).squeeze(-1).to(torch.float64)
            vd_01 = model(x_d01).squeeze(-1).to(torch.float64)

            phi_hist_dyn += 0.5 * ((vd_10 - vd_00) + (v_11 - vd_01))
            phi_curr_dyn += 0.5 * ((vd_01 - vd_00) + (v_11 - vd_10))

    h_all = float(torch.mean(torch.abs(phi_hist_all / float(nsim))).item())
    c_all = float(torch.mean(torch.abs(phi_curr_all / float(nsim))).item())
    h_dyn = float(torch.mean(torch.abs(phi_hist_dyn / float(nsim))).item())
    c_dyn = float(torch.mean(torch.abs(phi_curr_dyn / float(nsim))).item())

    return {
        "History_SHAP":          h_all,
        "Current_SHAP":          c_all,
        "History_Share_Pct":     100.0 * h_all / (h_all + c_all),
        "History_Dyn_Share_Pct": 100.0 * h_dyn / (h_dyn + c_dyn),
    }


def summarize_metrics_df(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scen, grp in metrics_df.groupby("Scenario", sort=False):
        n = len(grp)
        ci_mult = 1.96 / np.sqrt(n) if n > 1 else 0.0
        rows.append({
            "Scenario":  scen,
            "RMSE_Mean": grp["RMSE"].mean(),
            "RMSE_CI":   ci_mult * grp["RMSE"].std(ddof=1),
            "R2_Mean":   grp["R2"].mean(),
            "R2_CI":     ci_mult * grp["R2"].std(ddof=1),
            "ACC_Mean":  grp["ACC"].mean(),
            "ACC_CI":    ci_mult * grp["ACC"].std(ddof=1),
            "Prec_Mean": grp["Precision"].mean(),
            "Prec_CI":   ci_mult * grp["Precision"].std(ddof=1),
            "Rec_Mean":  grp["Recall"].mean(),
            "Rec_CI":    ci_mult * grp["Recall"].std(ddof=1),
            "F1_Mean":   grp["F1"].mean(),
            "F1_CI":     ci_mult * grp["F1"].std(ddof=1),
            "N_Splits":  int(n),
        })
    return pd.DataFrame(rows)


def summarize_shap_df(shap_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scen, feat), grp in shap_df.groupby(["Scenario", "Feature"], sort=False):
        n = len(grp)
        ci_mult = 1.96 / np.sqrt(n) if n > 1 else 0.0
        rows.append({
            "Scenario":  scen,
            "Feature":   feat,
            "SHAP_Mean": grp["SHAP_Mean_Abs"].mean(),
            "SHAP_CI":   ci_mult * grp["SHAP_Mean_Abs"].std(ddof=1),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["Scenario", "SHAP_Mean"], ascending=[True, False]).reset_index(drop=True)
