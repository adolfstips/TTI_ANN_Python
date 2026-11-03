# ==============================================================================
# Module: tti_common.py
# Title: Shared Utilities for TTI Deep Learning Framework (Data, Models, SHAP)
# Reference: Stips et al. (2026) - Environmental Modelling & Software
# License: GPL-3.0
# ==============================================================================

import os
import gc
import copy
import random
import zipfile
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# Flexible NetCDF backend support (netCDF4, xarray, or h5py)
try:
    import netCDF4 as nc
    HAS_NETCDF4 = True
except ImportError:
    nc = None
    HAS_NETCDF4 = False

VAR_COM = {
    "MSFD_Baseline":  ["Nit", "Pho", "Chla", "DO2_mg", "Secchi"],
    "Remote_Sensing": ["Temp", "Chla", "Depth", "Secchi"],
    "Comprehensive":  ["Nit", "Pho", "Chla", "DO2_mg", "Secchi", "Temp", "Depth"],
}


def set_seeds(seed: int = 123):
    """Set deterministic seeds across Python, NumPy, and PyTorch (CPU & CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU, clearing cache."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_nc_file(nc_path: str = "./GES_Baltic_5year_ALLE_Combined_4D.nc") -> str:
    """Locate or automatically unzip the input NetCDF dataset."""
    if os.path.exists(nc_path):
        return nc_path
    zip_path = nc_path + ".zip"
    if os.path.exists(zip_path):
        print(f"Unzipping {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(os.path.dirname(nc_path) or "./")
        if os.path.exists(nc_path):
            return nc_path
    # Check parent directory as fallback
    parent_nc = os.path.join("..", os.path.basename(nc_path))
    if os.path.exists(parent_nc):
        return parent_nc
    parent_zip = parent_nc + ".zip"
    if os.path.exists(parent_zip):
        print(f"Unzipping {parent_zip} ...")
        with zipfile.ZipFile(parent_zip, "r") as zf:
            zf.extractall("./")
        if os.path.exists(nc_path):
            return nc_path
    raise FileNotFoundError(
        f"Could not find {nc_path} or {zip_path}. "
        "Please place GES_Baltic_5year_ALLE_Combined_4D.nc (or .nc.zip) in the repository root."
    )


def load_baltic_data(
    nc_path: str = "./GES_Baltic_5year_ALLE_Combined_4D.nc",
    block_size: int = 10,
    ntime: int = 14,
):
    """
    Load the 4D GETM-ERGOM NetCDF file and orient the raster grid identically
    to R's `terra::sds()` (Row 1 = North / max latitude, Col 1 = West / min longitude,
    flattened row-wise).
    """
    nc_path = ensure_nc_file(nc_path)
    if not HAS_NETCDF4:
        try:
            import xarray as xr
            xds = xr.open_dataset(nc_path)
            coord_keys = list(xds.coords.keys()) + list(xds.variables.keys())
            lon_name = next((k for k in coord_keys if k.lower() in ("lon", "longitude", "x")), None)
            lat_name = next((k for k in coord_keys if k.lower() in ("lat", "latitude", "y")), None)
            lons = np.array(xds[lon_name].values, dtype=np.float64)
            lats = np.array(xds[lat_name].values, dtype=np.float64)
            flip_lat = lats[0] < lats[-1]
            flip_lon = lons[0] > lons[-1]
            if flip_lat:
                lats = lats[::-1]
            if flip_lon:
                lons = lons[::-1]
            nlat, nlon = len(lats), len(lons)
            var_names = [v for v in xds.data_vars.keys() if xds[v].ndim >= 2]
            def read_xr_3d(vname):
                arr = np.array(xds[vname].values, dtype=np.float64)
                arr[np.abs(arr) > 1e15] = np.nan
                arr = np.squeeze(arr)
                if arr.ndim == 2:
                    arr = np.repeat(arr[np.newaxis, :, :], ntime, axis=0)
                elif arr.ndim == 3:
                    if arr.shape[0] != ntime:
                        t_axis = next((i for i, s in enumerate(arr.shape) if s == ntime), 0)
                        arr = np.moveaxis(arr, t_axis, 0)
                    if arr.shape[1] == nlon and arr.shape[2] == nlat:
                        arr = np.swapaxes(arr, 1, 2)
                if flip_lat:
                    arr = arr[:, ::-1, :]
                if flip_lon:
                    arr = arr[:, :, ::-1]
                return arr
            depth_slice0 = read_xr_3d("Depth")[0, :, :]
            ii_good = np.flatnonzero(~np.isnan(depth_slice0.ravel(order="C")))
            ngoodp = len(ii_good)
            rows_0, cols_0 = np.unravel_index(ii_good, (nlat, nlon))
            spatial_blocks = np.array(
                [f"{int(np.ceil((r + 1) / block_size))}_{int(np.ceil((c + 1) / block_size))}" for r, c in zip(rows_0, cols_0)]
            )
            lon_lat_df = pd.DataFrame({"Longitude": lons[cols_0], "Latitude": lats[rows_0]})
            master_3d = np.full((ngoodp, ntime, len(var_names)), np.nan, dtype=np.float64)
            var_to_idx = {v: i for i, v in enumerate(var_names)}
            for i, vname in enumerate(var_names):
                flat_2d = read_xr_3d(vname).reshape(ntime, -1, order="C").T
                master_3d[:, :, i] = flat_2d[ii_good, :]
            xds.close()
            if "Chla" in var_to_idx:
                c_idx = var_to_idx["Chla"]
                master_3d[:, :, c_idx] = np.log10(np.maximum(master_3d[:, :, c_idx] + 0.01, 1e-6))
            return {
                "master_3d": master_3d, "var_names": var_names, "var_to_idx": var_to_idx,
                "ii_good": ii_good, "ngoodp": ngoodp, "ntime": ntime,
                "spatial_blocks": spatial_blocks, "lon_lat_df": lon_lat_df,
                "grid_shape": (nlat, nlon), "lats": lats, "lons": lons,
                "rows_0": rows_0, "cols_0": cols_0,
            }
        except ImportError:
            raise ImportError("Please install `netCDF4` or `xarray` (`pip install netCDF4 xarray`).")
    ds = nc.Dataset(nc_path, mode="r")

    # Identify coordinate variables
    coord_keys = list(ds.variables.keys())
    lon_name = next((k for k in coord_keys if k.lower() in ("lon", "longitude", "x")), None)
    lat_name = next((k for k in coord_keys if k.lower() in ("lat", "latitude", "y")), None)
    if lon_name is None or lat_name is None:
        raise ValueError(f"Could not identify lon/lat coordinates in {coord_keys}")

    lons = np.array(ds.variables[lon_name][:], dtype=np.float64)
    lats = np.array(ds.variables[lat_name][:], dtype=np.float64)

    # In terra::sds(), row 0 is North (descending latitude) and col 0 is West (ascending longitude)
    flip_lat = lats[0] < lats[-1]
    flip_lon = lons[0] > lons[-1]
    if flip_lat:
        lats = lats[::-1]
    if flip_lon:
        lons = lons[::-1]

    nlat = len(lats)
    nlon = len(lons)

    # Identify data variables (exclude dimension coordinate variables)
    dim_names = set(ds.dimensions.keys()) | {lon_name, lat_name, "time", "Time", "depth", "z", "lev"}
    var_names = [v for v in ds.variables.keys() if v not in dim_names and len(ds.variables[v].shape) >= 2]

    def read_aligned_3d(vname):
        var_obj = ds.variables[vname]
        raw = var_obj[:]
        if np.ma.isMaskedArray(raw):
            arr = np.array(raw.filled(np.nan), dtype=np.float64)
        else:
            arr = np.array(raw, dtype=np.float64)
        # Replace extreme fill values with NaN
        fill_val = getattr(var_obj, "_FillValue", None)
        missing_val = getattr(var_obj, "missing_value", None)
        if fill_val is not None:
            arr[np.isclose(arr, float(fill_val), equal_nan=True)] = np.nan
        if missing_val is not None:
            arr[np.isclose(arr, float(missing_val), equal_nan=True)] = np.nan
        arr[np.abs(arr) > 1e15] = np.nan

        # Squeeze singleton dimensions (e.g., surface depth = 1 in 4D NetCDF)
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            # Broadcast static 2D layer across ntime
            arr = np.repeat(arr[np.newaxis, :, :], ntime, axis=0)
        elif arr.ndim == 3:
            # Ensure shape is (ntime, nlat, nlon)
            if arr.shape[0] != ntime:
                t_axis = next((i for i, s in enumerate(arr.shape) if s == ntime), 0)
                arr = np.moveaxis(arr, t_axis, 0)
            if arr.shape[1] == nlon and arr.shape[2] == nlat:
                arr = np.swapaxes(arr, 1, 2)

        if flip_lat:
            arr = arr[:, ::-1, :]
        if flip_lon:
            arr = arr[:, :, ::-1]
        return arr

    depth_3d = read_aligned_3d("Depth")
    depth_slice0 = depth_3d[0, :, :]  # (nlat, nlon)

    # Flatten row-wise (C-order: West-to-East along each row from North to South, matching terra::values)
    ii_good = np.flatnonzero(~np.isnan(depth_slice0.ravel(order="C")))
    ngoodp = len(ii_good)

    rows_0, cols_0 = np.unravel_index(ii_good, (nlat, nlon))
    rows_1 = rows_0 + 1  # 1-based row index matching terra::rowColFromCell
    cols_1 = cols_0 + 1  # 1-based col index matching terra::rowColFromCell

    spatial_blocks = np.array(
        [f"{int(np.ceil(r / block_size))}_{int(np.ceil(c / block_size))}" for r, c in zip(rows_1, cols_1)]
    )

    lon_lat_df = pd.DataFrame({
        "Longitude": lons[cols_0],
        "Latitude":  lats[rows_0],
    })

    num_vars = len(var_names)
    master_3d = np.full((ngoodp, ntime, num_vars), np.nan, dtype=np.float64)
    var_to_idx = {v: i for i, v in enumerate(var_names)}

    for i, vname in enumerate(var_names):
        arr_3d = read_aligned_3d(vname)  # (ntime, nlat, nlon)
        # Flatten each time slice to (nlat*nlon) and index ii_good -> (ngoodp, ntime)
        flat_2d = arr_3d.reshape(ntime, -1, order="C").T  # (nlat*nlon, ntime)
        master_3d[:, :, i] = flat_2d[ii_good, :]

    ds.close()

    # Log-transform Chlorophyll-a identically to R scripts: log10(Chla + 0.01)
    if "Chla" in var_to_idx:
        c_idx = var_to_idx["Chla"]
        master_3d[:, :, c_idx] = np.log10(np.maximum(master_3d[:, :, c_idx] + 0.01, 1e-6))

    return {
        "master_3d":      master_3d,
        "var_names":      var_names,
        "var_to_idx":     var_to_idx,
        "ii_good":        ii_good,
        "ngoodp":         ngoodp,
        "ntime":          ntime,
        "spatial_blocks": spatial_blocks,
        "lon_lat_df":     lon_lat_df,
        "grid_shape":     (nlat, nlon),
        "lats":           lats,
        "lons":           lons,
        "rows_0":         rows_0,
        "cols_0":         cols_0,
    }


def assign_kfold_blocks(spatial_blocks: np.ndarray, n_splits: int = 10, seed: int = 123) -> np.ndarray:
    """
    Assign unique spatial blocks to `n_splits` non-overlapping folds (1..n_splits),
    returning the Fold_ID (1..n_splits) for each marine wet pixel.
    """
    unique_blocks = list(dict.fromkeys(spatial_blocks))
    n_blocks = len(unique_blocks)
    rep_folds = np.resize(np.arange(1, n_splits + 1), n_blocks)

    rng = np.random.default_rng(seed)
    fold_assignments = rng.permutation(rep_folds)
    block_to_fold = dict(zip(unique_blocks, fold_assignments))
    return np.array([block_to_fold[b] for b in spatial_blocks], dtype=np.int32)


def assign_60_20_20_blocks(spatial_blocks: np.ndarray, seed: int = 123):
    """
    Partition unique spatial blocks into 60% Train, 20% Val, and 20% Test subsets
    for the temporal rolling-origin hindcast scripts (03 and 04).
    """
    unique_blocks = np.array(list(dict.fromkeys(spatial_blocks)))
    n_blocks = len(unique_blocks)
    train_size = int(np.floor(n_blocks * 0.60))
    val_size = int(np.floor(n_blocks * 0.20))

    rng = np.random.default_rng(seed)
    perm_blocks = rng.permutation(unique_blocks)
    train_blocks = set(perm_blocks[:train_size])
    val_blocks = set(perm_blocks[train_size : train_size + val_size])
    test_blocks = set(perm_blocks[train_size + val_size :])

    train_idx = np.flatnonzero([b in train_blocks for b in spatial_blocks])
    val_idx = np.flatnonzero([b in val_blocks for b in spatial_blocks])
    test_idx = np.flatnonzero([b in test_blocks for b in spatial_blocks])
    return train_idx, val_idx, test_idx, train_size, len(test_blocks)


def flatten_2d(
    master_3d: np.ndarray,
    var_to_idx: dict,
    node_indices: np.ndarray,
    time_indices: np.ndarray,
    features: list,
    target_var: str = "TTI",
):
    """
    Flatten 3D (node, time, var) slices into 2D matrices (samples, features)
    in column-major time order (all nodes for t_1, then all nodes for t_2, ...),
    matching R's `apply(extracted_data, 3, c)`.
    """
    feat_idx = [var_to_idx[f] for f in features]
    targ_idx = var_to_idx[target_var]

    sub_x = master_3d[np.ix_(node_indices, time_indices, feat_idx)]  # (n_nodes, n_times, n_feat)
    sub_y = master_3d[np.ix_(node_indices, time_indices, [targ_idx])][:, :, 0]  # (n_nodes, n_times)

    # Transpose (n_times, n_nodes, n_feat) then reshape so node varies fastest per time step
    X = np.transpose(sub_x, (1, 0, 2)).reshape(-1, len(features))
    Y = sub_y.T.reshape(-1)
    return X.astype(np.float32), Y.astype(np.float32)


def create_sequences(
    master_3d: np.ndarray,
    var_to_idx: dict,
    node_indices: np.ndarray,
    target_time_steps: list,
    features: list,
    target_var: str = "TTI",
    sequence_length: int = 7,
):
    """
    Construct 3D overlapping chronological sequences of shape
    [samples, sequence_length = 7, features] for the LSTM architecture.
    Note: `target_time_steps` uses 0-based time indices (0..13, so valid t >= sequence_length - 1).
    """
    feat_idx = [var_to_idx[f] for f in features]
    targ_idx = var_to_idx[target_var]

    x_list, y_list = [], []
    for t in target_time_steps:
        if t < (sequence_length - 1):
            continue
        t_slice = list(range(t - sequence_length + 1, t + 1))
        seq_x = master_3d[np.ix_(node_indices, t_slice, feat_idx)]  # (n_nodes, seq_len, n_feat)
        seq_y = master_3d[node_indices, t, targ_idx]                # (n_nodes,)
        x_list.append(seq_x)
        y_list.append(seq_y)

    X = np.concatenate(x_list, axis=0).astype(np.float32)
    Y = np.concatenate(y_list, axis=0).astype(np.float32)
    return X, Y


class MLPNet(nn.Module):
    """Feedforward Multi-Layer Perceptron (128-64) with 0.2 Dropout."""

    def __init__(self, nvari: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nvari, 128),
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
    """Stacked Recurrent LSTM Network (128-64, Sequence-to-One) with 0.2 Dropout."""

    def __init__(self, nvari: int, sequence_length: int = 7):
        super().__init__()
        self.sequence_length = sequence_length
        self.lstm1 = nn.LSTM(input_size=nvari, hidden_size=128, batch_first=True)
        self.drop1 = nn.Dropout(p=0.2)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.drop2 = nn.Dropout(p=0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, _ = self.lstm1(x)
        out1 = self.drop1(out1)
        out2, _ = self.lstm2(out1)
        # Explicitly extract final time step (sequence_length - 1) for Sequence-to-One mapping
        last_step = out2[:, self.sequence_length - 1, :]
        last_step = self.drop2(last_step)
        return self.fc(last_step)


def train_neural_network(
    model: nn.Module,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
    num_epochs: int = 150,
    patience: int = 20,
    learning_rate: float = 0.002,
    fold_label: str = "",
) -> nn.Module:
    """
    Train MLP or LSTM model using Adam and MSE loss with dynamic early stopping
    evaluated strictly on the isolated validation set.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    x_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(Y_train, dtype=torch.float32, device=device).view(-1, 1)
    x_va = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_va = torch.tensor(Y_val, dtype=torch.float32, device=device).view(-1, 1)

    n_train = x_tr.shape[0]
    n_val = x_va.shape[0]
    n_train_batches = (n_train - 1) // batch_size + 1
    n_val_batches = (n_val - 1) // batch_size + 1

    best_loss = float("inf")
    best_state = None
    epochs_no_imp = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)

        for b in range(n_train_batches):
            start_i = b * batch_size
            end_i = min((b + 1) * batch_size, n_train)
            idx = perm[start_i:end_i]

            optimizer.zero_grad()
            preds = model(x_tr[idx])
            loss = criterion(preds, y_tr[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b in range(n_val_batches):
                start_i = b * batch_size
                end_i = min((b + 1) * batch_size, n_val)
                v_preds = model(x_va[start_i:end_i])
                v_loss = criterion(v_preds, y_va[start_i:end_i])
                val_loss += v_loss.item()

        avg_val_loss = val_loss / n_val_batches
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_no_imp = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_imp += 1

        if epochs_no_imp >= patience:
            if fold_label:
                print(f"      {fold_label} early stopped at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
    model.eval()
    return model


def predict_batches(model: nn.Module, X_data: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    """Run batched forward pass on CPU/GPU and return 1D numpy predictions."""
    model.eval()
    n = X_data.shape[0]
    preds_list = []
    with torch.no_grad():
        for start_i in range(0, n, batch_size):
            end_i = min(start_i + batch_size, n)
            batch_t = torch.tensor(X_data[start_i:end_i], dtype=torch.float32, device=device)
            out = model(batch_t).detach().cpu().numpy().reshape(-1)
            preds_list.append(out)
    return np.concatenate(preds_list, axis=0)


def evaluate_predictions(true_tti: np.ndarray, prob_preds: np.ndarray, tt_threshold: float = 0.71) -> dict:
    """Compute continuous (RMSE, R2) and binary GES classification metrics (ACC, Precision, Recall, F1)."""
    pred_ges = (prob_preds > tt_threshold).astype(np.int32)
    true_ges = (true_tti > tt_threshold).astype(np.int32)

    tp = int(np.sum((pred_ges == 1) & (true_ges == 1)))
    fp = int(np.sum((pred_ges == 1) & (true_ges == 0)))
    fn = int(np.sum((pred_ges == 0) & (true_ges == 1)))
    tn = int(np.sum((pred_ges == 0) & (true_ges == 0)))

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    mse_raw = float(np.mean((true_tti - prob_preds) ** 2))
    rmse_val = float(np.sqrt(mse_raw))
    ss_res = float(np.sum((true_tti - prob_preds) ** 2))
    ss_tot = float(np.sum((true_tti - np.mean(true_tti)) ** 2))
    r2_val = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "RMSE":      rmse_val,
        "R2":        r2_val,
        "ACC":       acc,
        "Precision": prec,
        "Recall":    rec,
        "F1":        f1,
        "pred_ges":  pred_ges,
        "true_ges":  true_ges,
    }


def compute_fastshap_mc(
    model: nn.Module,
    X_bg_2d: np.ndarray,
    X_test_2d: np.ndarray,
    feature_names: list,
    device: torch.device,
    nsim: int = 50,
    is_lstm: bool = False,
    sequence_length: int = 7,
    seed: int = 123,
):
    """
    GPU-accelerated Monte Carlo Shapley Value estimator (Strumbelj & Kononenko, 2014),
    mathematically identical to R's `fastshap::explain(..., nsim = 50)`.
    For LSTM models, replicates the 2D feature perturbation across `sequence_length`
    steps identically to the R `predict_fun` wrapper.
    """
    rng = np.random.default_rng(seed)
    n_test, n_feat = X_test_2d.shape
    n_bg = X_bg_2d.shape[0]

    shap_accum = np.zeros((n_test, n_feat), dtype=np.float64)

    def eval_2d(mat_2d: np.ndarray) -> np.ndarray:
        if is_lstm:
            # Replicate 2D [N, n_feat] across sequence_length -> [N, seq_len, n_feat]
            mat_3d = np.repeat(mat_2d[:, np.newaxis, :], sequence_length, axis=1)
            return predict_batches(model, mat_3d, device=device, batch_size=4096)
        return predict_batches(model, mat_2d, device=device, batch_size=4096)

    for _ in range(nsim):
        bg_indices = rng.integers(0, n_bg, size=n_test)
        X_bg_sample = X_bg_2d[bg_indices]  # (n_test, n_feat)

        # Generate random feature orderings per test sample
        perms = np.argsort(rng.random((n_test, n_feat)), axis=1)

        # Walk along the permutation chain from all-background to all-test
        X_curr = X_bg_sample.copy()
        prev_preds = eval_2d(X_curr)

        for step in range(n_feat):
            feat_col = perms[:, step]
            row_idx = np.arange(n_test)
            X_curr[row_idx, feat_col] = X_test_2d[row_idx, feat_col]
            new_preds = eval_2d(X_curr)
            diff = new_preds - prev_preds
            shap_accum[row_idx, feat_col] += diff
            prev_preds = new_preds

    shap_vals = (shap_accum / float(nsim)).astype(np.float64)
    feature_df = pd.DataFrame(X_test_2d, columns=feature_names)
    global_imp = {feat: float(np.mean(np.abs(shap_vals[:, i]))) for i, feat in enumerate(feature_names)}
    return shap_vals, feature_df, global_imp


def aggregate_summary_tables(metrics_df: pd.DataFrame, shap_df: pd.DataFrame):
    """Aggregate fold/origin metrics and SHAP values into Mean and 95% CI summaries."""
    def ci95(series):
        n = series.count()
        if n <= 1:
            return 0.0
        return float(1.96 * series.std(ddof=1) / np.sqrt(n))

    grouped_m = []
    for scen, grp in metrics_df.groupby("Scenario", sort=False):
        grouped_m.append({
            "Scenario":  scen,
            "RMSE_Mean": float(grp["RMSE"].mean()),
            "RMSE_CI":   ci95(grp["RMSE"]),
            "R2_Mean":   float(grp["R2"].mean()),
            "R2_CI":     ci95(grp["R2"]),
            "ACC_Mean":  float(grp["ACC"].mean()),
            "ACC_CI":    ci95(grp["ACC"]),
            "Prec_Mean": float(grp["Precision"].mean()),
            "Prec_CI":   ci95(grp["Precision"]),
            "Rec_Mean":  float(grp["Recall"].mean()),
            "Rec_CI":    ci95(grp["Recall"]),
            "F1_Mean":   float(grp["F1"].mean()),
            "F1_CI":     ci95(grp["F1"]),
            "N_Splits":  int(len(grp)),
        })
    final_metrics = pd.DataFrame(grouped_m)

    grouped_s = []
    for (scen, feat), grp in shap_df.groupby(["Scenario", "Feature"], sort=False):
        grouped_s.append({
            "Scenario":  scen,
            "Feature":   feat,
            "SHAP_Mean": float(grp["SHAP_Mean_Abs"].mean()),
            "SHAP_CI":   ci95(grp["SHAP_Mean_Abs"]),
        })
    final_shap = (
        pd.DataFrame(grouped_s)
        .sort_values(["Scenario", "SHAP_Mean"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return final_metrics, final_shap


def build_confusion_matrix_df(global_targets: list, global_preds: list) -> pd.DataFrame:
    """Build a 2x2 confusion matrix table compatible with cvms / Script 10."""
    t_arr = np.array(global_targets, dtype=np.int32)
    p_arr = np.array(global_preds, dtype=np.int32)
    rows = []
    for p_val in [1, 0]:
        for t_val in [1, 0]:
            count = int(np.sum((p_arr == p_val) & (t_arr == t_val)))
            rows.append({"Prediction": p_val, "Target": t_val, "N": count})
    return pd.DataFrame(rows)


def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path) or "./", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
