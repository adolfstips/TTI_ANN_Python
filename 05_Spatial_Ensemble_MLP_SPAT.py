# ==============================================================================
# Script: 05_Spatial_Ensemble_MLP_SPAT.py
# Title: 10-Fold Spatial Bootstrap Ensemble Reconstruction (Figures 5, 6, & S3)
# Description: Trains the 10-fold spatial MLP_SPAT (Comprehensive) ensemble and
#              reconstructs full-basin TTI trajectories (1955-2020) and spatial
#              maps (1970, 1995, 2020) with pixel-wise uncertainty (SD).
# Reference: Stips et al. (2026) - Environmental Modelling & Software
# License: GPL-3.0
# ==============================================================================

import os
import gc
import numpy as np
import pandas as pd
import torch
from tti_common import (
    set_seeds,
    get_device,
    load_baltic_data,
    assign_kfold_blocks,
    flatten_2d,
    MLPNet,
    train_neural_network,
    predict_batches,
    save_pickle,
)


def main():
    set_seeds(123)
    device = get_device()

    model_name = "MLP_SPAT"
    out_dir = "./Output"
    os.makedirs(out_dir, exist_ok=True)
    rfile = "./GES_Baltic_5year_ALLE_Combined_4D.nc"

    ntime = 14
    years = list(range(1955, 2025, 5))
    n_splits = 10
    block_size = 10
    target_var = "TTI"
    vari = ["Nit", "Pho", "Chla", "DO2_mg", "Secchi", "Temp", "Depth"]
    nvari = len(vari)

    batch_size = 2048
    num_epochs = 150
    patience = 20
    learning_rate = 0.002

    baltic = load_baltic_data(rfile, block_size=block_size, ntime=ntime)
    master_3d = baltic["master_3d"]
    var_to_idx = baltic["var_to_idx"]
    ngoodp = baltic["ngoodp"]
    lon_lat_df = baltic["lon_lat_df"]
    spatial_blocks = baltic["spatial_blocks"]

    fold_ids = assign_kfold_blocks(spatial_blocks, n_splits=n_splits, seed=123)
    all_nodes = np.arange(ngoodp)
    all_times = np.arange(ntime)

    all_predictions_matrix = np.full((ngoodp * ntime, n_splits), np.nan, dtype=np.float64)
    _, Y_all_true = flatten_2d(master_3d, var_to_idx, all_nodes, all_times, vari, target_var)

    print("\n" + "=" * 50)
    print(f"Generating {n_splits}-Fold Spatial Ensemble ({model_name} - Comprehensive)")
    print("=" * 50)

    for k in range(1, n_splits + 1):
        print(f"   Training Ensemble Member {k} / {n_splits}...")
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        val_k = 1 if k == n_splits else k + 1
        train_idx = np.flatnonzero((fold_ids != k) & (fold_ids != val_k))
        val_idx = np.flatnonzero(fold_ids == val_k)

        fold_3d = master_3d.copy()
        for p in vari:
            p_idx = var_to_idx[p]
            mean_tr = float(np.nanmean(fold_3d[train_idx, :, p_idx]))
            sd_tr = float(np.nanstd(fold_3d[train_idx, :, p_idx], ddof=1))
            fold_3d[:, :, p_idx] = (fold_3d[:, :, p_idx] - mean_tr) / sd_tr

        X_train, Y_train = flatten_2d(fold_3d, var_to_idx, train_idx, all_times, vari, target_var)
        X_val, Y_val = flatten_2d(fold_3d, var_to_idx, val_idx, all_times, vari, target_var)
        X_all, _ = flatten_2d(fold_3d, var_to_idx, all_nodes, all_times, vari, target_var)
        del fold_3d

        set_seeds(123 + k)
        model = MLPNet(nvari)
        model = train_neural_network(
            model, X_train, Y_train, X_val, Y_val,
            device=device, batch_size=batch_size, num_epochs=num_epochs,
            patience=patience, learning_rate=learning_rate, fold_label=f"Member {k}"
        )

        all_predictions_matrix[:, k - 1] = predict_batches(model, X_all, device=device)

    print("\nCalculating Fisher-Z ensemble statistics for Figures 5, 6, and S3...")
    z_matrix = np.arctanh(np.clip(all_predictions_matrix, -0.999, 0.999))

    pixel_z_mean = np.mean(z_matrix, axis=1)
    pixel_z_sd = np.std(z_matrix, axis=1, ddof=1)
    pixel_tti_mean = np.tanh(pixel_z_mean)
    pixel_tti_sd = np.tanh(pixel_z_sd)

    final_map_df = pd.DataFrame({
        "Longitude":         np.tile(lon_lat_df["Longitude"].values, ntime),
        "Latitude":          np.tile(lon_lat_df["Latitude"].values, ntime),
        "Year":              np.repeat(years, ngoodp),
        "True_TTI":          Y_all_true,
        "Ensemble_Mean_TTI": pixel_tti_mean,
        "Ensemble_SD_TTI":   pixel_tti_sd,
    })

    target_years = [1970, 1995, 2020]
    fig6_data = final_map_df[final_map_df["Year"].isin(target_years)].reset_index(drop=True)
    map_file = os.path.join(out_dir, f"Ensemble_Map_Data_{model_name}.pkl")
    save_pickle(fig6_data, map_file)

    # Area-Weighted Basin-Wide Time Series (1955-2020) for Figure 5
    cos_weights_single = np.cos(np.deg2rad(lon_lat_df["Latitude"].values))
    w_norm = cos_weights_single / np.sum(cos_weights_single)
    true_z_all = np.arctanh(np.clip(Y_all_true, -0.999, 0.999))

    fold_basin_means = np.zeros((ntime, n_splits), dtype=np.float64)
    true_basin_mean = np.zeros(ntime, dtype=np.float64)
    true_basin_sd = np.zeros(ntime, dtype=np.float64)

    for t_idx in range(ntime):
        r_start = t_idx * ngoodp
        r_end = (t_idx + 1) * ngoodp
        true_basin_mean[t_idx] = float(np.tanh(np.sum(w_norm * true_z_all[r_start:r_end])))
        true_basin_sd[t_idx] = float(np.nanstd(Y_all_true[r_start:r_end], ddof=1))
        for k_i in range(n_splits):
            fold_basin_means[t_idx, k_i] = float(np.tanh(np.sum(w_norm * z_matrix[r_start:r_end, k_i])))

    fig5_timeseries_df = pd.DataFrame({
        "Year":          years,
        "True_TTI_Mean": true_basin_mean,
        "True_TTI_SD":   true_basin_sd,
        "Pred_TTI_Mean": np.mean(fold_basin_means, axis=1),
        "Pred_TTI_CI":   1.96 * np.std(fold_basin_means, axis=1, ddof=1) / np.sqrt(n_splits),
    })

    ts_file = os.path.join(out_dir, f"Ensemble_TimeSeries_{model_name}.pkl")
    save_pickle(fig5_timeseries_df, ts_file)

    print(f"\nSpatial ensemble outputs successfully saved:\n  - {map_file}\n  - {ts_file}")


if __name__ == "__main__":
    main()
