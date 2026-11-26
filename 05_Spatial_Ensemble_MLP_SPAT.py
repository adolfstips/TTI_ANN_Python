# ==============================================================================
# Script: 05_Spatial_Ensemble_MLP_SPAT.py
# Title: 10-Fold Spatial Bootstrap Ensemble Reconstruction (Figures 5, 6, & S3)
# Description: Trains the 10-fold spatial MLP_SPAT (Comprehensive) ensemble and
#              reconstructs full-basin TTI trajectories (1955-2020) and spatial
#              maps (1970, 1995, 2020) with pixel-wise Fisher-Z uncertainty (SD).
# Reference: Stips et al. (2026) - Environmental Modelling & Software
# License: GPL-3.0
# ==============================================================================

import os
import gc
import numpy as np
import pandas as pd
import torch
from tti_common import (
    VAR_COM,
    init_torch_device,
    set_all_seeds,
    load_baltic_data,
    assign_kfold_blocks,
    flatten_2d,
    MLPNet,
    train_model_vram,
    predict_batches,
    save_pickle,
)


def main() -> None:
    device = init_torch_device(num_threads=4, seed=123)
    model_name = "MLP_SPAT"
    out_dir = "./Output"
    os.makedirs(out_dir, exist_ok=True)

    ntime = 14
    years = list(range(1955, 2025, 5))
    n_splits = 10
    block_size = 10
    target_var = "TTI"
    vari = VAR_COM["Comprehensive"]
    nvari = len(vari)
    batch_size = 2048
    pred_batch_size = 65536
    num_epochs = 150
    patience = 20
    learning_rate = 0.002

    baltic = load_baltic_data("./GES_Baltic_5year_ALLE_Combined_4D.nc", block_size=block_size, ntime=ntime)
    master_3d = baltic["master_3d"]
    var_idx = baltic["var_idx"]
    ngoodp = baltic["ngoodp"]
    cell_lons = baltic["cell_lons"]
    cell_lats = baltic["cell_lats"]
    fold_ids = assign_kfold_blocks(baltic["spatial_blocks"], n_splits=n_splits, seed=123)

    all_nodes = np.arange(ngoodp, dtype=np.int64)
    all_times_0 = list(range(ntime))
    _, true_tti_all = flatten_2d(master_3d, var_idx, all_nodes, all_times_0, vari, target_var)
    true_tti_all = true_tti_all.astype(np.float64)

    all_predictions_matrix = np.empty((ngoodp * ntime, n_splits), dtype=np.float64)

    print("\n" + "=" * 50)
    print(f"Generating {n_splits}-Fold Spatial Ensemble ({model_name} - Comprehensive)")
    print("=" * 50)

    for k in range(1, n_splits + 1):
        print(f"   Training Ensemble Member {k} / {n_splits}...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        val_k = 1 if k == n_splits else k + 1
        train_idx = np.where((fold_ids != k) & (fold_ids != val_k))[0]
        val_idx = np.where(fold_ids == val_k)[0]

        fold_3d = master_3d.copy()
        for p in vari:
            p_i = var_idx[p]
            sub_tr = fold_3d[train_idx, :, p_i]
            mean_tr = float(np.nanmean(sub_tr))
            sd_tr = float(np.nanstd(sub_tr, ddof=1))
            fold_3d[:, :, p_i] = (fold_3d[:, :, p_i] - mean_tr) / sd_tr

        X_tr, Y_tr = flatten_2d(fold_3d, var_idx, train_idx, all_times_0, vari, target_var)
        X_va, Y_va = flatten_2d(fold_3d, var_idx, val_idx, all_times_0, vari, target_var)
        X_all, _ = flatten_2d(fold_3d, var_idx, all_nodes, all_times_0, vari, target_var)
        del fold_3d

        set_all_seeds(123 + k)
        model = MLPNet(nvari)
        model = train_model_vram(
            model=model,
            X_train=X_tr,
            Y_train=Y_tr,
            X_val=X_va,
            Y_val=Y_va,
            device=device,
            batch_size=batch_size,
            val_batch_size=pred_batch_size,
            num_epochs=num_epochs,
            patience=patience,
            lr=learning_rate,
            fold_label=f"Member {k}",
        )

        all_predictions_matrix[:, k - 1] = predict_batches(
            model, X_all, device=device, batch_size=pred_batch_size
        )

    print("\nCalculating Fisher-Z ensemble statistics for Figures 5, 6, and S3...")
    z_matrix = np.arctanh(np.clip(all_predictions_matrix, -0.999, 0.999))
    pixel_z_mean = np.mean(z_matrix, axis=1)
    pixel_z_sd = np.std(z_matrix, axis=1, ddof=1)
    pixel_tti_mean = np.tanh(pixel_z_mean)
    pixel_tti_sd = np.tanh(pixel_z_sd)

    final_map_df = pd.DataFrame({
        "Longitude":         np.tile(cell_lons, ntime),
        "Latitude":          np.tile(cell_lats, ntime),
        "Year":              np.repeat(years, ngoodp),
        "True_TTI":          true_tti_all,
        "Ensemble_Mean_TTI": pixel_tti_mean,
        "Ensemble_SD_TTI":   pixel_tti_sd,
    })

    fig6_data = final_map_df[final_map_df["Year"].isin([1970, 1995, 2020])].reset_index(drop=True)
    map_file = os.path.join(out_dir, f"Ensemble_Map_Data_{model_name}.pkl")
    save_pickle(fig6_data, map_file)

    w_cell = np.cos(np.deg2rad(cell_lats))
    w_cell = w_cell / np.sum(w_cell)
    true_z = np.arctanh(np.clip(true_tti_all, -0.999, 0.999))

    fold_basin_means = np.empty((ntime, n_splits), dtype=np.float64)
    true_basin_mean = np.empty(ntime, dtype=np.float64)
    true_basin_sd = np.empty(ntime, dtype=np.float64)

    for t_idx in range(ntime):
        s_i = t_idx * ngoodp
        e_i = (t_idx + 1) * ngoodp
        true_basin_mean[t_idx] = float(np.tanh(np.sum(w_cell * true_z[s_i:e_i])))
        true_basin_sd[t_idx] = float(np.std(true_tti_all[s_i:e_i], ddof=1))
        for k_i in range(n_splits):
            fold_basin_means[t_idx, k_i] = float(np.tanh(np.sum(w_cell * z_matrix[s_i:e_i, k_i])))

    fig5_timeseries_df = pd.DataFrame({
        "Year":          years,
        "True_TTI_Mean": true_basin_mean,
        "True_TTI_SD":   true_basin_sd,
        "Pred_TTI_Mean": np.mean(fold_basin_means, axis=1),
        "Pred_TTI_CI":   1.96 * np.std(fold_basin_means, axis=1, ddof=1) / np.sqrt(n_splits),
    })

    ts_file = os.path.join(out_dir, f"Ensemble_TimeSeries_{model_name}.pkl")
    save_pickle(fig5_timeseries_df, ts_file)

    print("\nSpatial ensemble outputs successfully saved:")
    print(f"  - {map_file}\n  - {ts_file}")


if __name__ == "__main__":
    main()
