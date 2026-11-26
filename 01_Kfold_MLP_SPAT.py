# ==============================================================================
# Script: 01_Kfold_MLP_SPAT.py
# Title: 10-Fold Spatial Block Cross-Validation for Feedforward Network (MLP_SPAT)
# Description: Evaluates spatial gap-filling across 10 non-overlapping 10x10
#              pixel blocks (80% Train / 10% Val / 10% Test) with pure-GPU 2D
#              Monte Carlo Shapley attributions (Strumbelj & Kononenko, 2014).
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
    calc_eval_metrics,
    build_confusion_dict,
    compute_2d_shap_mlp,
    summarize_metrics_df,
    summarize_shap_df,
    save_pickle,
)


def main() -> None:
    device = init_torch_device(num_threads=4, seed=123)
    model_name = "MLP_SPAT"
    out_dir = "./Output"
    os.makedirs(out_dir, exist_ok=True)

    ntime = 14
    n_splits = 10
    block_size = 10
    target_var = "TTI"
    tt_threshold = 0.71
    batch_size = 2048
    pred_batch_size = 65536
    num_epochs = 150
    patience = 20
    learning_rate = 0.002
    nsim_shap = 50

    baltic = load_baltic_data("./GES_Baltic_5year_ALLE_Combined_4D.nc", block_size=block_size, ntime=ntime)
    master_3d = baltic["master_3d"]
    var_idx = baltic["var_idx"]
    fold_ids = assign_kfold_blocks(baltic["spatial_blocks"], n_splits=n_splits, seed=123)

    metrics_rows = []
    shap_rows = []
    raw_shap_export = {s: {"shap_vals": [], "feature_data": []} for s in VAR_COM}
    global_targets = []
    global_preds = []
    all_times_0 = list(range(ntime))

    for sim_name, vari in VAR_COM.items():
        nvari = len(vari)
        print("\n" + "=" * 50)
        print(f"Starting 10-Fold Spatial CV ({model_name}) for: {sim_name}")
        print("=" * 50)

        for k in range(1, n_splits + 1):
            print(f"   Running Fold {k} / {n_splits}...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            val_k = 1 if k == n_splits else k + 1
            train_idx = np.where((fold_ids != k) & (fold_ids != val_k))[0]
            val_idx = np.where(fold_ids == val_k)[0]
            test_idx = np.where(fold_ids == k)[0]

            fold_3d = master_3d.copy()
            for p in vari:
                p_i = var_idx[p]
                sub_tr = fold_3d[train_idx, :, p_i]
                mean_tr = float(np.nanmean(sub_tr))
                sd_tr = float(np.nanstd(sub_tr, ddof=1))
                fold_3d[:, :, p_i] = (fold_3d[:, :, p_i] - mean_tr) / sd_tr

            X_tr, Y_tr = flatten_2d(fold_3d, var_idx, train_idx, all_times_0, vari, target_var)
            X_va, Y_va = flatten_2d(fold_3d, var_idx, val_idx, all_times_0, vari, target_var)
            X_te, Y_te = flatten_2d(fold_3d, var_idx, test_idx, all_times_0, vari, target_var)
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
                fold_label=f"Fold {k}",
            )

            prob_preds = predict_batches(model, X_te, device=device, batch_size=pred_batch_size)
            ev = calc_eval_metrics(Y_te, prob_preds, threshold=tt_threshold)

            if sim_name == "Comprehensive":
                global_targets.append(ev["true_ges"])
                global_preds.append(ev["pred_ges"])

            metrics_rows.append({
                "Scenario":  sim_name,
                "Fold":      k,
                "RMSE":      ev["RMSE"],
                "R2":        ev["R2"],
                "ACC":       ev["ACC"],
                "Precision": ev["Precision"],
                "Recall":    ev["Recall"],
                "F1":        ev["F1"],
            })

            rng_sub = np.random.default_rng(123 + k)
            bg_idx = rng_sub.choice(X_tr.shape[0], size=min(500, X_tr.shape[0]), replace=False)
            te_idx = rng_sub.choice(X_te.shape[0], size=min(2000, X_te.shape[0]), replace=False)

            shap_res = compute_2d_shap_mlp(
                model=model,
                X_bg_2d=X_tr[bg_idx],
                X_test_2d=X_te[te_idx],
                feature_names=vari,
                device=device,
                nsim=nsim_shap,
                seed=123 + k,
            )

            raw_shap_export[sim_name]["shap_vals"].append(shap_res["shap_vals"])
            raw_shap_export[sim_name]["feature_data"].append(shap_res["feature_data"])

            for feat in vari:
                shap_rows.append({
                    "Scenario":      sim_name,
                    "Fold":          k,
                    "Feature":       feat,
                    "SHAP_Mean_Abs": shap_res["global_imp"][feat],
                })

    for scen in raw_shap_export:
        raw_shap_export[scen]["shap_vals"] = np.vstack(raw_shap_export[scen]["shap_vals"])
        raw_shap_export[scen]["feature_data"] = pd.concat(
            raw_shap_export[scen]["feature_data"], ignore_index=True
        )

    metrics_collection = pd.DataFrame(metrics_rows)
    shap_collection = pd.DataFrame(shap_rows)
    final_metrics = summarize_metrics_df(metrics_collection)
    final_shap = summarize_shap_df(shap_collection)

    print(f"\n--- FINAL SPATIAL METRICS ({model_name}) ---")
    print(final_metrics.to_string(index=False))
    print(f"\n--- FINAL SHAP IMPORTANCE ({model_name}) ---")
    print(final_shap.to_string(index=False))

    summary_export = {
        "metrics":     final_metrics,
        "raw_metrics": metrics_collection,
        "final_shap":  final_shap,
        "raw_shap":    shap_collection,
    }
    cf_eval = build_confusion_dict(
        np.concatenate(global_targets), np.concatenate(global_preds)
    )

    f_sum = os.path.join(out_dir, f"Summary_{model_name}.pkl")
    f_shp = os.path.join(out_dir, f"SHAP_RAW_{model_name}.pkl")
    f_cfm = os.path.join(out_dir, f"CF_Matrix_{model_name}.pkl")

    save_pickle(summary_export, f_sum)
    save_pickle(raw_shap_export, f_shp)
    save_pickle(cf_eval, f_cfm)

    print(f"\nAll outputs successfully saved for {model_name}:")
    print(f"  - {f_sum}\n  - {f_shp}\n  - {f_cfm}")


if __name__ == "__main__":
    main()
