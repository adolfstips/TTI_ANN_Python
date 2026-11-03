# ==============================================================================
# Script: 01_Kfold_MLP_SPAT.py
# Title: 10-Fold Spatial Block Cross-Validation for Feedforward Network (MLP_SPAT)
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
    set_seeds,
    get_device,
    load_baltic_data,
    assign_kfold_blocks,
    flatten_2d,
    MLPNet,
    train_neural_network,
    predict_batches,
    evaluate_predictions,
    compute_fastshap_mc,
    aggregate_summary_tables,
    build_confusion_matrix_df,
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
    n_splits = 10
    block_size = 10
    target_var = "TTI"
    tt_threshold = 0.71

    batch_size = 2048
    num_epochs = 150
    patience = 20
    learning_rate = 0.002

    baltic = load_baltic_data(rfile, block_size=block_size, ntime=ntime)
    master_3d = baltic["master_3d"]
    var_to_idx = baltic["var_to_idx"]
    spatial_blocks = baltic["spatial_blocks"]

    fold_ids = assign_kfold_blocks(spatial_blocks, n_splits=n_splits, seed=123)
    all_times = np.arange(ntime)

    metrics_rows = []
    shap_rows = []
    raw_shap_export = {
        scen: {"shap_vals": [], "feature_data": []} for scen in VAR_COM.keys()
    }
    global_targets = []
    global_preds = []

    for sim_name, vari in VAR_COM.items():
        nvari = len(vari)
        print("\n" + "=" * 50)
        print(f"Starting {n_splits}-Fold Spatial CV ({model_name}) for: {sim_name}")
        print("=" * 50)

        for k in range(1, n_splits + 1):
            print(f"   Running Fold {k} / {n_splits}...")
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

            val_k = 1 if k == n_splits else k + 1
            train_idx = np.flatnonzero((fold_ids != k) & (fold_ids != val_k))
            val_idx = np.flatnonzero(fold_ids == val_k)
            test_idx = np.flatnonzero(fold_ids == k)

            # Zero-Leakage Standardisation fit strictly on Training Fold
            fold_3d = master_3d.copy()
            for p in vari:
                p_idx = var_to_idx[p]
                mean_tr = float(np.nanmean(fold_3d[train_idx, :, p_idx]))
                sd_tr = float(np.nanstd(fold_3d[train_idx, :, p_idx], ddof=1))
                fold_3d[:, :, p_idx] = (fold_3d[:, :, p_idx] - mean_tr) / sd_tr

            X_train, Y_train = flatten_2d(fold_3d, var_to_idx, train_idx, all_times, vari, target_var)
            X_val, Y_val = flatten_2d(fold_3d, var_to_idx, val_idx, all_times, vari, target_var)
            X_test, Y_test = flatten_2d(fold_3d, var_to_idx, test_idx, all_times, vari, target_var)
            del fold_3d

            set_seeds(123 + k)
            model = MLPNet(nvari)
            model = train_neural_network(
                model, X_train, Y_train, X_val, Y_val,
                device=device, batch_size=batch_size, num_epochs=num_epochs,
                patience=patience, learning_rate=learning_rate, fold_label=f"Fold {k}"
            )

            prob_preds = predict_batches(model, X_test, device=device)
            eval_res = evaluate_predictions(Y_test, prob_preds, tt_threshold=tt_threshold)

            if sim_name == "Comprehensive":
                global_targets.extend(eval_res["true_ges"].tolist())
                global_preds.extend(eval_res["pred_ges"].tolist())

            metrics_rows.append({
                "Scenario":  sim_name,
                "Split":     k,
                "RMSE":      eval_res["RMSE"],
                "R2":        eval_res["R2"],
                "ACC":       eval_res["ACC"],
                "Precision": eval_res["Precision"],
                "Recall":    eval_res["Recall"],
                "F1":        eval_res["F1"],
            })

            # Monte Carlo SHAP on Independent Test Fold
            rng = np.random.default_rng(123 + k)
            bg_idx = rng.choice(X_train.shape[0], size=min(500, X_train.shape[0]), replace=False)
            te_idx = rng.choice(X_test.shape[0], size=min(2000, X_test.shape[0]), replace=False)

            shap_vals, feat_df, global_imp = compute_fastshap_mc(
                model, X_train[bg_idx], X_test[te_idx], vari,
                device=device, nsim=50, is_lstm=False, seed=123 + k
            )

            raw_shap_export[sim_name]["shap_vals"].append(shap_vals)
            raw_shap_export[sim_name]["feature_data"].append(feat_df)

            for feat, imp_val in global_imp.items():
                shap_rows.append({
                    "Scenario":      sim_name,
                    "Split":         k,
                    "Feature":       feat,
                    "SHAP_Mean_Abs": imp_val,
                })

    # Combine raw SHAP arrays across folds
    for scen in raw_shap_export:
        raw_shap_export[scen]["shap_vals"] = np.vstack(raw_shap_export[scen]["shap_vals"])
        raw_shap_export[scen]["feature_data"] = pd.concat(
            raw_shap_export[scen]["feature_data"], ignore_index=True
        )

    metrics_collection = pd.DataFrame(metrics_rows)
    shap_collection = pd.DataFrame(shap_rows)
    final_metrics, final_shap = aggregate_summary_tables(metrics_collection, shap_collection)

    print("\n--- FINAL SPATIAL METRICS (MLP_SPAT) ---")
    print(final_metrics.to_string(index=False))
    print("\n--- FINAL SHAP IMPORTANCE (MLP_SPAT) ---")
    print(final_shap.to_string(index=False))

    summary_export_data = {
        "metrics":     final_metrics,
        "raw_metrics": metrics_collection,
        "final_shap":  final_shap,
        "raw_shap":    shap_collection,
    }

    summary_file = os.path.join(out_dir, f"Summary_{model_name}.pkl")
    raw_shap_file = os.path.join(out_dir, f"SHAP_RAW_{model_name}.pkl")
    cf_file = os.path.join(out_dir, f"CF_Matrix_{model_name}.pkl")

    save_pickle(summary_export_data, summary_file)
    save_pickle(raw_shap_export, raw_shap_file)
    save_pickle(build_confusion_matrix_df(global_targets, global_preds), cf_file)

    print(f"\nAll outputs successfully saved for {model_name}:\n  - {summary_file}\n  - {raw_shap_file}\n  - {cf_file}")


if __name__ == "__main__":
    main()
