# ==============================================================================
# Script: 03_Rolling_MLP_TIME.py
# Title: Rolling-Origin Temporal Forecast for Feedforward Network (MLP_TIME)
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
    assign_60_20_20_blocks,
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

    model_name = "MLP_TIME"
    out_dir = "./Output"
    os.makedirs(out_dir, exist_ok=True)
    rfile = "./GES_Baltic_5year_ALLE_Combined_4D.nc"

    ntime = 14
    block_size = 20
    target_var = "TTI"
    tt_threshold = 0.71

    batch_size = 2048
    num_epochs = 250
    patience = 20
    learning_rate = 0.002

    baltic = load_baltic_data(rfile, block_size=block_size, ntime=ntime)
    master_3d = baltic["master_3d"]
    var_to_idx = baltic["var_to_idx"]
    spatial_blocks = baltic["spatial_blocks"]

    train_idx, val_idx, test_idx, train_size, test_blocks_len = assign_60_20_20_blocks(
        spatial_blocks, seed=123
    )

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
        print(f"Executing Rolling Forecast ({model_name}) for: {sim_name}")
        print("=" * 50)

        # In R: t_train_end in 7:12 (1-based), t_val = t_train_end + 1, t_test = t_train_end + 2
        for t_train_end_1b in range(7, 13):
            t_val_1b = t_train_end_1b + 1
            t_test_1b = t_train_end_1b + 2
            forecast_year = 1950 + (t_test_1b * 5)

            print(
                f"\n   --- Rolling Origin: Train 1-{t_train_end_1b} | Val {t_val_1b} | "
                f"Forecast {t_test_1b} ({forecast_year}) ---"
            )

            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

            train_t_0b = list(range(0, t_train_end_1b))
            val_t_0b = [t_val_1b - 1]
            test_t_0b = [t_test_1b - 1]

            # Dynamic Zero-Leakage Standardisation fit strictly on historical training window
            fold_3d = master_3d.copy()
            for p in vari:
                p_idx = var_to_idx[p]
                sub_tr = fold_3d[np.ix_(train_idx, train_t_0b, [p_idx])]
                mean_tr = float(np.nanmean(sub_tr))
                sd_tr = float(np.nanstd(sub_tr, ddof=1))
                fold_3d[:, :, p_idx] = (fold_3d[:, :, p_idx] - mean_tr) / sd_tr

            X_train, Y_train = flatten_2d(fold_3d, var_to_idx, train_idx, train_t_0b, vari, target_var)
            X_val, Y_val = flatten_2d(fold_3d, var_to_idx, val_idx, val_t_0b, vari, target_var)
            X_test, Y_test = flatten_2d(fold_3d, var_to_idx, test_idx, test_t_0b, vari, target_var)
            del fold_3d

            set_seeds(123 + t_train_end_1b)
            model = MLPNet(nvari)
            model = train_neural_network(
                model, X_train, Y_train, X_val, Y_val,
                device=device, batch_size=batch_size, num_epochs=num_epochs,
                patience=patience, learning_rate=learning_rate
            )

            prob_preds = predict_batches(model, X_test, device=device)
            eval_res = evaluate_predictions(Y_test, prob_preds, tt_threshold=tt_threshold)

            if sim_name == "Comprehensive":
                global_targets.extend(eval_res["true_ges"].tolist())
                global_preds.extend(eval_res["pred_ges"].tolist())

            metrics_rows.append({
                "Scenario":     sim_name,
                "Origin_Year":  forecast_year,
                "RMSE":         eval_res["RMSE"],
                "R2":           eval_res["R2"],
                "ACC":          eval_res["ACC"],
                "Precision":    eval_res["Precision"],
                "Recall":       eval_res["Recall"],
                "F1":           eval_res["F1"],
                "Train_Blocks": train_size,
                "Test_Blocks":  test_blocks_len,
            })

            rng = np.random.default_rng(123)
            bg_idx = rng.choice(X_train.shape[0], size=min(500, X_train.shape[0]), replace=False)
            te_idx = rng.choice(X_test.shape[0], size=min(2000, X_test.shape[0]), replace=False)

            shap_vals, feat_df, global_imp = compute_fastshap_mc(
                model, X_train[bg_idx], X_test[te_idx], vari,
                device=device, nsim=50, is_lstm=False, seed=123
            )

            raw_shap_export[sim_name]["shap_vals"].append(shap_vals)
            raw_shap_export[sim_name]["feature_data"].append(feat_df)

            for feat, imp_val in global_imp.items():
                shap_rows.append({
                    "Scenario":      sim_name,
                    "Origin_Year":   forecast_year,
                    "Feature":       feat,
                    "SHAP_Mean_Abs": imp_val,
                })

    for scen in raw_shap_export:
        raw_shap_export[scen]["shap_vals"] = np.vstack(raw_shap_export[scen]["shap_vals"])
        raw_shap_export[scen]["feature_data"] = pd.concat(
            raw_shap_export[scen]["feature_data"], ignore_index=True
        )

    metrics_collection = pd.DataFrame(metrics_rows)
    shap_collection = pd.DataFrame(shap_rows)
    final_metrics, final_shap = aggregate_summary_tables(metrics_collection, shap_collection)

    print("\n--- FINAL TEMPORAL METRICS (MLP_TIME) ---")
    print(final_metrics.to_string(index=False))
    print("\n--- FINAL SHAP IMPORTANCE (MLP_TIME) ---")
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
