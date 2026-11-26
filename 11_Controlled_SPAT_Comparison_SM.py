# ==============================================================================
# Script: 11_Controlled_SPAT_Comparison_SM.py
# Title: Controlled 10-Fold Spatial Comparison on Common Test Set (1985-2020)
# Description: Python companion to 11_Controlled_SPAT_Comparison_SM.R addressing
#              Reviewer 1 (Revision 1, Comment 2). Evaluates MLP_SPAT_Common8,
#              MLP_SEQ7_Windowed, and LSTM_SPAT_Common8 on identical held-out
#              10x10 spatial blocks and identical 8 target pentads (1985-2020),
#              reporting paired fold-level differences and two-sided paired t-tests.
# Reference: Stips et al. (2026) - Environmental Modelling & Software
# License: GPL-3.0
# ==============================================================================

import os
import gc
import time
import numpy as np
import pandas as pd
import torch
from scipy import stats
from tti_common import (
    VAR_COM,
    init_torch_device,
    set_all_seeds,
    load_baltic_data,
    assign_kfold_blocks,
    flatten_2d,
    create_sequences_3d,
    flatten_sequences_for_mlp_seq7,
    MLPNet,
    LSTMNet,
    train_model_vram,
    predict_batches,
    calc_eval_metrics,
    save_pickle,
)


def compute_paired_contrast(df_raw: pd.DataFrame, model_a: str, model_b: str, label: str, scen_order: list) -> pd.DataFrame:
    a_df = df_raw[df_raw["Architecture"] == model_a].set_index(["Scenario", "Fold"])
    b_df = df_raw[df_raw["Architecture"] == model_b].set_index(["Scenario", "Fold"])
    rows = []
    for scen in scen_order:
        sa = a_df.loc[scen].sort_index()
        sb = b_df.loc[scen].sort_index()
        n = len(sa)
        ci_m = 1.96 / np.sqrt(n)

        d_acc = sa["ACC"].values - sb["ACC"].values
        d_f1 = sa["F1"].values - sb["F1"].values
        d_rmse = sa["RMSE"].values - sb["RMSE"].values
        d_r2 = sa["R2"].values - sb["R2"].values

        p_rmse = float(stats.ttest_rel(sa["RMSE"].values, sb["RMSE"].values).pvalue)
        p_f1 = float(stats.ttest_rel(sa["F1"].values, sb["F1"].values).pvalue)

        rows.append({
            "Group":         label,
            "Scenario":      scen,
            "Accuracy":      f"{np.mean(d_acc):+.3f} (\u00b1{ci_m * np.std(d_acc, ddof=1):.3f})",
            "F1_Score":      f"{np.mean(d_f1):+.3f} (\u00b1{ci_m * np.std(d_f1, ddof=1):.3f})",
            "RMSE":          f"{np.mean(d_rmse):+.3f} (\u00b1{ci_m * np.std(d_rmse, ddof=1):.3f})",
            "R2":            f"{np.mean(d_r2):+.3f} (\u00b1{ci_m * np.std(d_r2, ddof=1):.3f})",
            "Paired_p_RMSE": f"{p_rmse:.2e}",
            "Paired_p_F1":   f"{p_f1:.2e}",
        })
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.perf_counter()
    device = init_torch_device(num_threads=4, seed=123)
    out_dir = "./Output"
    tab_dir = "./Tables"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    ntime = 14
    sequence_length = 7
    n_splits = 10
    block_size = 10
    target_var = "TTI"
    tt_threshold = 0.71
    batch_size = 2048
    pred_batch_size = 65536
    num_epochs = 150
    patience = 20
    learning_rate = 0.002

    baltic = load_baltic_data("./GES_Baltic_5year_ALLE_Combined_4D.nc", block_size=block_size, ntime=ntime)
    master_3d = baltic["master_3d"]
    var_idx = baltic["var_idx"]
    fold_ids = assign_kfold_blocks(baltic["spatial_blocks"], n_splits=n_splits, seed=123)

    all_times_0 = list(range(ntime))                       # 0..13 (1955..2020)
    common_times_0 = list(range(sequence_length - 1, ntime))  # 6..13 (1985..2020)

    records_common8 = []
    records_check14 = []

    for sim_name, vari in VAR_COM.items():
        t_scen = time.perf_counter()
        nvari = len(vari)
        print("\n" + "=" * 66)
        print(f"Controlled Spatial Comparison (1985-2020 Common Test Set): {sim_name}")
        print("=" * 66)

        for k in range(1, n_splits + 1):
            print(f"   Running Fold {k} / {n_splits} ...")
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

            # A. MLP_SPAT (1-pentad input; scored on both 1955-2020 and Common 1985-2020)
            X_mlp_tr, Y_mlp_tr = flatten_2d(fold_3d, var_idx, train_idx, all_times_0, vari, target_var)
            X_mlp_va, Y_mlp_va = flatten_2d(fold_3d, var_idx, val_idx, all_times_0, vari, target_var)
            X_mlp_te14, Y_mlp_te14 = flatten_2d(fold_3d, var_idx, test_idx, all_times_0, vari, target_var)
            X_mlp_te8, Y_mlp_te8 = flatten_2d(fold_3d, var_idx, test_idx, common_times_0, vari, target_var)

            set_all_seeds(123 + k)
            m_mlp = train_model_vram(
                MLPNet(nvari), X_mlp_tr, Y_mlp_tr, X_mlp_va, Y_mlp_va,
                device=device, batch_size=batch_size, val_batch_size=pred_batch_size,
                num_epochs=num_epochs, patience=patience, lr=learning_rate,
            )
            ev_14 = calc_eval_metrics(Y_mlp_te14, predict_batches(m_mlp, X_mlp_te14, device, pred_batch_size), tt_threshold)
            ev_8 = calc_eval_metrics(Y_mlp_te8, predict_batches(m_mlp, X_mlp_te8, device, pred_batch_size), tt_threshold)
            del m_mlp, X_mlp_tr, Y_mlp_tr, X_mlp_va, Y_mlp_va

            records_check14.append({
                "Architecture": "MLP_SPAT_14Pentads_Check", "Scenario": sim_name, "Fold": k,
                "ACC": ev_14["ACC"], "F1": ev_14["F1"], "RMSE": ev_14["RMSE"], "R2": ev_14["R2"],
            })
            records_common8.append({
                "Architecture": "MLP_SPAT_Common8", "Scenario": sim_name, "Fold": k,
                "ACC": ev_8["ACC"], "F1": ev_8["F1"], "RMSE": ev_8["RMSE"], "R2": ev_8["R2"],
            })

            # Shared 7-pentad 3D sequences for 1985-2020
            X_seq_tr, Y_seq_tr = create_sequences_3d(fold_3d, var_idx, train_idx, common_times_0, vari, target_var, sequence_length)
            X_seq_va, Y_seq_va = create_sequences_3d(fold_3d, var_idx, val_idx, common_times_0, vari, target_var, sequence_length)
            X_seq_te, Y_seq_te = create_sequences_3d(fold_3d, var_idx, test_idx, common_times_0, vari, target_var, sequence_length)
            del fold_3d

            # B. MLP_SEQ7_Windowed (flattened 7-pentad input; no recurrence)
            X_s7_tr = flatten_sequences_for_mlp_seq7(X_seq_tr)
            X_s7_va = flatten_sequences_for_mlp_seq7(X_seq_va)
            X_s7_te = flatten_sequences_for_mlp_seq7(X_seq_te)

            set_all_seeds(123 + k)
            m_seq7 = train_model_vram(
                MLPNet(sequence_length * nvari), X_s7_tr, Y_seq_tr, X_s7_va, Y_seq_va,
                device=device, batch_size=batch_size, val_batch_size=pred_batch_size,
                num_epochs=num_epochs, patience=patience, lr=learning_rate,
            )
            ev_seq7 = calc_eval_metrics(Y_seq_te, predict_batches(m_seq7, X_s7_te, device, pred_batch_size), tt_threshold)
            del m_seq7, X_s7_tr, X_s7_va, X_s7_te

            records_common8.append({
                "Architecture": "MLP_SEQ7_Windowed", "Scenario": sim_name, "Fold": k,
                "ACC": ev_seq7["ACC"], "F1": ev_seq7["F1"], "RMSE": ev_seq7["RMSE"], "R2": ev_seq7["R2"],
            })

            # C. LSTM_SPAT_Common8 (sequential 7-pentad recurrent input)
            set_all_seeds(123 + k)
            m_lstm = train_model_vram(
                LSTMNet(nvari=nvari, seq_len=sequence_length), X_seq_tr, Y_seq_tr, X_seq_va, Y_seq_va,
                device=device, batch_size=batch_size, val_batch_size=pred_batch_size,
                num_epochs=num_epochs, patience=patience, lr=learning_rate,
            )
            ev_lstm = calc_eval_metrics(Y_seq_te, predict_batches(m_lstm, X_seq_te, device, pred_batch_size), tt_threshold)
            del m_lstm, X_seq_tr, X_seq_va, X_seq_te

            records_common8.append({
                "Architecture": "LSTM_SPAT_Common8", "Scenario": sim_name, "Fold": k,
                "ACC": ev_lstm["ACC"], "F1": ev_lstm["F1"], "RMSE": ev_lstm["RMSE"], "R2": ev_lstm["R2"],
            })

        print(f"Completed scenario {sim_name} in {(time.perf_counter() - t_scen) / 60.0:.2f} minutes.")

    raw_common8_df = pd.DataFrame(records_common8)
    raw_check14_df = pd.DataFrame(records_check14)

    scen_order = ["Comprehensive", "MSFD_Baseline", "Remote_Sensing"]
    arch_order = ["MLP_SPAT_Common8", "MLP_SEQ7_Windowed", "LSTM_SPAT_Common8"]

    summary_rows = []
    for arch in arch_order:
        for scen in scen_order:
            sub = raw_common8_df[(raw_common8_df["Architecture"] == arch) & (raw_common8_df["Scenario"] == scen)]
            ci_m = 1.96 / np.sqrt(len(sub))
            summary_rows.append({
                "Group":         arch,
                "Scenario":      scen,
                "Accuracy":      f"{sub['ACC'].mean():.3f} (\u00b1{ci_m * sub['ACC'].std(ddof=1):.3f})",
                "F1_Score":      f"{sub['F1'].mean():.3f} (\u00b1{ci_m * sub['F1'].std(ddof=1):.3f})",
                "RMSE":          f"{sub['RMSE'].mean():.3f} (\u00b1{ci_m * sub['RMSE'].std(ddof=1):.3f})",
                "R2":            f"{sub['R2'].mean():.3f} (\u00b1{ci_m * sub['R2'].std(ddof=1):.3f})",
                "Paired_p_RMSE": "-",
            })
    summary_models_df = pd.DataFrame(summary_rows)

    paired_contrasts_df = pd.concat([
        compute_paired_contrast(raw_common8_df, "LSTM_SPAT_Common8", "MLP_SPAT_Common8", "LSTM_SPAT minus MLP_SPAT (Common 1985-2020)", scen_order),
        compute_paired_contrast(raw_common8_df, "MLP_SEQ7_Windowed", "MLP_SPAT_Common8", "MLP_SEQ7 minus MLP_SPAT (35-yr Window Effect)", scen_order),
        compute_paired_contrast(raw_common8_df, "LSTM_SPAT_Common8", "MLP_SEQ7_Windowed", "LSTM_SPAT minus MLP_SEQ7 (Pure Recurrence Effect)", scen_order),
    ], ignore_index=True)

    table_s1_export = pd.concat([
        summary_models_df,
        paired_contrasts_df[["Group", "Scenario", "Accuracy", "F1_Score", "RMSE", "R2", "Paired_p_RMSE"]],
    ], ignore_index=True)

    print("\n" + "=" * 66)
    print("COMMON 8-PENTAD TEST SET PERFORMANCE & PAIRED DIFFERENCES (1985-2020)")
    print("=" * 66)
    print(table_s1_export.to_string(index=False))

    pkl_path = os.path.join(out_dir, "Controlled_SPAT_1985_2020_SM.pkl")
    csv_path = os.path.join(tab_dir, "Supplementary_Table_S1_Controlled_SPAT_1985_2020.csv")

    save_pickle({
        "raw_common8":      raw_common8_df,
        "raw_check14":      raw_check14_df,
        "summary_models":   summary_models_df,
        "paired_contrasts": paired_contrasts_df,
        "table_s1":         table_s1_export,
    }, pkl_path)
    table_s1_export.to_csv(csv_path, index=False)

    print(f"\nCOMPLETED 11_Controlled_SPAT_Comparison_SM.py in {(time.perf_counter() - t0) / 60.0:.2f} minutes!")
    print(f"Saved:\n  - {pkl_path}\n  - {csv_path}")


if __name__ == "__main__":
    main()
