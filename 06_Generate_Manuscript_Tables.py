# ==============================================================================
# Script: 06_Generate_Manuscript_Tables.py
# Title: Compile and Export Manuscript Tables 2, 3, and 4 (Python)
# Description: Aggregates summary outputs from Scripts 01-04 into publication-
#              ready CSV (and optional .docx) tables for Sections 3.1-3.3.
# Reference: Stips et al. (2026) - Environmental Modelling & Software
# License: GPL-3.0
# ==============================================================================

import os
import pandas as pd
from tti_common import load_pickle


def format_perf_table(sum_lstm: dict, name_lstm: str, sum_mlp: dict, name_mlp: str) -> pd.DataFrame:
    scen_order = ["Comprehensive", "MSFD_Baseline", "Remote_Sensing"]
    rows = []
    for arch_name, sum_obj in [(name_lstm, sum_lstm), (name_mlp, sum_mlp)]:
        df_m = sum_obj["metrics"].set_index("Scenario")
        for scen in scen_order:
            r = df_m.loc[scen]
            rows.append({
                "Architecture": arch_name,
                "Scenario":     scen,
                "Accuracy":     f"{r['ACC_Mean']:.3f} (\u00b1{r['ACC_CI']:.3f})",
                "F1_Score":     f"{r['F1_Mean']:.3f} (\u00b1{r['F1_CI']:.3f})",
                "RMSE":         f"{r['RMSE_Mean']:.3f} (\u00b1{r['RMSE_CI']:.3f})",
                "R2":           f"{r['R2_Mean']:.3f} (\u00b1{r['R2_CI']:.3f})",
            })
    return pd.DataFrame(rows)


def main() -> None:
    in_dir = "./Output" if os.path.exists("./Output") else "./"
    tab_dir = "./Tables"
    os.makedirs(tab_dir, exist_ok=True)

    mlp_spat = load_pickle(os.path.join(in_dir, "Summary_MLP_SPAT.pkl"))
    lstm_spat = load_pickle(os.path.join(in_dir, "Summary_LSTM_SPAT.pkl"))
    mlp_time = load_pickle(os.path.join(in_dir, "Summary_MLP_TIME.pkl"))
    lstm_time = load_pickle(os.path.join(in_dir, "Summary_LSTM_TIME.pkl"))

    table2_df = format_perf_table(lstm_spat, "LSTM_SPAT", mlp_spat, "MLP_SPAT")
    table3_df = format_perf_table(lstm_time, "LSTM_TIME", mlp_time, "MLP_TIME")

    ms_comp = (
        mlp_spat["final_shap"]
        .query("Scenario == 'Comprehensive'")
        .sort_values("SHAP_Mean", ascending=False)
    )
    feat_order = ms_comp["Feature"].tolist()

    def get_shap_map(sum_obj: dict) -> dict:
        sub = sum_obj["final_shap"].query("Scenario == 'Comprehensive'").set_index("Feature")
        return {
            f: f"{sub.loc[f, 'SHAP_Mean']:.4f} (\u00b1{sub.loc[f, 'SHAP_CI']:.4f})"
            for f in sub.index
        }

    m_ms = get_shap_map(mlp_spat)
    m_ls = get_shap_map(lstm_spat)
    m_mt = get_shap_map(mlp_time)
    m_lt = get_shap_map(lstm_time)

    table4_df = pd.DataFrame([
        {
            "Feature":   f,
            "MLP_SPAT":  m_ms[f],
            "LSTM_SPAT": m_ls[f],
            "MLP_TIME":  m_mt[f],
            "LSTM_TIME": m_lt[f],
        }
        for f in feat_order
    ])

    print("\n" + "=" * 66)
    print("TABLE 2: SPATIAL EXTRAPOLATION PERFORMANCE (SECTION 3.1)")
    print("=" * 66)
    print(table2_df.to_string(index=False))

    print("\n" + "=" * 66)
    print("TABLE 3: OUT-OF-SAMPLE TEMPORAL EXTRAPOLATION PERFORMANCE (SECTION 3.2)")
    print("=" * 66)
    print(table3_df.to_string(index=False))

    print("\n" + "=" * 66)
    print("TABLE 4: GLOBAL FEATURE IMPORTANCE - SHAP (SECTION 3.3)")
    print("=" * 66)
    print(table4_df.to_string(index=False))

    table2_df.to_csv(os.path.join(tab_dir, "Table_2_Spatial_Performance.csv"), index=False)
    table3_df.to_csv(os.path.join(tab_dir, "Table_3_Temporal_Performance.csv"), index=False)
    table4_df.to_csv(os.path.join(tab_dir, "Table_4_SHAP_Feature_Importance.csv"), index=False)

    print(f"\nAll manuscript tables (Tables 2, 3, and 4) successfully saved to: {tab_dir}")


if __name__ == "__main__":
    main()
