# ==============================================================================
# Script: 06_Generate_Manuscript_Tables.py
# Title: Master Script to Generate Manuscript Tables 2, 3, and 4
# Description: Loads the harmonized Summary_*.pkl outputs from Scripts 01-04
#              and generates publication-ready CSV (.csv) and Word (.docx) tables
#              for Spatial CV (Table 2), Temporal Forecasting (Table 3), and
#              Global Feature Importance (Table 4).
# Reference: Stips et al. (2026) - Environmental Modelling & Software
# License: GPL-3.0
# ==============================================================================

import os
import pandas as pd
from tti_common import load_pickle

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False


def save_docx_table(df: pd.DataFrame, title: str, subtitle: str, out_path: str):
    if not HAS_DOCX:
        return
    doc = Document()
    h = doc.add_heading(title, level=2)
    p = doc.add_paragraph(subtitle)
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = "Table Grid"
    hdr_cells = table.rows[0].cells
    for i, col_name in enumerate(df.columns):
        hdr_cells[i].text = str(col_name)
    for _, row in df.iterrows():
        row_cells = table.add_row().cells
        for i, val in enumerate(row):
            row_cells[i].text = str(val)
    doc.save(out_path)


def format_perf_df(res_mlp: dict, res_lstm: dict, mlp_name: str, lstm_name: str) -> pd.DataFrame:
    scenario_order = ["Comprehensive", "MSFD_Baseline", "Remote_Sensing"]
    df_m = res_mlp["metrics"].copy()
    df_m["Architecture"] = mlp_name
    df_l = res_lstm["metrics"].copy()
    df_l["Architecture"] = lstm_name

    combined = pd.concat([df_m, df_l], ignore_index=True)
    combined["Scenario"] = pd.Categorical(combined["Scenario"], categories=scenario_order, ordered=True)
    combined = combined.sort_values(["Architecture", "Scenario"]).reset_index(drop=True)

    out_df = pd.DataFrame({
        "Architecture": combined["Architecture"],
        "Scenario":     combined["Scenario"].astype(str),
        "Accuracy":     [f"{m:.3f} (±{c:.3f})" for m, c in zip(combined["ACC_Mean"], combined["ACC_CI"])],
        "F1_Score":     [f"{m:.3f} (±{c:.3f})" for m, c in zip(combined["F1_Mean"], combined["F1_CI"])],
        "RMSE":         [f"{m:.3f} (±{c:.3f})" for m, c in zip(combined["RMSE_Mean"], combined["RMSE_CI"])],
        "R2":           [f"{m:.3f} (±{c:.3f})" for m, c in zip(combined["R2_Mean"], combined["R2_CI"])],
    })
    return out_df


def main():
    in_dir = "./Output" if os.path.exists("./Output") else "./"
    out_dir = "./Tables"
    os.makedirs(out_dir, exist_ok=True)

    mlp_spat_file = os.path.join(in_dir, "Summary_MLP_SPAT.pkl")
    lstm_spat_file = os.path.join(in_dir, "Summary_LSTM_SPAT.pkl")
    mlp_time_file = os.path.join(in_dir, "Summary_MLP_TIME.pkl")
    lstm_time_file = os.path.join(in_dir, "Summary_LSTM_TIME.pkl")

    for req in [mlp_spat_file, lstm_spat_file, mlp_time_file, lstm_time_file]:
        if not os.path.exists(req):
            raise FileNotFoundError(f"Missing summary file: {req}. Run scripts 01 to 04 first.")

    res_mlp_spat = load_pickle(mlp_spat_file)
    res_lstm_spat = load_pickle(lstm_spat_file)
    res_mlp_time = load_pickle(mlp_time_file)
    res_lstm_time = load_pickle(lstm_time_file)

    # 1. Table 2: Spatial Extrapolation Performance
    df_spat = format_perf_df(res_mlp_spat, res_lstm_spat, "MLP_SPAT", "LSTM_SPAT")
    print("\n" + "=" * 66)
    print("TABLE 2: SPATIAL EXTRAPOLATION PERFORMANCE (SECTION 3.1)")
    print("=" * 66)
    print(df_spat.to_string(index=False))

    df_spat.to_csv(os.path.join(out_dir, "Table_2_Spatial_Performance.csv"), index=False)
    save_docx_table(
        df_spat,
        "Table 2: Spatial Extrapolation Performance",
        "10-Fold Spatial Block Cross-Validation (Means ± 95% CI)",
        os.path.join(out_dir, "Table_2_Spatial_Performance.docx"),
    )

    # 2. Table 3: Temporal Forecasting Performance
    df_time = format_perf_df(res_mlp_time, res_lstm_time, "MLP_TIME", "LSTM_TIME")
    print("\n" + "=" * 66)
    print("TABLE 3: TEMPORAL FORECASTING PERFORMANCE (SECTION 3.2)")
    print("=" * 66)
    print(df_time.to_string(index=False))

    df_time.to_csv(os.path.join(out_dir, "Table_3_Temporal_Performance.csv"), index=False)
    save_docx_table(
        df_time,
        "Table 3: Temporal Forecasting Performance",
        "30-Year Rolling-Origin Hindcast (Means ± 95% CI across 6 forward pentads)",
        os.path.join(out_dir, "Table_3_Temporal_Performance.docx"),
    )

    # 3. Table 4: Global Feature Importance (Comprehensive Scenario)
    def extract_shap_col(res_obj: dict, col_name: str) -> pd.DataFrame:
        sub = res_obj["final_shap"]
        sub = sub[sub["Scenario"] == "Comprehensive"].copy()
        sub[col_name] = [
            f"{m:.4f} (±{c:.4f})" for m, c in zip(sub["SHAP_Mean"], sub["SHAP_CI"])
        ]
        return sub[["Feature", col_name]]

    s_mlp_spat = extract_shap_col(res_mlp_spat, "MLP_SPAT")
    s_lstm_spat = extract_shap_col(res_lstm_spat, "LSTM_SPAT")
    s_mlp_time = extract_shap_col(res_mlp_time, "MLP_TIME")
    s_lstm_time = extract_shap_col(res_lstm_time, "LSTM_TIME")

    sort_ref = res_mlp_spat["final_shap"]
    sort_ref = sort_ref[sort_ref["Scenario"] == "Comprehensive"][["Feature", "SHAP_Mean"]].rename(
        columns={"SHAP_Mean": "Sort_Val"}
    )

    df_shap = (
        s_mlp_spat.merge(s_lstm_spat, on="Feature")
        .merge(s_mlp_time, on="Feature")
        .merge(s_lstm_time, on="Feature")
        .merge(sort_ref, on="Feature")
        .sort_values("Sort_Val", ascending=False)
        .drop(columns=["Sort_Val"])
        .reset_index(drop=True)
    )

    print("\n" + "=" * 66)
    print("TABLE 4: GLOBAL FEATURE IMPORTANCE - SHAP (SECTION 3.3)")
    print("=" * 66)
    print(df_shap.to_string(index=False))

    df_shap.to_csv(os.path.join(out_dir, "Table_4_SHAP_Feature_Importance.csv"), index=False)
    save_docx_table(
        df_shap,
        "Table 4: Global Feature Importance (Comprehensive Scenario)",
        "Mean Absolute SHAP value (± 95% CI) representing the impact on continuous TTI prediction.",
        os.path.join(out_dir, "Table_4_SHAP_Feature_Importance.docx"),
    )

    print(f"\nAll manuscript tables (Tables 2, 3, and 4) successfully saved to: {out_dir}")


if __name__ == "__main__":
    main()
