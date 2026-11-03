# ==============================================================================
# Script: 10_Plot_FigsS1_S2_Confusion_Matrices.py
# Title: Generate Supplementary Figures S1 and S2 (Global Confusion Matrices)
# Description: Plots the global confusion matrices aggregated across all 10
#              independent spatial cross-validation test blocks for MLP_SPAT
#              (Figure S1) and LSTM_SPAT (Figure S2).
# Reference: Stips et al. (2026) - Environmental Modelling & Software
# License: GPL-3.0
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tti_common import load_pickle


def generate_cm_plot(model_name: str, fig_label: str, in_dir: str, fig_dir: str):
    cf_file = os.path.join(in_dir, f"CF_Matrix_{model_name}.pkl")
    if not os.path.exists(cf_file):
        raise FileNotFoundError(f"Confusion matrix file not found: {cf_file}")

    cf_df = load_pickle(cf_file)
    total_pred = int(cf_df["N"].sum())

    # Extract 2x2 grid ordered as Target = [1, 0] (columns), Prediction = [1, 0] (rows)
    def get_n(p_val, t_val):
        sub = cf_df[(cf_df["Prediction"] == p_val) & (cf_df["Target"] == t_val)]
        return int(sub["N"].iloc[0]) if len(sub) > 0 else 0

    mat_counts = np.array([
        [get_n(1, 1), get_n(1, 0)],
        [get_n(0, 1), get_n(0, 0)],
    ], dtype=np.float64)

    mat_pct = (mat_counts / max(total_pred, 1)) * 100.0
    col_sums = np.maximum(mat_counts.sum(axis=0), 1.0)
    row_sums = np.maximum(mat_counts.sum(axis=1), 1.0)

    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    im = ax.imshow(mat_pct, cmap="Blues", vmin=0.0, vmax=max(65.0, mat_pct.max() * 1.1))

    for r_i in range(2):
        for c_i in range(2):
            cnt = int(mat_counts[r_i, c_i])
            pct = mat_pct[r_i, c_i]
            col_p = (cnt / col_sums[c_i]) * 100.0
            row_p = (cnt / row_sums[r_i]) * 100.0

            ax.text(
                c_i, r_i - 0.06, f"{pct:.1f}%",
                ha="center", va="center", fontsize=15, fontweight="bold", color="black"
            )
            ax.text(
                c_i, r_i + 0.12, f"{cnt}",
                ha="center", va="center", fontsize=11, color="black"
            )
            ax.text(
                c_i, r_i + 0.36, f"Col: {col_p:.1f}% | Row: {row_p:.1f}%",
                ha="center", va="center", fontsize=8.5, style="italic", color="#262626"
            )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["1 (GES)", "0 (Non-GES)"], fontsize=12, fontweight="bold")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Target", fontsize=13, fontweight="bold", labelpad=10)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["1 (GES)", "0 (Non-GES)"], fontsize=12, fontweight="bold")
    ax.set_ylabel("Prediction", fontsize=13, fontweight="bold")

    fig.suptitle(
        f"Global Confusion Matrix ({model_name} - Comprehensive)",
        fontsize=14, fontweight="bold", y=0.97
    )
    ax.set_title(
        "Aggregated across all 10 independent spatial cross-validation test blocks.",
        fontsize=10.5, color="#4d4d4d", pad=28
    )
    fig.text(
        0.95, 0.02, f"Total Predictions: {total_pred}",
        ha="right", va="bottom", fontsize=11, fontweight="bold"
    )

    out_file = os.path.join(fig_dir, f"{fig_label}_Confusion_Matrix_{model_name}.png")
    plt.tight_layout(rect=[0, 0.04, 1, 0.92])
    fig.savefig(out_file, dpi=500, facecolor="white")
    plt.close(fig)
    print(f"{fig_label} ({model_name}) successfully saved to: {out_file}")


def main():
    in_dir = "./Output" if os.path.exists("./Output") else "./"
    fig_dir = "./Figures"
    os.makedirs(fig_dir, exist_ok=True)

    generate_cm_plot("MLP_SPAT", "Figure_S1", in_dir, fig_dir)
    generate_cm_plot("LSTM_SPAT", "Figure_S2", in_dir, fig_dir)


if __name__ == "__main__":
    main()
