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
import matplotlib.pyplot as plt
from tti_common import load_pickle


def _extract_counts(cf_obj) -> tuple:
    """Extract (TN, FP, FN, TP) from dict or 2x2 matrix."""
    if isinstance(cf_obj, dict) and all(k in cf_obj for k in ("TN", "FP", "FN", "TP")):
        return int(cf_obj["TN"]), int(cf_obj["FP"]), int(cf_obj["FN"]), int(cf_obj["TP"])
    arr = np.asarray(cf_obj)
    if arr.shape == (2, 2):
        return int(arr[0, 0]), int(arr[0, 1]), int(arr[1, 0]), int(arr[1, 1])
    raise ValueError(f"Unsupported confusion matrix structure: {type(cf_obj)}")


def generate_cm_plot(in_dir: str, fig_dir: str, model_name: str, fig_label: str) -> None:
    cf_file = os.path.join(in_dir, f"CF_Matrix_{model_name}.pkl")
    if not os.path.exists(cf_file):
        raise FileNotFoundError(f"Confusion matrix file not found: {cf_file}")

    tn, fp, fn, tp = _extract_counts(load_pickle(cf_file))
    total = max(tn + fp + fn + tp, 1)

    # 2x2 grid matching cvms layout: x = Target (0, 1), y = Prediction (0, 1)
    counts_grid = np.array([[tn, fn], [fp, tp]], dtype=np.float64)
    pct_grid = 100.0 * counts_grid / total

    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    im = ax.imshow(pct_grid, cmap="Blues", vmin=0.0, vmax=max(60.0, pct_grid.max() * 1.1))

    for r_i in range(2):      # Prediction: 0 (NOGES), 1 (GES)
        for c_i in range(2):  # Target:     0 (NOGES), 1 (GES)
            cnt = int(counts_grid[r_i, c_i])
            pct = pct_grid[r_i, c_i]
            col_sum = max(counts_grid[0, c_i] + counts_grid[1, c_i], 1.0)
            row_sum = max(counts_grid[r_i, 0] + counts_grid[r_i, 1], 1.0)
            col_pct = 100.0 * cnt / col_sum
            row_pct = 100.0 * cnt / row_sum

            txt_color = "white" if pct > 32.0 else "black"
            ax.text(
                c_i, r_i,
                f"{cnt:,}\n({pct:.1f}%)",
                ha="center", va="center", fontsize=14, fontweight="bold", color=txt_color
            )
            ax.text(
                c_i, r_i + 0.34,
                f"Col: {col_pct:.1f}%  |  Row: {row_pct:.1f}%",
                ha="center", va="center", fontsize=9.2, color=txt_color
            )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0 (NOGES)", "1 (GES)"], fontsize=12, fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0 (NOGES)", "1 (GES)"], fontsize=12, fontweight="bold")
    ax.set_xlabel("Target (Ecosystem Model True GES)", fontsize=12.5, fontweight="bold", labelpad=8)
    ax.set_ylabel("Prediction (ANN Reconstructed GES)", fontsize=12.5, fontweight="bold", labelpad=8)

    fig.suptitle(f"Global Confusion Matrix ({model_name} - Comprehensive)", fontsize=14, fontweight="bold", y=0.96)
    ax.set_title(
        "Aggregated across all 10 independent spatial cross-validation test blocks.",
        fontsize=10.8, color="#4d4d4d", pad=12
    )
    fig.text(0.95, 0.01, f"Total Predictions: {total:,}", ha="right", fontsize=10, color="#333333")

    out_file = os.path.join(fig_dir, f"{fig_label}_Confusion_Matrix_{model_name}.png")
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.13, top=0.86)
    fig.savefig(out_file, dpi=500, facecolor="white", bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(f"{fig_label} ({model_name}) successfully saved to: {out_file}")


def main() -> None:
    in_dir = "./Output" if os.path.exists("./Output") else "./"
    fig_dir = "./Figures"
    os.makedirs(fig_dir, exist_ok=True)

    generate_cm_plot(in_dir, fig_dir, "MLP_SPAT", "Figure_S1")
    generate_cm_plot(in_dir, fig_dir, "LSTM_SPAT", "Figure_S2")


if __name__ == "__main__":
    main()
