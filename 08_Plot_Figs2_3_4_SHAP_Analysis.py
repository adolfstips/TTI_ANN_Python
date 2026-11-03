# ==============================================================================
# Script: 08_Plot_Figs2_3_4_SHAP_Analysis.py
# Title: Master SHAP Visualization Suite (Figures 2, 3, and 4)
# Description: Generates Figure 2 (Temporal Rolling-Origin SHAP Stability),
#              Figure 3 (Spatial 10-Fold Block CV SHAP Stability), and
#              Figure 4 (Spatial Beeswarm Feature Interactions).
# Reference: Stips et al. (2026) - Environmental Modelling & Software
# License: GPL-3.0
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tti_common import load_pickle


def plot_figure_2(sum_mlp_time: dict, sum_lstm_time: dict, fig_dir: str):
    mlp_shap = sum_mlp_time["final_shap"].query("Scenario == 'Comprehensive'").copy()
    lstm_shap = sum_lstm_time["final_shap"].query("Scenario == 'Comprehensive'").copy()
    mlp_raw = sum_mlp_time["raw_shap"].query("Scenario == 'Comprehensive'").copy()
    lstm_raw = sum_lstm_time["raw_shap"].query("Scenario == 'Comprehensive'").copy()

    # Order features by MLP_TIME importance (top to bottom)
    ordered_features = list(mlp_shap.sort_values("SHAP_Mean", ascending=True)["Feature"])
    y_pos = np.arange(len(ordered_features))

    max_x = max(
        0.12,
        float(max(mlp_raw["SHAP_Mean_Abs"].max(), lstm_raw["SHAP_Mean_Abs"].max())) * 1.08,
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 5.5), sharey=True)
    panels = [
        (axes[0], lstm_shap, lstm_raw, "LSTM_TIME (Recurrent)", "#56B4E9"),
        (axes[1], mlp_shap, mlp_raw, "MLP_TIME (Feedforward)", "#E69F00"),
    ]

    rng = np.random.default_rng(123)
    for ax, df_sum, df_raw, title_str, color_hex in panels:
        df_s = df_sum.set_index("Feature").loc[ordered_features].reset_index()
        means = df_s["SHAP_Mean"].values
        cis = df_s["SHAP_CI"].values
        lower_err = np.minimum(means, cis)

        ax.barh(
            y_pos, means, height=0.72, color=color_hex, edgecolor="black",
            linewidth=0.6, alpha=0.88, zorder=2
        )
        ax.errorbar(
            means, y_pos, xerr=[lower_err, cis], fmt="none", ecolor="black",
            elinewidth=1.2, capsize=4, zorder=3
        )

        for idx_f, feat in enumerate(ordered_features):
            pts = df_raw[df_raw["Feature"] == feat]["SHAP_Mean_Abs"].values
            jitter = rng.uniform(-0.14, 0.14, size=len(pts))
            ax.scatter(
                pts, np.full_like(pts, idx_f, dtype=np.float64) + jitter,
                color="black", s=22, alpha=0.65, zorder=4
            )

        ax.set_title(title_str, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlim(0.0, max_x)
        ax.grid(axis="x", linestyle="--", color="grey", alpha=0.35, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(ordered_features, fontsize=11, fontweight="bold")
    fig.supxlabel("Mean Absolute SHAP Value (Impact on TTI Forecast)", fontsize=12, fontweight="bold", y=0.03)
    fig.suptitle(
        "Temporal Stability of Global Feature Importance (30-Year Rolling Forecast)\n"
        "Comprehensive Variable Suite (Means ± 95% CI with individual forecast origins overlaid)",
        fontsize=12.5, fontweight="bold", y=0.97
    )

    out_file = os.path.join(fig_dir, "Figure2_Temporal_SHAP_Stability.png")
    plt.tight_layout(rect=[0, 0.06, 1, 0.90])
    fig.savefig(out_file, dpi=500, facecolor="white")
    plt.close(fig)
    print(f"Figure 2 successfully saved to: {out_file}")


def plot_figure_3(sum_mlp_spat: dict, sum_lstm_spat: dict, fig_dir: str):
    mlp_sum = sum_mlp_spat["final_shap"].query("Scenario == 'Comprehensive'").copy()
    lstm_sum = sum_lstm_spat["final_shap"].query("Scenario == 'Comprehensive'").copy()
    mlp_raw = sum_mlp_spat["raw_shap"].query("Scenario == 'Comprehensive'").copy()
    lstm_raw = sum_lstm_spat["raw_shap"].query("Scenario == 'Comprehensive'").copy()

    ordered_features = list(mlp_sum.sort_values("SHAP_Mean", ascending=True)["Feature"])
    y_pos = np.arange(len(ordered_features))
    bar_h = 0.35

    fig, ax = plt.subplots(figsize=(7.8, 5.5))
    rng = np.random.default_rng(123)

    for offset, df_s_in, df_r_in, label_str, col_hex in [
        (+bar_h / 2.0, mlp_sum, mlp_raw, "MLP_SPAT (Feedforward)", "#E69F00"),
        (-bar_h / 2.0, lstm_sum, lstm_raw, "LSTM_SPAT (Recurrent)", "#56B4E9"),
    ]:
        df_s = df_s_in.set_index("Feature").loc[ordered_features].reset_index()
        means = df_s["SHAP_Mean"].values
        cis = df_s["SHAP_CI"].values
        lower_err = np.minimum(means, cis)

        ax.barh(
            y_pos + offset, means, height=bar_h * 0.92, color=col_hex,
            edgecolor="black", linewidth=0.5, alpha=0.88, label=label_str, zorder=2
        )
        ax.errorbar(
            means, y_pos + offset, xerr=[lower_err, cis], fmt="none",
            ecolor="black", elinewidth=1.2, capsize=3.5, zorder=3
        )

        for idx_f, feat in enumerate(ordered_features):
            pts = df_r_in[df_r_in["Feature"] == feat]["SHAP_Mean_Abs"].values
            jitter = rng.uniform(-bar_h * 0.25, bar_h * 0.25, size=len(pts))
            ax.scatter(
                pts, np.full_like(pts, idx_f + offset, dtype=np.float64) + jitter,
                facecolor=col_hex, edgecolor="black", linewidth=0.6, s=22, alpha=0.75, zorder=4
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered_features, fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on TTI)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", color="grey", alpha=0.35, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Spatial Stability of Global Feature Importance (10-Fold Block Cross-Validation)",
        fontsize=13, fontweight="bold", y=0.96
    )
    ax.set_title(
        "10-Fold Spatial Block Cross-Validation (Means ± 95% CI with fold-level iterations)",
        fontsize=11, color="#4d4d4d", pad=10
    )

    handles = [
        Patch(facecolor="#56B4E9", edgecolor="black", label="LSTM_SPAT (Recurrent)"),
        Patch(facecolor="#E69F00", edgecolor="black", label="MLP_SPAT (Feedforward)"),
    ]
    ax.legend(
        handles=handles, title="ANN Architecture:", title_fontsize=11, fontsize=10.5,
        loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False
    )

    out_file = os.path.join(fig_dir, "Figure3_Spatial_SHAP_Stability.png")
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    fig.savefig(out_file, dpi=500, facecolor="white")
    plt.close(fig)
    print(f"Figure 3 successfully saved to: {out_file}")


def plot_figure_4(raw_mlp_spat: dict, raw_lstm_spat: dict, fig_dir: str):
    n_rep = min(2000, raw_mlp_spat["Comprehensive"]["shap_vals"].shape[0])
    mlp_s = raw_mlp_spat["Comprehensive"]["shap_vals"][:n_rep, :]
    mlp_x = raw_mlp_spat["Comprehensive"]["feature_data"].iloc[:n_rep].copy()

    lstm_s = raw_lstm_spat["Comprehensive"]["shap_vals"][:n_rep, :]
    lstm_x = raw_lstm_spat["Comprehensive"]["feature_data"].iloc[:n_rep].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 6.0))
    rng = np.random.default_rng(123)

    for ax, s_mat, x_df, title_str, show_ylabel in [
        (axes[0], mlp_s, mlp_x, "A: MLP_SPAT (Feedforward)", True),
        (axes[1], lstm_s, lstm_x, "B: LSTM_SPAT (Recurrent)", False),
    ]:
        feat_names = list(x_df.columns)
        mean_abs = np.mean(np.abs(s_mat), axis=0)
        order_idx = np.argsort(mean_abs)  # ascending from bottom to top

        y_positions = np.arange(len(feat_names))
        ax.axvline(0.0, color="grey", linestyle="-", linewidth=0.9, zorder=1)

        for pos_i, f_i in enumerate(order_idx):
            s_col = s_mat[:, f_i]
            f_col = x_df.iloc[:, f_i].values
            # Min-max normalize feature values [0, 1] using 2nd-98th percentiles for clean Viridis color mapping
            q02, q98 = np.nanpercentile(f_col, [2, 98])
            norm_f = np.clip((f_col - q02) / max(q98 - q02, 1e-6), 0.0, 1.0)

            # Density-scaled vertical jitter to mimic beeswarm distribution
            hist_counts, bin_edges = np.histogram(s_col, bins=35)
            bin_idx = np.clip(np.digitize(s_col, bin_edges) - 1, 0, len(hist_counts) - 1)
            rel_width = hist_counts[bin_idx] / max(hist_counts.max(), 1)
            y_jitter = rng.uniform(-0.32, 0.32, size=len(s_col)) * rel_width

            sc = ax.scatter(
                s_col, np.full_like(s_col, pos_i, dtype=np.float64) + y_jitter,
                c=norm_f, cmap="viridis", vmin=0.0, vmax=1.0,
                s=10, alpha=0.75, edgecolors="none", zorder=3
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            [f"{feat_names[i]}   {mean_abs[i]:.4f}" for i in order_idx],
            fontsize=10.5, fontweight="bold"
        )
        ax.set_title(title_str, fontsize=13.5, fontweight="bold", loc="left", pad=10)
        ax.set_xlabel("SHAP Value (Impact on TTI)", fontsize=12, fontweight="bold")
        if show_ylabel:
            ax.set_ylabel("Environmental Predictors", fontsize=12, fontweight="bold")
        ax.grid(axis="x", linestyle="--", color="grey", alpha=0.35, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(sc, ax=axes[1], fraction=0.045, pad=0.04)
    cbar.set_label("Feature Value (Normalized)", fontsize=11, fontweight="bold")

    fig.suptitle("Spatial SHAP Distribution: Comprehensive Suite", fontsize=15.5, fontweight="bold", y=0.97)
    out_file = os.path.join(fig_dir, "Figure4_Spatial_SHAP_Beeswarm.png")
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.savefig(out_file, dpi=500, facecolor="white")
    plt.close(fig)
    print(f"Figure 4 successfully saved to: {out_file}")


def main():
    in_dir = "./Output" if os.path.exists("./Output") else "./"
    fig_dir = "./Figures"
    os.makedirs(fig_dir, exist_ok=True)

    sum_mlp_spat = load_pickle(os.path.join(in_dir, "Summary_MLP_SPAT.pkl"))
    sum_lstm_spat = load_pickle(os.path.join(in_dir, "Summary_LSTM_SPAT.pkl"))
    sum_mlp_time = load_pickle(os.path.join(in_dir, "Summary_MLP_TIME.pkl"))
    sum_lstm_time = load_pickle(os.path.join(in_dir, "Summary_LSTM_TIME.pkl"))
    raw_mlp_spat = load_pickle(os.path.join(in_dir, "SHAP_RAW_MLP_SPAT.pkl"))
    raw_lstm_spat = load_pickle(os.path.join(in_dir, "SHAP_RAW_LSTM_SPAT.pkl"))

    plot_figure_2(sum_mlp_time, sum_lstm_time, fig_dir)
    plot_figure_3(sum_mlp_spat, sum_lstm_spat, fig_dir)
    plot_figure_4(raw_mlp_spat, raw_lstm_spat, fig_dir)


if __name__ == "__main__":
    main()
