# ==============================================================================
# Script: 08_Plot_Figs2_3_4_SHAP_Analysis.py
# Title: Master SHAP Visualization Suite (Figures 2, 3, and 4 - Python)
# Description: Generates:
#                - Figure 2: Temporal Rolling-Origin SHAP Stability (30-Yr Extrapolation)
#                - Figure 3: Spatial 10-Fold Block CV SHAP Stability
#                - Figure 4: 10-Fold Stratified Spatial SHAP Beeswarm Distributions
# Reference: Stips et al. (2026) - Environmental Modelling & Software
# License: GPL-3.0
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tti_common import load_pickle


def get_stratified_indices(n_total: int, n_target: int = 2000, n_folds: int = 10, seed: int = 123) -> np.ndarray:
    """Draw 200 cells uniformly from each of the 10 spatial folds (2,000 total)."""
    if n_total <= n_target:
        return np.arange(n_total, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    fold_len = n_total // n_folds
    per_fold = n_target // n_folds
    idxs = []
    for k in range(n_folds):
        s_i = k * fold_len
        e_i = n_total if k == (n_folds - 1) else (k + 1) * fold_len
        idxs.append(rng.choice(np.arange(s_i, e_i), size=min(per_fold, e_i - s_i), replace=False))
    return np.concatenate(idxs)


def plot_figure_2(mlp_time: dict, lstm_time: dict, fig_dir: str) -> None:
    ms = mlp_time["final_shap"].query("Scenario == 'Comprehensive'").sort_values("SHAP_Mean", ascending=True)
    feat_order = ms["Feature"].tolist()  # bottom-to-top for barh
    ls = lstm_time["final_shap"].query("Scenario == 'Comprehensive'").set_index("Feature").loc[feat_order].reset_index()

    mr = mlp_time["raw_shap"].query("Scenario == 'Comprehensive'")
    lr = lstm_time["raw_shap"].query("Scenario == 'Comprehensive'")

    max_val = max(
        float(mr["SHAP_Mean_Abs"].max()),
        float(lr["SHAP_Mean_Abs"].max()),
        float((ms["SHAP_Mean"] + ms["SHAP_CI"]).max()),
        float((ls["SHAP_Mean"] + ls["SHAP_CI"]).max()),
    )
    max_x = max(0.12, np.ceil(max_val * 40.0) / 40.0)
    x_step = 0.03 if max_x <= 0.12 else 0.025
    xticks = np.round(np.arange(0.0, max_x + 1e-6, x_step), 3)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.5), sharey=True)
    y_pos = np.arange(len(feat_order))
    rng = np.random.default_rng(123)

    panels = [
        (axes[0], ls, lr, "#56B4E9", "LSTM_TIME (Recurrent)"),
        (axes[1], ms, mr, "#E69F00", "MLP_TIME (Feedforward)"),
    ]

    for ax, sum_df, raw_df, color, title_str in panels:
        means = sum_df["SHAP_Mean"].values
        cis = sum_df["SHAP_CI"].values
        lower_err = np.minimum(means, cis)

        ax.barh(y_pos, means, height=0.72, color=color, edgecolor="black", linewidth=0.6, alpha=0.88, zorder=2)
        ax.errorbar(
            means, y_pos, xerr=[lower_err, cis], fmt="none",
            ecolor="black", elinewidth=1.2, capsize=4.0, zorder=3
        )

        for i, f in enumerate(feat_order):
            pts = raw_df.loc[raw_df["Feature"] == f, "SHAP_Mean_Abs"].values
            jit = rng.uniform(-0.14, 0.14, size=len(pts))
            ax.scatter(pts, np.full_like(pts, y_pos[i], dtype=float) + jit,
                       color="black", s=18, alpha=0.65, zorder=4)

        ax.set_xlim(0.0, max_x * 1.04)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.3f}" if x_step == 0.025 else f"{x:.2f}" for x in xticks], fontsize=10.5)
        ax.grid(axis="x", linestyle="--", color="#d9d9d9", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(title_str, fontsize=13, fontweight="bold", pad=10)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(feat_order, fontsize=11.5, fontweight="bold")

    fig.suptitle(
        "Temporal Stability of Global Feature Importance (30-Year Rolling Extrapolation)",
        fontsize=13.5, fontweight="bold", y=0.965
    )
    fig.text(
        0.5, 0.905,
        "Comprehensive Variable Suite (Means \u00b1 95% CI across 6 unseen target pentads)",
        ha="center", fontsize=11.2, color="#4d4d4d"
    )
    fig.supxlabel("Mean Absolute SHAP Value (Impact on TTI)", fontsize=12.5, fontweight="bold", y=0.03)

    out_file = os.path.join(fig_dir, "Figure2_Temporal_SHAP_Stability.png")
    fig.subplots_adjust(left=0.11, right=0.96, bottom=0.14, top=0.83, wspace=0.12)
    fig.savefig(out_file, dpi=500, facecolor="white", bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(f"Figure 2 successfully saved to: {out_file}")


def plot_figure_3(mlp_spat: dict, lstm_spat: dict, fig_dir: str) -> None:
    ms = mlp_spat["final_shap"].query("Scenario == 'Comprehensive'").sort_values("SHAP_Mean", ascending=True)
    feat_order = ms["Feature"].tolist()
    ls = lstm_spat["final_shap"].query("Scenario == 'Comprehensive'").set_index("Feature").loc[feat_order].reset_index()

    mr = mlp_spat["raw_shap"].query("Scenario == 'Comprehensive'")
    lr = lstm_spat["raw_shap"].query("Scenario == 'Comprehensive'")

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    y_pos = np.arange(len(feat_order), dtype=float)
    bar_h = 0.35
    rng = np.random.default_rng(123)

    # Top dodged bar (+bar_h/2): MLP_SPAT; Bottom dodged bar (-bar_h/2): LSTM_SPAT
    ax.barh(y_pos + bar_h / 2, ms["SHAP_Mean"].values, height=bar_h, color="#E69F00",
            edgecolor="black", linewidth=0.6, alpha=0.88, zorder=2)
    ax.errorbar(
        ms["SHAP_Mean"].values, y_pos + bar_h / 2,
        xerr=[np.minimum(ms["SHAP_Mean"].values, ms["SHAP_CI"].values), ms["SHAP_CI"].values],
        fmt="none", ecolor="black", elinewidth=1.2, capsize=3.5, zorder=3
    )

    ax.barh(y_pos - bar_h / 2, ls["SHAP_Mean"].values, height=bar_h, color="#56B4E9",
            edgecolor="black", linewidth=0.6, alpha=0.88, zorder=2)
    ax.errorbar(
        ls["SHAP_Mean"].values, y_pos - bar_h / 2,
        xerr=[np.minimum(ls["SHAP_Mean"].values, ls["SHAP_CI"].values), ls["SHAP_CI"].values],
        fmt="none", ecolor="black", elinewidth=1.2, capsize=3.5, zorder=3
    )

    for i, f in enumerate(feat_order):
        pts_m = mr.loc[mr["Feature"] == f, "SHAP_Mean_Abs"].values
        pts_l = lr.loc[lr["Feature"] == f, "SHAP_Mean_Abs"].values
        ax.scatter(pts_m, y_pos[i] + bar_h / 2 + rng.uniform(-0.08, 0.08, len(pts_m)),
                   facecolor="#E69F00", edgecolor="black", linewidth=0.6, s=20, alpha=0.75, zorder=4)
        ax.scatter(pts_l, y_pos[i] - bar_h / 2 + rng.uniform(-0.08, 0.08, len(pts_l)),
                   facecolor="#56B4E9", edgecolor="black", linewidth=0.6, s=20, alpha=0.75, zorder=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_order, fontsize=11.5, fontweight="bold")
    ax.grid(axis="x", linestyle="--", color="#d9d9d9", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on TTI)", fontsize=12.5, fontweight="bold", labelpad=8)

    fig.suptitle(
        "Spatial Stability of Global Feature Importance (10-Fold Block Cross-Validation)",
        fontsize=13.2, fontweight="bold", y=0.965
    )
    ax.set_title(
        "10-Fold Spatial Block Cross-Validation (Means \u00b1 95% CI with fold-level iterations)",
        fontsize=11.2, color="#4d4d4d", pad=12
    )

    handles = [
        Patch(facecolor="#56B4E9", edgecolor="black", label="LSTM_SPAT (Recurrent)"),
        Patch(facecolor="#E69F00", edgecolor="black", label="MLP_SPAT (Feedforward)"),
    ]
    ax.legend(
        handles=handles, title="ANN Architecture:", title_fontsize=11.5, fontsize=11,
        loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False
    )

    out_file = os.path.join(fig_dir, "Figure3_Spatial_SHAP_Stability.png")
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.19, top=0.86)
    fig.savefig(out_file, dpi=500, facecolor="white", bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(f"Figure 3 successfully saved to: {out_file}")


def _draw_beeswarm_panel(ax, shap_mat: np.ndarray, feat_df: pd.DataFrame, title_str: str):
    mean_abs = np.mean(np.abs(shap_mat), axis=0)
    order = np.argsort(mean_abs)  # ascending (bottom to top)
    feat_names = list(feat_df.columns)
    rng = np.random.default_rng(123)

    sc = None
    for plot_y, col_i in enumerate(order):
        s_vals = shap_mat[:, col_i]
        f_vals = feat_df.iloc[:, col_i].values.astype(np.float64)
        # Robust 1st-99th percentile min-max normalization to [0, 1] matching shapviz
        p_lo, p_hi = np.nanpercentile(f_vals, [1.0, 99.0])
        if p_hi > p_lo:
            f_norm = np.clip((f_vals - p_lo) / (p_hi - p_lo), 0.0, 1.0)
        else:
            f_norm = np.full_like(f_vals, 0.5)

        # Density-weighted vertical jitter (beeswarm silhouette)
        bins = np.linspace(s_vals.min() - 1e-6, s_vals.max() + 1e-6, 45)
        bin_idx = np.clip(np.digitize(s_vals, bins) - 1, 0, len(bins) - 2)
        counts = np.bincount(bin_idx, minlength=len(bins) - 1)
        rel_width = counts[bin_idx] / max(counts.max(), 1)
        y_jit = plot_y + rng.uniform(-0.34, 0.34, size=len(s_vals)) * rel_width

        sc = ax.scatter(
            s_vals, y_jit, c=f_norm, cmap="viridis", vmin=0.0, vmax=1.0,
            s=11, alpha=0.80, linewidths=0.0, rasterized=True, zorder=3
        )

    ax.axvline(0.0, color="grey", linewidth=0.9, zorder=2)
    ax.grid(axis="x", linestyle="--", color="#d9d9d9", zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([feat_names[i] for i in order], fontsize=11.5, fontweight="bold")
    ax.set_xlabel("SHAP Value (Impact on TTI)", fontsize=12, fontweight="bold")
    ax.set_title(title_str, loc="left", fontsize=13.5, fontweight="bold", pad=10)

    x_min, x_max = ax.get_xlim()
    span = x_max - x_min
    ax.set_xlim(x_min - 0.16 * span, x_max + 0.04 * span)
    x_txt = x_min - 0.14 * span
    for plot_y, col_i in enumerate(order):
        ax.text(x_txt, plot_y, f"{mean_abs[col_i]:.4f}", va="center", ha="left", fontsize=9.5, color="#222222")

    return sc


def plot_figure_4(raw_mlp: dict, raw_lstm: dict, fig_dir: str) -> None:
    s_m_all = raw_mlp["Comprehensive"]["shap_vals"]
    x_m_all = raw_mlp["Comprehensive"]["feature_data"]
    s_l_all = raw_lstm["Comprehensive"]["shap_vals"]
    x_l_all = raw_lstm["Comprehensive"]["feature_data"]

    idx_m = get_stratified_indices(len(s_m_all), n_target=2000, n_folds=10, seed=123)
    idx_l = get_stratified_indices(len(s_l_all), n_target=2000, n_folds=10, seed=123)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.0))
    _draw_beeswarm_panel(axes[0], s_m_all[idx_m], x_m_all.iloc[idx_m], "A: MLP_SPAT (Feedforward)")
    axes[0].set_ylabel("Environmental Predictors", fontsize=12.5, fontweight="bold")

    sc = _draw_beeswarm_panel(axes[1], s_l_all[idx_l], x_l_all.iloc[idx_l], "B: LSTM_SPAT (Recurrent)")

    fig.suptitle("Spatial SHAP Distribution: Comprehensive Suite", fontsize=15.5, fontweight="bold", y=0.965)
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.12, top=0.86, wspace=0.25)

    cbar_ax = fig.add_axes([0.92, 0.22, 0.018, 0.52])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("Feature Value\n(Normalized)", fontsize=11, fontweight="bold")

    out_file = os.path.join(fig_dir, "Figure4_Spatial_SHAP_Beeswarm.png")
    fig.savefig(out_file, dpi=500, facecolor="white", bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(f"Figure 4 successfully saved to: {out_file}")


def main() -> None:
    in_dir = "./Output" if os.path.exists("./Output") else "./"
    fig_dir = "./Figures"
    os.makedirs(fig_dir, exist_ok=True)

    mlp_spat = load_pickle(os.path.join(in_dir, "Summary_MLP_SPAT.pkl"))
    lstm_spat = load_pickle(os.path.join(in_dir, "Summary_LSTM_SPAT.pkl"))
    mlp_time = load_pickle(os.path.join(in_dir, "Summary_MLP_TIME.pkl"))
    lstm_time = load_pickle(os.path.join(in_dir, "Summary_LSTM_TIME.pkl"))
    raw_mlp_spat = load_pickle(os.path.join(in_dir, "SHAP_RAW_MLP_SPAT.pkl"))
    raw_lstm_spat = load_pickle(os.path.join(in_dir, "SHAP_RAW_LSTM_SPAT.pkl"))

    plot_figure_2(mlp_time, lstm_time, fig_dir)
    plot_figure_3(mlp_spat, lstm_spat, fig_dir)
    plot_figure_4(raw_mlp_spat, raw_lstm_spat, fig_dir)


if __name__ == "__main__":
    main()
