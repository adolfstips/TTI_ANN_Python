# ==============================================================================
# Script: 07_Plot_Fig1_Spatial_Block_Map.py
# Title: Generate Figure 1 (Spatial Block Cross-Validation Design Map)
# Description: Maps the exact 10x10 pixel block allocation for Fold 1 (80% Train,
#              10% Validation, 10% Testing) with geographic context.
# Reference: Stips et al. (2026) - Environmental Modelling & Software
# License: GPL-3.0
# ==============================================================================

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from tti_common import load_baltic_data, assign_kfold_blocks

warnings.filterwarnings("ignore")

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


def main():
    fig_dir = "./Figures"
    os.makedirs(fig_dir, exist_ok=True)
    rfile = "./GES_Baltic_5year_ALLE_Combined_4D.nc"

    baltic = load_baltic_data(rfile, block_size=10, ntime=14)
    spatial_blocks = baltic["spatial_blocks"]
    nlat, nlon = baltic["grid_shape"]
    lats = baltic["lats"]
    lons = baltic["lons"]
    rows_0 = baltic["rows_0"]
    cols_0 = baltic["cols_0"]

    fold_ids = assign_kfold_blocks(spatial_blocks, n_splits=10, seed=123)
    k = 1       # Test Fold 1
    val_k = 2   # Validation Fold 2

    split_assignment = np.ones(len(spatial_blocks), dtype=np.float64)  # 1: Training (80%)
    split_assignment[fold_ids == val_k] = 2.0                          # 2: Validation (10%)
    split_assignment[fold_ids == k] = 3.0                              # 3: Testing (10%)

    grid_2d = np.full((nlat, nlon), np.nan, dtype=np.float64)
    grid_2d[rows_0, cols_0] = split_assignment

    cmap = ListedColormap(["#E69F00", "#56B4E9", "#009E73"])
    cmap.set_bad(color="white", alpha=0.0)

    dlon = abs(lons[1] - lons[0]) / 2.0 if len(lons) > 1 else 0.033
    dlat = abs(lats[1] - lats[0]) / 2.0 if len(lats) > 1 else 0.016
    extent = [lons.min() - dlon, lons.max() + dlon, lats.min() - dlat, lats.max() + dlat]

    # At mean latitude 59.75 deg N, 1 / cos(59.75 deg) = 1.985 (matches ggplot2::coord_sf)
    baltic_aspect = 1.985

    if HAS_CARTOPY:
        fig, ax = plt.subplots(
            figsize=(8.5, 9.5),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        ax.set_facecolor("#f0f8ff")
        ax.imshow(
            grid_2d,
            origin="upper",
            extent=extent,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=0.5,
            vmax=3.5,
            interpolation="nearest",
            aspect=baltic_aspect,
            zorder=2,
        )
        try:
            ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.5, edgecolor="black", zorder=3)
            ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.4, edgecolor="black", zorder=3)
            ax.add_feature(cfeature.RIVERS.with_scale("10m"), linewidth=0.25, edgecolor="#1f78b4", zorder=2)
        except Exception:
            pass
        ax.set_extent([9.5, 30.5, 53.5, 66.0], crs=ccrs.PlateCarree())
    else:
        fig, ax = plt.subplots(figsize=(8.5, 9.5))
        ax.set_facecolor("#f0f8ff")
        ax.imshow(
            grid_2d,
            origin="upper",
            extent=extent,
            cmap=cmap,
            vmin=0.5,
            vmax=3.5,
            interpolation="nearest",
            aspect=baltic_aspect,
        )
        ax.set_xlim(9.5, 30.5)
        ax.set_ylim(53.5, 66.0)

    # IMPORTANT: Set aspect AFTER imshow() because imshow() resets aspect to 'equal' (1.0) by default
    ax.set_aspect(baltic_aspect, adjustable="box")

    ax.set_xticks(range(10, 31, 5))
    ax.set_xticklabels([f"{x}°E" for x in range(10, 31, 5)], fontsize=11.5)
    ax.set_yticks(range(54, 67, 2))
    ax.set_yticklabels([f"{y}°N" for y in range(54, 67, 2)], fontsize=11.5)
    ax.grid(True, linestyle="--", color="grey", alpha=0.35, zorder=1)
    ax.set_xlabel("Longitude", fontsize=13.5, fontweight="bold", labelpad=8)
    ax.set_ylabel("Latitude", fontsize=13.5, fontweight="bold", labelpad=8)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    fig.suptitle("Spatial Block Cross-Validation Design", fontsize=16.5, fontweight="bold", y=0.965)
    ax.set_title(
        "Baltic Sea partitioned into 10x10 pixel blocks (40' Lon × 20' Lat; ~31-43 km)",
        fontsize=12.2,
        color="#4d4d4d",
        pad=12,
    )

    # Single-line bottom legend matching ggplot2's guides(fill = guide_legend(nrow = 1))
    legend_handles = [
        Patch(facecolor="none", edgecolor="none", label="10-Fold Split:"),
        Patch(facecolor="#E69F00", edgecolor="black", label="Training Blocks (80%)"),
        Patch(facecolor="#56B4E9", edgecolor="black", label="Validation Blocks (10%)"),
        Patch(facecolor="#009E73", edgecolor="black", label="Testing Blocks (10%)"),
    ]
    leg = ax.legend(
        handles=legend_handles,
        fontsize=11.5,
        loc="upper center",
        bbox_to_anchor=(0.48, -0.075),
        ncol=4,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.1,
    )
    leg.get_texts()[0].set_fontweight("bold")
    leg.get_texts()[0].set_fontsize(12)

    out_file = os.path.join(fig_dir, "Figure1_Spatial_Block_Geographic.png")
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.90)
    fig.savefig(out_file, dpi=500, facecolor="white", bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(f"Figure 1 successfully saved to: {out_file}")


if __name__ == "__main__":
    main()
