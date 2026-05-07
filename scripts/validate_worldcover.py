"""
ESA WorldCover 2021 vs RF Prediction 2023 — Validation Script
Reprojects WorldCover tiles to Hunza 5000x2978 grid, remaps classes,
compares pixel-by-pixel with rf_prediction_2023.tif, reports agreement stats.
"""

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = Path(r"e:\zaheer-work")
WC_DIR     = BASE / "Hunza_GIS_Data_Extraction" / "worldcover"
RF_PRED    = BASE / "Hunza_GIS_ML_Implementation" / "outputs" / "rf_prediction_2023.tif"
OUT_DIR    = BASE / "Hunza_GIS_ML_Implementation" / "outputs" / "worldcover_validation"
FIG_DIR    = BASE / "Hunza_GIS_ML_Implementation" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TILE_PATHS = [
    WC_DIR / "ESA_WorldCover_N36E072.tif",
    WC_DIR / "ESA_WorldCover_N36E075.tif",
]

# ── Class definitions ─────────────────────────────────────────────────────────
# RF classes (0-4)
RF_CLASSES = {0: "Snow/Ice", 1: "Bare Rock", 2: "Sparse Veg", 3: "Moderate Veg", 4: "Dense Veg"}

# ESA WorldCover 2021 class values
ESA_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse veg",
    70: "Snow and ice",
    80: "Permanent water",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

# ESA → RF class mapping (best-effort for high-altitude alpine region)
# This is the critical mapping — documented with rationale
ESA_TO_RF = {
    10:  4,   # Tree cover          → Dense Vegetation
    20:  2,   # Shrubland           → Sparse Vegetation (alpine shrubs)
    30:  2,   # Grassland           → Sparse Vegetation (alpine grassland)
    40:  3,   # Cropland            → Moderate Vegetation (irrigated fields)
    50: -1,   # Built-up            → EXCLUDED (urban, negligible in Hunza)
    60:  1,   # Bare/sparse veg     → Bare Rock
    70:  0,   # Snow and ice        → Snow/Ice
    80: -1,   # Permanent water     → EXCLUDED (Water class)
    90:  2,   # Herbaceous wetland  → Sparse Vegetation
    95: -1,   # Mangroves           → EXCLUDED (not present in Hunza)
    100: 2,   # Moss and lichen     → Sparse Vegetation
}

# RF class colors for maps
RF_COLORS = {
    0: "#A8D8EA",  # Snow/Ice       — light blue
    1: "#8B7355",  # Bare Rock      — brown
    2: "#90EE90",  # Sparse Veg     — light green
    3: "#228B22",  # Moderate Veg   — forest green
    4: "#006400",  # Dense Veg      — dark green
}

def load_and_merge_tiles():
    """Merge the two ESA WorldCover tiles into one mosaic."""
    print("Loading ESA WorldCover tiles...")
    datasets = [rasterio.open(p) for p in TILE_PATHS if p.exists()]
    if not datasets:
        raise FileNotFoundError(f"No WorldCover tiles found in {WC_DIR}")
    if len(datasets) == 1:
        print("  Only one tile found — using it directly")
        mosaic = datasets[0].read(1)
        profile = datasets[0].profile.copy()
        transform = datasets[0].transform
        crs = datasets[0].crs
        for d in datasets:
            d.close()
        return mosaic, transform, crs, profile
    mosaic_data, mosaic_transform = merge(datasets)
    profile = datasets[0].profile.copy()
    profile.update({"transform": mosaic_transform, "width": mosaic_data.shape[2], "height": mosaic_data.shape[1]})
    crs = datasets[0].crs
    for d in datasets:
        d.close()
    print(f"  Merged mosaic: {mosaic_data.shape[1]}×{mosaic_data.shape[2]} pixels")
    return mosaic_data[0], mosaic_transform, crs, profile


def reproject_to_rf_grid(mosaic, mosaic_transform, mosaic_crs):
    """Reproject and clip ESA WorldCover to exactly match RF prediction grid."""
    print("Reprojecting to RF prediction grid (5000×2978)...")
    with rasterio.open(RF_PRED) as rf_src:
        dst_crs       = rf_src.crs
        dst_transform = rf_src.transform
        dst_width     = rf_src.width
        dst_height    = rf_src.height
        rf_bounds     = rf_src.bounds

    wc_reprojected = np.zeros((dst_height, dst_width), dtype=np.uint8)

    reproject(
        source=mosaic,
        destination=wc_reprojected,
        src_transform=mosaic_transform,
        src_crs=mosaic_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,  # nearest-neighbour for categorical data
    )
    print(f"  Reprojected to {dst_height}×{dst_width}, CRS: {dst_crs}")
    return wc_reprojected, dst_transform, dst_crs, dst_width, dst_height


def remap_esa_to_rf(wc_array):
    """Convert ESA class values to RF class IDs (0-4), -1 = excluded."""
    print("Remapping ESA classes to RF classes...")
    rf_mapped = np.full_like(wc_array, -1, dtype=np.int8)
    for esa_val, rf_val in ESA_TO_RF.items():
        rf_mapped[wc_array == esa_val] = rf_val

    unique, counts = np.unique(wc_array, return_counts=True)
    print("  ESA class distribution in Hunza extent:")
    total = wc_array.size
    for u, c in zip(unique, counts):
        name = ESA_CLASSES.get(int(u), f"Unknown({u})")
        rf_eq = ESA_TO_RF.get(int(u), -1)
        rf_name = RF_CLASSES.get(rf_eq, "EXCLUDED") if rf_eq != -1 else "EXCLUDED"
        print(f"    ESA {u:3d} ({name:25s}) → RF {rf_eq} ({rf_name:15s}): {c/total*100:.2f}%")
    return rf_mapped


def compute_agreement(rf_array, wc_rf_array):
    """Compare RF predictions vs WorldCover remapped classes."""
    print("\nComputing agreement statistics...")

    # Mask: both must be valid (0-4), exclude nodata and excluded ESA classes
    valid_mask = (rf_array >= 0) & (rf_array <= 4) & (wc_rf_array >= 0) & (wc_rf_array <= 4)
    rf_valid  = rf_array[valid_mask]
    wc_valid  = wc_rf_array[valid_mask]

    n_valid   = valid_mask.sum()
    n_total   = rf_array.size
    n_excluded = n_total - n_valid
    print(f"  Valid comparison pixels: {n_valid:,} / {n_total:,} ({n_valid/n_total*100:.1f}%)")
    print(f"  Excluded (water/built-up/nodata): {n_excluded:,} ({n_excluded/n_total*100:.1f}%)")

    overall_acc = (rf_valid == wc_valid).mean() * 100
    kappa = cohen_kappa_score(wc_valid, rf_valid)

    print(f"\n  Overall Agreement: {overall_acc:.2f}%")
    print(f"  Cohen's Kappa:     {kappa:.4f}")

    # Per-class breakdown
    cm = confusion_matrix(wc_valid, rf_valid, labels=[0, 1, 2, 3, 4])
    report = classification_report(
        wc_valid, rf_valid,
        labels=[0, 1, 2, 3, 4],
        target_names=[RF_CLASSES[i] for i in range(5)],
        zero_division=0
    )
    print("\n  Per-class report (WorldCover as reference):")
    print(report)

    return rf_valid, wc_valid, cm, overall_acc, kappa, valid_mask


def save_outputs(wc_reprojected, wc_rf_mapped, valid_mask, dst_transform, dst_crs, dst_width, dst_height):
    """Save reprojected WorldCover rasters for QGIS inspection."""
    profile = {
        "driver": "GTiff", "dtype": "uint8", "width": dst_width,
        "height": dst_height, "count": 1, "crs": dst_crs,
        "transform": dst_transform, "compress": "lzw",
    }
    out_raw = OUT_DIR / "worldcover_reprojected_hunza.tif"
    with rasterio.open(out_raw, "w", **profile) as dst:
        dst.write(wc_reprojected, 1)
    print(f"\n  Saved reprojected WorldCover (raw ESA values): {out_raw}")

    profile["dtype"] = "int8"
    out_mapped = OUT_DIR / "worldcover_rf_classes_hunza.tif"
    with rasterio.open(out_mapped, "w", **profile) as dst:
        dst.write(wc_rf_mapped.astype(np.int8), 1)
    print(f"  Saved WorldCover remapped to RF classes:       {out_mapped}")


def plot_confusion_matrix(cm, overall_acc, kappa):
    """Save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    class_names = [RF_CLASSES[i] for i in range(5)]
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = cm_norm / row_sums * 100

    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                cbar_kws={"label": "% of WorldCover class"})
    ax.set_xlabel("RF Prediction 2023")
    ax.set_ylabel("ESA WorldCover 2021 (reference)")
    ax.set_title(f"RF vs ESA WorldCover Agreement\nOverall: {overall_acc:.1f}%  |  Kappa: {kappa:.3f}")
    plt.tight_layout()
    out = FIG_DIR / "worldcover_confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix: {out}")


def plot_class_comparison(rf_valid, wc_valid, overall_acc, kappa):
    """Bar chart: per-class agreement rates."""
    class_names = [RF_CLASSES[i] for i in range(5)]
    agreements = []
    counts = []
    for c in range(5):
        mask = wc_valid == c
        n = mask.sum()
        counts.append(n)
        agreements.append((rf_valid[mask] == c).mean() * 100 if n > 0 else 0)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(class_names, agreements,
                  color=[RF_COLORS[i] for i in range(5)], edgecolor="black", linewidth=0.8)
    for bar, count, pct in zip(bars, counts, agreements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{pct:.1f}%\n(n={count:,})", ha="center", va="bottom", fontsize=8)
    ax.axhline(overall_acc, color="red", linestyle="--", linewidth=1.5, label=f"Overall: {overall_acc:.1f}%")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Agreement with ESA WorldCover (%)")
    ax.set_title(f"Per-Class Agreement: RF Prediction 2023 vs ESA WorldCover 2021\nKappa: {kappa:.3f}")
    ax.legend()
    plt.tight_layout()
    out = FIG_DIR / "worldcover_per_class_agreement.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved per-class agreement chart: {out}")


def save_report(rf_valid, wc_valid, cm, overall_acc, kappa):
    """Save text report."""
    lines = [
        "=" * 60,
        "ESA WorldCover 2021 vs RF Prediction 2023 — Validation Report",
        "=" * 60,
        f"Date: 2026-05-02",
        f"RF prediction: {RF_PRED}",
        f"ESA WorldCover tiles: N36E072, N36E075 (v200, 2021)",
        "",
        "CLASS MAPPING (ESA → RF):",
        "  ESA 70 Snow and ice     → RF 0 Snow/Ice",
        "  ESA 60 Bare/sparse veg  → RF 1 Bare Rock",
        "  ESA 20 Shrubland        → RF 2 Sparse Veg",
        "  ESA 30 Grassland        → RF 2 Sparse Veg",
        "  ESA 40 Cropland         → RF 3 Moderate Veg",
        "  ESA 10 Tree cover       → RF 4 Dense Veg",
        "  ESA 80 Water            → EXCLUDED",
        "  ESA 50 Built-up         → EXCLUDED",
        "",
        "AGREEMENT STATISTICS:",
        f"  Overall Agreement: {overall_acc:.2f}%",
        f"  Cohen's Kappa:     {kappa:.4f}",
        f"  Valid pixels:      {len(rf_valid):,}",
        "",
        "CONFUSION MATRIX (rows=WorldCover, cols=RF):",
        str(pd.DataFrame(cm,
            index=[f"WC:{RF_CLASSES[i]}" for i in range(5)],
            columns=[f"RF:{RF_CLASSES[i]}" for i in range(5)])),
        "",
        "INTERPRETATION:",
        "  - Snow/Ice and Bare Rock typically show highest agreement",
        "    (spectrally distinct, ESA mapping is reliable)",
        "  - Vegetation sub-classes may show cross-class confusion",
        "    (ESA uses different thresholds than K-Means NDVI ranges)",
        "  - Discrepancies reflect: (1) different sensor resolution",
        "    (ESA=10m Sentinel-2 vs Landsat=30m), (2) different years",
        "    (ESA=2021 vs RF=2023), (3) class definition differences",
        "=" * 60,
    ]
    out = OUT_DIR / "worldcover_validation_report.txt"
    out.write_text("\n".join(lines))
    print(f"  Saved validation report: {out}")
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("ESA WorldCover 2021 Validation Pipeline")
    print("=" * 60)

    # Step 1: Load and merge ESA tiles
    mosaic, mosaic_transform, mosaic_crs, profile = load_and_merge_tiles()

    # Step 2: Reproject to RF grid
    wc_repr, dst_transform, dst_crs, dst_w, dst_h = reproject_to_rf_grid(mosaic, mosaic_transform, mosaic_crs)

    # Step 3: Remap ESA classes to RF classes
    wc_rf = remap_esa_to_rf(wc_repr)

    # Step 4: Load RF predictions
    print("\nLoading RF prediction 2023...")
    with rasterio.open(RF_PRED) as src:
        rf_pred = src.read(1).astype(np.int8)
    print(f"  RF prediction shape: {rf_pred.shape}")

    # Step 5: Compute agreement
    rf_v, wc_v, cm, acc, kappa, valid_mask = compute_agreement(rf_pred, wc_rf)

    # Step 6: Save outputs
    print("\nSaving outputs...")
    save_outputs(wc_repr, wc_rf, valid_mask, dst_transform, dst_crs, dst_w, dst_h)
    plot_confusion_matrix(cm, acc, kappa)
    plot_class_comparison(rf_v, wc_v, acc, kappa)
    report_text = save_report(rf_v, wc_v, cm, acc, kappa)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(report_text.split("AGREEMENT STATISTICS:")[1].split("CONFUSION MATRIX")[0])
    print("\nDone. Check outputs/worldcover_validation/ and figures/ for results.")


if __name__ == "__main__":
    main()
