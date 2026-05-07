"""
export_temporal_geotiff.py — Export temporal analysis as GeoTIFF for QGIS
========================================================================
Creates visualization rasters:
  - change_2020_2023.tif: 0=stable, 1=changed
  - class_2020.tif: 2020 classification
  - class_2023.tif: 2023 classification
  - persistence_2020_2023.tif: persistence percentage (0-100)

Run:  python export_temporal_geotiff.py
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds

# ============================================================================
# PATHS
# ============================================================================

PRED_DIR = r"e:\zaheer-work\Hunza_GIS_ML_Implementation\outputs"
OUTPUT_DIR = PRED_DIR  # Save GeoTIFFs in main outputs dir

CLASS_NAMES = {
    0: "Snow/Ice",
    1: "Bare Rock",
    2: "Sparse Vegetation",
    3: "Moderate Vegetation",
    4: "Dense Vegetation",
}

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print(" EXPORTING TEMPORAL ANALYSIS AS GEOTIFF")
    print("=" * 70)

    # Load 2020 and 2023 predictions with metadata
    print(f"\n[1/5] Loading 2020 and 2023 predictions...")
    pred_2020_path = os.path.join(PRED_DIR, "rf_prediction_2020.tif")
    pred_2023_path = os.path.join(PRED_DIR, "rf_prediction_2023.tif")

    with rasterio.open(pred_2020_path) as src:
        pred_2020 = src.read(1).astype(np.uint8)
        profile = src.profile.copy()

    with rasterio.open(pred_2023_path) as src:
        pred_2023 = src.read(1).astype(np.uint8)

    print(f"  2020 shape: {pred_2020.shape}")
    print(f"  2023 shape: {pred_2023.shape}")

    # 1. Change mask (0=stable, 1=changed)
    print(f"\n[2/5] Creating change mask...")
    change_mask = (pred_2020 != pred_2023).astype(np.uint8)
    changed_count = np.sum(change_mask)
    total_pixels = change_mask.size
    pct_changed = (changed_count / total_pixels) * 100

    out_path = os.path.join(OUTPUT_DIR, "change_2020_2023.tif")
    profile_uint8 = profile.copy()
    profile_uint8.update({"dtype": "uint8", "count": 1})
    with rasterio.open(out_path, "w", **profile_uint8) as dst:
        dst.write(change_mask, 1)
    print(f"  Saved: {out_path}")
    print(f"    Changed pixels: {changed_count:,} ({pct_changed:.2f}%)")
    print(f"    Stable pixels: {total_pixels - changed_count:,} ({100-pct_changed:.2f}%)")

    # 2. Persistence map (% of pixels that stayed same class by locality)
    print(f"\n[3/5] Creating persistence map (% stable by region)...")
    persistence = np.zeros_like(pred_2020, dtype=np.float32)
    window_size = 11  # 11x11 pixel window

    for i in range(pred_2020.shape[0]):
        if i % 500 == 0:
            print(f"    Row {i}/{pred_2020.shape[0]}...")
        for j in range(pred_2020.shape[1]):
            # Local window
            i_min = max(0, i - window_size // 2)
            i_max = min(pred_2020.shape[0], i + window_size // 2 + 1)
            j_min = max(0, j - window_size // 2)
            j_max = min(pred_2020.shape[1], j + window_size // 2 + 1)

            window_2020 = pred_2020[i_min:i_max, j_min:j_max]
            window_2023 = pred_2023[i_min:i_max, j_min:j_max]

            stable = np.sum(window_2020 == window_2023)
            total = window_2020.size
            persistence[i, j] = (stable / total) * 100

    out_path = os.path.join(OUTPUT_DIR, "persistence_2020_2023.tif")
    profile_float32 = profile.copy()
    profile_float32.update({"dtype": "float32", "count": 1})
    with rasterio.open(out_path, "w", **profile_float32) as dst:
        dst.write(persistence, 1)
    print(f"  Saved: {out_path}")
    print(f"    Mean persistence: {persistence.mean():.1f}%")
    print(f"    Min persistence: {persistence.min():.1f}%")
    print(f"    Max persistence: {persistence.max():.1f}%")

    # 3. Copy 2020 classification
    print(f"\n[4/5] Copying 2020 classification...")
    out_path = os.path.join(OUTPUT_DIR, "class_2020.tif")
    with rasterio.open(out_path, "w", **profile_uint8) as dst:
        dst.write(pred_2020, 1)
    print(f"  Saved: {out_path}")

    # 4. Copy 2023 classification
    print(f"\n[5/5] Copying 2023 classification...")
    out_path = os.path.join(OUTPUT_DIR, "class_2023.tif")
    with rasterio.open(out_path, "w", **profile_uint8) as dst:
        dst.write(pred_2023, 1)
    print(f"  Saved: {out_path}")

    # Summary
    print("\n" + "=" * 70)
    print(" EXPORT COMPLETE")
    print("=" * 70)
    print(f"\n  Files saved in: {OUTPUT_DIR}")
    print(f"\n  Open in QGIS:")
    print(f"    1. Add Raster Layer")
    print(f"    2. Select each .tif file below")
    print(f"\n  Files:")
    print(f"    • change_2020_2023.tif")
    print(f"      → 0 (black) = stable | 1 (white) = changed")
    print(f"    • class_2020.tif")
    print(f"      → 0=Snow/Ice | 1=Bare Rock | 2=Sparse Veg | 3=Moderate Veg | 4=Dense Veg")
    print(f"    • class_2023.tif")
    print(f"      → Same class mapping as 2020")
    print(f"    • persistence_2020_2023.tif")
    print(f"      → 0-100 = % of local pixels that stayed same class")
    print(f"\n  Color map suggestion (in QGIS):")
    print(f"    • change_2020_2023.tif: Render → Single band gray")
    print(f"    • class_*.tif: Render → Paletted, set colors manually")
    print(f"    • persistence_*.tif: Render → Singleband pseudocolor, invert")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
