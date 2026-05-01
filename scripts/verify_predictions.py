"""
verify_predictions.py — Compare RF predictions with original K-Means classification
===================================================================================
Checks if RF predictions match the original land_cover rasters (distillation validation).
If they differ significantly, there's an issue with feature extraction or band indexing.
"""

import os
import numpy as np
import rasterio
import pandas as pd

# ============================================================================
# PATHS
# ============================================================================

LAND_COVER_RASTERS = {
    2020: r"e:\zaheer-work\Hunza_GIS_Data_Extraction\geospatial_project\outputs\qgis\land_cover.tif",
    2021: r"e:\zaheer-work\Hunza_GIS_Data_Extraction\geospatial_project_2021\outputs\qgis\land_cover_2021.tif",
    2022: r"e:\zaheer-work\Hunza_GIS_Data_Extraction\geospatial_project_2022\outputs\qgis\land_cover_2022.tif",
    2023: r"e:\zaheer-work\Hunza_GIS_Data_Extraction\geospatial_project_2023\outputs\qgis\land_cover_2023.tif",
}

RF_PREDICTIONS = {
    2020: r"e:\zaheer-work\Hunza_GIS_ML_Implementation\outputs\rf_prediction_2020.tif",
    2021: r"e:\zaheer-work\Hunza_GIS_ML_Implementation\outputs\rf_prediction_2021.tif",
    2022: r"e:\zaheer-work\Hunza_GIS_ML_Implementation\outputs\rf_prediction_2022.tif",
    2023: r"e:\zaheer-work\Hunza_GIS_ML_Implementation\outputs\rf_prediction_2023.tif",
}

OUTPUT_DIR = r"e:\zaheer-work\Hunza_GIS_ML_Implementation\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = {
    0: "NoData/Background",
    1: "Snow/Ice",
    2: "Water",
    3: "Bare Rock",
    4: "Sparse Vegetation",
    5: "Moderate Vegetation",
    6: "Dense Vegetation",
}

# Raster value → ML class (verified in CLAUDE.md)
RASTER_TO_ML = {
    0: -1,  # NoData (excluded)
    1: 0,   # Snow/Ice → class 0
    2: -1,  # Water (excluded)
    3: 1,   # Bare Rock → class 1
    4: 2,   # Sparse Veg → class 2
    5: 3,   # Moderate Veg → class 3
    6: 4,   # Dense Veg → class 4
}

ML_CLASS_NAMES = {
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
    print("\n" + "=" * 80)
    print(" VERIFICATION: RF PREDICTIONS vs ORIGINAL K-MEANS CLASSIFICATION")
    print("=" * 80)

    for year in [2020, 2021, 2022, 2023]:
        print(f"\n[{year}]")
        print("-" * 80)

        # Check files exist
        kmeans_path = LAND_COVER_RASTERS[year]
        rf_path = RF_PREDICTIONS[year]

        if not os.path.exists(kmeans_path):
            print(f"  ERROR: K-Means raster not found: {kmeans_path}")
            continue
        if not os.path.exists(rf_path):
            print(f"  ERROR: RF prediction not found: {rf_path}")
            continue

        # Load rasters
        with rasterio.open(kmeans_path) as src:
            kmeans = src.read(1)

        with rasterio.open(rf_path) as src:
            rf_pred = src.read(1)

        # Check dimensions
        if kmeans.shape != rf_pred.shape:
            print(f"  ERROR: Dimension mismatch!")
            print(f"    K-Means shape: {kmeans.shape}")
            print(f"    RF shape:      {rf_pred.shape}")
            continue

        print(f"  Raster shape: {kmeans.shape}")

        # Convert K-Means raster values to ML classes
        kmeans_ml = np.full_like(kmeans, -1, dtype=np.int8)
        for raster_val, ml_class in RASTER_TO_ML.items():
            kmeans_ml[kmeans == raster_val] = ml_class

        # Count pixels per class
        print(f"\n  K-Means Classification (original):")
        print(f"  {'Class':<25} {'Count':>12} {'Percentage':>12}")
        print(f"  {'-'*50}")

        kmeans_counts = {}
        total_valid = 0
        for class_id in range(5):
            count = np.sum(kmeans_ml == class_id)
            kmeans_counts[class_id] = count
            total_valid += count
            pct = count / total_valid * 100 if total_valid > 0 else 0
            print(f"  {ML_CLASS_NAMES[class_id]:<25} {count:>12,} {pct:>11.2f}%")

        print(f"\n  RF Prediction (new):")
        print(f"  {'Class':<25} {'Count':>12} {'Percentage':>12}")
        print(f"  {'-'*50}")

        rf_counts = {}
        total_valid_rf = 0
        for class_id in range(5):
            count = np.sum(rf_pred == class_id)
            rf_counts[class_id] = count
            total_valid_rf += count
            pct = count / total_valid_rf * 100 if total_valid_rf > 0 else 0
            print(f"  {ML_CLASS_NAMES[class_id]:<25} {count:>12,} {pct:>11.2f}%")

        # Compute agreement
        print(f"\n  Comparison:")
        print(f"  {'Class':<25} {'K-Means %':>12} {'RF %':>12} {'Δ':>12}")
        print(f"  {'-'*62}")

        total_agreement = 0
        total_pixels = kmeans_ml.size

        for class_id in range(5):
            kmeans_pct = kmeans_counts[class_id] / total_valid * 100
            rf_pct = rf_counts[class_id] / total_valid_rf * 100
            delta = rf_pct - kmeans_pct
            agreement = np.sum((kmeans_ml == class_id) & (rf_pred == class_id))
            total_agreement += agreement

            print(f"  {ML_CLASS_NAMES[class_id]:<25} {kmeans_pct:>11.2f}% {rf_pct:>11.2f}% {delta:>+11.2f}%")

        # Overall agreement
        agreement_pct = total_agreement / total_valid * 100
        print(f"\n  Overall pixel agreement: {agreement_pct:.2f}%")

        if agreement_pct < 80:
            print(f"  ⚠ WARNING: Low agreement (<80%) suggests feature extraction issue!")
        elif agreement_pct > 99:
            print(f"  ✓ Excellent agreement (>99%) — predictions are very close to original")
        else:
            print(f"  ✓ Good agreement (80-99%) — some variation expected")

    print("\n" + "=" * 80)
    print(" INTERPRETATION")
    print("=" * 80)
    print("""
  • If agreement > 99%: RF successfully distilled K-Means. Ready for temporal analysis.
  • If agreement 80-99%: Some variation — check if trends are similar.
  • If agreement < 80%: Likely band indexing or feature computation issue.

  Common issues if agreement is low:
    1. Band indices off-by-one (checking band 2 instead of band 1, etc.)
    2. Calibration constants wrong (ML or AL values)
    3. NDVI/NDSI/NDWI computed differently than training
    4. Raster dimensions don't match (5376 vs 5000 width)
    """)
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
