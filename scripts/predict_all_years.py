"""
predict_all_years.py — Apply trained RF to 2020-2023 rasters
==============================================================
Mirrors extract_corrected.py exactly:
  - Bands B2-B7 read from raw rasters, resampled 5376→5000, calibrated
  - NDVI/NDSI/NDWI loaded from QGIS pre-computed rasters (same source as training)
  - Prediction written as GeoTIFF at 2978×5000

Run:  python predict_all_years.py
"""

import os
import time
import numpy as np
import joblib
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# PATHS  — must match extract_corrected.py exactly
# ============================================================================

MODEL_PATH = r"e:\zaheer-work\Hunza_GIS_ML_Implementation\models\rf_model_distillation.pkl"
OUTPUT_DIR = r"e:\zaheer-work\Hunza_GIS_ML_Implementation\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE = r"e:\zaheer-work\Hunza_GIS_Data_Extraction"

LAND_COVER = {
    2020: os.path.join(BASE, r"geospatial_project\outputs\qgis\land_cover.tif"),
    2021: os.path.join(BASE, r"geospatial_project_2021\outputs\qgis\land_cover_2021.tif"),
    2022: os.path.join(BASE, r"geospatial_project_2022\outputs\qgis\land_cover_2022.tif"),
    2023: os.path.join(BASE, r"geospatial_project_2023\outputs\qgis\land_cover_2023.tif"),
}
RAW_RASTERS = {
    2020: os.path.join(BASE, r"geospatial_project\data\raw\Hunza_2020-0000000000-0000000000.tif"),
    2021: os.path.join(BASE, r"geospatial_project_2021\data\raw\Hunza_2021-0000000000-0000000000.tif"),
    2022: os.path.join(BASE, r"geospatial_project_2022\data\raw\Hunza_2022-0000000000-0000000000.tif"),
    2023: os.path.join(BASE, r"geospatial_project_2023\data\raw\Hunza_2023-0000000000-0000000000.tif"),
}
NDVI_RASTERS = {
    2020: os.path.join(BASE, r"geospatial_project\outputs\qgis\ndvi.tif"),
    2021: os.path.join(BASE, r"geospatial_project_2021\outputs\qgis\ndvi_2021.tif"),
    2022: os.path.join(BASE, r"geospatial_project_2022\outputs\qgis\ndvi_2022.tif"),
    2023: os.path.join(BASE, r"geospatial_project_2023\outputs\qgis\ndvi_2023.tif"),
}
NDSI_RASTERS = {
    2020: os.path.join(BASE, r"geospatial_project\outputs\qgis\ndsi.tif"),
    2021: os.path.join(BASE, r"geospatial_project_2021\outputs\qgis\ndsi_2021.tif"),
    2022: os.path.join(BASE, r"geospatial_project_2022\outputs\qgis\ndsi_2022.tif"),
    2023: os.path.join(BASE, r"geospatial_project_2023\outputs\qgis\ndsi_2023.tif"),
}
NDWI_RASTERS = {
    2020: os.path.join(BASE, r"geospatial_project\outputs\qgis\ndwi.tif"),
    2021: os.path.join(BASE, r"geospatial_project_2021\outputs\qgis\ndwi_2021.tif"),
    2022: os.path.join(BASE, r"geospatial_project_2022\outputs\qgis\ndwi_2022.tif"),
    2023: os.path.join(BASE, r"geospatial_project_2023\outputs\qgis\ndwi_2023.tif"),
}

# Band indices (1-indexed, rasterio convention) — same as extract_corrected.py
BAND_INDICES = [2, 3, 4, 5, 6, 7]
BAND_NAMES   = ["B2_Blue", "B3_Green", "B4_Red", "B5_NIR", "B6_SWIR1", "B7_SWIR2"]
FEATURE_ORDER = BAND_NAMES + ["NDVI", "NDSI", "NDWI"]

# Landsat 8 Collection 2 Level-2 calibration — same as extract_corrected.py
DN_SCALE  = 0.0000275
DN_OFFSET = -0.2

CLASS_NAMES = {
    0: "Snow/Ice",
    1: "Bare Rock",
    2: "Sparse Vegetation",
    3: "Moderate Vegetation",
    4: "Dense Vegetation",
}

# ============================================================================
# UTILITY — copied from extract_corrected.py
# ============================================================================

def calibrate(dn_array):
    """Convert Landsat 8 SR DN to surface reflectance, clipped to [0, 1]."""
    refl = dn_array.astype(np.float32) * DN_SCALE + DN_OFFSET
    return np.clip(refl, 0.0, 1.0)


def resample_to_grid(src_path, target_meta, band_idx=1):
    """Read one band and reproject to match target_meta (height, width, crs, transform)."""
    dst = np.empty((target_meta["height"], target_meta["width"]), dtype=np.float32)
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, band_idx),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_meta["transform"],
            dst_crs=target_meta["crs"],
            resampling=Resampling.bilinear,
        )
    return dst


def load_features(year):
    """Load all 9 features for a year, exactly matching extract_corrected.py."""

    # 1. Land cover raster defines the target grid (2978×5000)
    with rasterio.open(LAND_COVER[year]) as src:
        lc_profile = src.profile.copy()
        target_meta = {
            "height":    src.height,
            "width":     src.width,
            "crs":       src.crs,
            "transform": src.transform,
        }
    print(f"    Target grid: {target_meta['height']} × {target_meta['width']}")

    # 2. Raw spectral bands (resample 5376→5000)
    features = {}
    raw_path = RAW_RASTERS[year]
    print(f"    Loading bands B2–B7 from raw raster...")
    for band_idx, band_name in zip(BAND_INDICES, BAND_NAMES):
        resampled = resample_to_grid(raw_path, target_meta, band_idx=band_idx)
        features[band_name] = calibrate(resampled)
        print(f"      {band_name}: min={features[band_name].min():.3f}  max={features[band_name].max():.3f}")

    # 3. NDVI/NDSI/NDWI from QGIS pre-computed rasters (same as training!)
    print(f"    Loading NDVI/NDSI/NDWI from QGIS rasters...")
    with rasterio.open(NDVI_RASTERS[year]) as src:
        features["NDVI"] = src.read(1).astype(np.float32)
    with rasterio.open(NDSI_RASTERS[year]) as src:
        features["NDSI"] = src.read(1).astype(np.float32)
    with rasterio.open(NDWI_RASTERS[year]) as src:
        features["NDWI"] = src.read(1).astype(np.float32)

    print(f"      NDVI: min={features['NDVI'].min():.3f}  max={features['NDVI'].max():.3f}")
    print(f"      NDSI: min={features['NDSI'].min():.3f}  max={features['NDSI'].max():.3f}")
    print(f"      NDWI: min={features['NDWI'].min():.3f}  max={features['NDWI'].max():.3f}")

    return features, lc_profile, target_meta


def predict_year(rf_model, features, target_meta):
    """Stack features and predict all pixels."""
    h, w = target_meta["height"], target_meta["width"]
    print(f"    Stacking {len(FEATURE_ORDER)} features ({h} × {w})...")

    # Stack in exact training order: B2, B3, B4, B5, B6, B7, NDVI, NDSI, NDWI
    stack = np.stack([features[f] for f in FEATURE_ORDER], axis=0)  # (9, H, W)
    X = stack.reshape(9, -1).T  # (H*W, 9)

    print(f"    Predicting {X.shape[0]:,} pixels...")
    y_pred = rf_model.predict(X)
    return y_pred.reshape(h, w).astype(np.uint8)


def save_geotiff(prediction, lc_profile, output_path):
    """Save prediction as GeoTIFF with same CRS/transform as land cover raster."""
    profile = lc_profile.copy()
    profile.update({"dtype": "uint8", "count": 1, "nodata": None})

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(prediction, 1)

    print(f"    Saved: {output_path}  ({os.path.getsize(output_path)/1e6:.1f} MB)")


def save_stats(prediction, year):
    """Save class pixel counts and percentages."""
    unique, counts = np.unique(prediction, return_counts=True)
    total = counts.sum()
    stats_path = os.path.join(OUTPUT_DIR, f"prediction_stats_{year}.txt")

    with open(stats_path, "w") as f:
        f.write(f"PREDICTION STATISTICS — {year}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'Class':<25} {'Count':>12} {'Percentage':>12}\n")
        f.write("-" * 50 + "\n")
        for class_id, count in zip(unique, counts):
            name = CLASS_NAMES.get(class_id, f"Unknown_{class_id}")
            pct = count / total * 100
            f.write(f"{name:<25} {count:>12,} {pct:>11.2f}%\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'TOTAL':<25} {total:>12,} {'100.00%':>11}\n")

    print(f"    Stats: {stats_path}")

    # Also print to console
    print(f"\n    Class distribution ({year}):")
    for class_id, count in zip(unique, counts):
        name = CLASS_NAMES.get(class_id, f"Unknown_{class_id}")
        pct = count / total * 100
        print(f"      {name:<25} {count:>10,}  ({pct:5.2f}%)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print(" HUNZA VALLEY — RF PREDICTION (fixed: using QGIS index rasters)")
    print("=" * 70)

    # Load model
    print(f"\n[1] Loading trained model...")
    rf_model = joblib.load(MODEL_PATH)
    print(f"  Model: {MODEL_PATH}")
    print(f"  Estimators: {rf_model.n_estimators}")
    print(f"  Feature order: {FEATURE_ORDER}")

    # Predict per year
    for year in [2020, 2021, 2022, 2023]:
        print(f"\n{'='*70}")
        print(f" Year {year}")
        print(f"{'='*70}")
        t0 = time.time()

        # Check required files
        missing = []
        for label, paths in [("raw", RAW_RASTERS), ("lc", LAND_COVER),
                              ("ndvi", NDVI_RASTERS), ("ndsi", NDSI_RASTERS), ("ndwi", NDWI_RASTERS)]:
            if not os.path.exists(paths[year]):
                missing.append(f"{label}: {paths[year]}")
        if missing:
            print(f"  SKIP — missing files:\n" + "\n".join(f"    {m}" for m in missing))
            continue

        print(f"  Loading features...")
        features, lc_profile, target_meta = load_features(year)

        print(f"\n  Predicting...")
        prediction = predict_year(rf_model, features, target_meta)

        print(f"\n  Saving...")
        out_path = os.path.join(OUTPUT_DIR, f"rf_prediction_{year}.tif")
        save_geotiff(prediction, lc_profile, out_path)
        save_stats(prediction, year)

        print(f"\n  Year {year} done in {time.time()-t0:.1f}s")

    print("\n" + "=" * 70)
    print(" PREDICTION COMPLETE")
    print("=" * 70)
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Next: python verify_predictions.py  →  then temporal_analysis.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
