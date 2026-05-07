"""
extract_corrected.py — Hunza Valley ML Feature Extraction
==========================================================
Reads land_cover_{year}.tif + raw Landsat 8 stacked .tif + NDVI/NDSI/NDWI
rasters for each year (2020–2023), applies the correct class mapping, performs
stratified pixel sampling, and writes clean CSVs ready for Random Forest training.

Correct raster-value → class mapping (verified against actual pixel counts):
  0 → NoData/background  (~17–42 px)    → EXCLUDED
  1 → Snow/Ice           (~24–29%)      → ML class 0
  2 → Water              (<0.01%)       → EXCLUDED (too rare)
  3 → Bare Rock          (~57–65%)      → ML class 1
  4 → Sparse Vegetation  (~10–12%)      → ML class 2
  5 → Moderate Vegetation (~2–2.5%)     → ML class 3
  6 → Dense Vegetation   (~0.07–0.16%) → ML class 4

Output CSVs: e:\zaheer-work\Hunza_GIS_Data_Extraction\training_data\
  hunza_training_{year}.csv  — sampled pixels for RF training
  hunza_training_all.csv     — combined across all years
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = r"e:\zaheer-work\Hunza_GIS_Data_Extraction"
OUT_DIR = os.path.join(BASE, "training_data")
os.makedirs(OUT_DIR, exist_ok=True)

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
NDVI = {
    2020: os.path.join(BASE, r"geospatial_project\outputs\qgis\ndvi.tif"),
    2021: os.path.join(BASE, r"geospatial_project_2021\outputs\qgis\ndvi_2021.tif"),
    2022: os.path.join(BASE, r"geospatial_project_2022\outputs\qgis\ndvi_2022.tif"),
    2023: os.path.join(BASE, r"geospatial_project_2023\outputs\qgis\ndvi_2023.tif"),
}
NDSI = {
    2020: os.path.join(BASE, r"geospatial_project\outputs\qgis\ndsi.tif"),
    2021: os.path.join(BASE, r"geospatial_project_2021\outputs\qgis\ndsi_2021.tif"),
    2022: os.path.join(BASE, r"geospatial_project_2022\outputs\qgis\ndsi_2022.tif"),
    2023: os.path.join(BASE, r"geospatial_project_2023\outputs\qgis\ndsi_2023.tif"),
}
NDWI = {
    2020: os.path.join(BASE, r"geospatial_project\outputs\qgis\ndwi.tif"),
    2021: os.path.join(BASE, r"geospatial_project_2021\outputs\qgis\ndwi_2021.tif"),
    2022: os.path.join(BASE, r"geospatial_project_2022\outputs\qgis\ndwi_2022.tif"),
    2023: os.path.join(BASE, r"geospatial_project_2023\outputs\qgis\ndwi_2023.tif"),
}

# ── Class mapping ──────────────────────────────────────────────────────────────
# Verified against actual pixel counts in all four land_cover rasters.
# Raster value 0 = background/nodata, value 2 = Water (too rare) → both excluded.
EXCLUDE_VALUES = {0, 2}

CLASS_NAMES = {
    1: "Snow/Ice",
    3: "Bare Rock",
    4: "Sparse Vegetation",
    5: "Moderate Vegetation",
    6: "Dense Vegetation",
}
# Raster value → compact ML integer label (0-based, contiguous for sklearn)
RASTER_TO_ML = {1: 0, 3: 1, 4: 2, 5: 3, 6: 4}
ML_TO_NAME = {v: CLASS_NAMES[k] for k, v in RASTER_TO_ML.items()}

# Raw bands to extract: SR_B2 (Blue) … SR_B7 (SWIR2) = rasterio band indices 2–7
BAND_INDICES = [2, 3, 4, 5, 6, 7]
BAND_NAMES   = ["B2_Blue", "B3_Green", "B4_Red", "B5_NIR", "B6_SWIR1", "B7_SWIR2"]

# Landsat 8 Collection 2 Level-2 reflectance calibration
DN_SCALE  = 0.0000275
DN_OFFSET = -0.2

# Stratified sampling cap per class per year.
# Dense Veg is tiny (~16–25k pixels) — always take all of it.
MAX_PER_CLASS = 100_000
RANDOM_SEED   = 42


def calibrate(dn_array: np.ndarray) -> np.ndarray:
    """Convert Landsat 8 SR DN to surface reflectance, clipped to [0, 1]."""
    refl = dn_array.astype(np.float32) * DN_SCALE + DN_OFFSET
    return np.clip(refl, 0.0, 1.0)


def resample_to_grid(src_path: str, target_meta: dict,
                     band_idx: int = 1) -> np.ndarray:
    """
    Read one band from src_path and reproject/resample it to match
    the grid described by target_meta (height, width, crs, transform).
    Returns float32 array of shape (height, width).
    """
    dst = np.empty(
        (target_meta["height"], target_meta["width"]),
        dtype=np.float32,
    )
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


def pixel_coords(transform, rows: np.ndarray, cols: np.ndarray):
    """Convert pixel (row, col) to geographic (lon, lat) using affine transform."""
    lons = transform.c + cols * transform.a + rows * transform.b
    lats = transform.f + cols * transform.d + rows * transform.e
    return lons, lats


def extract_year(year: int) -> pd.DataFrame:
    """Extract, align, and sample pixels for one year. Returns a DataFrame."""
    print(f"\n{'='*60}")
    print(f" Year {year}")
    print(f"{'='*60}")
    t0 = time.time()

    # ── 1. Land cover raster → defines the target grid ────────────────────────
    print(f"  Reading land cover…", end=" ", flush=True)
    with rasterio.open(LAND_COVER[year]) as lc_src:
        lc = lc_src.read(1)          # shape: (H, W)
        target_meta = {
            "height":    lc_src.height,
            "width":     lc_src.width,
            "crs":       lc_src.crs,
            "transform": lc_src.transform,
        }
    print(f"shape={lc.shape}")

    # Print class distribution for this year
    unique, counts = np.unique(lc, return_counts=True)
    for v, c in zip(unique, counts):
        name = CLASS_NAMES.get(v, f"value={v}")
        pct = 100 * c / lc.size
        status = "EXCLUDE" if v in EXCLUDE_VALUES else "→ keep"
        print(f"    raster value {v} ({name:20s}): {c:>10,} px  ({pct:5.2f}%)  {status}")

    # ── 2. Build valid-pixel mask ──────────────────────────────────────────────
    valid_mask = ~np.isin(lc, list(EXCLUDE_VALUES))
    valid_rows, valid_cols = np.where(valid_mask)
    valid_lc = lc[valid_rows, valid_cols]
    print(f"  Valid pixels after exclusion: {valid_rows.size:,}")

    # ── 3. Stratified sampling ─────────────────────────────────────────────────
    rng = np.random.default_rng(RANDOM_SEED)
    sampled_idx = []
    for rv in sorted(CLASS_NAMES.keys()):
        class_mask = (valid_lc == rv)
        class_idx  = np.where(class_mask)[0]
        n_avail    = len(class_idx)
        n_take     = min(n_avail, MAX_PER_CLASS)
        chosen     = rng.choice(class_idx, size=n_take, replace=False)
        sampled_idx.append(chosen)
        print(f"    {CLASS_NAMES[rv]:22s}: {n_avail:>8,} avail → sampling {n_take:>7,}")

    sampled_idx = np.concatenate(sampled_idx)
    s_rows = valid_rows[sampled_idx]
    s_cols = valid_cols[sampled_idx]
    s_lc   = valid_lc[sampled_idx]
    print(f"  Total sampled: {len(s_rows):,}")

    # ── 4. Extract raw spectral bands (resample 5376→5000) ────────────────────
    print(f"  Extracting bands B2–B7 (resampling 5376→5000)…", flush=True)
    band_data = {}
    raw_path = RAW_RASTERS[year]
    for band_idx, band_name in zip(BAND_INDICES, BAND_NAMES):
        print(f"    {band_name}…", end=" ", flush=True)
        resampled = resample_to_grid(raw_path, target_meta, band_idx=band_idx)
        band_data[band_name] = calibrate(resampled)[s_rows, s_cols]
        print("done")

    # ── 5. Extract index rasters ───────────────────────────────────────────────
    print(f"  Reading NDVI/NDSI/NDWI…", end=" ", flush=True)
    with rasterio.open(NDVI[year]) as src:
        ndvi_arr = src.read(1).astype(np.float32)
    with rasterio.open(NDSI[year]) as src:
        ndsi_arr = src.read(1).astype(np.float32)
    with rasterio.open(NDWI[year]) as src:
        ndwi_arr = src.read(1).astype(np.float32)
    print("done")

    # ── 6. Compute lon/lat for sampled pixels ─────────────────────────────────
    lons, lats = pixel_coords(target_meta["transform"], s_rows, s_cols)

    # ── 7. Assemble DataFrame ─────────────────────────────────────────────────
    df = pd.DataFrame({
        "year":       year,
        "lon":        lons.astype(np.float32),
        "lat":        lats.astype(np.float32),
        **{k: v for k, v in band_data.items()},
        "NDVI":       ndvi_arr[s_rows, s_cols],
        "NDSI":       ndsi_arr[s_rows, s_cols],
        "NDWI":       ndwi_arr[s_rows, s_cols],
        "raster_val": s_lc.astype(np.int8),
        "class_id":   np.array([RASTER_TO_ML[v] for v in s_lc], dtype=np.int8),
        "class_name": [CLASS_NAMES[v] for v in s_lc],
    })

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  |  rows: {len(df):,}")
    return df


def main():
    print("Hunza Valley — Corrected Feature Extraction")
    print(f"Output dir: {OUT_DIR}")
    print(f"Max samples per class per year: {MAX_PER_CLASS:,}")
    print(f"Classes included: {list(CLASS_NAMES.values())}")
    print(f"Classes excluded: raster values {sorted(EXCLUDE_VALUES)} (NoData, Water)")

    all_dfs = []
    for year in [2020, 2021, 2022, 2023]:
        df = extract_year(year)
        out_path = os.path.join(OUT_DIR, f"hunza_training_{year}.csv")
        df.to_csv(out_path, index=False)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  Saved: {out_path}  ({size_mb:.1f} MB)")
        all_dfs.append(df)

    # ── Combined dataset ───────────────────────────────────────────────────────
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = os.path.join(OUT_DIR, "hunza_training_all.csv")
    combined.to_csv(combined_path, index=False)
    size_mb = os.path.getsize(combined_path) / 1e6
    print(f"\nCombined dataset saved: {combined_path}  ({size_mb:.1f} MB)")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(combined.groupby(["year", "class_name"])["class_id"]
          .count().rename("n_pixels").to_string())
    print(f"\nTotal rows: {len(combined):,}")
    print(f"Columns:    {list(combined.columns)}")
    print("\nClass label map for Random Forest training:")
    for ml_id, name in sorted(ML_TO_NAME.items()):
        print(f"  class_id={ml_id} → {name}")
    print("\nDone. Next step: train_random_forest.py")


if __name__ == "__main__":
    main()
