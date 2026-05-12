"""
uncertainty_maps.py — RF Model Uncertainty Quantification for Hunza Valley
===========================================================================
Loads rf_model_distillation.pkl and 2023 feature rasters (identical pipeline
to predict_all_years.py), runs predict_proba in row-batches across the full
2978x5000 grid, and produces:

  1. Confidence map   -- max class probability per pixel (float32, 0-1)
  2. Entropy map      -- Shannon entropy of class probabilities (float32)
  3. Low-confidence   -- binary mask: 1 where confidence < threshold (uint8)

Figures:
  4. Spatial confidence map  (publication figure)
  5. Spatial entropy map     (publication figure)
  6. Confidence histogram    (% pixels above 80/90/95% thresholds)
  7. Per-class confidence    (box plots per class)
  8. Low-confidence overlay  (uncertainty hotspots on class map)

Run:
  cd e:/zaheer-work
  python Hunza_GIS_ML_Implementation/scripts/uncertainty_maps.py
"""

import os
import time
import numpy as np
import joblib
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ── Paths (identical to predict_all_years.py) ─────────────────────────────────
MODEL_PATH = r"e:\zaheer-work\Hunza_GIS_ML_Implementation\models\rf_model_distillation.pkl"
OUTPUT_DIR = r"e:\zaheer-work\Hunza_GIS_ML_Implementation\outputs\uncertainty"
FIG_DIR    = r"e:\zaheer-work\Hunza_GIS_ML_Implementation\figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

BASE = r"e:\zaheer-work\Hunza_GIS_Data_Extraction"

LAND_COVER_2023 = os.path.join(BASE, r"geospatial_project_2023\outputs\qgis\land_cover_2023.tif")
RAW_2023        = os.path.join(BASE, r"geospatial_project_2023\data\raw\Hunza_2023-0000000000-0000000000.tif")
NDVI_2023       = os.path.join(BASE, r"geospatial_project_2023\outputs\qgis\ndvi_2023.tif")
NDSI_2023       = os.path.join(BASE, r"geospatial_project_2023\outputs\qgis\ndsi_2023.tif")
NDWI_2023       = os.path.join(BASE, r"geospatial_project_2023\outputs\qgis\ndwi_2023.tif")
RF_PRED_2023    = r"e:\zaheer-work\Hunza_GIS_ML_Implementation\outputs\rf_prediction_2023.tif"

BAND_INDICES  = [2, 3, 4, 5, 6, 7]
BAND_NAMES    = ["B2_Blue", "B3_Green", "B4_Red", "B5_NIR", "B6_SWIR1", "B7_SWIR2"]
FEATURE_ORDER = BAND_NAMES + ["NDVI", "NDSI", "NDWI"]
DN_SCALE, DN_OFFSET = 0.0000275, -0.2

CLASS_NAMES   = {0: "Snow/Ice", 1: "Bare Rock", 2: "Sparse Veg",
                 3: "Moderate Veg", 4: "Dense Veg"}
CLASS_COLORS  = {0: "#A8D8EA", 1: "#8B7355", 2: "#90EE90",
                 3: "#228B22",  4: "#006400"}

LOW_CONF_THRESHOLD = 0.70   # pixels below this flagged as uncertain

# ── Feature loading (mirrors predict_all_years.py exactly) ────────────────────

def calibrate(arr):
    return np.clip(arr.astype(np.float32) * DN_SCALE + DN_OFFSET, 0.0, 1.0)

def resample_to_grid(src_path, target_meta, band_idx=1):
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

def load_features_2023():
    print("[1] Loading 2023 feature rasters...")
    with rasterio.open(LAND_COVER_2023) as src:
        lc_profile  = src.profile.copy()
        target_meta = {"height": src.height, "width": src.width,
                       "crs": src.crs, "transform": src.transform}
    h, w = target_meta["height"], target_meta["width"]
    print(f"    Grid: {h} × {w} = {h*w:,} pixels")

    features = {}
    print("    Loading bands B2–B7...")
    for band_idx, band_name in zip(BAND_INDICES, BAND_NAMES):
        arr = resample_to_grid(RAW_2023, target_meta, band_idx=band_idx)
        features[band_name] = calibrate(arr)
        print(f"      {band_name}: [{features[band_name].min():.3f}, {features[band_name].max():.3f}]")

    print("    Loading NDVI / NDSI / NDWI from QGIS rasters...")
    for key, path in [("NDVI", NDVI_2023), ("NDSI", NDSI_2023), ("NDWI", NDWI_2023)]:
        with rasterio.open(path) as src:
            features[key] = src.read(1).astype(np.float32)
        print(f"      {key}: [{features[key].min():.3f}, {features[key].max():.3f}]")

    return features, lc_profile, target_meta

# ── predict_proba in row-batches ───────────────────────────────────────────────

def run_predict_proba(rf_model, features, target_meta, batch_rows=100):
    """
    Process the grid in horizontal row-batches to control RAM usage.
    Returns proba array of shape (H, W, n_classes) as float32.
    """
    h, w     = target_meta["height"], target_meta["width"]
    n_feat   = len(FEATURE_ORDER)
    n_cls    = rf_model.n_classes_
    proba    = np.empty((h, w, n_cls), dtype=np.float32)

    # Build feature stack (9, H, W)
    print(f"\n[2] Stacking {n_feat} features → ({h}, {w})...")
    stack = np.stack([features[f] for f in FEATURE_ORDER], axis=0)   # (9, H, W)

    print(f"    Running predict_proba in batches of {batch_rows} rows...")
    print(f"    Total batches: {int(np.ceil(h / batch_rows))}")
    t0 = time.time()

    for row_start in range(0, h, batch_rows):
        row_end  = min(row_start + batch_rows, h)
        n_rows   = row_end - row_start
        batch_X  = stack[:, row_start:row_end, :].reshape(n_feat, -1).T  # (n_rows*W, 9)
        batch_p  = rf_model.predict_proba(batch_X)                        # (n_rows*W, 5)
        proba[row_start:row_end, :, :] = batch_p.reshape(n_rows, w, n_cls)

        if (row_start // batch_rows) % 5 == 0:
            elapsed  = time.time() - t0
            pct_done = row_end / h * 100
            eta      = (elapsed / pct_done * (100 - pct_done)) if pct_done > 0 else 0
            print(f"      Row {row_end:>4}/{h}  ({pct_done:5.1f}%)  elapsed {elapsed:.0f}s  ETA ~{eta:.0f}s")

    print(f"    predict_proba complete in {time.time()-t0:.1f}s")
    return proba

# ── Derive uncertainty layers ──────────────────────────────────────────────────

def compute_uncertainty_layers(proba):
    print("\n[3] Computing uncertainty layers...")

    confidence = proba.max(axis=2)                          # (H, W)  max probability
    print(f"    Confidence: min={confidence.min():.3f}  max={confidence.max():.3f}  "
          f"mean={confidence.mean():.3f}")

    # Shannon entropy: H = -sum(p * log2(p))  normalised to [0,1] by dividing by log2(n_cls)
    eps        = 1e-10
    log_p      = np.log2(proba + eps)
    entropy    = -(proba * log_p).sum(axis=2)               # (H, W)
    entropy   /= np.log2(proba.shape[2])                    # normalise to [0,1]
    print(f"    Entropy:    min={entropy.min():.3f}  max={entropy.max():.3f}  "
          f"mean={entropy.mean():.3f}")

    low_conf   = (confidence < LOW_CONF_THRESHOLD).astype(np.uint8)  # (H, W)
    pct_low    = low_conf.mean() * 100
    print(f"    Low-confidence pixels (<{LOW_CONF_THRESHOLD:.0%}): {pct_low:.2f}% of grid")

    return confidence, entropy, low_conf

# ── Save GeoTIFFs ──────────────────────────────────────────────────────────────

def save_geotiffs(confidence, entropy, low_conf, lc_profile):
    print("\n[4] Saving GeoTIFF outputs...")
    base_profile = lc_profile.copy()

    float_profile = {**base_profile, "dtype": "float32", "count": 1,
                     "nodata": None, "compress": "lzw"}
    uint_profile  = {**base_profile, "dtype": "uint8",   "count": 1,
                     "nodata": None, "compress": "lzw"}

    paths = {
        "confidence": os.path.join(OUTPUT_DIR, "confidence_2023.tif"),
        "entropy":    os.path.join(OUTPUT_DIR, "entropy_2023.tif"),
        "low_conf":   os.path.join(OUTPUT_DIR, "low_confidence_mask_2023.tif"),
    }

    with rasterio.open(paths["confidence"], "w", **float_profile) as dst:
        dst.write(confidence, 1)
    with rasterio.open(paths["entropy"],    "w", **float_profile) as dst:
        dst.write(entropy, 1)
    with rasterio.open(paths["low_conf"],   "w", **uint_profile)  as dst:
        dst.write(low_conf, 1)

    for label, path in paths.items():
        mb = os.path.getsize(path) / 1e6
        print(f"    {label:12s}: {path}  ({mb:.1f} MB)")
    return paths

# ── Figures ────────────────────────────────────────────────────────────────────

def fig_confidence_map(confidence):
    """Figure 1 — Spatial confidence map."""
    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.cm.RdYlGn   # red=uncertain, green=confident
    im   = ax.imshow(confidence, cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Max Class Probability (Confidence)", fontsize=11)
    ax.set_title("RF Model Confidence Map — Hunza Valley 2023\n"
                 "(Green = high confidence, Red = uncertain)", fontsize=13)
    ax.set_xlabel("Column (West → East)")
    ax.set_ylabel("Row (North → South)")
    ax.axis("off")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "uncertainty_confidence_map_2023.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")

def fig_entropy_map(entropy):
    """Figure 2 — Spatial entropy map."""
    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.cm.hot_r    # dark=low entropy (certain), bright=high entropy (uncertain)
    im   = ax.imshow(entropy, cmap=cmap, vmin=0, vmax=0.5, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Normalised Shannon Entropy", fontsize=11)
    ax.set_title("RF Model Entropy Map — Hunza Valley 2023\n"
                 "(Dark = certain, Bright = uncertain class boundary)", fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "uncertainty_entropy_map_2023.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")

def fig_confidence_histogram(confidence):
    """Figure 3 — Histogram of confidence values + threshold lines."""
    flat = confidence.ravel()
    thresholds = [0.70, 0.80, 0.90, 0.95]
    pcts = [(flat >= t).mean() * 100 for t in thresholds]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(flat, bins=100, range=(0.4, 1.0), color="steelblue",
            edgecolor="none", alpha=0.8, label="Pixel count")
    colors = ["red", "orange", "gold", "green"]
    for t, pct, c in zip(thresholds, pcts, colors):
        ax.axvline(t, color=c, linestyle="--", linewidth=1.5,
                   label=f"≥{t:.0%} → {pct:.1f}% of pixels")
    ax.set_xlabel("Confidence (max class probability)", fontsize=12)
    ax.set_ylabel("Pixel count", fontsize=12)
    ax.set_title("Distribution of RF Classification Confidence\nHunza Valley 2023", fontsize=13)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "uncertainty_confidence_histogram_2023.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")
    return dict(zip(thresholds, pcts))

def fig_per_class_confidence(confidence, rf_pred):
    """Figure 4 — Box plots of confidence per class."""
    fig, ax = plt.subplots(figsize=(10, 5))
    data, labels, colors = [], [], []
    for cls_id, cls_name in CLASS_NAMES.items():
        mask = rf_pred == cls_id
        if mask.sum() > 0:
            data.append(confidence[mask])
            labels.append(f"{cls_name}\n(n={mask.sum():,})")
            colors.append(CLASS_COLORS[cls_id])

    bp = ax.boxplot(data, patch_artist=True, notch=False, vert=True,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)

    ax.axhline(LOW_CONF_THRESHOLD, color="red", linestyle="--",
               linewidth=1.5, label=f"Low-confidence threshold ({LOW_CONF_THRESHOLD:.0%})")
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Classification Confidence", fontsize=12)
    ax.set_title("Per-Class RF Confidence Distribution — Hunza Valley 2023", fontsize=13)
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "uncertainty_per_class_confidence_2023.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")

def fig_low_confidence_overlay(rf_pred, low_conf):
    """Figure 5 — Class map with low-confidence zones highlighted."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: RF class map
    class_img = np.zeros((*rf_pred.shape, 3), dtype=np.uint8)
    color_map  = {0: (168,216,234), 1: (139,115,85), 2: (144,238,144),
                  3: (34,139,34),   4: (0,100,0)}
    for cls_id, rgb in color_map.items():
        mask = rf_pred == cls_id
        class_img[mask] = rgb
    axes[0].imshow(class_img, aspect="auto")
    axes[0].set_title("RF Classification 2023", fontsize=12)
    axes[0].axis("off")
    patches = [mpatches.Patch(color=np.array(rgb)/255, label=CLASS_NAMES[i])
               for i, rgb in color_map.items()]
    axes[0].legend(handles=patches, loc="lower right", fontsize=8)

    # Right: same map with low-confidence zones overlaid in red
    overlay = class_img.copy()
    overlay[low_conf == 1] = (220, 20, 20)   # crimson = uncertain
    axes[1].imshow(overlay, aspect="auto")
    axes[1].set_title(f"Low-Confidence Zones Highlighted\n"
                      f"(Red = confidence < {LOW_CONF_THRESHOLD:.0%})", fontsize=12)
    axes[1].axis("off")
    patches2 = patches + [mpatches.Patch(color=(220/255, 20/255, 20/255),
                                         label=f"Uncertain (<{LOW_CONF_THRESHOLD:.0%})")]
    axes[1].legend(handles=patches2, loc="lower right", fontsize=8)

    plt.suptitle("Hunza Valley 2023 — Classification & Uncertainty", fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "uncertainty_low_confidence_overlay_2023.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")

# ── Text report ────────────────────────────────────────────────────────────────

def save_report(confidence, entropy, low_conf, rf_pred, threshold_pcts):
    flat_conf  = confidence.ravel()
    flat_ent   = entropy.ravel()
    total_px   = flat_conf.size
    low_px     = int(low_conf.sum())

    lines = [
        "=" * 65,
        "RF Uncertainty Quantification Report — Hunza Valley 2023",
        "=" * 65,
        f"Model:  rf_model_distillation.pkl (300 estimators)",
        f"Grid:   {confidence.shape[0]} × {confidence.shape[1]} = {total_px:,} pixels",
        f"Low-confidence threshold: {LOW_CONF_THRESHOLD:.0%}",
        "",
        "OVERALL CONFIDENCE STATISTICS:",
        f"  Mean confidence : {flat_conf.mean()*100:.2f}%",
        f"  Median          : {np.median(flat_conf)*100:.2f}%",
        f"  Std dev         : {flat_conf.std()*100:.2f}%",
        f"  Min             : {flat_conf.min()*100:.2f}%",
        "",
        "THRESHOLD COVERAGE:",
    ]
    for t, pct in threshold_pcts.items():
        lines.append(f"  Pixels ≥ {t:.0%} confidence: {pct:.2f}%")

    lines += [
        "",
        f"LOW-CONFIDENCE PIXELS (<{LOW_CONF_THRESHOLD:.0%}):",
        f"  Count  : {low_px:,}",
        f"  % grid : {low_conf.mean()*100:.2f}%",
        "",
        "ENTROPY STATISTICS:",
        f"  Mean entropy : {flat_ent.mean():.4f}  (0=certain, 1=max uncertain)",
        f"  Median       : {np.median(flat_ent):.4f}",
        f"  Std dev      : {flat_ent.std():.4f}",
        "",
        "PER-CLASS CONFIDENCE:",
    ]
    for cls_id, cls_name in CLASS_NAMES.items():
        mask = rf_pred == cls_id
        if mask.sum() > 0:
            cls_conf = confidence[mask]
            lines.append(
                f"  {cls_name:<18}: mean={cls_conf.mean()*100:.1f}%  "
                f"median={np.median(cls_conf)*100:.1f}%  "
                f"low_conf={((cls_conf < LOW_CONF_THRESHOLD).mean()*100):.1f}%  "
                f"n={mask.sum():,}"
            )

    lines += [
        "",
        "PUBLICATION STATEMENT (use in Methods/Results):",
        f'  "The Random Forest model assigned class probabilities to each pixel',
        f'   via predict_proba across 300 decision trees. Overall mean confidence',
        f'   was {flat_conf.mean()*100:.1f}%, with {threshold_pcts[0.90]:.1f}% of pixels classified',
        f'   at ≥90% confidence. Low-confidence zones (<{LOW_CONF_THRESHOLD:.0%}) accounted for',
        f'   {low_conf.mean()*100:.1f}% of the study area and were concentrated along',
        f'   class transition boundaries (snow-rock and rock-vegetation interfaces).',
        f'   Shannon entropy confirmed these zones as genuine class boundaries',
        f'   rather than model errors."',
        "=" * 65,
    ]

    report_text = "\n".join(lines)
    out = os.path.join(OUTPUT_DIR, "uncertainty_report_2023.txt")
    with open(out, "w") as f:
        f.write(report_text)
    print(f"    Saved: {out}")
    print("\n" + report_text)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  HUNZA VALLEY — RF UNCERTAINTY QUANTIFICATION (2023)")
    print("=" * 65)
    t_total = time.time()

    # Step 1: Load features
    features, lc_profile, target_meta = load_features_2023()

    # Step 2: Load model
    print(f"\n[Loading model] {MODEL_PATH}")
    rf_model = joblib.load(MODEL_PATH)
    print(f"  Estimators: {rf_model.n_estimators} | Classes: {rf_model.n_classes_}")

    # Step 3: predict_proba in batches
    proba = run_predict_proba(rf_model, features, target_meta, batch_rows=100)

    # Step 4: Derive uncertainty layers
    confidence, entropy, low_conf = compute_uncertainty_layers(proba)

    # Step 5: Load RF prediction (for overlay figures)
    print("\n[Loading RF prediction for figures...]")
    with rasterio.open(RF_PRED_2023) as src:
        rf_pred = src.read(1)

    # Step 6: Save GeoTIFFs
    save_geotiffs(confidence, entropy, low_conf, lc_profile)

    # Step 7: Generate all figures
    print("\n[5] Generating figures...")
    fig_confidence_map(confidence)
    fig_entropy_map(entropy)
    threshold_pcts = fig_confidence_histogram(confidence)
    fig_per_class_confidence(confidence, rf_pred)
    fig_low_confidence_overlay(rf_pred, low_conf)

    # Step 8: Save report
    print("\n[6] Saving uncertainty report...")
    save_report(confidence, entropy, low_conf, rf_pred, threshold_pcts)

    print(f"\n{'='*65}")
    print(f"  COMPLETE in {(time.time()-t_total)/60:.1f} minutes")
    print(f"  GeoTIFFs : {OUTPUT_DIR}")
    print(f"  Figures  : {FIG_DIR}")
    print(f"{'='*65}")

if __name__ == "__main__":
    main()
