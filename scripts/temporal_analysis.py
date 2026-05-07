"""
temporal_analysis.py — Pixel-level change detection 2020→2023
==============================================================
Analyzes transitions between land cover classes across 4 years:
  - Transition matrices (from-to class changes)
  - Net change per class (gain/loss)
  - Persistence (pixels that didn't change)
  - Visualizations (sankey, heatmaps, change maps)

Run:  python temporal_analysis.py
"""

import os
import time
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# ============================================================================
# CONFIGURATION
# ============================================================================

PRED_DIR = r"e:\zaheer-work\Hunza_GIS_ML_Implementation\outputs"
OUTPUT_DIR = os.path.join(PRED_DIR, "temporal_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PREDICTIONS = {
    2020: os.path.join(PRED_DIR, "rf_prediction_2020.tif"),
    2021: os.path.join(PRED_DIR, "rf_prediction_2021.tif"),
    2022: os.path.join(PRED_DIR, "rf_prediction_2022.tif"),
    2023: os.path.join(PRED_DIR, "rf_prediction_2023.tif"),
}

CLASS_NAMES = {
    0: "Snow/Ice",
    1: "Bare Rock",
    2: "Sparse Vegetation",
    3: "Moderate Vegetation",
    4: "Dense Vegetation",
}

CLASS_COLORS = {
    0: "#FFFFFF",  # White
    1: "#808080",  # Gray
    2: "#FFD700",  # Yellow
    3: "#00C800",  # Green
    4: "#006400",  # Dark Green
}

# ============================================================================
# LOAD PREDICTIONS
# ============================================================================

def load_predictions():
    """Load all 4 prediction rasters."""
    predictions = {}
    for year in [2020, 2021, 2022, 2023]:
        path = PREDICTIONS[year]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        with rasterio.open(path) as src:
            predictions[year] = src.read(1).astype(np.uint8)
    return predictions

# ============================================================================
# TRANSITION ANALYSIS
# ============================================================================

def compute_transition_matrix(pred_t1, pred_t2, year1, year2):
    """Compute transition matrix from year1 to year2."""
    flat_t1 = pred_t1.flatten()
    flat_t2 = pred_t2.flatten()

    # Transition matrix: rows=from, cols=to
    trans = np.zeros((5, 5), dtype=np.int64)
    for from_class in range(5):
        for to_class in range(5):
            mask = (flat_t1 == from_class) & (flat_t2 == to_class)
            trans[from_class, to_class] = np.sum(mask)

    return trans


def compute_all_transitions(predictions):
    """Compute transition matrices for all year pairs."""
    transitions = {}
    year_pairs = [(2020, 2021), (2021, 2022), (2022, 2023), (2020, 2023)]

    for y1, y2 in year_pairs:
        transitions[(y1, y2)] = compute_transition_matrix(
            predictions[y1], predictions[y2], y1, y2
        )

    return transitions


def analyze_transitions(transitions):
    """Compute net gains/losses and persistence."""
    y1, y2 = 2020, 2023
    trans = transitions[(y1, y2)]

    # Total pixels per class per year
    total_2020 = trans.sum(axis=1)  # Row sums
    total_2023 = trans.sum(axis=0)  # Col sums

    # Persistence (diagonal)
    persistence = np.diag(trans)

    # Net change
    net_change = total_2023 - total_2020

    # Percentage change
    pct_change = (net_change / total_2020) * 100

    return {
        "total_2020": total_2020,
        "total_2023": total_2023,
        "persistence": persistence,
        "net_change": net_change,
        "pct_change": pct_change,
        "trans": trans,
    }


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_transition_heatmap(trans, year1, year2, output_dir):
    """Plot transition matrix as heatmap."""
    labels = [CLASS_NAMES[i][:8] for i in range(5)]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        trans,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Pixel count"},
        ax=ax,
    )
    ax.set_xlabel(f"Class {year2} (to)")
    ax.set_ylabel(f"Class {year1} (from)")
    ax.set_title(f"Transition Matrix: {year1} → {year2}")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"transition_matrix_{year1}_{year2}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")


def plot_net_change(analysis, output_dir):
    """Plot net gain/loss per class."""
    classes = [CLASS_NAMES[i] for i in range(5)]
    net_change = analysis["net_change"]
    pct_change = analysis["pct_change"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Absolute change
    colors = ["green" if x > 0 else "red" for x in net_change]
    ax1.barh(classes, net_change, color=colors, alpha=0.7)
    ax1.set_xlabel("Pixel count change")
    ax1.set_title("Net Change (2020→2023) — Absolute")
    ax1.axvline(0, color="black", linestyle="-", linewidth=0.5)
    for i, v in enumerate(net_change):
        ax1.text(v, i, f" {v:+,}", va="center", fontsize=9)

    # Percentage change
    colors = ["green" if x > 0 else "red" for x in pct_change]
    ax2.barh(classes, pct_change, color=colors, alpha=0.7)
    ax2.set_xlabel("Percentage change")
    ax2.set_title("Net Change (2020→2023) — Percentage")
    ax2.axvline(0, color="black", linestyle="-", linewidth=0.5)
    for i, v in enumerate(pct_change):
        ax2.text(v, i, f" {v:+.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "net_change_2020_2023.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")


def plot_change_map(pred_2020, pred_2023, output_dir):
    """Plot maps showing changed vs unchanged pixels."""
    changed = pred_2020 != pred_2023

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 2020 classification
    im1 = axes[0].imshow(pred_2020, cmap="tab10", vmin=0, vmax=9)
    axes[0].set_title("Classification 2020")
    axes[0].set_ylabel("Latitude")
    plt.colorbar(im1, ax=axes[0], label="Class")

    # 2023 classification
    im2 = axes[1].imshow(pred_2023, cmap="tab10", vmin=0, vmax=9)
    axes[1].set_title("Classification 2023")
    axes[1].set_xlabel("Longitude")
    plt.colorbar(im2, ax=axes[1], label="Class")

    # Change map
    im3 = axes[2].imshow(changed, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[2].set_title("Changed Pixels (2020→2023)\nRed=Changed, Green=Stable")
    plt.colorbar(im3, ax=axes[2], label="Changed")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "change_map_2020_2023.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")


def plot_persistence(analysis, output_dir):
    """Plot persistence (stability) per class."""
    classes = [CLASS_NAMES[i] for i in range(5)]
    total_2020 = analysis["total_2020"]
    persistence = analysis["persistence"]

    persistence_pct = (persistence / total_2020) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4CAF50" if x > 80 else "#FFC107" if x > 50 else "#F44336" for x in persistence_pct]
    ax.barh(classes, persistence_pct, color=colors, alpha=0.8)
    ax.set_xlabel("Persistence (%)")
    ax.set_title("Pixel Persistence by Class (2020→2023)\nGreen: Stable (>80%), Yellow: Moderate (50-80%), Red: Unstable (<50%)")
    ax.set_xlim([0, 105])

    for i, v in enumerate(persistence_pct):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "persistence_2020_2023.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")


# ============================================================================
# REPORTS
# ============================================================================

def save_transition_report(transitions, analysis, output_dir):
    """Save detailed transition reports."""
    report_path = os.path.join(output_dir, "transition_report.txt")

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("HUNZA VALLEY — TEMPORAL CHANGE ANALYSIS (2020-2023)\n")
        f.write("=" * 80 + "\n\n")

        # Overall summary
        f.write("OVERALL SUMMARY (2020 → 2023)\n")
        f.write("-" * 80 + "\n")
        total_pixels = analysis["trans"].sum()
        changed_pixels = total_pixels - analysis["persistence"].sum()
        pct_changed = (changed_pixels / total_pixels) * 100

        f.write(f"Total pixels: {total_pixels:,}\n")
        f.write(f"Changed pixels: {changed_pixels:,} ({pct_changed:.2f}%)\n")
        f.write(f"Stable pixels: {analysis['persistence'].sum():,} ({100-pct_changed:.2f}%)\n\n")

        # Per-class analysis
        f.write("PER-CLASS ANALYSIS (2020 vs 2023)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<25} {'2020 Count':>12} {'2023 Count':>12} {'Change':>12} {'% Change':>12}\n")
        f.write("-" * 80 + "\n")

        for i in range(5):
            name = CLASS_NAMES[i]
            cnt_2020 = analysis["total_2020"][i]
            cnt_2023 = analysis["total_2023"][i]
            change = analysis["net_change"][i]
            pct = analysis["pct_change"][i]
            f.write(f"{name:<25} {cnt_2020:>12,} {cnt_2023:>12,} {change:>+12,} {pct:>+11.2f}%\n")

        f.write("\n")

        # Transition matrix (2020→2023)
        f.write("DETAILED TRANSITION MATRIX (2020 → 2023)\n")
        f.write("Rows: Source class (2020), Cols: Target class (2023)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'From / To':<15}" + "".join(f"{CLASS_NAMES[i][:8]:>12}" for i in range(5)) + "\n")
        f.write("-" * 80 + "\n")

        trans = analysis["trans"]
        for i in range(5):
            f.write(f"{CLASS_NAMES[i]:<15}")
            for j in range(5):
                f.write(f"{trans[i, j]:>12,}")
            f.write("\n")

        f.write("\n")

        # Persistence
        f.write("PERSISTENCE BY CLASS (% of 2020 pixels that remained same class)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<25} {'Persistence':>12} {'Changed':>12}\n")
        f.write("-" * 80 + "\n")

        for i in range(5):
            persist = analysis["persistence"][i]
            total = analysis["total_2020"][i]
            persist_pct = (persist / total) * 100 if total > 0 else 0
            changed = total - persist
            f.write(f"{CLASS_NAMES[i]:<25} {persist_pct:>11.2f}% {changed:>12,}\n")

        f.write("\n")

        # Trends
        f.write("TRENDS BY YEAR PAIR\n")
        f.write("-" * 80 + "\n")

        for (y1, y2) in [(2020, 2021), (2021, 2022), (2022, 2023)]:
            trans = transitions[(y1, y2)]
            total_pixels = trans.sum()
            changed = total_pixels - np.trace(trans)
            f.write(f"\n{y1} → {y2}:\n")
            f.write(f"  Changed: {changed:,} / {total_pixels:,} ({changed/total_pixels*100:.2f}%)\n")

    print(f"    Saved: {report_path}")


def save_csv_exports(transitions, analysis, output_dir):
    """Export transition matrices as CSV."""
    # Full 2020→2023 transition matrix
    trans_df = pd.DataFrame(
        analysis["trans"],
        index=[CLASS_NAMES[i] for i in range(5)],
        columns=[CLASS_NAMES[i] for i in range(5)],
    )
    csv_path = os.path.join(output_dir, "transition_matrix_2020_2023.csv")
    trans_df.to_csv(csv_path)
    print(f"    Saved: {csv_path}")

    # Per-year transitions
    for (y1, y2), trans in transitions.items():
        trans_df = pd.DataFrame(
            trans,
            index=[CLASS_NAMES[i] for i in range(5)],
            columns=[CLASS_NAMES[i] for i in range(5)],
        )
        csv_path = os.path.join(output_dir, f"transition_matrix_{y1}_{y2}.csv")
        trans_df.to_csv(csv_path)

    # Summary table
    summary_df = pd.DataFrame({
        "Class": [CLASS_NAMES[i] for i in range(5)],
        "2020_Count": analysis["total_2020"],
        "2023_Count": analysis["total_2023"],
        "Net_Change": analysis["net_change"],
        "Pct_Change": analysis["pct_change"],
        "Persistence_2020_2023": (analysis["persistence"] / analysis["total_2020"] * 100).round(2),
    })
    summary_path = os.path.join(output_dir, "summary_2020_2023.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"    Saved: {summary_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print(" HUNZA VALLEY — TEMPORAL CHANGE ANALYSIS (2020-2023)")
    print("=" * 70)

    # Load predictions
    print(f"\n[1/5] Loading predictions...")
    t0 = time.time()
    predictions = load_predictions()
    print(f"  Loaded 4 rasters in {time.time()-t0:.1f}s")
    for year, pred in predictions.items():
        print(f"    {year}: {pred.shape} | classes: {np.unique(pred).tolist()}")

    # Compute transitions
    print(f"\n[2/5] Computing transition matrices...")
    t0 = time.time()
    transitions = compute_all_transitions(predictions)
    analysis = analyze_transitions(transitions)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Computed {len(transitions)} transition matrices")

    # Print summary
    print(f"\n[3/5] Summary (2020 → 2023):")
    total_pixels = analysis["trans"].sum()
    changed_pixels = total_pixels - analysis["persistence"].sum()
    pct_changed = (changed_pixels / total_pixels) * 100
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Changed: {changed_pixels:,} ({pct_changed:.2f}%)")
    print(f"  Stable: {analysis['persistence'].sum():,} ({100-pct_changed:.2f}%)")
    print(f"\n  Per-class changes:")
    for i in range(5):
        name = CLASS_NAMES[i]
        change = analysis["net_change"][i]
        pct = analysis["pct_change"][i]
        print(f"    {name:<20} {change:>+9,} pixels  ({pct:>+6.2f}%)")

    # Visualizations
    print(f"\n[4/5] Generating visualizations...")
    t0 = time.time()
    plot_transition_heatmap(analysis["trans"], 2020, 2023, OUTPUT_DIR)
    plot_net_change(analysis, OUTPUT_DIR)
    plot_change_map(predictions[2020], predictions[2023], OUTPUT_DIR)
    plot_persistence(analysis, OUTPUT_DIR)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Reports & exports
    print(f"\n[5/5] Saving reports...")
    save_transition_report(transitions, analysis, OUTPUT_DIR)
    save_csv_exports(transitions, analysis, OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 70)
    print(" TEMPORAL ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"\n  Generated files:")
    print(f"    transition_matrix_2020_2023.png")
    print(f"    net_change_2020_2023.png")
    print(f"    change_map_2020_2023.png")
    print(f"    persistence_2020_2023.png")
    print(f"    transition_report.txt")
    print(f"    summary_2020_2023.csv")
    print(f"    transition_matrix_*.csv (all year pairs)")
    print(f"\n  Ready for report to Dr. Zaheer!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
