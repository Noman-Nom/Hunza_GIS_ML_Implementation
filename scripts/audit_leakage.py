"""
audit_leakage.py — Hunza Valley leakage/circularity diagnostics
===============================================================
Runs read-only audits on the combined training CSV to detect:
  1) Feature duplicates across train/val/test splits
  2) Coordinate overlap across splits (exact and approximate)
  3) Trivial separability using simple baseline models

Outputs:
  - models/leakage_audit_report.txt
  - figures/audit_ndvi_by_class.png
  - figures/audit_ndsi_by_class.png
"""

from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier


DATA_PATH = r"e:\zaheer-work\Hunza_GIS_Data_Extraction\training_data\hunza_training_all.csv"
MODELS_DIR = r"e:\zaheer-work\Hunza_GIS_ML_Implementation\models"
FIGS_DIR = r"e:\zaheer-work\Hunza_GIS_ML_Implementation\figures"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

FEATURES = [
    "B2_Blue",
    "B3_Green",
    "B4_Red",
    "B5_NIR",
    "B6_SWIR1",
    "B7_SWIR2",
    "NDVI",
    "NDSI",
    "NDWI",
]
TARGET = "class_id"


def _split(df: pd.DataFrame):
    df_trainval = df[df["year"] != 2023].copy()
    df_test = df[df["year"] == 2023].copy()

    train_df, val_df = train_test_split(
        df_trainval,
        test_size=0.20,
        stratify=df_trainval[TARGET],
        random_state=42,
    )
    return train_df, val_df, df_test


def _overlap_ratio(left: pd.Series, right: pd.Series) -> float:
    left_set = set(left.tolist())
    right_set = set(right.tolist())
    if not left_set:
        return 0.0
    return len(left_set.intersection(right_set)) / len(left_set)


def _feature_signature(df: pd.DataFrame, decimals: int = 6) -> pd.Series:
    rounded = df[FEATURES].round(decimals)
    return rounded.astype(str).agg("|".join, axis=1)


def _coord_signature(df: pd.DataFrame, decimals: int = 8) -> pd.Series:
    coord = df[["lon", "lat"]].round(decimals)
    return coord.astype(str).agg("|".join, axis=1)


def _plot_index_distributions(df: pd.DataFrame, index_name: str, save_name: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="class_name", y=index_name, ax=ax)
    ax.set_title(f"{index_name} distribution by class")
    ax.set_xlabel("Class")
    ax.set_ylabel(index_name)
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, save_name), dpi=150, bbox_inches="tight")
    plt.close()


def run_audit() -> None:
    t0 = time.time()
    print("=" * 66)
    print(" Hunza Valley — Leakage/Circularity Audit")
    print("=" * 66)
    print(f"\nLoading: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Rows loaded: {len(df):,}")

    train_df, val_df, test_df = _split(df)
    print(f"Train rows: {len(train_df):,}")
    print(f"Val rows:   {len(val_df):,}")
    print(f"Test rows:  {len(test_df):,} (2023 temporal holdout)")

    # 1) Feature duplication overlap
    print("\n[1/3] Feature overlap across splits...")
    train_feat = _feature_signature(train_df)
    val_feat = _feature_signature(val_df)
    test_feat = _feature_signature(test_df)

    feat_overlap_train_val = _overlap_ratio(train_feat, val_feat)
    feat_overlap_train_test = _overlap_ratio(train_feat, test_feat)
    feat_overlap_val_test = _overlap_ratio(val_feat, test_feat)

    # 2) Coordinate overlap (exact and rounded)
    print("[2/3] Coordinate overlap checks...")
    train_coord_exact = _coord_signature(train_df, decimals=8)
    val_coord_exact = _coord_signature(val_df, decimals=8)
    test_coord_exact = _coord_signature(test_df, decimals=8)

    coord_exact_train_val = _overlap_ratio(train_coord_exact, val_coord_exact)
    coord_exact_train_test = _overlap_ratio(train_coord_exact, test_coord_exact)
    coord_exact_val_test = _overlap_ratio(val_coord_exact, test_coord_exact)

    train_coord_approx = _coord_signature(train_df, decimals=4)
    val_coord_approx = _coord_signature(val_df, decimals=4)
    test_coord_approx = _coord_signature(test_df, decimals=4)

    coord_approx_train_val = _overlap_ratio(train_coord_approx, val_coord_approx)
    coord_approx_train_test = _overlap_ratio(train_coord_approx, test_coord_approx)
    coord_approx_val_test = _overlap_ratio(val_coord_approx, test_coord_approx)

    # 3) Trivial separability tests
    print("[3/3] Trivial baseline separability tests...")
    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values
    X_test = test_df[FEATURES].values
    y_test = test_df[TARGET].values

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_acc = accuracy_score(y_test, dummy.predict(X_test))

    stump = DecisionTreeClassifier(max_depth=1, random_state=42)
    stump.fit(X_train, y_train)
    stump_acc = accuracy_score(y_test, stump.predict(X_test))

    tree3 = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree3.fit(X_train, y_train)
    tree3_acc = accuracy_score(y_test, tree3.predict(X_test))

    _plot_index_distributions(df, "NDVI", "audit_ndvi_by_class.png")
    _plot_index_distributions(df, "NDSI", "audit_ndsi_by_class.png")

    report_path = os.path.join(MODELS_DIR, "leakage_audit_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Hunza Valley — Leakage/Circularity Audit Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Rows: {len(df):,}\n")
        f.write(f"Train rows: {len(train_df):,}\n")
        f.write(f"Val rows: {len(val_df):,}\n")
        f.write(f"Test rows: {len(test_df):,}\n\n")

        f.write("[Feature signature overlap ratio]\n")
        f.write(f"train -> val  : {feat_overlap_train_val:.6f}\n")
        f.write(f"train -> test : {feat_overlap_train_test:.6f}\n")
        f.write(f"val -> test   : {feat_overlap_val_test:.6f}\n\n")

        f.write("[Coordinate overlap ratio | exact (8 decimals)]\n")
        f.write(f"train -> val  : {coord_exact_train_val:.6f}\n")
        f.write(f"train -> test : {coord_exact_train_test:.6f}\n")
        f.write(f"val -> test   : {coord_exact_val_test:.6f}\n\n")

        f.write("[Coordinate overlap ratio | approximate (4 decimals)]\n")
        f.write(f"train -> val  : {coord_approx_train_val:.6f}\n")
        f.write(f"train -> test : {coord_approx_train_test:.6f}\n")
        f.write(f"val -> test   : {coord_approx_val_test:.6f}\n\n")

        f.write("[Trivial baseline test accuracy on temporal holdout (2023)]\n")
        f.write(f"Dummy most_frequent: {dummy_acc:.6f}\n")
        f.write(f"Decision tree depth=1: {stump_acc:.6f}\n")
        f.write(f"Decision tree depth=3: {tree3_acc:.6f}\n\n")

        f.write("Interpretation note:\n")
        f.write(
            "- Very high RF accuracy with low leakage overlap still can indicate circular supervision\n"
        )
        f.write(
            "  when training labels are generated from the same spectral/index feature space.\n"
        )

    print(f"Report saved: {report_path}")
    print(f"Plots saved: {os.path.join(FIGS_DIR, 'audit_ndvi_by_class.png')}")
    print(f"Plots saved: {os.path.join(FIGS_DIR, 'audit_ndsi_by_class.png')}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_audit()
