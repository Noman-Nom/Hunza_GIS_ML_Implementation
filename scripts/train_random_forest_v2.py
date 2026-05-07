"""
train_random_forest_v2.py — Hunza RF training (code only, no auto-fit)
======================================================================
Split-and-save workflow:
  1) Load and split data (temporal: train on 2020-2022, test on 2023)
  2) Run 5-fold stratified CV on train split (no model save yet)
  3) Train final RF on full train split
  4) Save model, evaluate on val/test, plot confusion matrices, feature importance

Run:  python train_random_forest_v2.py --help
  or: python train_random_forest_v2.py --mode distillation
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, cohen_kappa_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, train_test_split

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_DATA_PATH = r"e:\zaheer-work\Hunza_GIS_Data_Extraction\training_data\hunza_training_all.csv"
MODELS_DIR = r"e:\zaheer-work\Hunza_GIS_ML_Implementation\models"
FIGS_DIR = r"e:\zaheer-work\Hunza_GIS_ML_Implementation\figures"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

SPECTRAL_FEATURES = [
    "B2_Blue", "B3_Green", "B4_Red", "B5_NIR", "B6_SWIR1", "B7_SWIR2",
    "NDVI", "NDSI", "NDWI",
]
TARGET = "class_id"
CLASS_NAMES = {0: "Snow/Ice", 1: "Bare Rock", 2: "Sparse Veg", 3: "Moderate Veg", 4: "Dense Veg"}

RF_PARAMS = {
    "n_estimators": 300,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
    "oob_score": True,
}

CV_FOLDS = 5
VAL_FRACTION = 0.20
RANDOM_SEED = 42

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RF for Hunza land cover (code-only, ready to run)"
    )
    parser.add_argument(
        "--mode",
        default="distillation",
        choices=["distillation", "independent"],
        help="distillation=train on 2020-2022, test on 2023 (temporal holdout)"
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Path to training CSV"
    )
    return parser.parse_args()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def class_labels(y):
    """Get class names for a set of class IDs."""
    labels = sorted(np.unique(y).tolist())
    return [CLASS_NAMES.get(i, f"class_{i}") for i in labels]

def plot_confusion_matrix(cm, labels, title, save_path):
    """Plot confusion matrix in counts and percentages."""
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title(f"{title} — Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title(f"{title} — Row % (recall)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")

def plot_cv_scores(cv_scores, title, save_path):
    """Plot CV fold scores."""
    fig, ax = plt.subplots(figsize=(8, 5))
    folds = range(1, len(cv_scores) + 1)
    ax.bar(folds, cv_scores * 100, color="#4CAF50", alpha=0.85)
    ax.axhline(cv_scores.mean() * 100, color="red", linestyle="--",
              label=f"Mean={cv_scores.mean()*100:.2f}%")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim([90, 101])
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")

def plot_feature_importance(rf, save_path):
    """Plot feature importance."""
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(SPECTRAL_FEATURES)), importances[order],
          yerr=std[order], capsize=4, color="#2196F3")
    ax.set_xticks(range(len(SPECTRAL_FEATURES)))
    ax.set_xticklabels([SPECTRAL_FEATURES[i] for i in order], rotation=30, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importance — Hunza Valley RF")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")

def evaluate_split(rf, X, y, split_name, plot_name):
    """Evaluate RF on a split and plot confusion matrix."""
    y_pred = rf.predict(X)
    acc = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    numeric_labels = sorted(np.unique(y).tolist())
    labels = class_labels(y)
    cm = confusion_matrix(y, y_pred, labels=numeric_labels)
    report = classification_report(y, y_pred, labels=numeric_labels,
                                  target_names=labels, digits=4)

    print(f"\n  ── {split_name} ──")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  Kappa:    {kappa:.4f}")
    print(f"\n{report}")

    plot_confusion_matrix(cm, labels, split_name, os.path.join(FIGS_DIR, plot_name))
    return acc, kappa, report

# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print(" HUNZA VALLEY — RANDOM FOREST TRAINING (v2)")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Data: {args.data_path}")

    # [1] Load
    print(f"\n[1/6] Loading dataset...")
    t0 = time.time()
    df = pd.read_csv(args.data_path)
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    print(f"  Features: {SPECTRAL_FEATURES}")
    print(f"\n  Class distribution (full dataset):")
    print(df["class_name"].value_counts().sort_index().to_string())

    # Validate columns
    for col in SPECTRAL_FEATURES + [TARGET]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # [2] Split
    print(f"\n[2/6] Splitting data...")
    if args.mode == "distillation":
        if "year" not in df.columns:
            raise ValueError("distillation mode needs 'year' column")
        df_trainval = df[df["year"] != 2023].copy()
        df_test = df[df["year"] == 2023].copy()
        X_train, X_val, y_train, y_val = train_test_split(
            df_trainval[SPECTRAL_FEATURES].values,
            df_trainval[TARGET].values,
            test_size=VAL_FRACTION,
            stratify=df_trainval[TARGET].values,
            random_state=RANDOM_SEED,
        )
        X_test, y_test = df_test[SPECTRAL_FEATURES].values, df_test[TARGET].values
        split_info = "Distillation: train=2020-2022, val=2020-2022 (random), test=2023 (temporal holdout)"
    else:
        # Independent mode: random split
        train_data, test_data = train_test_split(df, test_size=0.20,
                                                  stratify=df[TARGET],
                                                  random_state=RANDOM_SEED)
        X_train, X_val, y_train, y_val = train_test_split(
            train_data[SPECTRAL_FEATURES].values,
            train_data[TARGET].values,
            test_size=VAL_FRACTION,
            stratify=train_data[TARGET].values,
            random_state=RANDOM_SEED,
        )
        X_test, y_test = test_data[SPECTRAL_FEATURES].values, test_data[TARGET].values
        split_info = "Independent: random 60/20/20 train/val/test split"

    print(f"  {split_info}")
    print(f"  Train: {len(X_train):,} rows")
    print(f"  Val:   {len(X_val):,} rows")
    print(f"  Test:  {len(X_test):,} rows")

    # [3] CV
    print(f"\n[3/6] Running {CV_FOLDS}-fold stratified CV on train split...")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_train[tr_idx], y_train[tr_idx])
        pred = rf.predict(X_train[va_idx])
        score = accuracy_score(y_train[va_idx], pred)
        cv_scores.append(score)
        print(f"  Fold {fold}: {score*100:.2f}%")

    cv_scores = np.array(cv_scores)
    plot_cv_scores(cv_scores, f"{CV_FOLDS}-Fold Stratified CV",
                   os.path.join(FIGS_DIR, "cv_scores_stratified.png"))
    print(f"  Mean: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # [4] Train
    print(f"\n[4/6] Training final RF on full train split ({len(X_train):,} rows)...")
    rf_final = RandomForestClassifier(**RF_PARAMS)
    t_fit = time.time()
    rf_final.fit(X_train, y_train)
    fit_time = time.time() - t_fit
    print(f"  Time: {fit_time:.1f}s")
    print(f"  OOB score: {rf_final.oob_score_*100:.2f}%")

    # Save model
    model_name = f"rf_model_{args.mode}.pkl"
    model_path = os.path.join(MODELS_DIR, model_name)
    joblib.dump(rf_final, model_path)
    size_mb = os.path.getsize(model_path) / 1e6
    print(f"  Model saved: {model_path} ({size_mb:.1f} MB)")

    # [5] Evaluate
    print(f"\n[5/6] Evaluating on validation and test splits...")
    val_acc, val_kappa, _ = evaluate_split(rf_final, X_val, y_val,
                                           "Validation split",
                                           f"confusion_matrix_val_{args.mode}.png")
    test_title = "Test (2023 holdout)" if args.mode == "distillation" else "Test split"
    test_acc, test_kappa, test_report = evaluate_split(rf_final, X_test, y_test,
                                                       test_title,
                                                       f"confusion_matrix_test_{args.mode}.png")

    # [6] Feature importance
    print(f"\n[6/6] Feature importance...")
    importances = rf_final.feature_importances_
    order = np.argsort(importances)[::-1]
    print(f"  {'Feature':<15} {'Importance':>10}")
    print(f"  {'-'*27}")
    for idx in order:
        print(f"  {SPECTRAL_FEATURES[idx]:<15} {importances[idx]:>10.4f}")
    plot_feature_importance(rf_final, os.path.join(FIGS_DIR, f"feature_importance_{args.mode}.png"))

    # Save report
    report_name = f"classification_report_{args.mode}.txt"
    report_path = os.path.join(MODELS_DIR, report_name)
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("HUNZA VALLEY — RANDOM FOREST CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Training time: {fit_time:.1f}s\n\n")
        f.write(f"Train rows: {len(X_train):,}\n")
        f.write(f"Val rows:   {len(X_val):,}\n")
        f.write(f"Test rows:  {len(X_test):,}\n\n")
        f.write(f"OOB score:      {rf_final.oob_score_:.6f}\n")
        f.write(f"Val accuracy:   {val_acc:.6f}  |  Kappa: {val_kappa:.6f}\n")
        f.write(f"Test accuracy:  {test_acc:.6f}  |  Kappa: {test_kappa:.6f}\n\n")
        f.write(f"Stratified CV:  {cv_scores.mean():.6f} ± {cv_scores.std():.6f}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("TEST CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(test_report)
    print(f"  Report saved: {report_path}")

    # Summary
    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Model:  {model_path}")
    print(f"  OOB:    {rf_final.oob_score_*100:.2f}%")
    print(f"  Val:    {val_acc*100:.2f}% (Kappa: {val_kappa:.4f})")
    print(f"  Test:   {test_acc*100:.2f}% (Kappa: {test_kappa:.4f})")
    print(f"  CV:     {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    print(f"  Figs:   {FIGS_DIR}")
    print(f"\n  Next: python predict_all_years.py")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
