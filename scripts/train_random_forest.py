"""
train_random_forest.py — Hunza Valley Random Forest training and evaluation
===========================================================================
Dual-track workflow:
  1) distillation mode: agreement with source land_cover labels
  2) independent mode: classification accuracy against external reference labels
"""

from __future__ import annotations

import argparse
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split


DEFAULT_DATA_PATH = r"e:\zaheer-work\Hunza_GIS_Data_Extraction\training_data\hunza_training_all.csv"
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

CLASS_NAMES = {
    0: "Snow/Ice",
    1: "Bare Rock",
    2: "Sparse Veg",
    3: "Moderate Veg",
    4: "Dense Veg",
}

RF_PARAMS = dict(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
    oob_score=True,
)

VAL_FRACTION = 0.20
CV_FOLDS = 5
RANDOM_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Train RF for Hunza land cover")
    parser.add_argument(
        "--mode",
        default="distillation",
        choices=["distillation", "independent"],
        help="distillation=agreement with land_cover labels; independent=external reference labels",
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Input CSV path. For independent mode, pass your external label CSV.",
    )
    parser.add_argument(
        "--spatial-grid-size",
        type=int,
        default=20,
        help="Number of bins per axis for spatial block CV grouping.",
    )
    return parser.parse_args()


def class_labels_from_data(y: np.ndarray) -> list[str]:
    labels = sorted(np.unique(y).tolist())
    return [CLASS_NAMES.get(i, f"class_{i}") for i in labels]


def make_spatial_groups(df: pd.DataFrame, grid_size: int) -> np.ndarray:
    lon_bins = pd.qcut(df["lon"], q=grid_size, labels=False, duplicates="drop")
    lat_bins = pd.qcut(df["lat"], q=grid_size, labels=False, duplicates="drop")
    lon_bins = lon_bins.fillna(0).astype(int)
    lat_bins = lat_bins.fillna(0).astype(int)
    return (lat_bins * 1000 + lon_bins).to_numpy()


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str, save_path: str):
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title(f"{title} — Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title(f"{title} — Row % (recall per class)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_importance(rf: RandomForestClassifier, save_path: str):
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2196F3" if i < 6 else "#FF9800" for i in range(len(FEATURES))]
    ax.bar(
        range(len(FEATURES)),
        importances[order],
        yerr=std[order],
        capsize=4,
        color=[colors[i] for i in order],
    )
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels([FEATURES[i] for i in order], rotation=30, ha="right")
    ax.set_ylabel("Mean Decrease in Impurity")
    ax.set_title("Random Forest Feature Importance — Hunza Valley")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_cv_scores(cv_scores: np.ndarray, title: str, save_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    folds = range(1, len(cv_scores) + 1)
    ax.bar(folds, cv_scores * 100, color="#4CAF50", alpha=0.85)
    ax.axhline(cv_scores.mean() * 100, color="red", linestyle="--", label=f"Mean = {cv_scores.mean()*100:.2f}%")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def evaluate_split(rf: RandomForestClassifier, X: np.ndarray, y: np.ndarray, split_title: str, plot_name: str):
    y_pred = rf.predict(X)
    labels = class_labels_from_data(y)
    numeric_labels = sorted(np.unique(y).tolist())
    acc = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    cm = confusion_matrix(y, y_pred, labels=numeric_labels)
    report = classification_report(y, y_pred, labels=numeric_labels, target_names=labels, digits=4)

    print(f"\n  ── {split_title} ──")
    print(f"  Accuracy      : {acc*100:.2f}%")
    print(f"  Cohen's Kappa : {kappa:.4f}")
    print(f"\n{report}")

    plot_confusion_matrix(cm, labels, split_title, os.path.join(FIGS_DIR, plot_name))
    return acc, kappa, report


def run_spatial_block_cv(df_train: pd.DataFrame, grid_size: int) -> np.ndarray:
    X = df_train[FEATURES].values
    y = df_train[TARGET].values
    groups = make_spatial_groups(df_train, grid_size=grid_size)

    gkf = GroupKFold(n_splits=CV_FOLDS)
    scores = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X[tr_idx], y[tr_idx])
        pred = rf.predict(X[va_idx])
        score = accuracy_score(y[va_idx], pred)
        print(f"  Spatial fold {fold}: {score*100:.2f}%")
        scores.append(score)
    return np.array(scores)


def split_data(df: pd.DataFrame, mode: str):
    if mode == "distillation":
        if "year" not in df.columns:
            raise ValueError("distillation mode expects a 'year' column for temporal holdout.")
        df_trainval = df[df["year"] != 2023].copy()
        df_test = df[df["year"] == 2023].copy()
        train_df, val_df = train_test_split(
            df_trainval,
            test_size=VAL_FRACTION,
            stratify=df_trainval[TARGET],
            random_state=RANDOM_SEED,
        )
        split_note = "Distillation mode: agreement with source land_cover labels."
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=0.20,
            stratify=df[TARGET],
            random_state=RANDOM_SEED,
        )
        train_df, val_df = train_test_split(
            train_df,
            test_size=VAL_FRACTION,
            stratify=train_df[TARGET],
            random_state=RANDOM_SEED,
        )
        df_test = test_df
        split_note = "Independent mode: external-reference classification accuracy."

    return train_df, val_df, df_test, split_note


def main():
    args = parse_args()

    print("=" * 65)
    print(" Hunza Valley — Random Forest Training")
    print("=" * 65)
    print(f"Mode: {args.mode}")
    print(f"Data: {args.data_path}")

    print(f"\n[1/7] Loading dataset...")
    t0 = time.time()
    df = pd.read_csv(args.data_path)
    print(f"  Loaded {len(df):,} rows in {time.time() - t0:.1f}s")
    print(f"  Features: {FEATURES}")
    if "class_name" in df.columns:
        print(f"  Class distribution:\n{df['class_name'].value_counts().to_string()}")
    else:
        print(f"  Class distribution:\n{df[TARGET].value_counts().sort_index().to_string()}")

    for col in FEATURES + [TARGET]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    if args.mode == "independent":
        print("  Note: metrics below are interpreted as independent classification accuracy.")
    else:
        print("  Note: metrics below are agreement with source land_cover labels (distillation).")

    print(f"\n[2/7] Splitting data...")
    train_df, val_df, test_df, split_note = split_data(df, args.mode)
    print(f"  {split_note}")
    print(f"  Train rows: {len(train_df):,}")
    print(f"  Val rows:   {len(val_df):,}")
    print(f"  Test rows:  {len(test_df):,}")

    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values
    X_val = val_df[FEATURES].values
    y_val = val_df[TARGET].values
    X_test = test_df[FEATURES].values
    y_test = test_df[TARGET].values

    print(f"\n[3/7] Stratified {CV_FOLDS}-fold CV on train split...")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_train[tr_idx], y_train[tr_idx])
        pred = rf.predict(X_train[va_idx])
        score = accuracy_score(y_train[va_idx], pred)
        print(f"  Stratified fold {fold}: {score*100:.2f}%")
        cv_scores.append(score)
    cv_scores = np.array(cv_scores)
    plot_cv_scores(cv_scores, f"{CV_FOLDS}-Fold Stratified CV — Accuracy", os.path.join(FIGS_DIR, "cv_scores_stratified.png"))

    print(f"\n[4/7] Spatial block CV on train split...")
    if {"lon", "lat"}.issubset(train_df.columns):
        spatial_scores = run_spatial_block_cv(train_df, grid_size=args.spatial_grid_size)
        plot_cv_scores(
            spatial_scores,
            f"{CV_FOLDS}-Fold Spatial Block CV — Accuracy",
            os.path.join(FIGS_DIR, "cv_scores_spatial.png"),
        )
    else:
        spatial_scores = np.array([])
        print("  Skipped spatial block CV (lon/lat columns are missing).")

    print(f"\n[5/7] Training final RF on train split...")
    rf_final = RandomForestClassifier(**RF_PARAMS)
    t_fit = time.time()
    rf_final.fit(X_train, y_train)
    print(f"  Training time: {time.time() - t_fit:.1f}s")
    print(f"  OOB score: {rf_final.oob_score_*100:.2f}%")

    model_name = "rf_model_distillation.pkl" if args.mode == "distillation" else "rf_model_independent.pkl"
    model_path = os.path.join(MODELS_DIR, model_name)
    joblib.dump(rf_final, model_path)
    print(f"  Model saved: {model_path} ({os.path.getsize(model_path)/1e6:.1f} MB)")

    print(f"\n[6/7] Evaluating validation and test splits...")
    val_title = "Validation split"
    test_title = "Temporal holdout 2023" if args.mode == "distillation" and "year" in test_df.columns else "Test split"
    val_acc, val_kappa, _ = evaluate_split(rf_final, X_val, y_val, val_title, f"confusion_matrix_val_{args.mode}.png")
    test_acc, test_kappa, test_report = evaluate_split(rf_final, X_test, y_test, test_title, f"confusion_matrix_test_{args.mode}.png")

    print(f"\n[7/7] Feature importance...")
    importances = rf_final.feature_importances_
    order = np.argsort(importances)[::-1]
    print(f"  {'Feature':<15} {'Importance':>10}")
    print(f"  {'-'*27}")
    for idx in order:
        print(f"  {FEATURES[idx]:<15} {importances[idx]:>10.4f}")
    plot_feature_importance(rf_final, os.path.join(FIGS_DIR, f"feature_importance_{args.mode}.png"))

    report_name = "classification_report_distillation.txt" if args.mode == "distillation" else "classification_report_independent.txt"
    report_path = os.path.join(MODELS_DIR, report_name)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Hunza Valley — Random Forest Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Mode: {args.mode}\n")
        if args.mode == "distillation":
            f.write("Interpretation: Scores represent agreement with source land_cover labels, not independent ground-truth accuracy.\n")
        else:
            f.write("Interpretation: Scores represent independent classification accuracy against external labels.\n")
        f.write(f"Train rows: {len(train_df):,}\n")
        f.write(f"Val rows: {len(val_df):,}\n")
        f.write(f"Test rows: {len(test_df):,}\n")
        f.write(f"OOB score: {rf_final.oob_score_:.6f}\n")
        f.write(f"Validation accuracy: {val_acc:.6f}\n")
        f.write(f"Validation kappa: {val_kappa:.6f}\n")
        f.write(f"Test accuracy: {test_acc:.6f}\n")
        f.write(f"Test kappa: {test_kappa:.6f}\n")
        f.write(f"Stratified CV scores: {[round(x, 6) for x in cv_scores.tolist()]}\n")
        f.write(f"Stratified CV mean±std: {cv_scores.mean():.6f} ± {cv_scores.std():.6f}\n")
        if spatial_scores.size:
            f.write(f"Spatial CV scores: {[round(x, 6) for x in spatial_scores.tolist()]}\n")
            f.write(f"Spatial CV mean±std: {spatial_scores.mean():.6f} ± {spatial_scores.std():.6f}\n")
        f.write("\nTest classification report\n")
        f.write(test_report)
    print(f"  Report saved: {report_path}")

    print("\n" + "=" * 65)
    print(" TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Mode: {args.mode}")
    print(f"  Model: {model_path}")
    print(f"  OOB: {rf_final.oob_score_*100:.2f}%")
    print(f"  Val: {val_acc*100:.2f}% | Kappa: {val_kappa:.4f}")
    print(f"  Test: {test_acc*100:.2f}% | Kappa: {test_kappa:.4f}")
    print(f"  Stratified CV mean±std: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    if spatial_scores.size:
        print(f"  Spatial CV mean±std: {spatial_scores.mean()*100:.2f}% ± {spatial_scores.std()*100:.2f}%")
    print(f"  Figures: {FIGS_DIR}")
    print(f"  Next step: predict_all_years.py")


if __name__ == "__main__":
    main()
