"""
train_random_forest.py — Hunza Valley Random Forest Classifier
===============================================================
Loads the corrected training CSVs produced by extract_corrected.py,
trains a Random Forest with class_weight='balanced', evaluates with
5-fold stratified cross-validation and a temporal holdout (2023).

Split strategy (temporally aware):
  Train : 2020 + 2021 + 2022  (80% of those rows → train, 20% → val)
  Test  : 2023 (full year held out — true temporal generalisation test)
  CV    : 5-fold stratified on the train split only

Features (9): B2_Blue, B3_Green, B4_Red, B5_NIR, B6_SWIR1, B7_SWIR2,
              NDVI, NDSI, NDWI

Classes (5):
  0 → Snow/Ice
  1 → Bare Rock
  2 → Sparse Vegetation
  3 → Moderate Vegetation
  4 → Dense Vegetation

Outputs (all in Hunza_GIS_ML_Implementation/):
  models/rf_model.pkl              — trained model (joblib)
  models/classification_report.txt — per-class precision/recall/F1
  figures/confusion_matrix_val.png  — validation set CM
  figures/confusion_matrix_test.png — 2023 temporal holdout CM
  figures/feature_importance.png    — RF feature importance bar chart
  figures/cv_scores.png             — 5-fold CV accuracy distribution
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, cohen_kappa_score,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = r"e:\zaheer-work\Hunza_GIS_Data_Extraction\training_data\hunza_training_all.csv"
MODELS_DIR = r"e:\zaheer-work\Hunza_GIS_ML_Implementation\models"
FIGS_DIR   = r"e:\zaheer-work\Hunza_GIS_ML_Implementation\figures"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR,   exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
FEATURES = ["B2_Blue", "B3_Green", "B4_Red", "B5_NIR",
            "B6_SWIR1", "B7_SWIR2", "NDVI", "NDSI", "NDWI"]
TARGET   = "class_id"

CLASS_NAMES = {
    0: "Snow/Ice",
    1: "Bare Rock",
    2: "Sparse Veg",
    3: "Moderate Veg",
    4: "Dense Veg",
}
CLASS_LABELS = [CLASS_NAMES[i] for i in range(5)]

RF_PARAMS = dict(
    n_estimators=300,
    max_features="sqrt",      # standard for classification
    class_weight="balanced",  # corrects for rare Dense Veg
    n_jobs=-1,
    random_state=42,
    oob_score=True,           # free accuracy estimate on unseen bootstrap samples
)

VAL_FRACTION = 0.20   # fraction of 2020–2022 data held out for validation
CV_FOLDS     = 5
RANDOM_SEED  = 42


# ── Plotting helpers ───────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, title, save_path):
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=axes[0])
    axes[0].set_title(f"{title} — Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Percentages
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=axes[1])
    axes[1].set_title(f"{title} — Row % (recall per class)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_importance(rf, save_path):
    importances = rf.feature_importances_
    std = np.std([t.feature_importances_ for t in rf.estimators_], axis=0)
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2196F3" if i < 6 else "#FF9800" for i in range(len(FEATURES))]
    bars = ax.bar(range(len(FEATURES)), importances[order],
                  yerr=std[order], capsize=4,
                  color=[colors[i] for i in order])
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels([FEATURES[i] for i in order], rotation=30, ha="right")
    ax.set_ylabel("Mean Decrease in Impurity")
    ax.set_title("Random Forest Feature Importance — Hunza Valley Land Cover")

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#2196F3", label="Spectral bands (B2–B7)"),
        Patch(color="#FF9800", label="Spectral indices (NDVI/NDSI/NDWI)"),
    ], loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_cv_scores(cv_scores, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    folds = range(1, len(cv_scores) + 1)
    ax.bar(folds, cv_scores * 100, color="#4CAF50", alpha=0.8)
    ax.axhline(cv_scores.mean() * 100, color="red", linestyle="--",
               label=f"Mean = {cv_scores.mean()*100:.2f}%")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(85, 100)
    ax.set_title(f"{CV_FOLDS}-Fold Stratified CV — Accuracy per Fold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print(" Hunza Valley — Random Forest Training")
    print("=" * 65)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    print(f"\n[1/6] Loading {DATA_PATH} …")
    t0 = time.time()
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    print(f"  Features: {FEATURES}")
    print(f"  Class distribution:\n{df['class_name'].value_counts().to_string()}")

    # ── 2. Temporal split: train+val = 2020–2022, test = 2023 ─────────────────
    print(f"\n[2/6] Splitting data (train/val: 2020–2022  |  test: 2023)…")
    df_trainval = df[df["year"] != 2023].copy()
    df_test     = df[df["year"] == 2023].copy()

    X_trainval = df_trainval[FEATURES].values
    y_trainval = df_trainval[TARGET].values
    X_test     = df_test[FEATURES].values
    y_test     = df_test[TARGET].values

    # Val split (stratified) within 2020–2022
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=VAL_FRACTION,
        stratify=y_trainval,
        random_state=RANDOM_SEED,
    )
    print(f"  Train : {len(X_train):>8,} rows  (years 2020–2022, 80%)")
    print(f"  Val   : {len(X_val):>8,} rows  (years 2020–2022, 20%)")
    print(f"  Test  : {len(X_test):>8,} rows  (year 2023, temporal holdout)")

    # ── 3. 5-fold CV on train split ───────────────────────────────────────────
    print(f"\n[3/6] 5-fold stratified cross-validation on train split…")
    rf_cv = RandomForestClassifier(**RF_PARAMS)
    skf   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    t_cv  = time.time()
    cv_scores = cross_val_score(rf_cv, X_train, y_train, cv=skf,
                                scoring="accuracy", n_jobs=-1)
    print(f"  CV scores: {[f'{s*100:.2f}%' for s in cv_scores]}")
    print(f"  Mean: {cv_scores.mean()*100:.2f}%  ±  {cv_scores.std()*100:.2f}%")
    print(f"  Time: {time.time()-t_cv:.1f}s")
    plot_cv_scores(cv_scores, os.path.join(FIGS_DIR, "cv_scores.png"))

    # ── 4. Train final RF on full train split ─────────────────────────────────
    print(f"\n[4/6] Training final RF on full train split ({len(X_train):,} rows)…")
    rf = RandomForestClassifier(**RF_PARAMS)
    t_fit = time.time()
    rf.fit(X_train, y_train)
    print(f"  Training time: {time.time()-t_fit:.1f}s")
    print(f"  OOB accuracy:  {rf.oob_score_*100:.2f}%")

    # Save model
    model_path = os.path.join(MODELS_DIR, "rf_model.pkl")
    joblib.dump(rf, model_path)
    size_mb = os.path.getsize(model_path) / 1e6
    print(f"  Model saved: {model_path}  ({size_mb:.1f} MB)")

    # ── 5. Evaluate on val and test ───────────────────────────────────────────
    print(f"\n[5/6] Evaluating…")

    for split_name, X_eval, y_eval in [
        ("Validation (2020–2022)", X_val, y_val),
        ("Test — Temporal holdout (2023)", X_test, y_test),
    ]:
        y_pred = rf.predict(X_eval)
        acc    = accuracy_score(y_eval, y_pred)
        kappa  = cohen_kappa_score(y_eval, y_pred)
        cm     = confusion_matrix(y_eval, y_pred)
        report = classification_report(y_eval, y_pred,
                                       target_names=CLASS_LABELS, digits=4)

        print(f"\n  ── {split_name} ──")
        print(f"  Overall accuracy : {acc*100:.2f}%")
        print(f"  Cohen's Kappa    : {kappa:.4f}")
        print(f"\n{report}")

        tag = "val" if "Val" in split_name else "test"
        plot_confusion_matrix(
            cm,
            title=split_name,
            save_path=os.path.join(FIGS_DIR, f"confusion_matrix_{tag}.png"),
        )

    # Save full classification report
    report_path = os.path.join(MODELS_DIR, "classification_report.txt")
    y_pred_test = rf.predict(X_test)
    with open(report_path, "w") as f:
        f.write(f"Hunza Valley — Random Forest Classification Report\n")
        f.write(f"Train: 2020–2022 ({len(X_train):,} rows)  |  Test: 2023 ({len(X_test):,} rows)\n")
        f.write(f"OOB accuracy: {rf.oob_score_*100:.2f}%\n")
        f.write(f"Test accuracy: {accuracy_score(y_test, y_pred_test)*100:.2f}%\n")
        f.write(f"Cohen Kappa:  {cohen_kappa_score(y_test, y_pred_test):.4f}\n\n")
        f.write(f"CV scores: {[round(s,4) for s in cv_scores.tolist()]}\n")
        f.write(f"CV mean±std: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
        f.write(classification_report(y_test, y_pred_test,
                                      target_names=CLASS_LABELS, digits=4))
    print(f"\n  Report saved: {report_path}")

    # ── 6. Feature importance ─────────────────────────────────────────────────
    print(f"\n[6/6] Feature importance…")
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]
    print(f"  {'Feature':<15} {'Importance':>10}")
    print(f"  {'-'*27}")
    for i in order:
        print(f"  {FEATURES[i]:<15} {importances[i]:>10.4f}")
    plot_feature_importance(rf, os.path.join(FIGS_DIR, "feature_importance.png"))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(" TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Model     : {model_path}")
    print(f"  OOB acc   : {rf.oob_score_*100:.2f}%")
    y_pred_val = rf.predict(X_val)
    print(f"  Val acc   : {accuracy_score(y_val, y_pred_val)*100:.2f}%  |  Kappa: {cohen_kappa_score(y_val, y_pred_val):.4f}")
    print(f"  Test acc  : {accuracy_score(y_test, y_pred_test)*100:.2f}%  |  Kappa: {cohen_kappa_score(y_test, y_pred_test):.4f}")
    print(f"  CV mean   : {cv_scores.mean()*100:.2f}%  ±  {cv_scores.std()*100:.2f}%")
    print(f"\n  Outputs in {FIGS_DIR}:")
    for f in ["cv_scores.png", "confusion_matrix_val.png",
              "confusion_matrix_test.png", "feature_importance.png"]:
        print(f"    {f}")
    print(f"\n  Next step: predict_all_years.py")


if __name__ == "__main__":
    main()
