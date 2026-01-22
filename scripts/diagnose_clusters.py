#!/usr/bin/env python3
"""
Diagnose K-Means Cluster Assignments
"""

import pandas as pd
import numpy as np

# Load your feature data
df = pd.read_csv('data/processed/combined_features.csv')

print("="*80)
print("🔍 CLUSTER/CLASS DIAGNOSTIC ANALYSIS")
print("="*80)

# Analyze each class
for cls in sorted(df['Class'].unique()):
    if cls == 6:
        continue
    
    df_cls = df[df['Class'] == cls]
    n_pixels = len(df_cls)
    pct = n_pixels / len(df) * 100
    
    print(f"\n{'='*80}")
    print(f"Class {cls}: {n_pixels:,} pixels ({pct:.2f}%)")
    print(f"{'='*80}")
    
    # Spectral signature
    print("📊 Mean Spectral Values:")
    print(f"   Blue:  {df_cls['Blue'].mean():.4f}")
    print(f"   Green: {df_cls['Green'].mean():.4f}")
    print(f"   Red:   {df_cls['Red'].mean():.4f}")
    print(f"   NIR:   {df_cls['NIR'].mean():.4f}")
    print(f"   SWIR1: {df_cls['SWIR1'].mean():.4f}")
    print(f"   SWIR2: {df_cls['SWIR2'].mean():.4f}")
    
    print("\n📈 Mean Index Values:")
    print(f"   NDVI:  {df_cls['NDVI'].mean():.4f}")
    print(f"   NDSI:  {df_cls['NDSI'].mean():.4f}")
    print(f"   NDWI:  {df_cls['NDWI'].mean():.4f}")
    
    # Determine what this class ACTUALLY is
    ndvi = df_cls['NDVI'].mean()
    ndsi = df_cls['NDSI'].mean()
    ndwi = df_cls['NDWI'].mean()
    nir = df_cls['NIR'].mean()
    red = df_cls['Red'].mean()
    swir1 = df_cls['SWIR1'].mean()
    
    print("\n🎯 LIKELY ACTUAL CLASS:")
    if ndsi > 0.4:
        print("   ❄️  SNOW/ICE (high NDSI)")
    elif ndwi > 0.3:
        print("   💧 WATER (high NDWI)")
    elif ndvi < 0.0 and ndsi < 0.4:
        print("   🪨 BARE ROCK/SOIL (negative NDVI, low NDSI)")
    elif 0.0 <= ndvi < 0.15:
        print("   🌾 SPARSE VEGETATION (low positive NDVI)")
    elif 0.15 <= ndvi < 0.3:
        print("   🌿 MODERATE VEGETATION (medium NDVI)")
    elif ndvi >= 0.3:
        print("   🌲 DENSE VEGETATION (high NDVI)")
    else:
        print("   ❓ UNCLEAR - needs visual inspection")

print("\n" + "="*80)
print("✅ DIAGNOSTIC COMPLETE")
print("="*80)