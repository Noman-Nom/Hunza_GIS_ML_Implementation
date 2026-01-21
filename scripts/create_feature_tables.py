#!/usr/bin/env python3
"""
Feature Extraction Script for Hunza Land Cover ML
Extracts features from rasters and creates tabular datasets
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path("/home/muhammad-noman/zaheerproject02-hunza-dataextraction/hunza-ml-analysis")
RAW_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR = BASE_DIR / "data/processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2020, 2021, 2022, 2023]
BAND_NAMES = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
BAND_FILES = ['Band_2_Blue', 'Band_3_Green', 'Band_4_Red', 
              'Band_5_NIR', 'Band_6_SWIR1', 'Band_7_SWIR2']
INDEX_NAMES = ['NDVI', 'NDSI', 'NDWI']

# Landsat 8 Surface Reflectance scaling
SCALE_FACTOR = 0.0000275
OFFSET = -0.2

def scale_band(dn_values):
    """Convert DN to surface reflectance"""
    reflectance = (dn_values * SCALE_FACTOR) + OFFSET
    return np.clip(reflectance, 0, 1)

def load_band(year, band_file):
    """Load and scale a single band"""
    path = RAW_DIR / str(year) / "bands" / f"{band_file}_{year}.tif"
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        data[data < 0] = np.nan  # Mark invalid as NaN
        return scale_band(data)

def load_index(year, index_name):
    """Load an index (already normalized)"""
    path = RAW_DIR / str(year) / "indices" / f"{index_name}_{year}.tif"
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        return data

def load_classification(year):
    """Load classification labels"""
    path = RAW_DIR / str(year) / f"classification_{year}.tif"
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.int8)
        return data

def extract_features_year(year, max_samples=None):
    """
    Extract all features for one year
    
    Parameters:
    -----------
    year : int
        Year to process
    max_samples : int, optional
        Maximum pixels to sample (None = all pixels)
    
    Returns:
    --------
    pd.DataFrame
        Feature table
    """
    print(f"\n{'='*60}")
    print(f"📅 Processing Year {year}")
    print(f"{'='*60}")
    
    # Load bands
    print("📊 Loading bands...")
    bands_data = {}
    for band_name, band_file in zip(BAND_NAMES, BAND_FILES):
        print(f"   Loading {band_name}...", end=' ')
        bands_data[band_name] = load_band(year, band_file).flatten()
        print("✅")
    
    # Load indices
    print("📈 Loading indices...")
    for index_name in INDEX_NAMES:
        print(f"   Loading {index_name}...", end=' ')
        bands_data[index_name] = load_index(year, index_name).flatten()
        print("✅")
    
    # Load classification
    print("🗺️  Loading classification...", end=' ')
    classification = load_classification(year).flatten()
    print("✅")
    
    # Create DataFrame
    print("🔧 Creating feature table...")
    n_pixels = len(classification)
    
    df = pd.DataFrame({
        'Pixel_ID': np.arange(n_pixels),
        'Year': year,
        **bands_data,
        'Class': classification
    })
    
    print(f"   Initial size: {len(df):,} pixels")
    
    # Remove Class 6 (NoData/unclassified)
    df = df[df['Class'] != 6]
    print(f"   After removing Class 6: {len(df):,} pixels")
    
    # Remove NaN/Inf
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    removed = before - len(df)
    print(f"   After removing NoData: {len(df):,} pixels ({removed:,} removed, {removed/before*100:.2f}%)")
    
    # Sample if requested
    if max_samples and len(df) > max_samples:
        print(f"   Sampling {max_samples:,} pixels...")
        df = df.sample(n=max_samples, random_state=42)
        print(f"   Final size: {len(df):,} pixels")
    
    # Class distribution
    print("\n📊 Class Distribution:")
    class_counts = df['Class'].value_counts().sort_index()
    class_names = {
        0: 'Snow/Ice',
        1: 'Water',
        2: 'Bare Rock',
        3: 'Sparse Vegetation',
        4: 'Moderate Vegetation',
        5: 'Dense Vegetation'
    }
    for cls, count in class_counts.items():
        pct = count / len(df) * 100
        print(f"   Class {int(cls)} ({class_names[int(cls)]:20s}): {count:>10,} ({pct:>5.2f}%)")
    
    return df

def main():
    """Main extraction pipeline"""
    print("🚀 Starting Feature Extraction Pipeline")
    print(f"📁 Output directory: {PROCESSED_DIR}")
    
    all_years_data = []
    
    # Process each year
    for year in YEARS:
        try:
            # Extract features (sample 3M pixels per year for memory efficiency)
            df_year = extract_features_year(year, max_samples=3_000_000)
            
            # Save individual year
            output_path = PROCESSED_DIR / f"feature_table_{year}.csv"
            print(f"\n💾 Saving to: {output_path}")
            df_year.to_csv(output_path, index=False)
            print(f"   File size: {output_path.stat().st_size / 1024**2:.2f} MB")
            
            all_years_data.append(df_year)
            
        except Exception as e:
            print(f"❌ ERROR processing year {year}: {str(e)}")
            continue
    
    # Combine all years
    if all_years_data:
        print(f"\n{'='*60}")
        print("🔗 Combining all years...")
        df_combined = pd.concat(all_years_data, ignore_index=True)
        
        combined_path = PROCESSED_DIR / "combined_features.csv"
        print(f"💾 Saving combined dataset: {combined_path}")
        df_combined.to_csv(combined_path, index=False)
        
        print(f"\n✅ Pipeline Complete!")
        print(f"   Total pixels: {len(df_combined):,}")
        print(f"   File size: {combined_path.stat().st_size / 1024**2:.2f} MB")
        print(f"\n📊 Overall Class Distribution:")
        class_counts = df_combined['Class'].value_counts().sort_index()
        for cls, count in class_counts.items():
            pct = count / len(df_combined) * 100
            print(f"   Class {int(cls)}: {count:>12,} ({pct:>5.2f}%)")
    
    print(f"\n{'='*60}")
    print("🎉 All Done! Next: Run sampling and train/test split")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()