#!/usr/bin/env python3
"""
Verify classification between original and ML-extracted data
Compare Hunza_Data_Analysis with hunza-ml-analysis
"""

import pandas as pd
import numpy as np
import rasterio
from pathlib import Path

# Paths
ORIGINAL_BASE = Path("/home/muhammad-noman/zaheerproject02-hunza-dataextraction/Hunza_Data_Analysis")
ML_BASE = Path("/home/muhammad-noman/zaheerproject02-hunza-dataextraction/hunza-ml-analysis")

YEARS = [2020, 2021, 2022, 2023]

def check_original_structure():
    """Check what's in the original Hunza_Data_Analysis folder"""
    print("="*80)
    print("📁 ORIGINAL DATA STRUCTURE (Hunza_Data_Analysis)")
    print("="*80)
    
    for year in YEARS:
        year_dir = ORIGINAL_BASE / f"geospatial_project_{year}"
        
        print(f"\n📅 Year {year}:")
        
        if year_dir.exists():
            print(f"   ✅ Directory exists: {year_dir}")
            
            # Check for classification files
            outputs_dir = year_dir / "outputs"
            if outputs_dir.exists():
                print(f"   📂 outputs/")
                for subdir in outputs_dir.iterdir():
                    if subdir.is_dir():
                        print(f"      📂 {subdir.name}/")
                        for f in list(subdir.iterdir())[:5]:
                            print(f"         📄 {f.name}")
                        if len(list(subdir.iterdir())) > 5:
                            print(f"         ... and {len(list(subdir.iterdir()))-5} more files")
            
            # Check for CSV validation files
            csv_files = list(year_dir.glob("**/*classification*.csv"))
            if csv_files:
                print(f"\n   📊 Classification CSVs found:")
                for csv in csv_files:
                    print(f"      {csv.relative_to(year_dir)}")
        else:
            print(f"   ❌ Directory NOT found")

def compare_classifications():
    """Compare classification statistics between original and ML-extracted"""
    print("\n" + "="*80)
    print("🔬 COMPARING CLASSIFICATION STATISTICS")
    print("="*80)
    
    for year in YEARS:
        print(f"\n{'='*60}")
        print(f"📅 Year {year}")
        print(f"{'='*60}")
        
        # Load original validation CSV
        orig_csv = ORIGINAL_BASE / f"geospatial_project_{year}" / "outputs" / "datasets" / f"classification_validation_{year}.csv"
        
        if orig_csv.exists():
            print(f"\n📄 Original Classification ({orig_csv.name}):")
            df_orig = pd.read_csv(orig_csv)
            print(df_orig.to_string(index=False))
            
            # Calculate percentages
            total = df_orig['Count'].sum()
            print(f"\n   Total pixels: {total:,}")
            for _, row in df_orig.iterrows():
                pct = row['Count'] / total * 100
                print(f"   {row['Class']:15s}: {pct:>6.2f}%")
        else:
            print(f"   ❌ Original CSV not found: {orig_csv}")
        
        # Load ML-extracted data
        ml_csv = ML_BASE / "data" / "processed" / f"feature_table_{year}.csv"
        
        if ml_csv.exists():
            print(f"\n📄 ML-Extracted Data ({ml_csv.name}):")
            df_ml = pd.read_csv(ml_csv)
            
            class_names = {
                0: 'Snow/Ice',
                1: 'Water',
                2: 'Bare Rock',
                3: 'Sparse Veg',
                4: 'Moderate Veg',
                5: 'Dense Veg'
            }
            
            print(f"   Total pixels: {len(df_ml):,}")
            for cls in sorted(df_ml['Class'].unique()):
                count = (df_ml['Class'] == cls).sum()
                pct = count / len(df_ml) * 100
                name = class_names.get(int(cls), f'Unknown({cls})')
                
                # Get mean indices
                df_cls = df_ml[df_ml['Class'] == cls]
                ndvi = df_cls['NDVI'].mean()
                ndsi = df_cls['NDSI'].mean()
                
                print(f"   Class {int(cls)} ({name:12s}): {pct:>6.2f}% | NDVI={ndvi:+.3f} | NDSI={ndsi:+.3f}")
        else:
            print(f"   ❌ ML CSV not found")

def analyze_discrepancy():
    """Analyze why the percentages don't match"""
    print("\n" + "="*80)
    print("🔍 ANALYZING DISCREPANCY")
    print("="*80)
    
    # Load 2021 data from both sources
    orig_csv = ORIGINAL_BASE / "geospatial_project_2021" / "outputs" / "datasets" / "classification_validation_2021.csv"
    ml_csv = ML_BASE / "data" / "processed" / "feature_table_2021.csv"
    
    if orig_csv.exists() and ml_csv.exists():
        df_orig = pd.read_csv(orig_csv)
        df_ml = pd.read_csv(ml_csv)
        
        print("\n📊 Side-by-Side Comparison (2021):")
        print("-"*80)
        print(f"{'Class':<15} {'Original %':>12} {'ML-Extracted %':>15} {'Match?':>10}")
        print("-"*80)
        
        class_order = ['Snow/Ice', 'Water', 'Bare Rock', 'Sparse Veg', 'Moderate Veg', 'Dense Veg']
        class_to_id = {
            'Snow/Ice': 0,
            'Water': 1,
            'Bare Rock': 2,
            'Sparse Veg': 3,
            'Moderate Veg': 4,
            'Dense Veg': 5
        }
        
        orig_total = df_orig['Count'].sum()
        ml_total = len(df_ml)
        
        for cls_name in class_order:
            # Original
            orig_row = df_orig[df_orig['Class'] == cls_name]
            orig_pct = orig_row['Count'].values[0] / orig_total * 100 if len(orig_row) > 0 else 0
            
            # ML
            cls_id = class_to_id[cls_name]
            ml_count = (df_ml['Class'] == cls_id).sum()
            ml_pct = ml_count / ml_total * 100
            
            # Compare
            match = "✅" if abs(orig_pct - ml_pct) < 2 else "❌"
            print(f"{cls_name:<15} {orig_pct:>10.2f}% {ml_pct:>13.2f}% {match:>10}")
        
        print("-"*80)
        print(f"{'TOTAL':<15} {100:>10.2f}% {100:>13.2f}%")
        
        print("\n🚨 KEY FINDING:")
        print("   If 'Bare Rock' in original = 'Class 3 (Sparse Veg)' in ML-extracted,")
        print("   then the class IDs in your classification raster are DIFFERENT")
        print("   from what the CSV labels suggest!")

def main():
    print("🔬 VERIFICATION: Comparing Original vs ML-Extracted Classifications")
    print("="*80)
    
    check_original_structure()
    compare_classifications()
    analyze_discrepancy()
    
    print("\n" + "="*80)
    print("✅ VERIFICATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()