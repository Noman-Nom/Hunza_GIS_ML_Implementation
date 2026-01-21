#!/usr/bin/env python3
"""
Fix Snow/Water Classification by Swapping Classes 0 and 1
"""

import rasterio
import numpy as np
from pathlib import Path

BASE_DIR = Path("/home/muhammad-noman/zaheerproject02-hunza-dataextraction/hunza-ml-analysis/data/raw")
YEARS = [2020, 2021, 2022, 2023]

def remap_classification(year):
    """
    Remap classes:
    - Old Class 0 (wrong snow) → Class 1 (water)
    - Old Class 1 (wrong water) → Class 0 (snow)
    - Other classes stay the same
    """
    # Paths
    input_path = BASE_DIR / str(year) / f"classification_{year}.tif"
    backup_path = BASE_DIR / str(year) / f"classification_{year}_BACKUP.tif"
    
    # Backup original
    print(f"   Creating backup: {backup_path.name}")
    import shutil
    shutil.copy(input_path, backup_path)
    
    # Read
    with rasterio.open(input_path) as src:
        data = src.read(1)
        meta = src.meta.copy()
    
    # Count before
    old_counts = {
        0: np.sum(data == 0),
        1: np.sum(data == 1)
    }
    
    # Remap: swap 0 and 1
    remapped = data.copy()
    remapped[data == 0] = 99  # Temporary value
    remapped[data == 1] = 0   # Old water (1) → new snow (0)
    remapped[remapped == 99] = 1  # Old snow (0) → new water (1)
    
    # Count after
    new_counts = {
        0: np.sum(remapped == 0),
        1: np.sum(remapped == 1)
    }
    
    # Write
    with rasterio.open(input_path, 'w', **meta) as dst:
        dst.write(remapped, 1)
    
    return old_counts, new_counts

def main():
    print("="*70)
    print("🔧 FIXING SNOW/WATER CLASSIFICATION")
    print("="*70)
    print("\nRemapping logic:")
    print("   Old Class 0 (17 pixels, wrong)        → New Class 1 (Water)")
    print("   Old Class 1 (4.3M pixels, wrong)      → New Class 0 (Snow/Ice)")
    print("   Classes 2-6 unchanged")
    print()
    
    for year in YEARS:
        print(f"{'='*70}")
        print(f"📅 Processing Year {year}")
        print(f"{'='*70}")
        
        try:
            old_counts, new_counts = remap_classification(year)
            
            print(f"\n   ✅ Remapping complete!")
            print(f"\n   📊 Before:")
            print(f"      Class 0 (Snow): {old_counts[0]:>10,} pixels")
            print(f"      Class 1 (Water): {old_counts[1]:>10,} pixels")
            print(f"\n   📊 After:")
            print(f"      Class 0 (Snow): {new_counts[0]:>10,} pixels  ✅ FIXED!")
            print(f"      Class 1 (Water): {new_counts[1]:>10,} pixels")
            print()
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            continue
    
    print("="*70)
    print("✅ ALL YEARS FIXED!")
    print("="*70)
    print("\n📁 Backups saved as: classification_YEAR_BACKUP.tif")
    print("   (In case you need to revert)")
    print("\n🎯 Next: Run feature extraction again!")
    print("="*70)

if __name__ == "__main__":
    main()