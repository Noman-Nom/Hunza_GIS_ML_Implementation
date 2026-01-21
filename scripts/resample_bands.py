#!/usr/bin/env python3
"""
Resample bands to match indices/classification dimensions
Fixes the 5376×2978 → 5000×2978 mismatch
"""

import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as ResamplingEnum
from pathlib import Path
import numpy as np
import shutil

BASE_DIR = Path("/home/muhammad-noman/zaheerproject02-hunza-dataextraction/hunza-ml-analysis/data/raw")
YEARS = [2020, 2021, 2022, 2023]
BAND_FILES = ['Band_2_Blue', 'Band_3_Green', 'Band_4_Red', 
              'Band_5_NIR', 'Band_6_SWIR1', 'Band_7_SWIR2']

def get_reference_profile(year):
    """Get reference dimensions from classification"""
    ref_path = BASE_DIR / str(year) / f"classification_{year}.tif"
    with rasterio.open(ref_path) as src:
        return src.profile.copy(), src.bounds, src.transform

def resample_band(year, band_file):
    """Resample a band to match classification dimensions"""
    
    # Paths
    band_path = BASE_DIR / str(year) / "bands" / f"{band_file}_{year}.tif"
    backup_path = BASE_DIR / str(year) / "bands" / f"{band_file}_{year}_ORIGINAL.tif"
    temp_path = BASE_DIR / str(year) / "bands" / f"{band_file}_{year}_TEMP.tif"
    
    # Check if already resampled (backup exists)
    if backup_path.exists():
        print(f"⚠️  Already resampled (backup exists), skipping...")
        return True
    
    # Get reference profile
    ref_profile, ref_bounds, ref_transform = get_reference_profile(year)
    
    # Read source band
    with rasterio.open(band_path) as src:
        src_data = src.read(1)
        src_profile = src.profile.copy()
        src_dtype = src.dtypes[0]  # FIX: Use dtypes[0] instead of dtype
        
        # Check if already correct size
        if src.width == ref_profile['width'] and src.height == ref_profile['height']:
            print(f"✅ Already correct size ({src.width}×{src.height}), skipping...")
            return True
        
        print(f"📏 Resampling {src.width}×{src.height} → {ref_profile['width']}×{ref_profile['height']}...", end=' ')
        
        # Prepare output array
        resampled = np.empty((ref_profile['height'], ref_profile['width']), dtype=src_dtype)
        
        # Reproject/resample
        reproject(
            source=src_data,
            destination=resampled,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_profile['crs'],
            resampling=ResamplingEnum.bilinear
        )
        
        # Update profile
        out_profile = src_profile.copy()
        out_profile.update({
            'width': ref_profile['width'],
            'height': ref_profile['height'],
            'transform': ref_transform,
            'dtype': src_dtype
        })
        
        # Write to temporary file
        with rasterio.open(temp_path, 'w', **out_profile) as dst:
            dst.write(resampled, 1)
    
    # Backup original
    shutil.move(band_path, backup_path)
    
    # Replace with resampled
    shutil.move(temp_path, band_path)
    
    print("✅")
    return True

def verify_dimensions(year):
    """Verify all files have matching dimensions"""
    
    # Get classification dimensions
    with rasterio.open(BASE_DIR / str(year) / f"classification_{year}.tif") as src:
        target_width = src.width
        target_height = src.height
    
    print(f"\n   🔍 Verification for {year}:")
    print(f"      Target dimensions: {target_width} × {target_height}")
    
    all_match = True
    
    # Check bands
    for band_file in BAND_FILES:
        band_path = BASE_DIR / str(year) / "bands" / f"{band_file}_{year}.tif"
        with rasterio.open(band_path) as src:
            if src.width == target_width and src.height == target_height:
                print(f"      ✅ {band_file}: {src.width} × {src.height}")
            else:
                print(f"      ❌ {band_file}: {src.width} × {src.height} (MISMATCH!)")
                all_match = False
    
    # Check indices
    for index_name in ['NDVI', 'NDSI', 'NDWI']:
        index_path = BASE_DIR / str(year) / "indices" / f"{index_name}_{year}.tif"
        with rasterio.open(index_path) as src:
            if src.width == target_width and src.height == target_height:
                print(f"      ✅ {index_name}: {src.width} × {src.height}")
            else:
                print(f"      ❌ {index_name}: {src.width} × {src.height} (MISMATCH!)")
                all_match = False
    
    return all_match

def main():
    print("="*80)
    print("🔧 RESAMPLING BANDS TO MATCH INDICES/CLASSIFICATION")
    print("="*80)
    print("\nThis will resample all bands from 5376×2978 to 5000×2978")
    print("Original bands will be backed up with '_ORIGINAL' suffix")
    print()
    
    for year in YEARS:
        print(f"{'='*80}")
        print(f"📅 Year {year}")
        print(f"{'='*80}")
        
        try:
            # Check dimensions before
            print(f"\n   📊 Before resampling:")
            band_path = BASE_DIR / str(year) / "bands" / f"Band_2_Blue_{year}.tif"
            with rasterio.open(band_path) as src:
                print(f"      Bands: {src.width} × {src.height}")
            
            class_path = BASE_DIR / str(year) / f"classification_{year}.tif"
            with rasterio.open(class_path) as src:
                print(f"      Classification: {src.width} × {src.height}")
            
            # Resample each band
            print(f"\n   🔄 Resampling bands:")
            for band_file in BAND_FILES:
                band_name = band_file.split('_')[2]  # Extract Blue, Green, etc.
                print(f"      {band_name}... ", end='')
                try:
                    resample_band(year, band_file)
                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue
            
            # Verify
            if verify_dimensions(year):
                print(f"\n   ✅ All files verified - dimensions match!")
            else:
                print(f"\n   ⚠️  Some files still have mismatched dimensions")
            
            print()
            
        except Exception as e:
            print(f"   ❌ ERROR processing year {year}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("="*80)
    print("✅ RESAMPLING COMPLETE!")
    print("="*80)
    print("\n📁 Original bands backed up with '_ORIGINAL' suffix")
    print("🎯 Next: Run feature extraction")
    print("   Command: python scripts/create_feature_tables.py")
    print("="*80)

if __name__ == "__main__":
    main()