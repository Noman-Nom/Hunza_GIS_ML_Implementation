#!/bin/bash
# Batch extraction script for 2021-2023

BASE_DIR="/home/muhammad-noman/zaheerproject02-hunza-dataextraction"
HUNZA_DIR="$BASE_DIR/Hunza_Data_Analysis"
ML_DIR="$BASE_DIR/hunza-ml-analysis/data/raw"

echo "🚀 Starting batch extraction for years 2021-2023..."

# Years and their source directories
declare -A YEAR_DIRS=(
    [2021]="geospatial_project_2021"
    [2022]="geospatial_project_2022"
    [2023]="geospatial_project_2023"
)

for year in 2021 2022 2023; do
    echo ""
    echo "📅 Processing year $year..."
    
    # Create directories
    mkdir -p "$ML_DIR/$year/bands"
    mkdir -p "$ML_DIR/$year/indices"
    
    # Source paths
    RAW_FILE="$HUNZA_DIR/${YEAR_DIRS[$year]}/data/raw/Hunza_${year}-0000000000-0000000000.tif"
    INDICES_DIR="$HUNZA_DIR/${YEAR_DIRS[$year]}/outputs/qgis"
    
    # Extract bands 2-7
    echo "  📊 Extracting bands..."
    gdal_translate -q -b 2 "$RAW_FILE" "$ML_DIR/$year/bands/Band_2_Blue_$year.tif"
    gdal_translate -q -b 3 "$RAW_FILE" "$ML_DIR/$year/bands/Band_3_Green_$year.tif"
    gdal_translate -q -b 4 "$RAW_FILE" "$ML_DIR/$year/bands/Band_4_Red_$year.tif"
    gdal_translate -q -b 5 "$RAW_FILE" "$ML_DIR/$year/bands/Band_5_NIR_$year.tif"
    gdal_translate -q -b 6 "$RAW_FILE" "$ML_DIR/$year/bands/Band_6_SWIR1_$year.tif"
    gdal_translate -q -b 7 "$RAW_FILE" "$ML_DIR/$year/bands/Band_7_SWIR2_$year.tif"
    echo "  ✅ Bands extracted"
    
    # Copy indices
    echo "  📈 Copying indices..."
    cp "$INDICES_DIR/ndvi_$year.tif" "$ML_DIR/$year/indices/NDVI_$year.tif"
    cp "$INDICES_DIR/ndsi_$year.tif" "$ML_DIR/$year/indices/NDSI_$year.tif"
    cp "$INDICES_DIR/ndwi_$year.tif" "$ML_DIR/$year/indices/NDWI_$year.tif"
    echo "  ✅ Indices copied"
    
    # Copy classification
    echo "  🗺️  Copying classification..."
    cp "$INDICES_DIR/land_cover_$year.tif" "$ML_DIR/$year/classification_$year.tif"
    echo "  ✅ Classification copied"
    
    echo "✅ Year $year complete!"
done

echo ""
echo "🎉 All years processed successfully!"
echo ""
echo "📁 Data structure:"
tree -L 3 "$ML_DIR"