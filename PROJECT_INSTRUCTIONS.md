# Project Instructions - Hunza GIS ML Implementation

## Audience
- Primary: Self

## Project Overview
This folder contains the machine learning pipeline and supporting assets for land cover classification in the Hunza Valley (2020-2023) using Landsat 8 imagery and derived spectral indices. The workflow covers preprocessing, feature extraction, classification, validation, and reporting.

## Directory Map
- environment.yml: Conda environment definition for the ML pipeline.
- COMPLETE_DATA_PROCESSING_REPORT.md: End-to-end processing report and results summary.
- ml-pipeline.md: Detailed ML pipeline documentation.
- complete-update.md: Consolidated technical report and updates.
- scripts/: Python utilities for feature generation, diagnostics, and validation.
- figures/: Figures used in reports and analysis.
- *.png: Key diagnostic plots and verification images.

## Data Sources (High-Level)
- Satellite imagery: Landsat 8 OLI, summer scenes (2020-2023).
- Spectral bands: Blue, Green, Red, NIR, SWIR1, SWIR2.
- Indices: NDVI, NDSI, NDWI.

## Methods Summary
- Preprocessing: DN to surface reflectance conversion, clipping, and NoData handling.
- Feature extraction: 6 spectral bands + 3 indices (9 features).
- Classification: K-Means for initial clustering and Random Forest for supervised ML.
- Validation: Accuracy checks and class labeling verification.
- Temporal analysis: Multi-year comparisons (2020-2023).

## Setup
1. Install Conda (Miniconda or Anaconda).
2. Create the environment:

```bash
conda env create -f environment.yml
```

3. Activate the environment:

```bash
conda activate hunza-ml
```

## How to Run (Scripts)
Scripts are in scripts/ and are intended to be run from the Hunza_GIS_ML_Implementation folder.

- create_feature_tables.py: Build feature tables from processed rasters.
- diagnose_clusters.py: Analyze clustering outputs and class separability.
- fix_classification.py: Apply corrections to class labels or masks.
- verify_classification.py: Generate validation metrics and checks.
- resample_bands.py: Resample band rasters if needed.
- extract_all_years.sh: Batch extraction pipeline (bash). Use in a Unix-like shell.

Check each script header or docstring for required arguments and expected inputs.

## Outputs
- figures/: Charts, plots, and diagnostics used in reports.
- Report outputs are described in:
  - COMPLETE_DATA_PROCESSING_REPORT.md
  - ml-pipeline.md
  - complete-update.md

## Notes
- For data extraction and year-specific preprocessing, see the Hunza_GIS_Data_Extraction folder in the workspace.
- If citations or bibliography appear inconsistent in your thesis, validate references in your Overleaf project and ensure the BibTeX pipeline is configured correctly.
