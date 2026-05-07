# Distillation-Only Results Note

## Current Model Status

The current Random Forest outputs are **distillation results only**.

All reported scores represent **agreement with source `land_cover` labels** and are **not independent ground-truth accuracy**.

Use this wording in reports/thesis:

> "The Random Forest model achieved very high agreement with source land-cover labels in distillation mode. These metrics should be interpreted as classifier agreement, not independent real-world classification accuracy."

## Keep Current Outputs

Keep the following generated artifacts as valid distillation outputs:

- `e:\zaheer-work\Hunza_GIS_ML_Implementation\models\rf_model_distillation.pkl`
- `e:\zaheer-work\Hunza_GIS_ML_Implementation\models\classification_report_distillation.txt`
- `e:\zaheer-work\Hunza_GIS_ML_Implementation\figures\cv_scores_stratified.png`
- `e:\zaheer-work\Hunza_GIS_ML_Implementation\figures\cv_scores_spatial.png`
- `e:\zaheer-work\Hunza_GIS_ML_Implementation\figures\confusion_matrix_val_distillation.png`
- `e:\zaheer-work\Hunza_GIS_ML_Implementation\figures\confusion_matrix_test_distillation.png`
- `e:\zaheer-work\Hunza_GIS_ML_Implementation\figures\feature_importance_distillation.png`

These files should not be deleted; they are useful for documenting the distillation track.
