# Independent Validation Sampling Rules

Use these rules when building the first 500 independent reference points.

## 1) Class Targets (Initial Benchmark)

- Snow/Ice: 120
- Bare Rock: 120
- Sparse Vegetation: 120
- Moderate Vegetation: 100
- Dense Vegetation: 40 (or all available if fewer)

Total target: **500 points**

## 2) Source Requirements

- Use manual interpretation from high-resolution imagery plus Landsat composites.
- Record the imagery/source name in the `source` field.
- Prefer scenes close in date to each `year` label.

## 3) Spatial Spread Rules

- Distribute points across the full study area (not clustered in one valley/zone).
- Keep minimum spacing between points where possible (avoid dense local clusters).
- Include different elevation/aspect zones to improve representativeness.

## 4) Boundary and Ambiguity Rules

- Do not sample class-edge pixels.
- Avoid mixed pixels, shadow-edge zones, cloud haze, or terrain distortion areas.
- Keep only points with clear visual agreement between source imagery and class.

## 5) Confidence Rules

- Use only `confidence=high` for the first benchmark dataset.
- If unsure, skip the point rather than forcing a label.
- Optional: maintain a separate candidate file for medium/low confidence points.

## 6) Required Output Schema

Each reference point must include:

- `lon`
- `lat`
- `year`
- `class_id`
- `class_name`
- `confidence`
- `source`

Template file:

- `e:\zaheer-work\Hunza_GIS_Data_Extraction\training_data\reference_labels_template.csv`

## 7) QA Checklist Before Training

- Confirm class IDs match:
  - `0=Snow/Ice`, `1=Bare Rock`, `2=Sparse Vegetation`, `3=Moderate Vegetation`, `4=Dense Vegetation`
- Confirm no missing values in required columns.
- Confirm all benchmark points are `confidence=high`.
- Confirm class counts are close to the target distribution.
- Confirm points are spatially spread and not boundary-biased.
