# Script entrypoints

This skill bundles direct plotting scripts that can be executed with the project layout rooted at `project/`.

## 1. Input scanner

```bash
python scripts/detect_inputs.py project
```

Use this first when the available files are unclear.

## 2. Raw-data figures

```bash
python scripts/raw_data_figures.py --project-root project
python scripts/raw_data_figures.py --project-root project --dataset project/data/my_data.xlsx --target-col Bond_Strength_MPa
```

### Required input types
- one compatible CSV or Excel dataset
- a numeric target column for feature-target figures

### Outputs
- `Figure_1_Distributions`
- `Figure_2_Correlations`
- `Figure_3_KeyRelationships`
- `Figure_4_CategoricalEffects` when compatible categorical variables exist
- `Figure_S1_Outliers`

## 3. Model-evaluation figures

```bash
python scripts/model_evaluation_figures.py --project-root project
```

Optional explicit inputs:

```bash
python scripts/model_evaluation_figures.py \
  --project-root project \
  --cv-file project/results/cv_scores.csv \
  --bootstrap-file project/results/bootstrap_results.csv \
  --residual-file project/results/residual_diagnostics.csv \
  --prediction-file project/results/enhanced_predictions_test.csv
```

### Required input types by subfigure
- CV stability: score table with `Model` and `R2`
- bootstrap: table with `Model` and at least one metric such as `R2` or `RMSE`
- Q-Q: residual table with `Standardized_Residual` or `Residual`
- prediction interval: table with `Actual`, `Pred_*`, `Q5_*`, `Q95_*`

## 4. Pipeline flowchart

```bash
python scripts/pipeline_flowchart.py --project-root project
```

To override the default structure, provide a JSON spec:

```bash
python scripts/pipeline_flowchart.py --project-root project --spec-json project/scripts/flowchart_spec.json
```

### JSON spec structure
- `title`
- `subtitle`
- `boxes`: list of box objects with `x`, `y`, `w`, `h`, `label`, optional `fc`
- `arrows`: list of `[x1, y1, x2, y2]`

## 5. SHAP dependence figures

```bash
python scripts/shap_dependence.py --project-root project
python scripts/shap_dependence.py --project-root project --top-k 20 --page-size 20
```

### Required input types
- SHAP values `.npy`
- feature metadata `.json`
- aligned explanation matrix `.csv`

## 6. SHAP advanced summary

```bash
python scripts/shap_advanced_summary.py --project-root project
python scripts/shap_advanced_summary.py --project-root project --scheme viridis --max-display 25
```

### Required input types
- SHAP values `.npy`
- feature metadata `.json`
- aligned explanation matrix `.csv`

## Output policy

All direct plotting scripts save to `project/figures/` unless the user explicitly requests a different location.
All plots should be treated as manuscript-ready defaults, but style or labeling can still be adjusted if the user asks.
