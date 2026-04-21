# Script adaptation notes

These notes are derived from representative plotting scripts for raw-data figures, model-evaluation figures, flowcharts, and SHAP visualizations.

## Common adaptation issues

### 1. Hard-coded local paths
Many research scripts are project-specific and point to absolute Windows paths. Replace them with:
- the uploaded file paths in the current working directory, or
- the standardized project layout:
  - `project/data`
  - `project/results`
  - `project/shap`
  - `project/scripts`
  - `project/figures`

### 2. Rigid filename expectations
Do not require exact filenames. Instead, detect by:
- extension
- folder placement
- schema/column patterns
- SHAP artifact signatures

### 3. Multi-file figure families
Model-evaluation and SHAP figures often require several coordinated inputs. Before running, verify the full set of required input types is available.

## Representative figure families from user-provided examples

### Raw data visualization
Representative capabilities include:
- numerical distributions with KDE and histograms
- Pearson and Spearman correlation matrices
- feature-target relationship plots
- categorical effects
- outlier analysis

### Model-result visualization
Representative capabilities include:
- repeated K-fold stability figures
- bootstrap CI distributions
- Q-Q plots with normality checks
- prediction interval visualization

### Flowchart generation
Representative capabilities include:
- pipeline stage boxes
- directional arrows
- grouped legend for data, preprocessing, model, output

### SHAP dependence and advanced figures
Representative capabilities include:
- top-k dependence grids
- paginated dependence plots
- ranked importance bars
- beeswarm plots
- composite layouts with inset contribution view

## Output requirement
For successful runs, always export PNG and SVG and then draft bilingual caption plus bilingual methods or figure note.
