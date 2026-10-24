# Input requirements by figure family

This reference explains which **input types** are needed for each figure family. Exact filenames are not required.

## 1. Raw data figures

### Typical requests
- "Generate paper figures from this dataset"
- "Draw distribution, correlation, and feature-target plots"
- "Create descriptive figures for my raw dataset"

### Required input types
- A tabular dataset in CSV or Excel format
- At least one numeric target column if feature-target plots are requested
- Categorical columns only if categorical-effect plots are requested

### Strong signals for compatibility
- Spreadsheet or CSV with multiple numeric columns
- A likely target column such as bond strength, strength, response, target, y, label

### Common outputs
- distributions
- Pearson and Spearman correlation matrices
- feature versus target plots
- categorical-effect boxplots
- outlier analysis

## 2. Model evaluation figures

### Typical requests
- "Draw model performance figures"
- "Make train-vs-test parity plots for my models"
- "Generate publication figures from model results"

### Input types

#### Per-model JointGrid + all-models panel
Required input types:
- test prediction table (always needed)
- train prediction table (needed for Train points and Train R²; the script degrades to test-only when absent)

Compatibility signals:
- columns like `Actual`, `Pred_<model>` (e.g. `Pred_XGB-Optuna`, `Pred_Boost-Avg`)
- filenames like `enhanced_predictions_train.csv`, `enhanced_predictions_test.csv`
- one `Pred_*` column per model; every such column produces one JointGrid and one panel in the combined figure

Optional overrides:
- `--target-name`, `--x-label`, `--y-label` for axis and info-box text

## 3. Pipeline or flowchart figures

### Typical requests
- "Draw a model pipeline flowchart"
- "Make an SCI-style workflow diagram"

### Required input types
One of:
- an existing Python plotting script for the flowchart
- a textual pipeline description from the user
- a method section or structured list of steps

### Strong signals for compatibility
- script includes boxes/arrows or stage labels
- user mentions train/test split, feature engineering, tuning, ensemble, outputs

## 4. SHAP figures

The **advanced composite figure** is the default SHAP analysis output; dependence plots are supplementary.

### Typical requests
- "Generate SHAP analysis figures"
- "Create a combined SHAP figure"
- "Generate dependence plots" (supplementary)

### Required input types
Core set:
- SHAP values array
- feature-name mapping or ordered feature list
- explanation matrix for the corresponding rows and columns (`X_explain_subset.csv` preferred, `X_explain.csv` fallback; must contain every feature column and have exactly as many rows as the SHAP array)

### Compatibility signals
- `.npy` file containing SHAP values
- `.json` file describing features
- CSV with feature columns aligned to SHAP columns

### Subfigure requirements

#### Advanced composite SHAP figure (default)
- SHAP values array
- explanation matrix
- feature names
- enough features to rank and visualize importance distribution

#### Dependence plots (supplementary)
- SHAP values array
- feature list
- explanation matrix

## 5. Existing-script execution

### Typical requests
- "Run this plotting script"
- "Adapt my script to the current data"
- "Use my script but fix the paths"

### Required input types
- a Python plotting script
- every data artifact referenced by that script, or a project directory from which equivalent artifacts can be inferred

### First checks
- hard-coded local paths
- required packages
- expected directory layout
- expected file schemas

## Detection policy

When inputs are missing, report them like this:
- present input types
- missing input types
- feasible figure families now
- additional input types needed for unavailable figure families
