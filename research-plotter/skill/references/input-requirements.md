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
- "Make CV, bootstrap, Q-Q, and prediction-interval plots"
- "Generate publication figures from model results"

### Input types by subfigure

#### CV stability
Required input type:
- repeated CV scores or fold-wise scores with model identifier and metric values

Compatibility signals:
- columns like `Model`, `R2`, `RMSE`, `MAE`, `Fold`, `Repeat`

#### Bootstrap confidence intervals
Required input type:
- bootstrap resampling results for a target metric

Compatibility signals:
- columns like `Model`, `R2`, `RMSE`, `MAE`, `Bootstrap`, `Iteration`

#### Q-Q plot / residual diagnostics
Required input type:
- residual file with standardized or raw residuals

Compatibility signals:
- columns like `Residual`, `Standardized_Residual`, `Error`

#### Prediction interval figure
Required input types:
- actual values
- predicted values
- lower interval bound
- upper interval bound

Compatibility signals:
- columns like `Actual`, `Pred_*`, `Q5_*`, `Q95_*`, `Lower`, `Upper`, `In_PI_*`

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

### Typical requests
- "Generate dependence plots"
- "Make SHAP summary figures"
- "Create a combined SHAP figure"

### Required input types
Core set:
- SHAP values array
- feature-name mapping or ordered feature list
- explanation matrix for the corresponding rows and columns

### Compatibility signals
- `.npy` file containing SHAP values
- `.json` file describing features
- CSV with feature columns aligned to SHAP columns

### Subfigure requirements

#### Dependence plots
- SHAP values array
- feature list
- explanation matrix

#### Beeswarm summary plot
- SHAP values array
- explanation matrix
- feature names

#### Advanced composite SHAP figure
- same as beeswarm summary plot
- enough features to rank and visualize importance distribution

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
