# Research Plotter

A GitHub-ready research plotting skill and script bundle for publication-quality figures.

## What this project does

This project provides a ChatGPT Skill plus executable Python scripts for generating research figures from:

- tabular raw data (`.csv`, `.xlsx`, `.xls`)
- model evaluation results (prediction outputs, CV scores, bootstrap metrics, residual diagnostics, interval results)
- SHAP artifacts (`.npy`, feature metadata JSON, explanation matrices)
- existing plotting scripts that need light path adaptation

Outputs are written as both **PNG** and **SVG**. The Skill is designed to also help draft bilingual caption and methods / figure-note text in **Chinese and English**.

## Design principles

This version standardizes the repository around:

- **relative paths** instead of machine-specific absolute paths
- **parameterized input** via `--project-root` and optional file arguments
- **automatic directory detection** for common project layouts
- **GitHub-friendly documentation** with dependency and usage instructions

## Recommended project layout

```text
project/
  data/
  results/
  shap/
  scripts/
  figures/
```

The scripts prefer this layout, but exact filenames do **not** need to match. Detection is based on file type, folder role, and schema hints.

## Main entrypoints

All bundled scripts support relative-path usage.

```bash
python scripts/detect_inputs.py project
python scripts/raw_data_figures.py --project-root project
python scripts/model_evaluation_figures.py --project-root project
python scripts/pipeline_flowchart.py --project-root project
python scripts/shap_dependence.py --project-root project
python scripts/shap_advanced_summary.py --project-root project
```

## What each script expects

### 1. Raw data figures
Input types:
- one tabular dataset with multiple numeric columns
- optional categorical columns
- optional target column (auto-detected where possible)

Command:

```bash
python scripts/raw_data_figures.py --project-root project
```

### 2. Model evaluation figures
Input types:
- prediction-results table with columns like `Actual`, `Pred_*`
- CV table with `Model`, `R2`
- bootstrap table with `Model`, `R2`, `RMSE`
- residual-diagnostics table with `Standardized_Residual`
- optional prediction-interval table with `Q5_*`, `Q95_*`, `In_PI_*`

Command:

```bash
python scripts/model_evaluation_figures.py --project-root project
```

### 3. Pipeline flowchart
Input types:
- optional JSON spec describing stages and labels
- or use the default scientific flowchart template

Command:

```bash
python scripts/pipeline_flowchart.py --project-root project
python scripts/pipeline_flowchart.py --project-root project --spec-json path/to/spec.json
```

### 4. SHAP dependence figures
Input types:
- SHAP values array (`.npy`)
- feature-name metadata JSON
- explanation matrix (`.csv`)

Command:

```bash
python scripts/shap_dependence.py --project-root project
```

### 5. SHAP advanced summary figure
Input types:
- SHAP values array (`.npy`)
- feature-name metadata JSON
- explanation matrix (`.csv`)

Command:

```bash
python scripts/shap_advanced_summary.py --project-root project
```

## Automatic detection behavior

`detect_inputs.py` scans a root directory recursively and reports:

- detected file roles
- compatible figure families
- whether the recommended project layout exists

Example:

```bash
python scripts/detect_inputs.py project
```

## Environment and dependencies

Recommended environment:

- Python **3.10+**
- OS: Windows, macOS, or Linux
- A local font fallback may be used if `Times New Roman` is unavailable

Install dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `shap`
- `openpyxl`

## Notes on style fidelity

The bundled plotting scripts are intentionally written to preserve the reference style as closely as possible:

- serif / Times-style typography
- pastel raw-data palette
- SCI-style model-evaluation layouts
- high-fidelity SHAP figure structure
- publication-friendly file naming and folder output

## GitHub publishing notes

Before pushing publicly:

1. keep only relative paths
2. avoid committing private data or secrets
3. avoid committing generated `project/figures/*` unless they are intentional examples
4. keep fonts external; do not bundle font files

## Suggested repository structure

```text
research-plotter/
  README.md
  LICENSE
  .gitignore
  requirements.txt
  skill/
    SKILL.md
    agents/
    scripts/
    references/
    assets/
  examples/
    project/
      data/
      results/
      shap/
      scripts/
      figures/
```

## License

Choose the license you want before public release. MIT is a common default for code projects.
