---
name: research-plotter
description: generate publication-quality research figures from uploaded files or a project directory. use when chatgpt needs to create paper figures from csv, excel, shap artifacts, prediction results, or existing plotting scripts; detect required input types for a requested figure family; auto-match compatible files in project/data, project/results, project/shap, or project/scripts; run or adapt plotting scripts; export png and svg; and draft bilingual caption and methods/figure notes in chinese and english.
---

# Research Plotter

## Overview

Create publication-quality figures for research projects from uploaded tabular data, model-result files, SHAP artifacts, or existing plotting scripts. Always produce figure files in both PNG and SVG when generation succeeds, then draft a bilingual caption and a bilingual methods or figure note.

## Workflow decision tree

1. Determine the request type.
   - **Raw data figures**: distributions, correlation matrices, feature-target relationships, categorical effects, outlier figures.
   - **Model evaluation figures**: CV stability, bootstrap CI, Q-Q plots, residual diagnostics, prediction interval figures.
   - **Pipeline or flowchart figures**: model workflow diagrams or method diagrams.
   - **SHAP figures**: dependence plots, beeswarm summary plots, composite SHAP layouts.
   - **Existing-script execution**: the user uploaded a plotting script and expects it to run with minimal edits.

2. Determine the input mode.
   - **Uploaded files only**: use files in the conversation.
   - **Project directory**: prefer the standardized layout `project/data`, `project/results`, `project/shap`, `project/scripts`, `project/figures`.
   - **Hybrid**: combine uploaded files with files already present in the project directory.

3. Before plotting, inspect inputs.
   - Run `scripts/detect_inputs.py` against the working directory or the user-provided project root.
   - Read `references/input-requirements.md` to map figure families to required input types.
   - Tell the user which input types were found, which are missing, and which figure families are currently feasible.
   - Do not require exact filenames. Match by file role, suffix, and column/content patterns.

4. Choose execution path.
   - **User uploaded a working plotting script**: run it first if dependencies and paths can be resolved safely.
   - **User uploaded data but no script**: generate or adapt a plotting script based on the requested figure family.
   - **User asked for a figure family only**: select the closest template and explain the inferred inputs used.

5. Generate outputs.
   - Save figures under `project/figures` when using the project layout; otherwise save next to the working data under a `figures/` folder.
   - Export **both** `.png` and `.svg`.
   - Return a concise build summary with output paths.
   - Draft bilingual caption and bilingual methods/figure note using the templates below.

## Input inspection rules

- Prefer flexible matching over rigid filename matching.
- Infer file roles from any of:
  - file extension: `.csv`, `.xlsx`, `.xls`, `.npy`, `.json`, `.py`
  - folder placement: `data/`, `results/`, `shap/`, `scripts/`
  - schema hints: columns such as `Actual`, `Pred_*`, `Q5_*`, `Q95_*`, `R2`, `RMSE`, `Standardized_Residual`
  - SHAP artifact signatures such as `shap_values.npy`, feature-name json, or explanation matrices.
- If multiple candidate files match the same role, prefer the file whose schema best matches the requested figure family and explain the choice.
- If inputs are incomplete, explicitly list missing **input types**, not just missing filenames.

## Figure-family guidance

### Raw data figures

Use when the user uploads a dataset and asks for descriptive or exploratory paper figures.

Default outputs may include:
- numerical-variable distributions
- Pearson and Spearman correlation matrices
- key feature versus target relationships
- categorical-variable effect plots
- outlier diagnostics

See `references/input-requirements.md` for expected input types.

### Model evaluation figures

Use when the user uploads prediction outputs, CV scores, bootstrap metrics, residual diagnostics, or interval results.

Default outputs may include:
- repeated CV stability figures
- bootstrap confidence interval figures
- Q-Q plot and normality diagnostics
- prediction interval coverage figures

### Pipeline or flowchart figures

Use when the user asks for a workflow diagram or uploads a model-pipeline script. Prefer a clean scientific style with grouped stages such as data, preprocessing, model/tuning, and outputs.

### SHAP figures

Use when the user provides SHAP values plus feature metadata and the corresponding explanation matrix.

Default outputs may include:
- top-k dependence plots
- paginated dependence plots for all features
- beeswarm summary plot
- advanced composite layout combining importance bars, beeswarm, and contribution inset

## Existing-script execution workflow

1. Inspect the uploaded script for hard-coded paths, required packages, and expected input files.
2. Patch hard-coded paths to the current working directory or the standardized project layout.
3. Run the script.
4. If execution fails, fix path assumptions first, then obvious environment or schema issues.
5. Preserve the script's plotting intent and style. For the bundled high-fidelity templates, match the reference scripts as closely as possible in palette, font family, line weights, layout, naming, and output folder structure unless the user explicitly asks for changes.
6. Report any assumptions or unresolved gaps plainly.


## Repository and environment notes

- Use relative paths throughout bundled scripts and examples.
- Prefer parameterized inputs such as `--project-root` over hard-coded file locations.
- Auto-detect compatible files from folder role, extension, and schema hints.
- Refer to `README.md` for dependency installation and GitHub-ready repository usage.

## Script execution rules

- Run `scripts/detect_inputs.py` first when the available inputs are uncertain.
- Use the most specific bundled plotting script that matches the user request.
- Prefer the standardized project layout rooted at `project/`.
- Unless the user explicitly wants a custom location, save outputs to `project/figures/`.
- When a bundled script succeeds, report the exact output files and then draft bilingual caption plus bilingual methods or figure note.
- When a bundled script fails because the input schema is incomplete, tell the user which **input types** are missing and which script would become available once they are provided.

### Direct entrypoints

```bash
python scripts/detect_inputs.py project
python scripts/raw_data_figures.py --project-root project
python scripts/model_evaluation_figures.py --project-root project
python scripts/pipeline_flowchart.py --project-root project
python scripts/shap_dependence.py --project-root project
python scripts/shap_advanced_summary.py --project-root project
```

Read `references/script-entrypoints.md` when you need the full CLI options and the matching input-type rules for each entrypoint.

## Output templates

### Build summary

Use this structure after generation:

```markdown
## Figure build summary
- Figure family: [raw data / model evaluation / flowchart / shap / existing-script execution]
- Inputs detected: [types and selected files]
- Missing input types: [none or list]
- Outputs: [png path], [svg path]
- Notes: [important assumptions, patches, or limitations]
```

### Caption template

Provide both languages.

```markdown
## Caption
**中文：** 图X. [一句话说明图展示了什么、对象是什么、关键视觉编码是什么。]

**English:** Figure X. [One-sentence caption stating what the figure shows, what data or model it covers, and the main visual encoding.]
```

### Methods / figure note template

Provide both languages.

```markdown
## Methods / Figure note
**中文：** 基于[数据/模型结果/SHAP结果]生成该图。图中使用了[关键统计量、分箱、相关系数、置信区间、标准化残差、颜色映射等]，输出为 PNG 与 SVG 以满足论文排版与矢量编辑需求。

**English:** This figure was generated from [data/model outputs/SHAP artifacts]. The visualization uses [key statistics, binning, correlation measures, confidence intervals, standardized residuals, color mapping, etc.], and was exported to PNG and SVG for manuscript layout and vector editing.
```

## Resources

- `references/input-requirements.md`: figure-family input matrix and detection hints.
- `references/script-adaptation-notes.md`: adaptation notes derived from representative plotting scripts.
- `references/script-entrypoints.md`: direct execution guide for the bundled plotting scripts.
- `scripts/detect_inputs.py`: scan a project directory and summarize detected input roles.
- `scripts/raw_data_figures.py`: build high-fidelity raw-data figures matching the reference pastel white-background style.
- `scripts/model_evaluation_figures.py`: build high-fidelity SCI-style model evaluation figures matching the reference layout.
- `scripts/pipeline_flowchart.py`: build a high-fidelity pipeline flowchart matching the reference XGB-Optuna style.
- `scripts/shap_dependence.py`: build high-fidelity top-k and paginated SHAP dependence figures matching the reference layout.
- `scripts/shap_advanced_summary.py`: build a high-fidelity composite SHAP summary figure matching the reference layout.
