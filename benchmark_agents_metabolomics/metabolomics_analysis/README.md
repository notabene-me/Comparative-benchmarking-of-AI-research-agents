# mbx — Untargeted Metabolomics Analysis Pipeline

A self-contained, reproducible Python pipeline for analyzing untargeted
metabolomics intensity tables end-to-end:

**preprocessing → differential analysis → multivariate analysis → pathway &
gene analysis → automated biological interpretation.**

It was developed on a HILIC untargeted-metabolomics study comparing **fibrotic
vs. control atrial tissue** (a paired, 17-patient design with pooled QC samples),
but it is dataset-agnostic and driven entirely by `mbx/config.py`.

## Quick start

```bash
pip install -r requirements.txt

python run_analysis.py \
  --input "Copy of AF-1 Results Table NB_full list.xlsx" \
  --output results
```

For an independent-groups (non-paired) design, or to change the group prefixes:

```bash
python run_analysis.py --input data.csv --output results \
  --control-prefix Healthy --case-prefix Disease \
  --control-label Healthy --case-label Disease --unpaired
```

## Input format

A feature × sample table (Excel or CSV) where:

- the first/feature column (default `Molecule`) holds metabolite names;
- QC columns contain the QC prefix (default `QC`);
- the two experimental groups are detected by column-name prefix
  (defaults `Contr`/`Fibr`);
- in a paired design, matched samples share a trailing index
  (`Contr-1 ↔ Fibr-1`).

## What the pipeline does

| Stage | Module | Key methods |
|-------|--------|-------------|
| 1. Preprocessing | `mbx/preprocessing.py` | drop all-missing & high-missingness features; **QC-CV filter** (analytical reproducibility); half-minimum imputation; **PQN** sample normalization; log2 transform; Pareto scaling (multivariate only) |
| 2. Differential analysis | `mbx/statistics.py` | paired t-test + Wilcoxon (or Welch + Mann-Whitney); **Benjamini-Hochberg FDR**; log2 fold change; Cohen's d; univariate **ROC AUC** |
| 3. Multivariate | `mbx/multivariate.py` | **PCA** (with QC clustering check); **PLS-DA** with **VIP** scores and leave-one-out cross-validated **Q²** |
| 4. Pathway & gene analysis | `mbx/pathways.py`, `mbx/knowledge_base.py` | metabolite→pathway mapping; **over-representation analysis** (Fisher exact + BH); metabolite→enzyme **gene** nomination |
| 5. Interpretation | `mbx/interpretation.py` | evidence-anchored Markdown narrative linking findings to cardiac-fibrosis biology |
| 6. Figures | `mbx/plotting.py` | QC-CV histogram, sample boxplots, PCA, PLS-DA, volcano, top-feature heatmap, pathway barplot, ROC |

## Outputs

```
results/
├── config.json                 # exact parameters used (reproducibility)
├── run_summary.json            # machine-readable summary of the whole run
├── tables/                     # 12 CSVs (preprocessed matrices, stats, enrichment, genes…)
├── figures/                    # 8 PNGs
└── report/
    └── biological_interpretation.md
```

## Design notes & caveats

- The pathway/gene knowledge base (`mbx/knowledge_base.py`) is a **curated,
  offline** approximation of KEGG/SMPDB restricted to HILIC-detectable
  metabolites — it needs no network access but should be cross-checked against
  MetaboAnalyst/KEGG for publication.
- Gene nominations are a **hypothesis-generating** metabolite→enzyme bridge, not
  measured expression.
- Untargeted intensities are **semi-quantitative**; fold changes are relative.
- Reproducible: fixed random seed (42) and a serialized config per run.
