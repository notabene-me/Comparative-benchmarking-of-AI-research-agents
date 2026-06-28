# Benchmark Agents Proteomics — AI Agent Comparison Framework

A scientific pipeline for comparing how different AI agents process the same proteomics dataset.

## Overview

This framework benchmarks AI research agents on untargeted proteomics tasks by comparing
their outputs at three levels:

1. **File level** — per-file text, structure, and embedding
2. **Run level** — aggregated across all files in a run
3. **Agent level** — aggregated across all runs of an agent

### Agents benchmarked

| Agent | Runs |
|-------|------|
| ChatGPT | 8 |
| Biomni | 8 |
| K-Dense | 8 |
| Finch | 8 |

## Input directory structure

```
agent_outputs_proteomics/
├── ChatGPT/
│   ├── Run1/
│   │   ├── report.txt
│   │   └── results.csv
│   ├── Run2/
│   ...
│   └── Run8/
├── Biomni/
│   ├── Run1/
│   ...
│   └── Run8/
├── KDense/
│   ├── Run1/
│   ...
│   └── Run8/
└── Finch/
    ├── Run1/
    ...
    └── Run8/
```

Alternatively, files may be named `AgentName_run_N_filename.ext`.

## Usage

```bash
cd /Users/nbasov/benchmark_agents_proteomics

python main.py \
  --input_dir ./agent_outputs_proteomics \
  --output_dir ./benchmark_results_proteomics \
  --n_mc 10000 \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2

# With manual scores:
python main.py \
  --input_dir ./agent_outputs_proteomics \
  --output_dir ./benchmark_results_proteomics \
  --scores ./manual_scores.csv

# Fallback without embeddings:
python main.py \
  --input_dir ./agent_outputs_proteomics \
  --output_dir ./benchmark_results_proteomics \
  --no_embeddings

# Generate synthetic demo data for testing:
python main.py --generate_demo --input_dir ./agent_outputs_proteomics
```

## Manual scores CSV format

```csv
agent,run,R,P,D
ChatGPT,1,85,90,80
Biomni,1,80,75,85
KDense,1,70,80,75
Finch,1,75,82,78
```

Where:
- **R** = Result accuracy (0–100)
- **P** = Process quality (0–100)
- **D** = Interpretation depth (0–100)

## AgentScore formula

```
AgentScore = P × (0.50 + 0.30 × D/100 + 0.20 × R/100)
```

## Installation

```bash
pip install -r requirements.txt

# For embeddings support:
pip install sentence-transformers

# For UMAP:
pip install umap-learn

# For PDF parsing:
pip install PyMuPDF

# For DOCX parsing:
pip install python-docx

# For Excel parsing:
pip install openpyxl
```

## Output structure

```
benchmark_results_proteomics/
├── tables/
│   ├── file_inventory.csv
│   ├── agent_scores.csv
│   ├── domain_proteomics_score_components.csv
│   ├── consensus_method_terms.csv
│   ├── consensus_biological_terms.csv
│   └── validation_summary.csv
├── figures/
│   ├── pca_2d.png / pca_2d.svg
│   ├── pca_3d.html
│   ├── agent_score_errorbar.png / .svg
│   └── ...
└── report/
    ├── benchmark_report.md
    └── benchmark_report.html
```

## Proteomics-specific scoring

The pipeline includes domain-aware proteomics criteria (16 items, graded 0/1/2):

- LC-MS/MS platform context (DDA, DIA, label-free)
- QC strategy and pooled QC
- Normalization, imputation, multiple testing correction
- Effect sizes (fold change, log2FC)
- PCA with appropriate caution
- Significant protein reporting
- Protein ID / annotation uncertainty
- GO / pathway / functional interpretation
- Biological mechanism and limitations

Consensus analysis extracts proteomics method terms (MaxQuant, Perseus, limma, MSstats, etc.)
and biological themes (kinases, PTMs, immune, mitochondrial proteins, etc.).

## Fairness audit

Optional `agent_outputs_proteomics/run_registry.csv` with columns:
`agent,run,interactive_mode,asked_additional_questions,requested_metadata,additional_metadata_provided,notes`

## Reproducibility

Random seed: 42

## Notes

- All 2D figures are saved in both **PNG** and **SVG** formats.
- Proxy scores and bootstrap CIs are descriptive (8 real runs per agent).
- Monte Carlo satellites are uncertainty probes, not real observations.
- Domain-aware proteomics score is a rule-based sensitivity analysis, not expert review.

## Windows 64-bit

See **[RUN_WINDOWS.md](RUN_WINDOWS.md)**. This repo uses **proteomics folders only**:

| Platform | Input | Output |
|----------|-------|--------|
| macOS / Linux | `./agent_outputs_proteomics` | `./benchmark_results_proteomics` |
| Windows 64-bit | `./agent_outputs_proteomics_Win64` | `./benchmark_results_proteomics_Win64` |

```cmd
python setup_win64_folders.py
python smoke_test_windows.py
python main.py --input_dir .\agent_outputs_proteomics_Win64 --output_dir .\benchmark_results_proteomics_Win64 --n_mc 10000
```
