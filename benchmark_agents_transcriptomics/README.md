# Benchmark Agents Transcriptomics — AI Agent Comparison Framework

A scientific pipeline for comparing how different AI agents process the same transcriptomics dataset.

## Overview

This framework benchmarks AI research agents on bulk RNA-seq / transcriptomics tasks by comparing
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
agent_outputs_transcriptomics/
├── ChatGPT/
│   ├── Run1/
│   │   ├── report.txt
│   │   └── results.csv
│   ...
│   └── Run8/
├── Biomni/
├── Finch/
└── KDense/
```

Alternatively, files may be named `AgentName_run_N_filename.ext`.

## Usage

```bash
cd /Users/nbasov/benchmark_agents_transcriptomics

python main.py \
  --input_dir ./agent_outputs_transcriptomics \
  --output_dir ./benchmark_results_transcriptomics \
  --n_mc 10000

# With manual scores:
python main.py \
  --input_dir ./agent_outputs_transcriptomics \
  --output_dir ./benchmark_results_transcriptomics \
  --scores ./manual_scores.csv

# Fallback without embeddings:
python main.py \
  --input_dir ./agent_outputs_transcriptomics \
  --output_dir ./benchmark_results_transcriptomics \
  --no_embeddings

# Generate synthetic demo data for testing:
python main.py --generate_demo --input_dir ./agent_outputs_transcriptomics
```

## Manual scores CSV format

```csv
agent,run,R,P,D
ChatGPT,1,85,90,80
Biomni,1,80,75,85
KDense,1,70,80,75
Finch,1,75,82,78
```

## AgentScore formula

```
AgentScore = P × (0.50 + 0.30 × D/100 + 0.20 × R/100)
```

## Installation

```bash
pip install -r requirements.txt
pip install sentence-transformers umap-learn PyMuPDF python-docx openpyxl
```

## Output structure

```
benchmark_results_transcriptomics/
├── tables/
│   ├── agent_scores.csv
│   ├── domain_transcriptomics_score_components.csv
│   ├── consensus_method_terms.csv
│   └── ...
├── figures/
│   ├── pca_2d.png / pca_2d.svg
│   └── ...
└── report/
    ├── benchmark_report.md
    └── benchmark_report.html
```

## Transcriptomics-specific scoring

Domain-aware criteria (16 items, graded 0/1/2):

- RNA-seq platform context (STAR, salmon, kallisto, featureCounts)
- QC strategy (FastQC, MultiQC)
- Normalization and batch correction
- Differential expression (DESeq2, edgeR, limma-voom)
- Multiple testing correction and effect sizes
- PCA with appropriate caution
- Significant gene reporting
- GO / GSEA / pathway interpretation
- Biological mechanism and limitations

Consensus analysis extracts transcriptomics method terms (DESeq2, edgeR, STAR, GSEA, etc.)
and biological themes (immune, cell cycle, transcription factors, splicing, etc.).

## Fairness audit

Optional `agent_outputs_transcriptomics/run_registry.csv` with columns:
`agent,run,interactive_mode,asked_additional_questions,requested_metadata,additional_metadata_provided,notes`

## Notes

- All 2D figures are saved in both **PNG** and **SVG** formats.
- Proxy scores and bootstrap CIs are descriptive (8 real runs per agent).
- Monte Carlo satellites are uncertainty probes, not real observations.

## Windows 64-bit

See **[RUN_WINDOWS.md](RUN_WINDOWS.md)**. This repo uses **transcriptomics folders only**:

| Platform | Input | Output |
|----------|-------|--------|
| macOS / Linux | `./agent_outputs_transcriptomics` | `./benchmark_results_transcriptomics` |
| Windows 64-bit | `./agent_outputs_transcriptomics_Win64` | `./benchmark_results_transcriptomics_Win64` |

```cmd
python setup_win64_folders.py
python smoke_test_windows.py
python main.py --input_dir .\agent_outputs_transcriptomics_Win64 --output_dir .\benchmark_results_transcriptomics_Win64 --n_mc 10000
```
