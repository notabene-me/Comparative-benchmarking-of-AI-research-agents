# Benchmark Agents — AI Agent Comparison Framework

A scientific pipeline for comparing how different AI agents process the same omics dataset.

## Overview

This framework benchmarks AI research agents on untargeted metabolomics tasks by comparing
their outputs at three levels:

1. **File level** — per-file text, structure, and embedding
2. **Run level** — aggregated across all files in a run
3. **Agent level** — aggregated across all 8 runs of an agent

### Agents benchmarked

| Agent | Runs |
|-------|------|
| ChatGPT | 8 |
| Biomni | 8 |
| K-Dense | 8 |

## Input directory structure

```
agent_outputs_metabolomics/
├── ChatGPT/
│   ├── run_1/
│   │   ├── report.txt
│   │   └── results.csv
│   ├── run_2/
│   ...
│   └── run_8/
├── Biomni/
│   ├── run_1/
│   ...
│   └── run_8/
└── KDense/
    ├── run_1/
    ...
    └── run_8/
```

Alternatively, files may be named `AgentName_run_N_filename.ext`.

## Usage

```bash
python main.py \
  --input_dir ./agent_outputs_metabolomics \
  --output_dir ./benchmark_results_metabolomics \
  --n_mc 10000 \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2

# With manual scores:
python main.py \
  --input_dir ./agent_outputs_metabolomics \
  --output_dir ./benchmark_results_metabolomics \
  --scores ./manual_scores.csv

# Fallback without embeddings:
python main.py \
  --input_dir ./agent_outputs_metabolomics \
  --output_dir ./benchmark_results_metabolomics \
  --no_embeddings
```

## Manual scores CSV format

```csv
agent,run,R,P,D
ChatGPT,1,85,90,80
Biomni,1,80,75,85
KDense,1,70,80,75
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

# For advanced statistics:
pip install pingouin
```

## Output structure

```
benchmark_results_metabolomics/
├── tables/
│   ├── file_inventory.csv
│   ├── file_level_features.csv
│   ├── run_level_features.csv
│   ├── agent_level_features.csv
│   ├── pairwise_run_similarity.csv
│   ├── within_agent_reproducibility.csv
│   ├── between_agent_similarity.csv
│   ├── monte_carlo_centroids.csv
│   ├── monte_carlo_summary.csv
│   ├── agent_scores.csv
│   ├── ranking_summary.csv
│   ├── ranking_uncertainty.csv
│   ├── pairwise_win_probabilities.csv
│   └── validation_summary.csv
├── figures/
│   ├── pca_2d.png
│   ├── pca_3d.html
│   ├── mds_2d.png
│   ├── tsne_2d.png
│   ├── umap_2d.png
│   ├── run_similarity_heatmap.png
│   ├── within_agent_similarity_boxplot.png
│   ├── monte_carlo_satellites_3d.html
│   ├── score_distribution_boxplot.png
│   └── ranking_probability_barplot.png
└── report/
    ├── benchmark_report.md
    └── benchmark_report.html
```

## Modules

| Module | Description |
|--------|-------------|
| `src/scanner.py` | Scan input directory, detect agents and run numbers |
| `src/parser.py` | Parse files of various types (txt, csv, xlsx, pdf, etc.) |
| `src/inventory.py` | Generate file inventory |
| `src/chunking.py` | Split text into overlapping chunks |
| `src/embeddings.py` | Compute sentence-transformer embeddings |
| `src/features.py` | Extract structural and methodological features |
| `src/similarity.py` | Compute pairwise similarity at file/run/agent level |
| `src/dimensionality.py` | PCA, MDS, t-SNE, UMAP dimensionality reduction |
| `src/scoring.py` | AgentScore calculation |
| `src/monte_carlo.py` | Bootstrap / Monte Carlo satellite generation |
| `src/visualization.py` | Figures and plots (incl. uncertainty-aware figures) |
| `src/validation.py` | Ranking validation and reliability statistics |
| `src/report.py` | Generate Markdown and HTML report |
| `src/domain_scoring.py` | Domain-aware metabolomics score (0/1/2 criteria) |
| `src/score_audit.py` | Output-volume metrics, score correlations, ranking sensitivity, volume-penalised score |
| `src/fairness.py` | Fairness / information-access audit (`run_registry.csv`) |
| `src/error_analysis.py` | Bootstrap error, ranking, reproducibility & Monte Carlo uncertainty |

## Scoring audit & uncertainty layer

The pipeline includes a scoring-validation layer:

- **Proxy decomposition** — each proxy R/P/D score is the mean of explicit binary
  components (fully auditable) → `tables/proxy_score_components.csv`.
- **Domain-aware metabolomics score** — 16 criteria graded 0/1/2, reported
  *alongside* AgentScore as an independent sensitivity analysis (not a replacement)
  → `tables/domain_metabolomics_score_components.csv`.
- **Output-volume audit** — per-run volume metrics + Spearman correlations with
  AgentScore → `tables/output_volume_metrics.csv`, `tables/score_volume_correlation.csv`.
- **Ranking sensitivity** — rankings under 7 scoring definitions (A–G) →
  `tables/ranking_sensitivity_analysis.csv`.
- **Error estimation** — bootstrap absolute/relative error, ranking uncertainty,
  reproducibility & Monte Carlo error tables.
- **Fairness audit** — optional `agent_outputs_metabolomics/run_registry.csv`
  (columns: `agent,run,interactive_mode,asked_additional_questions,requested_metadata,additional_metadata_provided,notes`).
  Information-seeking behaviour is treated as part of agentic performance.

All scores are clearly marked **manual** or **proxy-estimated**. Proxy scores and
all error/CI estimates are descriptive (only 8 real runs per agent). Monte Carlo
satellites are uncertainty probes, not real observations.

### Manual scores CSV (extended)

```csv
agent,run,R,P,D,notes
ChatGPT,1,85,90,80,optional note
```

## Reproducibility

Random seed: 42

## Notes

- Biomni had an interactive dialogue function and requested additional clinical metadata.
- Agents that did not ask additional questions did not receive this information.
- Information-seeking behaviour is treated as part of agentic performance.
- Synthetic Monte Carlo satellite points are not real observations.
- Reliability statistics are descriptive because the number of real runs is limited.

## Windows 64-bit

See **[RUN_WINDOWS.md](RUN_WINDOWS.md)**. This repo uses **metabolomics folders only**:

| Platform | Input | Output |
|----------|-------|--------|
| macOS / Linux | `./agent_outputs_metabolomics` | `./benchmark_results_metabolomics` |
| Windows 64-bit | `./agent_outputs_metabolomics_Win64` | `./benchmark_results_metabolomics_Win64` |

Proteomics and transcriptomics are separate projects (`benchmark_agents_proteomics`,
`benchmark_agents_transcriptomics`) with their own folders and `RUN_WINDOWS.md`.

```cmd
python setup_win64_folders.py
python smoke_test_windows.py
python main.py --input_dir .\agent_outputs_metabolomics_Win64 --output_dir .\benchmark_results_metabolomics_Win64 --n_mc 10000
```
