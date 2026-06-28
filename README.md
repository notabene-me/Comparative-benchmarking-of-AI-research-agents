Omics AI Agent Benchmarking Framework

A reproducible Python framework for comparing how different AI research agents interpret the **same omics dataset** across three modalities: **transcriptomics**, **proteomics**, and **metabolomics**.

Each modality is a self-contained project with the same core pipeline and modality-specific domain scoring. The framework evaluates agent outputs as scientific documents — not as black-box model accuracy — using semantic embeddings, keyword coverage, structural features, reproducibility metrics, and bootstrap ranking uncertainty.

---

## Repository structure

This GitHub repository contains **three sibling projects**. Run each pipeline independently inside its own folder:

```
.
├── README.md                              ← this file (repository root)
├── benchmark_agents_transcriptomics/      ← bulk RNA-seq / transcriptomics
├── benchmark_agents_proteomics/           ← untargeted proteomics
└── benchmark_agents_metabolomics/         ← untargeted metabolomics
```

| Folder | Omics domain | Default input | Default output |
|--------|--------------|---------------|----------------|
| `benchmark_agents_transcriptomics/` | Transcriptomics | `agent_outputs_transcriptomics/` | `benchmark_results_transcriptomics/` |
| `benchmark_agents_proteomics/` | Proteomics | `agent_outputs_proteomics/` | `benchmark_results_proteomics/` |
| `benchmark_agents_metabolomics/` | Metabolomics | `agent_outputs_metabolomics/` | `benchmark_results_metabolomics/` |

Each folder has its own `main.py`, `src/`, `requirements.txt`, and `RUN_WINDOWS.md`.

---

## What is being compared

### Agents

| Agent | Independent runs per modality |
|-------|-------------------------------|
| ChatGPT | 8 |
| Biomni | 8 |
| K-Dense | 8 |
| Finch | 8 |

### Evaluation levels

1. **File level** — parsed text, structural counts, chunk embeddings
2. **Run level** — aggregated across all files produced in one agent run
3. **Agent level** — aggregated across all 8 runs of one agent

### Core metrics

| Metric | What it measures |
|--------|------------------|
| **AgentScore** | Composite proxy score from process quality (P), result accuracy (R), and interpretation depth (D) |
| **DomainScore** | Modality-specific rubric: 16 criteria graded 0 / 1 / 2 |
| **Within-agent reproducibility** \(\rho_a\) | Semantic consistency of an agent across its 8 runs |
| **Between-agent similarity** | Cosine similarity of run embeddings and agent centroids |
| **Bootstrap ranking uncertainty** | Probability that each agent occupies each rank; pairwise win probabilities |
| **Monte Carlo centroids** | Uncertainty cloud around agent embedding centroids |

---

## Pipeline overview

All three projects share the same sequential workflow:

```
Agent output files
    → directory scan & file inventory
    → text parsing (txt, csv, xlsx, pdf, docx, ipynb, json, …)
    → tokenisation & overlapping chunking (800 tokens, overlap 150)
    → parallel feature extraction
         ├─ structural counts (tables, p-values, code blocks, …)
         ├─ methodological keyword counts (DESeq2, FDR, PCA, …)
         └─ biological term counts (pathways, genes, immune, …)
    → semantic embeddings (sentence-transformers/all-MiniLM-L6-v2; TF-IDF+SVD fallback)
    → hierarchical mean pooling: chunk → file → run → agent centroid
    → pairwise similarity (cosine, Euclidean, Jaccard)
    → dimensionality reduction (PCA, MDS, t-SNE, UMAP)
    → AgentScore + DomainScore
    → bootstrap / Monte Carlo uncertainty
    → tables, figures, HTML/Markdown report
```

Detailed step-by-step algorithm with formulas:  
`benchmark_agents_transcriptomics/Radar_plot_discussion/Algorithm_benchmark_detailed_mathematical.md`

Manuscript Materials & Methods:  
`benchmark_agents_transcriptomics/MM AI comparison paper/`

---

## Quick start

### 1. Choose a modality and enter its folder

```bash
# Transcriptomics
cd benchmark_agents_transcriptomics

# Proteomics
cd benchmark_agents_proteomics

# Metabolomics
cd benchmark_agents_metabolomics
```

### 2. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Recommended (semantic embeddings + full file parsing):
pip install sentence-transformers umap-learn PyMuPDF python-docx openpyxl
```

### 3. Place agent outputs in the input folder

Expected layout (run folder names are flexible: `Run1`, `run_1`, `r3`, etc.):

```
agent_outputs_<omics>/
├── ChatGPT/
│   ├── Run1/
│   │   ├── report.txt
│   │   └── results.csv
│   └── Run8/
├── Biomni/
├── KDense/
└── Finch/
```

Alternatively, files may be named `AgentName_run_N_filename.ext`.

### 4. Run the benchmark

```bash
python main.py \
  --input_dir ./agent_outputs_transcriptomics \
  --output_dir ./benchmark_results_transcriptomics \
  --n_mc 10000 \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2
```

Replace folder names for proteomics or metabolomics:

```bash
# Proteomics
python main.py \
  --input_dir ./agent_outputs_proteomics \
  --output_dir ./benchmark_results_proteomics \
  --n_mc 10000

# Metabolomics
python main.py \
  --input_dir ./agent_outputs_metabolomics \
  --output_dir ./benchmark_results_metabolomics \
  --n_mc 10000
```

### 5. Optional flags

```bash
# Use manual expert scores instead of proxy scores:
python main.py --input_dir ./agent_outputs_transcriptomics \
               --output_dir ./benchmark_results_transcriptomics \
               --scores ./manual_scores.csv

# Skip neural embeddings (TF-IDF fallback):
python main.py --input_dir ./agent_outputs_transcriptomics \
               --output_dir ./benchmark_results_transcriptomics \
               --no_embeddings

# Generate synthetic demo data for testing:
python main.py --generate_demo --input_dir ./agent_outputs_transcriptomics

# Scan and validate inputs only (no scoring or embeddings):
python main.py --validate_only \
               --input_dir ./agent_outputs_transcriptomics \
               --output_dir ./benchmark_results_transcriptomics
```

---

## Scoring

### AgentScore (all modalities)

Proxy subscores \(P, R, D \in [0, 100]\) are computed from auditable binary components (process quality, result accuracy, interpretation depth):

```
AgentScore = P × (0.50 + 0.30 × D/100 + 0.20 × R/100)
```

Process quality acts as a **gate**: if \(P = 0\), the final score is 0 regardless of results or interpretation.

### Manual scores CSV

```csv
agent,run,R,P,D
ChatGPT,1,85,90,80
Biomni,1,80,75,85
KDense,1,70,80,75
Finch,1,75,82,78
```

- **R** = Result accuracy (0–100)
- **P** = Process quality (0–100)
- **D** = Interpretation depth (0–100)

### DomainScore (modality-specific rubric)

Each omics project applies **16 domain criteria** graded 0 (absent), 1 (partial), or 2 (clearly present). Maximum raw score = 32; normalised to 0–100:

```
DomainScore = 100 × (sum of grades) / 32
```

| Modality | Example criteria |
|----------|------------------|
| **Transcriptomics** | RNA-seq platform (STAR, salmon, kallisto), QC (FastQC, MultiQC), DESeq2/edgeR/limma, multiple testing, significant genes, GO/GSEA, limitations |
| **Proteomics** | LC-MS/MS platform (DDA, DIA, label-free), pooled QC, MaxQuant/Perseus, significant proteins, protein ID uncertainty, pathway interpretation |
| **Metabolomics** | LC-MS context, pooled QC, missing-value imputation, PLS-DA/OPLS-DA, pathway class, significant features, annotation caution |

Domain scores are reported **alongside** AgentScore as an independent sensitivity analysis — not as a replacement for expert review.

---

## Output structure

Each run produces:

```
benchmark_results_<omics>/
├── tables/
│   ├── file_inventory.csv
│   ├── run_level_features.csv
│   ├── agent_scores.csv
│   ├── domain_<omics>_score_components.csv
│   ├── pairwise_run_similarity.csv
│   ├── within_agent_reproducibility.csv
│   ├── between_agent_similarity.csv
│   ├── ranking_uncertainty.csv
│   ├── pairwise_win_probabilities.csv
│   ├── monte_carlo_centroids.csv
│   └── validation_summary.csv
├── figures/
│   ├── pca_2d.png / .svg
│   ├── tsne_2d.png / .svg
│   ├── domain_score_radar.png / .svg
│   ├── agent_score_errorbar.png / .svg
│   └── ...
└── report/
    ├── benchmark_report.md
    └── benchmark_report.html
```

All 2D figures are saved in both **PNG** and **SVG** formats.

---

## Source code modules

| Module | Role |
|--------|------|
| `src/scanner.py` | Scan input directory; detect agent and run labels |
| `src/parser.py` | Parse txt, csv, xlsx, pdf, docx, ipynb, json, svg, … |
| `src/chunking.py` | Overlapping text chunking (800 tokens, 150 overlap) |
| `src/embeddings.py` | Sentence-transformer embeddings; TF-IDF+SVD fallback |
| `src/features.py` | Structural counts + methodological/biological keyword extraction |
| `src/similarity.py` | Cosine, Euclidean, Jaccard pairwise similarity |
| `src/dimensionality.py` | PCA, MDS, t-SNE, UMAP |
| `src/scoring.py` | AgentScore calculation |
| `src/domain_scoring.py` | Modality-specific 0/1/2 domain rubric |
| `src/monte_carlo.py` | Bootstrap centroids + Gaussian satellite sampling |
| `src/validation.py` | Bootstrap ranking, Kendall's W, Cronbach's α |
| `src/visualization.py` | All figures including radar plots |
| `src/report.py` | Markdown and HTML benchmark report |

---

## Windows 64-bit

Each project folder contains `RUN_WINDOWS.md` and helper scripts:

```cmd
python setup_win64_folders.py
python smoke_test_windows.py
python main.py --input_dir .\agent_outputs_transcriptomics_Win64 ^
               --output_dir .\benchmark_results_transcriptomics_Win64 ^
               --n_mc 10000
```

Use the corresponding `_Win64` input/output folder names for proteomics and metabolomics.

---

## Fairness audit (optional)

Place `run_registry.csv` inside the agent outputs folder:

```csv
agent,run,interactive_mode,asked_additional_questions,requested_metadata,additional_metadata_provided,notes
Biomni,1,yes,yes,yes,yes,requested clinical metadata
ChatGPT,1,no,no,no,no,
```

Information-seeking behaviour is treated as part of agentic performance.

---

## Reproducibility and limitations

- **Random seed:** `SEED = 42` (bootstrap, PCA, t-SNE, UMAP, Monte Carlo)
- **Bootstrap / Monte Carlo intervals** describe variability over the **observed 8 runs per agent**, not population-level agent behaviour
- **Monte Carlo satellite points** are synthetic uncertainty probes, not real experimental outputs
- **Proxy scores** are keyword-detectable features in text — not verified biological truth
- **DomainScore** operationalises community best-practice checklists as computable functions; it is not a substitute for expert peer review

---

## What is novel vs. routine

**Routine (established methods used as-is):** whitespace tokenisation, sentence-transformer embeddings, TF-IDF+SVD, cosine/Euclidean/Jaccard similarity, PCA/MDS/t-SNE/UMAP, bootstrap CIs, Kendall's W, Cronbach's α.

**Novel contribution of this framework:**

1. **Experimental design** — replicated multi-agent, multi-omics evaluation (4 agents × 8 runs × 3 omics domains)
2. **Domain-specific ordinal rubrics** — 16 computable criteria per omics modality reflecting community best practices
3. **AgentScore formula** — process quality as a multiplicative gate; depth and accuracy as amplifiers
4. **Within-agent semantic reproducibility** \(\rho_a\) — behavioural stability metric independent of answer correctness
5. **Bootstrap rank probability distributions** — full uncertainty over agent rankings, not a single point estimate

---

## Related documentation

| Path | Content |
|------|---------|
| `benchmark_agents_transcriptomics/Radar_plot_discussion/` | Radar plot analysis, mathematical algorithm description, figure captions |
| `benchmark_agents_transcriptomics/MM AI comparison paper/` | Manuscript Materials & Methods (biologist + technical versions) |
| `benchmark_agents_*/RUN_WINDOWS.md` | Windows-specific setup per modality |

---

## Citation

If you use this framework, please cite the associated research article (link to be added upon publication).

---

## License

[Specify license here, e.g. MIT]
