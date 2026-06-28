# Running the metabolomics benchmark on Windows 64-bit

Project: **`benchmark_agents_metabolomics`**

Proteomics and transcriptomics are **separate sibling projects** with their own
`RUN_WINDOWS.md` files and folder names. Run commands only from this repo's root.

## Recommended Python version

- **Python 3.10, 3.11, or 3.12** (64-bit)
- Verify: `python --version` or `py -3 --version`

## One-time setup

### PowerShell

```powershell
cd path\to\benchmark_agents_metabolomics
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python setup_win64_folders.py
python smoke_test_windows.py
```

### CMD

```cmd
cd path\to\benchmark_agents_metabolomics
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python setup_win64_folders.py
python smoke_test_windows.py
```

### Optional packages

```text
pip install sentence-transformers umap-learn python-docx PyMuPDF streamlit
```

### HuggingFace model download issues

**PowerShell (session):** `$env:HF_HUB_DISABLE_XET="1"`

**CMD (session):** `set HF_HUB_DISABLE_XET=1`

**Permanent:** `setx HF_HUB_DISABLE_XET 1`

Use `--no_embeddings` if the model still cannot load (TF-IDF fallback is automatic).

## Project folders (this repo only)

`python setup_win64_folders.py` creates:

```text
agent_outputs_metabolomics/              ← macOS / Linux inputs
benchmark_results_metabolomics/            ← macOS / Linux outputs
agent_outputs_metabolomics_Win64/        ← Windows inputs (copy files here)
benchmark_results_metabolomics_Win64/      ← Windows outputs
```

Expected input layout:

```text
agent_outputs_metabolomics_Win64/
├── Biomni/
│   ├── run_1/
│   └── ... run_8/
├── ChatGPT/
├── Finch/
└── KDense/
```

## macOS / Linux benchmark run

From this project root:

```bash
python main.py \
  --input_dir ./agent_outputs_metabolomics \
  --output_dir ./benchmark_results_metabolomics \
  --n_mc 10000
```

## Win64 benchmark runs

From **`benchmark_agents_metabolomics`** project root:

**CMD:**

```cmd
python main.py --input_dir .\agent_outputs_metabolomics_Win64 --output_dir .\benchmark_results_metabolomics_Win64 --n_mc 10000
```

**PowerShell:**

```powershell
python main.py --input_dir .\agent_outputs_metabolomics_Win64 --output_dir .\benchmark_results_metabolomics_Win64 --n_mc 10000
```

If `python` is not on PATH:

```cmd
py -3 main.py --input_dir .\agent_outputs_metabolomics_Win64 --output_dir .\benchmark_results_metabolomics_Win64 --n_mc 10000
```

### Quick test

```cmd
python main.py --input_dir .\agent_outputs_metabolomics_Win64 --output_dir .\benchmark_results_metabolomics_Win64 --n_mc 200 --no_embeddings
```

### Validate input only

```cmd
python main.py --validate_only --input_dir .\agent_outputs_metabolomics_Win64 --output_dir .\benchmark_results_metabolomics_Win64
```

## Streamlit dashboard

```cmd
streamlit run app.py
```

## Smoke test

```cmd
python smoke_test_windows.py
```

## Other omics projects

| Project | Repo folder | Windows input | Windows output |
|---------|-------------|---------------|----------------|
| Proteomics | `benchmark_agents_proteomics` | `agent_outputs_proteomics_Win64` | `benchmark_results_proteomics_Win64` |
| Transcriptomics | `benchmark_agents_transcriptomics` | `agent_outputs_transcriptomics_Win64` | `benchmark_results_transcriptomics_Win64` |

See each project's own `RUN_WINDOWS.md` for its launch command.
