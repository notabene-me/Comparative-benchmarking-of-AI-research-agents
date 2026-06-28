# Running the transcriptomics benchmark on Windows 64-bit

Project: **`benchmark_agents_transcriptomics`**

Metabolomics and proteomics are **separate sibling projects** with their own
`RUN_WINDOWS.md` files and folder names. Run commands only from this repo's root.

## Recommended Python version

- **Python 3.10, 3.11, or 3.12** (64-bit)

## One-time setup

### PowerShell

```powershell
cd path\to\benchmark_agents_transcriptomics
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python setup_win64_folders.py
python smoke_test_windows.py
```

### CMD

```cmd
cd path\to\benchmark_agents_transcriptomics
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python setup_win64_folders.py
python smoke_test_windows.py
```

### HuggingFace download issues

**PowerShell:** `$env:HF_HUB_DISABLE_XET="1"` · **CMD:** `set HF_HUB_DISABLE_XET=1`

## Project folders (this repo only)

```text
agent_outputs_transcriptomics/
benchmark_results_transcriptomics/
agent_outputs_transcriptomics_Win64/
benchmark_results_transcriptomics_Win64/
```

## macOS / Linux benchmark run

```bash
python main.py \
  --input_dir ./agent_outputs_transcriptomics \
  --output_dir ./benchmark_results_transcriptomics \
  --n_mc 10000
```

## Win64 benchmark runs

From **`benchmark_agents_transcriptomics`** project root:

**CMD / PowerShell:**

```cmd
python main.py --input_dir .\agent_outputs_transcriptomics_Win64 --output_dir .\benchmark_results_transcriptomics_Win64 --n_mc 10000
```

### Quick test

```cmd
python main.py --input_dir .\agent_outputs_transcriptomics_Win64 --output_dir .\benchmark_results_transcriptomics_Win64 --n_mc 200 --no_embeddings
```

### Validate input only

```cmd
python main.py --validate_only --input_dir .\agent_outputs_transcriptomics_Win64 --output_dir .\benchmark_results_transcriptomics_Win64
```

## Streamlit dashboard

```cmd
streamlit run app.py
```

## Smoke test

```cmd
python smoke_test_windows.py
```
