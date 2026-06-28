#!/usr/bin/env python3
"""
setup_win64_folders.py — create macOS and Win64 folder trees for this project.

Run from the metabolomics project root (PowerShell or CMD):

    python setup_win64_folders.py

Creates (if missing):
  agent_outputs_metabolomics/
  benchmark_results_metabolomics/{tables,figures,report}
  agent_outputs_metabolomics_Win64/
  benchmark_results_metabolomics_Win64/{tables,figures,report}
"""

from pathlib import Path

from src.platform_utils import create_project_folders


def main() -> None:
    root = Path(__file__).resolve().parent
    print(f"Project root: {root}")
    print("Metabolomics folders only (this repo):")
    print("-" * 60)
    for folder, created in create_project_folders(root):
        status = "Created" if created else "Already exists"
        print(f"  [{status}] {folder.name}")
    print("-" * 60)
    print("Done. Copy agent outputs into agent_outputs_metabolomics (macOS)")
    print("or agent_outputs_metabolomics_Win64 (Windows), then run main.py.")
    print("See RUN_WINDOWS.md for Windows commands.")


if __name__ == "__main__":
    main()
