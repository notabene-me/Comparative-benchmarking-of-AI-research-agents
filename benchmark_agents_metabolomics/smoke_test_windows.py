#!/usr/bin/env python3
"""
smoke_test_windows.py — minimal environment and folder smoke test (metabolomics).

Run from the metabolomics project root:

    python smoke_test_windows.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def _check_imports(errors: list[str]) -> None:
    required = [
        "numpy", "pandas", "scipy", "sklearn", "matplotlib",
        "seaborn", "plotly", "tqdm", "openpyxl",
    ]
    optional = {
        "sentence_transformers": "sentence-transformers (optional)",
        "umap": "umap-learn (optional)",
        "fitz": "PyMuPDF (optional)",
        "docx": "python-docx (optional)",
        "streamlit": "streamlit (optional)",
    }
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  OK  import {pkg}")
        except ImportError as exc:
            errors.append(f"Required package missing: {pkg} ({exc})")
            print(f"  FAIL import {pkg}: {exc}")
    for pkg, label in optional.items():
        try:
            __import__(pkg)
            print(f"  OK  import {pkg} (optional)")
        except ImportError:
            print(f"  skip {label}")


def _check_project_modules(errors: list[str]) -> None:
    try:
        from src.platform_utils import configure_matplotlib_backend, is_windows
        import main  # noqa: F401
        configure_matplotlib_backend()
        print(f"  OK  project modules (platform={sys.platform}, windows={is_windows()})")
    except Exception as exc:
        errors.append(f"Project import failed: {exc}")
        print(f"  FAIL project modules: {exc}")


def _check_project_folders(root: Path, errors: list[str]) -> None:
    from src.platform_utils import (
        AGENT_INPUT_FOLDER,
        AGENT_INPUT_FOLDER_WIN64,
        BENCHMARK_OUTPUT_FOLDER,
        BENCHMARK_OUTPUT_FOLDER_WIN64,
        create_project_folders,
    )

    create_project_folders(root)
    for name in (
        AGENT_INPUT_FOLDER,
        BENCHMARK_OUTPUT_FOLDER,
        AGENT_INPUT_FOLDER_WIN64,
        BENCHMARK_OUTPUT_FOLDER_WIN64,
    ):
        p = root / name
        if not p.is_dir():
            errors.append(f"Folder missing after setup: {name}")
            print(f"  FAIL folder {name}")
        else:
            n_items = sum(1 for _ in p.rglob("*") if _.is_file())
            status = f"{n_items} file(s)" if n_items else "empty (copy agent outputs here)"
            print(f"  OK  {name} — {status}")


def _check_write_outputs(errors: list[str]) -> None:
    from src.platform_utils import ensure_benchmark_output_layout

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "smoke_test_output"
        tables, figures, report = ensure_benchmark_output_layout(out)
        test_csv = tables / "smoke_test.csv"
        test_csv.write_text("agent,run,value\nChatGPT,1,1\n", encoding="utf-8")
        test_png = figures / "smoke_test.png"
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.plot([0, 1], [0, 1])
            fig.savefig(test_png, dpi=72)
            plt.close(fig)
        except Exception as exc:
            errors.append(f"Could not write PNG figure: {exc}")
            print(f"  FAIL figure write: {exc}")
            return
        test_md = report / "smoke_test.md"
        test_md.write_text("# smoke test\n", encoding="utf-8")
        if not test_csv.exists() or not test_png.exists() or not test_md.exists():
            errors.append("Output write check failed")
            print("  FAIL output write")
        else:
            print("  OK  write tables / figures / report")


def main() -> int:
    root = Path(__file__).resolve().parent
    errors: list[str] = []

    print("=== Smoke test: imports ===")
    _check_imports(errors)
    print("\n=== Smoke test: project modules ===")
    _check_project_modules(errors)
    print("\n=== Smoke test: metabolomics folders ===")
    _check_project_folders(root, errors)
    print("\n=== Smoke test: output write ===")
    _check_write_outputs(errors)

    print("\n" + "=" * 60)
    if errors:
        print("SMOKE TEST FAILED")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("SMOKE TEST PASSED")
    print("macOS:  python main.py --input_dir ./agent_outputs_metabolomics \\")
    print("          --output_dir ./benchmark_results_metabolomics --n_mc 10000")
    print("Windows: see RUN_WINDOWS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
