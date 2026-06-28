"""
platform_utils.py — cross-platform helpers (Windows / macOS / Linux).

Transcriptomics project: folder names are scoped to this repo only.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

AGENT_INPUT_FOLDER = "agent_outputs_transcriptomics"
BENCHMARK_OUTPUT_FOLDER = "benchmark_results_transcriptomics"
AGENT_INPUT_FOLDER_WIN64 = "agent_outputs_transcriptomics_Win64"
BENCHMARK_OUTPUT_FOLDER_WIN64 = "benchmark_results_transcriptomics_Win64"

IGNORED_FILE_NAMES = frozenset({
    ".ds_store",
    "thumbs.db",
    "desktop.ini",
})

IGNORED_DIR_NAMES = frozenset({
    "__macosx",
    "$recycle.bin",
    "system volume information",
})

CANONICAL_AGENTS_LOWER = {
    "chatgpt":  "ChatGPT",
    "biomni":   "Biomni",
    "kdense":   "KDense",
    "k-dense":  "KDense",
    "k_dense":  "KDense",
    "finch":    "Finch",
    "fich":     "Finch",
}


def is_windows() -> bool:
    return os.name == "nt" or sys.platform.startswith("win")


def is_ignored_file(path: Path) -> bool:
    name = path.name.lower()
    if name in IGNORED_FILE_NAMES:
        return True
    if name.startswith("~$"):
        return True
    if name == "desktop.ini":
        return True
    return False


def is_ignored_dir(name: str) -> bool:
    lower = name.lower()
    if lower in IGNORED_DIR_NAMES:
        return True
    if lower.startswith("__macosx"):
        return True
    return False


def resolve_agent_folder_name(dir_name: str) -> str | None:
    return CANONICAL_AGENTS_LOWER.get(dir_name.lower())


def ensure_directory(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_benchmark_output_layout(
    output_dir: os.PathLike | str,
) -> Tuple[Path, Path, Path]:
    root = Path(output_dir)
    tables = ensure_directory(root / "tables")
    figures = ensure_directory(root / "figures")
    report = ensure_directory(root / "report")
    return tables, figures, report


def configure_matplotlib_backend() -> None:
    if not (is_windows() or not os.environ.get("DISPLAY")):
        return
    try:
        import matplotlib
        backend = matplotlib.get_backend().lower()
        if backend not in ("agg", "svg", "pdf", "ps", "template"):
            matplotlib.use("Agg")
            logger.debug("matplotlib backend set to Agg (%s)", sys.platform)
    except Exception as exc:
        logger.debug("matplotlib backend configuration skipped: %s", exc)


def _touch_folder(base: Path, name: str, *, with_output_layout: bool) -> Tuple[Path, bool]:
    p = base / name
    existed = p.exists()
    if with_output_layout:
        ensure_benchmark_output_layout(p)
    else:
        ensure_directory(p)
    return p, not existed


def create_project_folders(root: os.PathLike | str | None = None) -> List[Tuple[Path, bool]]:
    base = Path(root or Path.cwd())
    results: List[Tuple[Path, bool]] = []
    for name, layout in (
        (AGENT_INPUT_FOLDER, False),
        (BENCHMARK_OUTPUT_FOLDER, True),
        (AGENT_INPUT_FOLDER_WIN64, False),
        (BENCHMARK_OUTPUT_FOLDER_WIN64, True),
    ):
        results.append(_touch_folder(base, name, with_output_layout=layout))
    return results


def create_win64_folders(root: os.PathLike | str | None = None) -> List[Tuple[Path, bool]]:
    base = Path(root or Path.cwd())
    return [
        _touch_folder(base, AGENT_INPUT_FOLDER_WIN64, with_output_layout=False),
        _touch_folder(base, BENCHMARK_OUTPUT_FOLDER_WIN64, with_output_layout=True),
    ]
