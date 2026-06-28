"""
scanner.py — scan input directory, detect agents and run numbers,
produce a flat list of (agent, run, filepath) records.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from .platform_utils import is_ignored_dir, is_ignored_file, resolve_agent_folder_name

logger = logging.getLogger(__name__)

KNOWN_AGENTS = ["ChatGPT", "Biomni", "KDense", "K-Dense", "K_Dense", "Finch"]
AGENT_ALIASES: Dict[str, str] = {
    "k-dense": "KDense",
    "k_dense": "KDense",
    "kdense":  "KDense",
    "chatgpt": "ChatGPT",
    "biomni":  "Biomni",
    "finch":   "Finch",
    "fich":    "Finch",  # legacy alias; canonical label is Finch
}
# Folders that are never treated as agent names (metadata / pipeline dirs).
_RESERVED_DIRS = frozenset({
    "tables", "figures", "report", "raw_archives", "benchmark_results",
})

TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".tsv", ".scv", ".json", ".jsonl",
    ".html", ".htm", ".py", ".r", ".rmd", ".ipynb", ".xml",
    ".log", ".tex",
    # LaTeX helper files (all plain-text)
    ".bib", ".bbl", ".blg", ".aux", ".sty", ".toc", ".out",
    # config / structured text
    ".toml",
}
TABLE_EXTENSIONS = {".csv", ".tsv", ".scv", ".xlsx", ".xls", ".xlsm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".tiff", ".bmp"}
DOC_EXTENSIONS   = {".pdf", ".docx", ".doc"}
CODE_EXTENSIONS  = {".py", ".r", ".rmd", ".ipynb", ".sh", ".bash"}

ALL_SUPPORTED = TEXT_EXTENSIONS | TABLE_EXTENSIONS | IMAGE_EXTENSIONS | DOC_EXTENSIONS | CODE_EXTENSIONS


def _normalise_agent(raw: str) -> str:
    return AGENT_ALIASES.get(raw.lower(), raw)


def _detect_run_number(name: str) -> Optional[int]:
    """Extract the first integer that looks like a run index from a string."""
    patterns = [
        r"run[_\-\s]*(\d+)",
        r"r(\d+)",
        r"(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, name, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _detect_agent_from_name(name: str) -> Optional[str]:
    """Try to find a known agent alias inside a filename or directory name."""
    lower = name.lower()
    for alias, canonical in AGENT_ALIASES.items():
        if alias in lower:
            return canonical
    return None


def _resolve_agent_dir(dir_name: str) -> Optional[str]:
    """
    Resolve an agent label from a top-level directory name.

    Known aliases (ChatGPT, Biomni, KDense, …) are normalised first.
    Any other non-reserved folder name is accepted as a custom agent label
    when using the standard layout: <AgentName>/run_N/...
    """
    if not dir_name or dir_name.lower() in _RESERVED_DIRS:
        return None
    # Case-insensitive match for standard agent folders (important on Windows)
    canonical = resolve_agent_folder_name(dir_name)
    if canonical:
        return canonical
    alias = _detect_agent_from_name(dir_name)
    if alias:
        return alias
    # Do not treat run-only folders as agent names.
    if re.match(r"^run[_\-\s]*\d+$", dir_name, re.IGNORECASE):
        return None
    return _normalise_agent(dir_name)


class FileRecord:
    __slots__ = ("agent", "run", "filepath", "extension", "file_type", "size_bytes")

    def __init__(self, agent: str, run: int, filepath: Path):
        self.agent = agent
        self.run = run
        self.filepath = filepath
        self.extension = filepath.suffix.lower()
        self.file_type = _classify(self.extension)
        self.size_bytes = filepath.stat().st_size if filepath.exists() else 0

    def to_dict(self) -> dict:
        return {
            "agent":      self.agent,
            "run":        self.run,
            "filepath":   str(self.filepath),
            "filename":   self.filepath.name,
            "extension":  self.extension,
            "file_type":  self.file_type,
            "size_bytes": self.size_bytes,
        }


def _classify(ext: str) -> str:
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in TABLE_EXTENSIONS:
        return "table"
    if ext in DOC_EXTENSIONS:
        return "document"
    if ext in CODE_EXTENSIONS:
        return "code"
    if ext in TEXT_EXTENSIONS:
        return "text"
    return "other"


def scan(input_dir: str | Path) -> List[FileRecord]:
    """
    Walk input_dir and return a list of FileRecord objects.

    Supported layouts
    -----------------
    1. Hierarchical:
         input_dir/AgentName/run_N/file.ext
         input_dir/AgentName/runN/file.ext

    2. Flat with encoded name:
         input_dir/AgentName_run_N_something.ext
         input_dir/AgentName/AgentName_run_N_something.ext

    3. Mixed.
    """
    root = Path(input_dir)
    if not root.exists():
        logger.warning("Input directory does not exist: %s", root)
        return []

    records: List[FileRecord] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Do not descend into OS/archive artifact directories
        dirnames[:] = sorted(d for d in dirnames if not is_ignored_dir(d))
        dp = Path(dirpath)
        rel = dp.relative_to(root)
        parts = list(rel.parts)

        for fname in sorted(filenames):
            fp = dp / fname
            if is_ignored_file(fp):
                continue
            ext = fp.suffix.lower()
            if fp.name.startswith("."):
                continue

            # --- Try hierarchical layout first ---
            agent: Optional[str] = None
            run: Optional[int] = None

            if parts:
                # Level-0 directory holds agent name (known alias or custom folder)
                candidate_agent = _resolve_agent_dir(parts[0])
                if candidate_agent:
                    agent = candidate_agent

            if len(parts) >= 2:
                # Level-1 directory often holds run number
                run = _detect_run_number(parts[1])
                if run is None:
                    run = _detect_run_number(parts[0])

            elif len(parts) == 1:
                run = _detect_run_number(parts[0])

            # --- Fallback: try to decode from filename ---
            if agent is None:
                agent = _detect_agent_from_name(fname)
            if run is None:
                run = _detect_run_number(fname)

            # If still no agent or run, try parent directory names
            if agent is None:
                for p in parts:
                    a = _resolve_agent_dir(p) if p == parts[0] else _detect_agent_from_name(p)
                    if a:
                        agent = a
                        break

            if run is None:
                for p in parts:
                    r = _detect_run_number(p)
                    if r is not None:
                        run = r
                        break

            if agent is None or run is None:
                logger.debug("Skipping unrecognised file: %s (agent=%s run=%s)", fp, agent, run)
                continue

            agent = _normalise_agent(agent)
            records.append(FileRecord(agent=agent, run=run, filepath=fp))
            logger.debug("Found: agent=%s run=%d file=%s", agent, run, fp.name)

    logger.info("Scanner found %d files across %d unique (agent, run) pairs.",
                len(records),
                len({(r.agent, r.run) for r in records}))
    return records


def summarise(records: List[FileRecord]) -> Dict[str, Dict[int, List[FileRecord]]]:
    """Group records: agent → run → [FileRecord, ...]."""
    out: Dict[str, Dict[int, List[FileRecord]]] = {}
    for rec in records:
        out.setdefault(rec.agent, {}).setdefault(rec.run, []).append(rec)
    return out
