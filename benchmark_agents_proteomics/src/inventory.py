"""
inventory.py — build and save a file inventory from scanner records.
"""

import logging
from pathlib import Path
from typing import List

import pandas as pd

from .scanner import FileRecord

logger = logging.getLogger(__name__)


def build_inventory(records: List[FileRecord]) -> pd.DataFrame:
    """Return a DataFrame with one row per FileRecord."""
    rows = [r.to_dict() for r in records]
    if not rows:
        logger.warning("No files found — inventory is empty.")
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["agent", "run", "filename"]).reset_index(drop=True)
    logger.info("Inventory: %d files, agents: %s",
                len(df),
                sorted(df["agent"].unique().tolist()))
    return df


def save_inventory(df: pd.DataFrame, output_dir: Path) -> None:
    out = output_dir / "tables" / "file_inventory.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Saved inventory → %s", out)


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No files found.")
        return
    print("\n=== File Inventory Summary ===")
    for agent in sorted(df["agent"].unique()):
        adf = df[df["agent"] == agent]
        runs = sorted(adf["run"].unique())
        print(f"  {agent}: {len(runs)} runs, {len(adf)} files")
        for run in runs:
            rdf = adf[adf["run"] == run]
            types = rdf["file_type"].value_counts().to_dict()
            print(f"    Run {run}: {len(rdf)} files — {types}")
    print()
