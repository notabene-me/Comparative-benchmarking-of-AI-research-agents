"""
fairness.py — fairness / information-access audit.

Reads an optional run_registry.csv describing each run's interaction mode
and whether additional metadata was requested/provided. Information-seeking
behaviour is treated as part of agentic performance — NOT a protocol
violation.

Expected run_registry.csv columns:
    agent, run, interactive_mode, asked_additional_questions,
    requested_metadata, additional_metadata_provided, notes
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGISTRY_COLUMNS = [
    "agent", "run", "interactive_mode", "asked_additional_questions",
    "requested_metadata", "additional_metadata_provided", "notes",
]

_BOOL_COLS = [
    "interactive_mode", "asked_additional_questions",
    "requested_metadata", "additional_metadata_provided",
]


def _to_bool(series: pd.Series) -> pd.Series:
    truthy = {"1", "true", "yes", "y", "t"}
    return series.astype(str).str.strip().str.lower().isin(truthy)


def load_run_registry(input_dir: str | Path) -> Optional[pd.DataFrame]:
    """
    Look for run_registry.csv in input_dir. Returns a normalised DataFrame
    or None if absent / unreadable.
    """
    p = Path(input_dir) / "run_registry.csv"
    if not p.exists():
        logger.info("No run_registry.csv found in %s — fairness audit skipped.", input_dir)
        return None
    try:
        df = pd.read_csv(p)
        if "agent" not in df.columns or "run" not in df.columns:
            logger.warning("run_registry.csv missing agent/run columns; ignoring.")
            return None
        df["agent"] = df["agent"].astype(str).str.strip()
        for col in _BOOL_COLS:
            if col in df.columns:
                df[col] = _to_bool(df[col])
            else:
                df[col] = False
        if "notes" not in df.columns:
            df["notes"] = ""
        logger.info("Loaded run_registry.csv: %d rows from %s", len(df), p)
        return df
    except Exception as exc:
        logger.error("Failed to read run_registry.csv: %s", exc)
        return None


def merge_registry(run_level: pd.DataFrame, registry: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Left-merge registry fields onto a run-level DataFrame (agent, run)."""
    if registry is None or run_level.empty:
        return run_level
    keep = ["agent", "run"] + [c for c in _BOOL_COLS if c in registry.columns] + (
        ["notes"] if "notes" in registry.columns else [])
    return run_level.merge(registry[keep], on=["agent", "run"], how="left")


def fairness_sensitivity_summary(
    run_scores: pd.DataFrame,
    registry: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Summarise scores stratified by information-access status.
    Returns an empty DataFrame if no registry is available.
    """
    if registry is None or run_scores.empty:
        return pd.DataFrame()

    merged = merge_registry(run_scores, registry)
    rows = []

    for agent in sorted(merged["agent"].unique()):
        adf = merged[merged["agent"] == agent]
        interactive = bool(adf["interactive_mode"].fillna(False).any()) \
            if "interactive_mode" in adf.columns else False
        asked = bool(adf["asked_additional_questions"].fillna(False).any()) \
            if "asked_additional_questions" in adf.columns else False
        provided = bool(adf["additional_metadata_provided"].fillna(False).any()) \
            if "additional_metadata_provided" in adf.columns else False
        rows.append({
            "agent": agent,
            "interactive_mode_any": interactive,
            "asked_additional_questions_any": asked,
            "additional_metadata_provided_any": provided,
            "n_runs": len(adf),
            "mean_AgentScore": round(float(adf["AgentScore"].mean()), 3),
            "sd_AgentScore": round(float(adf["AgentScore"].std(ddof=1)) if len(adf) > 1 else 0.0, 3),
        })

    summary = pd.DataFrame(rows)

    # Group-level contrast: metadata-provided vs not
    if "additional_metadata_provided" in merged.columns:
        grp = merged.groupby(merged["additional_metadata_provided"].fillna(False))["AgentScore"]
        contrast = grp.agg(["mean", "std", "count"]).reset_index()
        contrast.rename(columns={
            "additional_metadata_provided": "metadata_provided",
            "mean": "mean_AgentScore", "std": "sd_AgentScore", "count": "n_runs",
        }, inplace=True)
        contrast["sd_AgentScore"] = contrast["sd_AgentScore"].fillna(0.0)
        contrast["agent"] = "__GROUP_CONTRAST__"
        contrast["interactive_mode_any"] = np.nan
        contrast["asked_additional_questions_any"] = np.nan
        contrast["additional_metadata_provided_any"] = contrast["metadata_provided"]
        contrast = contrast[[c for c in summary.columns if c in contrast.columns]
                            + ["metadata_provided"]]
        summary = pd.concat([summary, contrast], ignore_index=True)

    return summary
