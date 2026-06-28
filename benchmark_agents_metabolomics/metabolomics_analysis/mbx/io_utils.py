"""Data loading and sample/group assignment."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config


@dataclass
class Dataset:
    """A loaded metabolomics dataset.

    `data` is a feature x sample matrix (rows = metabolites, cols = samples)
    holding raw intensities. `sample_groups` maps every biological-sample column
    to its group label, and `pairs` (paired designs only) lists matched
    (control_col, case_col) tuples.
    """
    data: pd.DataFrame
    qc_cols: list[str]
    group_cols: dict[str, list[str]]      # label -> sample columns
    sample_groups: pd.Series              # sample col -> group label
    pairs: list[tuple[str, str]]
    cfg: Config

    @property
    def bio_cols(self) -> list[str]:
        cols: list[str] = []
        for g in self.group_cols.values():
            cols.extend(g)
        return cols


def _trailing_index(name: str) -> str | None:
    m = re.search(r"(\d+)\s*$", str(name))
    return m.group(1) if m else None


def load_dataset(cfg: Config) -> Dataset:
    """Read the intensity table and assign columns to QC / experimental groups."""
    path = cfg.input_path
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name=cfg.sheet)
    else:
        df = pd.read_csv(path)

    if cfg.feature_col not in df.columns:
        raise ValueError(
            f"feature column '{cfg.feature_col}' not found; columns: {list(df.columns)[:5]}..."
        )

    # De-duplicate feature names (some HILIC tables repeat names with suffixes).
    df[cfg.feature_col] = df[cfg.feature_col].astype(str)
    df = df[df[cfg.feature_col].str.lower() != "nan"]
    df = _dedupe_feature_names(df, cfg.feature_col)
    df = df.set_index(cfg.feature_col)

    # Coerce all sample columns to numeric.
    df = df.apply(pd.to_numeric, errors="coerce")

    cols = list(df.columns)
    qc_cols = [c for c in cols if cfg.qc_prefix.lower() in str(c).lower()]

    ctrl_pre, case_pre = cfg.group_prefixes
    ctrl_label, case_label = cfg.group_labels
    ctrl_cols = [c for c in cols if str(c).lower().startswith(ctrl_pre.lower())]
    case_cols = [c for c in cols if str(c).lower().startswith(case_pre.lower())]

    group_cols = {ctrl_label: ctrl_cols, case_label: case_cols}
    sample_groups = pd.Series(
        {**{c: ctrl_label for c in ctrl_cols}, **{c: case_label for c in case_cols}}
    )

    pairs: list[tuple[str, str]] = []
    if cfg.paired:
        ctrl_by_idx = {_trailing_index(c): c for c in ctrl_cols}
        case_by_idx = {_trailing_index(c): c for c in case_cols}
        for idx in sorted(set(ctrl_by_idx) & set(case_by_idx), key=lambda x: int(x)):
            pairs.append((ctrl_by_idx[idx], case_by_idx[idx]))

    return Dataset(
        data=df,
        qc_cols=qc_cols,
        group_cols=group_cols,
        sample_groups=sample_groups,
        pairs=pairs,
        cfg=cfg,
    )


def _dedupe_feature_names(df: pd.DataFrame, col: str) -> pd.DataFrame:
    names = df[col].tolist()
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}#{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    df = df.copy()
    df[col] = out
    return df


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
