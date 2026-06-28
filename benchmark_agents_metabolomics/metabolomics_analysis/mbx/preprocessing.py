"""Preprocessing: QC filtering, missing-value handling, normalization, transform.

The output of `preprocess()` is a `Processed` bundle holding several matrices at
different stages so that downstream steps can choose the appropriate one:

    raw_bio       feature x bio-sample, raw intensities (post feature filtering)
    normalized    after sample normalization + imputation (linear scale)
    transformed   after log transform (used for univariate statistics)
    scaled        after per-feature scaling (used for PCA / PLS-DA)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .io_utils import Dataset


@dataclass
class Processed:
    raw_bio: pd.DataFrame
    normalized: pd.DataFrame
    transformed: pd.DataFrame
    scaled: pd.DataFrame
    qc_data: pd.DataFrame
    feature_qc: pd.DataFrame      # per-feature QC metrics
    steps: list[str]
    cfg: Config


def coefficient_of_variation(df: pd.DataFrame) -> pd.Series:
    """Per-feature CV (std/mean) computed row-wise, ignoring NaNs."""
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = std / mean
    return cv.replace([np.inf, -np.inf], np.nan)


def _impute(df: pd.DataFrame, method: str) -> pd.DataFrame:
    out = df.copy()
    if method in ("half_min", "min"):
        factor = 0.5 if method == "half_min" else 1.0
        row_min = out.min(axis=1, skipna=True)
        for feat in out.index:
            fill = row_min[feat] * factor
            if not np.isfinite(fill):
                fill = 0.0
            out.loc[feat] = out.loc[feat].fillna(fill)
    elif method == "knn":
        from sklearn.impute import KNNImputer
        imp = KNNImputer(n_neighbors=5)
        arr = imp.fit_transform(out.T.values).T
        out = pd.DataFrame(arr, index=out.index, columns=out.columns)
    else:
        raise ValueError(f"unknown impute method: {method}")
    return out


def _pqn_normalize(df: pd.DataFrame, reference: pd.Series) -> pd.DataFrame:
    """Probabilistic Quotient Normalization (Dieterle et al. 2006).

    df: feature x sample (linear, positive). reference: per-feature reference
    profile (e.g. median across all samples).
    """
    ref = reference.replace(0, np.nan)
    quotients = df.div(ref, axis=0)               # feature x sample
    factors = quotients.median(axis=0, skipna=True)
    factors = factors.replace(0, np.nan).fillna(1.0)
    return df.div(factors, axis=1)


def _normalize(df: pd.DataFrame, method: str | None) -> tuple[pd.DataFrame, pd.Series]:
    if method is None:
        return df, pd.Series(1.0, index=df.columns)
    if method == "sum":
        factors = df.sum(axis=0)
        factors = factors / factors.median()
        return df.div(factors, axis=1), factors
    if method == "median":
        factors = df.median(axis=0)
        factors = factors / factors.median()
        return df.div(factors, axis=1), factors
    if method == "pqn":
        reference = df.median(axis=1)
        out = _pqn_normalize(df, reference)
        factors = (df / out).median(axis=0)
        return out, factors
    raise ValueError(f"unknown normalization: {method}")


def _transform(df: pd.DataFrame, method: str | None) -> pd.DataFrame:
    if method is None:
        return df
    shifted = df.clip(lower=0) + 1.0   # +1 offset keeps zeros finite
    if method == "log2":
        return np.log2(shifted)
    if method == "log10":
        return np.log10(shifted)
    if method == "ln":
        return np.log(shifted)
    raise ValueError(f"unknown transform: {method}")


def scale_features(df: pd.DataFrame, method: str | None) -> pd.DataFrame:
    """Scale rows (features). Operates on the transformed matrix."""
    if method is None:
        return df
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=1).replace(0, np.nan)
    centered = df.sub(mean, axis=0)
    if method == "auto":
        out = centered.div(std, axis=0)
    elif method == "pareto":
        out = centered.div(np.sqrt(std), axis=0)
    elif method == "range":
        rng = (df.max(axis=1) - df.min(axis=1)).replace(0, np.nan)
        out = centered.div(rng, axis=0)
    else:
        raise ValueError(f"unknown scaling: {method}")
    return out.fillna(0.0)


def preprocess(ds: Dataset) -> Processed:
    cfg = ds.cfg
    steps: list[str] = []

    bio = ds.data[ds.bio_cols].copy()
    qc = ds.data[ds.qc_cols].copy() if ds.qc_cols else pd.DataFrame(index=ds.data.index)

    n_feat0 = bio.shape[0]

    # 1) Drop features that are entirely missing.
    all_missing = bio.isna().all(axis=1)
    bio = bio[~all_missing]
    qc = qc.reindex(bio.index)
    steps.append(f"Removed {int(all_missing.sum())} all-missing features "
                 f"({n_feat0} -> {bio.shape[0]}).")

    # 2) Missing-value filter on biological samples.
    miss_frac = bio.isna().mean(axis=1)
    keep_missing = miss_frac <= cfg.max_missing_frac
    n_drop_missing = int((~keep_missing).sum())
    bio = bio[keep_missing]
    qc = qc.reindex(bio.index)
    steps.append(f"Removed {n_drop_missing} features with >"
                 f"{cfg.max_missing_frac:.0%} missingness "
                 f"(-> {bio.shape[0]}).")

    # 3) QC CV filter (analytical reproducibility).
    qc_cv = coefficient_of_variation(qc) if qc.shape[1] >= 2 else pd.Series(np.nan, index=bio.index)
    if cfg.qc_cv_threshold is not None and qc.shape[1] >= 2:
        keep_cv = (qc_cv <= cfg.qc_cv_threshold) | qc_cv.isna()
        n_drop_cv = int((~keep_cv).sum())
        bio = bio[keep_cv]
        qc = qc.reindex(bio.index)
        qc_cv = qc_cv.reindex(bio.index)
        steps.append(f"Removed {n_drop_cv} features with QC CV>"
                     f"{cfg.qc_cv_threshold:.0%} (-> {bio.shape[0]}).")
    else:
        steps.append("QC CV filter skipped (insufficient QC samples or disabled).")

    # 4) Sample normalization (linear scale, before imputation reference).
    bio_imp = _impute(bio, cfg.impute_method)
    steps.append(f"Imputed missing values with '{cfg.impute_method}'.")
    bio_norm, _ = _normalize(bio_imp, cfg.normalization)
    steps.append(f"Sample normalization: '{cfg.normalization}'.")

    # 5) Transform + scale.
    bio_trans = _transform(bio_norm, cfg.transform)
    steps.append(f"Transform: '{cfg.transform}'.")
    bio_scaled = scale_features(bio_trans, cfg.scaling)
    steps.append(f"Scaling (multivariate only): '{cfg.scaling}'.")

    # Per-feature QC table.
    feature_qc = pd.DataFrame({
        "qc_cv": qc_cv,
        "missing_frac": bio.isna().mean(axis=1),
        "mean_intensity": bio.mean(axis=1),
    })

    return Processed(
        raw_bio=bio,
        normalized=bio_norm,
        transformed=bio_trans,
        scaled=bio_scaled,
        qc_data=qc,
        feature_qc=feature_qc,
        steps=steps,
        cfg=cfg,
    )
