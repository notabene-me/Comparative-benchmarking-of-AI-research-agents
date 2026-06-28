"""Multivariate analysis: PCA and PLS-DA with VIP scores."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut

from .config import Config
from .io_utils import Dataset
from .preprocessing import Processed


@dataclass
class MultivariateResult:
    pca_scores: pd.DataFrame
    pca_explained: np.ndarray
    pls_scores: pd.DataFrame
    vip: pd.Series
    pls_q2: float
    pls_r2: float
    labels: pd.Series


def _design_matrix(ds: Dataset, proc: Processed, include_qc: bool):
    cols = list(proc.scaled.columns)
    if include_qc and ds.qc_cols:
        qc_scaled = _scale_like(proc, ds.qc_cols)
        X = pd.concat([proc.scaled, qc_scaled], axis=1).T
        labels = pd.Series(
            {**ds.sample_groups.to_dict(), **{c: "QC" for c in ds.qc_cols}}
        ).reindex(X.index)
    else:
        X = proc.scaled.T
        labels = ds.sample_groups.reindex(X.index)
    return X.fillna(0.0), labels


def _scale_like(proc: Processed, qc_cols):
    """Project QC samples into the same per-feature scaling used for biology."""
    from .preprocessing import _transform
    cfg = proc.cfg
    qc = proc.qc_data.reindex(proc.transformed.index)
    qc_norm = qc  # QC already on raw scale; transform only for projection
    qc_trans = _transform(qc_norm.fillna(qc_norm.min(axis=1).fillna(0.0)), cfg.transform)
    mean = proc.transformed.mean(axis=1)
    std = proc.transformed.std(axis=1, ddof=1).replace(0, np.nan)
    centered = qc_trans.sub(mean, axis=0)
    if cfg.scaling == "auto":
        out = centered.div(std, axis=0)
    elif cfg.scaling == "pareto":
        out = centered.div(np.sqrt(std), axis=0)
    else:
        out = centered
    return out.fillna(0.0)


def _vip_scores(pls: PLSRegression, X: np.ndarray) -> np.ndarray:
    """Variable Importance in Projection for a fitted PLS model."""
    t = pls.x_scores_           # n x A
    w = pls.x_weights_          # p x A
    q = pls.y_loadings_         # 1 x A
    p_feat, A = w.shape
    ssy = np.array([(q[0, a] ** 2) * (t[:, a] @ t[:, a]) for a in range(A)])
    total_ssy = ssy.sum()
    if total_ssy == 0:
        return np.zeros(p_feat)
    vip = np.zeros(p_feat)
    w_norm = w / np.linalg.norm(w, axis=0, keepdims=True)
    for j in range(p_feat):
        weight = np.array([(w_norm[j, a] ** 2) * ssy[a] for a in range(A)]).sum()
        vip[j] = np.sqrt(p_feat * weight / total_ssy)
    return vip


def run_multivariate(ds: Dataset, proc: Processed) -> MultivariateResult:
    cfg: Config = ds.cfg
    rng = cfg.seed

    # ---- PCA (include QC to visualise analytical clustering) ----
    X_all, labels_all = _design_matrix(ds, proc, include_qc=True)
    n_comp = min(5, X_all.shape[0] - 1, X_all.shape[1])
    pca = PCA(n_components=n_comp, random_state=rng)
    scores = pca.fit_transform(X_all.values)
    pca_scores = pd.DataFrame(
        scores, index=X_all.index,
        columns=[f"PC{i+1}" for i in range(n_comp)],
    )

    # ---- PLS-DA (biological samples only) ----
    X_bio, labels_bio = _design_matrix(ds, proc, include_qc=False)
    ctrl_label, case_label = cfg.group_labels
    y = (labels_bio == case_label).astype(float).values

    A = min(cfg.pls_components, X_bio.shape[0] - 1, X_bio.shape[1])
    pls = PLSRegression(n_components=A, scale=False)
    pls.fit(X_bio.values, y)
    pls_scores = pd.DataFrame(
        pls.x_scores_, index=X_bio.index,
        columns=[f"LV{i+1}" for i in range(A)],
    )
    vip = pd.Series(_vip_scores(pls, X_bio.values), index=X_bio.columns,
                    name="VIP").sort_values(ascending=False)

    # ---- Cross-validated Q2 (leave-one-out) ----
    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    for tr, te in loo.split(X_bio.values):
        m = PLSRegression(n_components=A, scale=False)
        m.fit(X_bio.values[tr], y[tr])
        preds[te] = m.predict(X_bio.values[te]).ravel()
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    q2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    r2 = float(pls.score(X_bio.values, y))

    return MultivariateResult(
        pca_scores=pca_scores,
        pca_explained=pca.explained_variance_ratio_,
        pls_scores=pls_scores,
        vip=vip,
        pls_q2=q2,
        pls_r2=r2,
        labels=labels_all,
    )
