"""
dimensionality.py — PCA, MDS, t-SNE, UMAP on run-level embedding matrix.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

SEED = 42

try:
    import umap
    _HAS_UMAP = True
except Exception as _umap_exc:  # ImportError, RuntimeError (numba), etc.
    _HAS_UMAP = False
    logger.debug("umap-learn unavailable (%s); UMAP will be skipped.", _umap_exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_df(
    coords: np.ndarray,
    keys: List[Tuple[str, int]],
    col_prefix: str,
    n_cols: int,
) -> pd.DataFrame:
    cols = [f"{col_prefix}{i+1}" for i in range(n_cols)]
    df = pd.DataFrame(coords[:, :n_cols], columns=cols)
    df.insert(0, "run",   [k[1] for k in keys])
    df.insert(0, "agent", [k[0] for k in keys])
    return df


def _safe_scale(mat: np.ndarray) -> np.ndarray:
    """StandardScale but handle zero-variance columns."""
    scaler = StandardScaler()
    return scaler.fit_transform(mat)


# ---------------------------------------------------------------------------
# Individual reductions
# ---------------------------------------------------------------------------

def run_pca(
    mat: np.ndarray,
    keys: List[Tuple[str, int]],
    n_components: int = 3,
) -> Tuple[pd.DataFrame, PCA]:
    """PCA on run matrix. Returns (DataFrame with PC1..PCn, fitted PCA)."""
    n_samples, n_feats = mat.shape
    n_comp = min(n_components, n_samples, n_feats)
    pca = PCA(n_components=n_comp, random_state=SEED)
    coords = pca.fit_transform(_safe_scale(mat))
    df = _label_df(coords, keys, "PC", n_comp)
    evr = pca.explained_variance_ratio_
    logger.info("PCA: %.1f%% variance in %d components",
                sum(evr[:n_comp]) * 100, n_comp)
    return df, pca


def run_mds(
    mat: np.ndarray,
    keys: List[Tuple[str, int]],
    n_components: int = 2,
) -> pd.DataFrame:
    """Metric MDS."""
    n_samples = mat.shape[0]
    n_comp = min(n_components, n_samples - 1)
    mds = MDS(n_components=n_comp, random_state=SEED, normalized_stress="auto")
    coords = mds.fit_transform(_safe_scale(mat))
    return _label_df(coords, keys, "MDS", n_comp)


def run_tsne(
    mat: np.ndarray,
    keys: List[Tuple[str, int]],
    n_components: int = 2,
    perplexity: float = 5.0,
) -> Optional[pd.DataFrame]:
    """t-SNE. Returns None if matrix too small."""
    n_samples = mat.shape[0]
    if n_samples < 4:
        logger.warning("t-SNE requires ≥4 samples; skipping.")
        return None
    perp = min(perplexity, (n_samples - 1) / 3)
    perp = max(1.0, perp)
    try:
        tsne = TSNE(n_components=n_components, random_state=SEED,
                    perplexity=perp, max_iter=1000, init="random")
        coords = tsne.fit_transform(_safe_scale(mat))
        return _label_df(coords, keys, "TSNE", n_components)
    except Exception as exc:
        logger.warning("t-SNE failed: %s", exc)
        return None


def run_umap(
    mat: np.ndarray,
    keys: List[Tuple[str, int]],
    n_components: int = 2,
    n_neighbors: int = 5,
) -> Optional[pd.DataFrame]:
    """UMAP (requires umap-learn)."""
    if not _HAS_UMAP:
        logger.info("umap-learn not available; UMAP skipped.")
        return None
    n_samples = mat.shape[0]
    nn = min(n_neighbors, n_samples - 1)
    try:
        reducer = umap.UMAP(n_components=n_components, n_neighbors=nn,
                            random_state=SEED)
        coords = reducer.fit_transform(_safe_scale(mat))
        return _label_df(coords, keys, "UMAP", n_components)
    except Exception as exc:
        logger.warning("UMAP failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# PCA in reduced space for Monte Carlo (high-dim → k dims)
# ---------------------------------------------------------------------------

def reduce_for_mc(
    mat: np.ndarray,
    variance_threshold: float = 0.95,
) -> Tuple[np.ndarray, PCA]:
    """
    Reduce high-dimensional run matrix to a lower-dimensional PCA space
    that captures at least *variance_threshold* of variance.
    Used for Monte Carlo covariance simulation.
    """
    n_samples, n_feats = mat.shape
    max_comp = min(n_samples - 1, n_feats, 50)
    if max_comp < 1:
        max_comp = 1

    pca = PCA(n_components=max_comp, random_state=SEED)
    coords = pca.fit_transform(_safe_scale(mat))

    evr_cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(evr_cum, variance_threshold)) + 1
    k = min(k, max_comp)
    k = max(k, 1)

    logger.info("PCA for MC: %d components → %.1f%% variance",
                k, evr_cum[k - 1] * 100)
    return coords[:, :k], pca
