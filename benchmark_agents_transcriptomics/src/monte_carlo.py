"""
monte_carlo.py — bootstrap centroid and regularised Gaussian satellite
generation around the 8 real run vectors of each agent.

Synthetic points are uncertainty probes, NOT real observations.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .dimensionality import reduce_for_mc

logger = logging.getLogger(__name__)

SEED = 42


# ---------------------------------------------------------------------------
# Covariance regularisation
# ---------------------------------------------------------------------------

def _regularise_cov(cov: np.ndarray, alpha: float = 1e-4) -> np.ndarray:
    """Add ridge regularisation to make covariance invertible."""
    return cov + alpha * np.eye(cov.shape[0])


# ---------------------------------------------------------------------------
# Bootstrap centroid sampling
# ---------------------------------------------------------------------------

def bootstrap_centroids(
    run_vecs: np.ndarray,   # shape (n_runs, D)
    n_mc: int = 10_000,
    seed: int = SEED,
) -> np.ndarray:
    """
    Resample run vectors with replacement n_mc times,
    compute the centroid each time.

    Returns array of shape (n_mc, D).
    """
    rng = np.random.default_rng(seed)
    n = run_vecs.shape[0]
    centroids = np.empty((n_mc, run_vecs.shape[1]), dtype=np.float32)
    for i in range(n_mc):
        idx = rng.integers(0, n, size=n)
        centroids[i] = run_vecs[idx].mean(axis=0)
    return centroids


# ---------------------------------------------------------------------------
# Regularised multivariate Gaussian
# ---------------------------------------------------------------------------

def gaussian_satellites(
    run_vecs: np.ndarray,   # shape (n_runs, D)
    n_mc: int = 10_000,
    seed: int = SEED,
    variance_threshold: float = 0.95,
) -> np.ndarray:
    """
    Generate n_mc synthetic points from a regularised Gaussian fitted
    to run_vecs.  If dimensionality > n_runs-1, first reduce via PCA.

    Returns array of shape (n_mc, D_original).
    """
    rng = np.random.default_rng(seed)
    n_runs, D = run_vecs.shape

    if D >= n_runs:
        # Reduce to PCA space
        reduced, pca = reduce_for_mc(run_vecs.astype(np.float64),
                                     variance_threshold=variance_threshold)
        mu  = reduced.mean(axis=0)
        cov = np.cov(reduced, rowvar=False) if reduced.shape[0] > 1 else np.eye(reduced.shape[1]) * 1e-6
        cov = _regularise_cov(np.atleast_2d(cov))

        samples_low = rng.multivariate_normal(mu, cov, size=n_mc)
        # Project back to original space via inverse PCA transform
        # pca.components_ shape: (k, D_original)
        k = samples_low.shape[1]
        components = pca.components_[:k]
        mean_vec = pca.mean_
        samples = samples_low @ components + mean_vec
    else:
        mu  = run_vecs.mean(axis=0).astype(np.float64)
        cov = np.cov(run_vecs.astype(np.float64), rowvar=False) if n_runs > 1 else np.eye(D) * 1e-6
        cov = _regularise_cov(np.atleast_2d(cov))
        samples = rng.multivariate_normal(mu, cov, size=n_mc)

    return samples.astype(np.float32)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def run_monte_carlo(
    run_embs: Dict[Tuple[str, int], np.ndarray],
    n_mc: int = 10_000,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    For each agent, generate bootstrap centroids and Gaussian satellites.

    Returns
    -------
    centroid_df : DataFrame with all MC centroid rows (agent, mc_type, PC1, PC2, PC3)
    summary_df  : per-agent summary statistics
    boot_by_agent : {agent: (n_mc, D) bootstrap centroids}
    gauss_by_agent: {agent: (n_mc, D) Gaussian satellites}
    """
    # Group run vecs by agent
    agent_vecs: Dict[str, List[np.ndarray]] = {}
    for (agent, run), vec in run_embs.items():
        agent_vecs.setdefault(agent, []).append(vec)

    boot_by_agent:  Dict[str, np.ndarray] = {}
    gauss_by_agent: Dict[str, np.ndarray] = {}
    centroid_rows: list = []
    summary_rows:  list = []

    for agent in sorted(agent_vecs.keys()):
        vecs = np.stack(agent_vecs[agent], axis=0).astype(np.float32)
        logger.info("Monte Carlo for %s: %d real runs, %d iterations.", agent, len(vecs), n_mc)

        boot  = bootstrap_centroids(vecs, n_mc=n_mc, seed=seed)
        gauss = gaussian_satellites(vecs, n_mc=n_mc, seed=seed)

        boot_by_agent[agent]  = boot
        gauss_by_agent[agent] = gauss

        # Summary
        for arr, mc_type in [(boot, "bootstrap"), (gauss, "gaussian")]:
            mu  = arr.mean(axis=0)
            sd  = arr.std(axis=0)
            ci_lo = np.percentile(arr, 2.5, axis=0)
            ci_hi = np.percentile(arr, 97.5, axis=0)
            summary_rows.append({
                "agent":    agent,
                "mc_type":  mc_type,
                "n_mc":     n_mc,
                "mean_norm":    float(np.linalg.norm(mu)),
                "mean_sd":      float(sd.mean()),
                "ci95_lo_norm": float(np.linalg.norm(ci_lo)),
                "ci95_hi_norm": float(np.linalg.norm(ci_hi)),
            })

        # Store first 3 PCA-projected dims for plotting
        # Reduce boot + gauss together for consistent axes
        combined   = np.vstack([boot, gauss])
        from sklearn.decomposition import PCA as _PCA
        n_comp = min(3, combined.shape[1], combined.shape[0] - 1)
        if n_comp < 1:
            n_comp = 1
        pca_vis = _PCA(n_components=n_comp, random_state=SEED)
        combined_red = pca_vis.fit_transform(combined)

        boot_red  = combined_red[:n_mc]
        gauss_red = combined_red[n_mc:]

        for i, row in enumerate(boot_red):
            r = {"agent": agent, "mc_type": "bootstrap"}
            for d in range(n_comp):
                r[f"PC{d+1}"] = float(row[d])
            centroid_rows.append(r)

        for i, row in enumerate(gauss_red):
            r = {"agent": agent, "mc_type": "gaussian"}
            for d in range(n_comp):
                r[f"PC{d+1}"] = float(row[d])
            centroid_rows.append(r)

    centroid_df = pd.DataFrame(centroid_rows)
    summary_df  = pd.DataFrame(summary_rows)

    return centroid_df, summary_df, boot_by_agent, gauss_by_agent
