"""
similarity.py — compute pairwise similarity at file, run, and agent level.

Metrics:
  - Cosine similarity of run-level embeddings
  - Euclidean distance of standardised structural features
  - Jaccard similarity of methodological keyword sets
  - Biological term overlap (Jaccard)
"""

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean
from sklearn.preprocessing import StandardScaler

from .features import keyword_set, bio_term_set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cosine similarity helpers
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [−1, 1] (1 = identical)."""
    if a is None or b is None:
        return float("nan")
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    inter = set_a & set_b
    return len(inter) / len(union)


# ---------------------------------------------------------------------------
# Run-level pairwise similarity
# ---------------------------------------------------------------------------

def pairwise_run_similarity(
    run_embs: Dict[Tuple[str, int], np.ndarray],
    run_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute pairwise similarity between all (agent, run) pairs.

    Returns a DataFrame with columns:
        agent_a, run_a, agent_b, run_b,
        cosine_sim, euclidean_dist, jaccard_kw, jaccard_bio
    """
    keys = sorted(run_embs.keys())
    if not keys:
        logger.warning("No run embeddings — pairwise similarity is empty.")
        return pd.DataFrame()

    # Standardise structural features for Euclidean distance
    numeric_cols = [c for c in run_features.columns
                    if c not in {"agent", "run"} and run_features[c].dtype != object]
    feat_df = run_features[["agent", "run"] + numeric_cols].copy()
    if not feat_df.empty and numeric_cols:
        scaler = StandardScaler()
        feat_df[numeric_cols] = scaler.fit_transform(feat_df[numeric_cols].fillna(0))

    def _get_feat_row(agent: str, run: int) -> Optional[dict]:
        mask = (feat_df["agent"] == agent) & (feat_df["run"] == run)
        rows = feat_df[mask]
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()

    rows = []
    for (a1, r1), (a2, r2) in combinations(keys, 2):
        cs = cosine_sim(run_embs[(a1, r1)], run_embs[(a2, r2)])

        feat1 = _get_feat_row(a1, r1)
        feat2 = _get_feat_row(a2, r2)

        if feat1 and feat2:
            v1 = np.array([feat1.get(c, 0) for c in numeric_cols], dtype=float)
            v2 = np.array([feat2.get(c, 0) for c in numeric_cols], dtype=float)
            eu = float(np.linalg.norm(v1 - v2))
            jk = jaccard(keyword_set(feat1), keyword_set(feat2))
            jb = jaccard(bio_term_set(feat1), bio_term_set(feat2))
        else:
            eu = float("nan")
            jk = float("nan")
            jb = float("nan")

        rows.append({
            "agent_a": a1, "run_a": r1,
            "agent_b": a2, "run_b": r2,
            "cosine_sim": cs,
            "euclidean_dist": eu,
            "jaccard_kw": jk,
            "jaccard_bio": jb,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Within-agent reproducibility
# ---------------------------------------------------------------------------

def within_agent_reproducibility(pairwise: pd.DataFrame) -> pd.DataFrame:
    """
    For each agent, compute reproducibility statistics from intra-agent
    pairwise similarities.

    Returns a DataFrame with one row per agent.
    """
    if pairwise.empty:
        return pd.DataFrame()

    records = []
    agents = sorted(set(pairwise["agent_a"]) | set(pairwise["agent_b"]))
    for agent in agents:
        mask = (pairwise["agent_a"] == agent) & (pairwise["agent_b"] == agent)
        sims = pairwise[mask]["cosine_sim"].dropna().values

        if len(sims) == 0:
            continue

        mean_s = float(np.mean(sims))
        std_s  = float(np.std(sims, ddof=1)) if len(sims) > 1 else 0.0
        cv_s   = std_s / mean_s * 100 if mean_s != 0 else float("nan")
        repro  = max(0, 1 - cv_s / 100)

        records.append({
            "agent":           agent,
            "n_pairs":         len(sims),
            "mean_sim":        mean_s,
            "std_sim":         std_s,
            "cv_pct":          cv_s,
            "min_sim":         float(np.min(sims)),
            "max_sim":         float(np.max(sims)),
            "reproducibility": repro,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Between-agent comparison
# ---------------------------------------------------------------------------

def between_agent_similarity(
    run_embs: Dict[Tuple[str, int], np.ndarray],
    agent_embs: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Compare each pair of agents:
      ChatGPT vs Biomni, ChatGPT vs KDense, Biomni vs KDense
    """
    agents = sorted(agent_embs.keys())
    rows = []
    for a1, a2 in combinations(agents, 2):
        cs = cosine_sim(agent_embs.get(a1), agent_embs.get(a2))

        # Average cosine between all runs of a1 and all runs of a2
        run_pairs = []
        for (ag, rn), vec in run_embs.items():
            if ag == a1:
                for (ag2, rn2), vec2 in run_embs.items():
                    if ag2 == a2:
                        run_pairs.append(cosine_sim(vec, vec2))

        mean_cross = float(np.mean(run_pairs)) if run_pairs else float("nan")

        rows.append({
            "agent_a":           a1,
            "agent_b":           a2,
            "centroid_cosine_sim": cs,
            "mean_run_cross_sim":  mean_cross,
        })

    return pd.DataFrame(rows)
