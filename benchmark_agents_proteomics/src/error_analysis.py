"""
error_analysis.py — bootstrap-based error estimation and uncertainty.

All uncertainty estimates are DESCRIPTIVE: they quantify variability induced
by the observed set of repeated agent runs (only 8 real runs per agent).
They are NOT population-level confidence intervals for all possible agent
behaviours. Monte Carlo satellites are uncertainty probes, not real data.
"""

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEED = 42
PRACTICAL_EQUIV = 2.0  # |Δ| < 2 score points = practically equivalent


# ---------------------------------------------------------------------------
# Core per-agent error statistics
# ---------------------------------------------------------------------------

def agent_error_stats(values: np.ndarray, n_boot: int, seed: int = SEED) -> dict:
    """
    Descriptive + bootstrap error statistics for a 1-D array of run values.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return {k: float("nan") for k in (
            "mean", "sd", "sem", "median", "iqr", "mad",
            "robust_absolute_error", "bootstrap_mean", "bootstrap_sd",
            "ci95_lo", "ci95_hi", "absolute_error", "relative_error_pct",
            "cv_pct")} | {"n_runs": 0, "n_bootstrap": n_boot}

    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
    sem = sd / np.sqrt(n) if n > 0 else 0.0
    median = float(np.median(values))
    q1, q3 = np.percentile(values, [25, 75])
    iqr = float(q3 - q1)
    mad = float(np.median(np.abs(values - median)))
    robust_abs_err = 1.4826 * mad / np.sqrt(n) if n > 0 else float("nan")

    rng = np.random.default_rng(seed)
    boot_means = np.array([
        np.mean(rng.choice(values, size=n, replace=True)) for _ in range(n_boot)
    ]) if n > 0 else np.array([])

    boot_mean = float(np.mean(boot_means))
    boot_sd = float(np.std(boot_means, ddof=1)) if len(boot_means) > 1 else 0.0
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    abs_err = boot_sd  # primary absolute error = bootstrap SD of the mean
    rel_err = abs_err / mean * 100 if mean != 0 else float("nan")
    cv = sd / mean * 100 if mean != 0 else float("nan")

    return {
        "mean": round(mean, 4),
        "sd": round(sd, 4),
        "sem": round(sem, 4),
        "median": round(median, 4),
        "iqr": round(iqr, 4),
        "mad": round(mad, 4),
        "robust_absolute_error": round(robust_abs_err, 4),
        "bootstrap_mean": round(boot_mean, 4),
        "bootstrap_sd": round(boot_sd, 4),
        "ci95_lo": round(ci_lo, 4),
        "ci95_hi": round(ci_hi, 4),
        "absolute_error": round(abs_err, 4),
        "relative_error_pct": round(rel_err, 4) if not np.isnan(rel_err) else float("nan"),
        "cv_pct": round(cv, 4) if not np.isnan(cv) else float("nan"),
        "n_runs": n,
        "n_bootstrap": n_boot,
    }


# ---------------------------------------------------------------------------
# 1. Score error summary across scoring definitions
# ---------------------------------------------------------------------------

def score_error_summary(
    definitions: Dict[str, Tuple[pd.DataFrame, str]],
    n_boot: int,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    definitions: {definition_label: (run_level_df, value_column)}
    Returns one row per (definition, agent).
    """
    rows = []
    for label, (df, col) in definitions.items():
        if df is None or df.empty or col not in df.columns:
            continue
        for agent in sorted(df["agent"].unique()):
            vals = df[df["agent"] == agent][col].to_numpy(dtype=float)
            stats = agent_error_stats(vals, n_boot, seed)
            stats.update({"definition": label, "agent": agent})
            rows.append(stats)
    if not rows:
        return pd.DataFrame()
    cols = ["definition", "agent"] + [c for c in rows[0] if c not in {"definition", "agent"}]
    return pd.DataFrame(rows)[cols]


def bootstrap_error_summary(
    run_scores: pd.DataFrame,
    n_boot: int,
    seed: int = SEED,
    score_col: str = "AgentScore",
) -> pd.DataFrame:
    """Focused bootstrap error table for the primary AgentScore."""
    if run_scores.empty:
        return pd.DataFrame()
    rows = []
    for agent in sorted(run_scores["agent"].unique()):
        vals = run_scores[run_scores["agent"] == agent][score_col].to_numpy(dtype=float)
        stats = agent_error_stats(vals, n_boot, seed)
        stats["agent"] = agent
        rows.append(stats)
    cols = ["agent"] + [c for c in rows[0] if c != "agent"]
    return pd.DataFrame(rows)[cols]


# ---------------------------------------------------------------------------
# 2. Pairwise score difference error
# ---------------------------------------------------------------------------

def pairwise_score_difference_error(
    run_scores: pd.DataFrame,
    n_boot: int,
    seed: int = SEED,
    score_col: str = "AgentScore",
) -> pd.DataFrame:
    """Bootstrap error of pairwise agent score differences."""
    if run_scores.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    agents = sorted(run_scores["agent"].unique())
    by_agent = {a: run_scores[run_scores["agent"] == a][score_col].to_numpy(dtype=float)
                for a in agents}

    rows = []
    for a1, a2 in combinations(agents, 2):
        v1, v2 = by_agent[a1], by_agent[a2]
        n1, n2 = len(v1), len(v2)
        if n1 == 0 or n2 == 0:
            continue
        obs_diff = float(np.mean(v1) - np.mean(v2))
        boot_diffs = np.array([
            np.mean(rng.choice(v1, n1, replace=True)) -
            np.mean(rng.choice(v2, n2, replace=True))
            for _ in range(n_boot)
        ])
        sd_diff = float(np.std(boot_diffs, ddof=1)) if len(boot_diffs) > 1 else 0.0
        ci_lo = float(np.percentile(boot_diffs, 2.5))
        ci_hi = float(np.percentile(boot_diffs, 97.5))
        p_gt0 = float(np.mean(boot_diffs > 0))
        p_equiv = float(np.mean(np.abs(boot_diffs) < PRACTICAL_EQUIV))
        rel_err = sd_diff / abs(obs_diff) * 100 if abs(obs_diff) > 1e-9 else float("nan")
        rows.append({
            "agent_a": a1, "agent_b": a2,
            "mean_difference": round(obs_diff, 4),
            "bootstrap_sd_difference": round(sd_diff, 4),
            "diff_ci95_lo": round(ci_lo, 4),
            "diff_ci95_hi": round(ci_hi, 4),
            "p_diff_gt_0": round(p_gt0, 4),
            "p_practical_equivalence": round(p_equiv, 4),
            "absolute_error_difference": round(sd_diff, 4),
            "relative_error_difference_pct":
                round(rel_err, 4) if not np.isnan(rel_err) else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Ranking error summary
# ---------------------------------------------------------------------------

def ranking_error_summary(
    run_scores: pd.DataFrame,
    n_boot: int,
    seed: int = SEED,
    score_col: str = "AgentScore",
) -> pd.DataFrame:
    """
    Bootstrap ranking uncertainty: per-agent rank probabilities, rank
    entropy, expected rank, SD of rank, rank-switch probability, and a
    95% bootstrap interval of rank.
    """
    if run_scores.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    agents = sorted(run_scores["agent"].unique())
    n_agents = len(agents)
    by_agent = {a: run_scores[run_scores["agent"] == a][score_col].to_numpy(dtype=float)
                for a in agents}

    # Observed ranking (1 = best)
    obs_means = np.array([np.mean(by_agent[a]) for a in agents])
    obs_rank = (-obs_means).argsort().argsort() + 1
    obs_rank_map = {a: int(obs_rank[i]) for i, a in enumerate(agents)}

    rank_draws = {a: [] for a in agents}
    for _ in range(n_boot):
        means = np.array([
            np.mean(rng.choice(by_agent[a], len(by_agent[a]), replace=True))
            if len(by_agent[a]) else 0.0
            for a in agents
        ])
        ranks = (-means).argsort().argsort() + 1
        for i, a in enumerate(agents):
            rank_draws[a].append(int(ranks[i]))

    rows = []
    for a in agents:
        draws = np.array(rank_draws[a])
        probs = {r: float(np.mean(draws == r)) for r in range(1, n_agents + 1)}
        # entropy
        p = np.array([probs[r] for r in range(1, n_agents + 1)])
        p_nz = p[p > 0]
        entropy = float(-np.sum(p_nz * np.log2(p_nz))) if len(p_nz) else 0.0
        switch_prob = float(np.mean(draws != obs_rank_map[a]))
        row = {
            "agent": a,
            "observed_rank": obs_rank_map[a],
            "expected_rank": round(float(np.mean(draws)), 4),
            "sd_rank": round(float(np.std(draws, ddof=1)) if len(draws) > 1 else 0.0, 4),
            "rank_ci95_lo": int(np.percentile(draws, 2.5)),
            "rank_ci95_hi": int(np.percentile(draws, 97.5)),
            "rank_entropy_bits": round(entropy, 4),
            "rank_switch_probability": round(switch_prob, 4),
            "n_bootstrap": n_boot,
        }
        for r in range(1, n_agents + 1):
            row[f"p_rank_{r}"] = round(probs[r], 4)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Reproducibility error summary
# ---------------------------------------------------------------------------

def reproducibility_error_summary(
    pairwise_sim: pd.DataFrame,
    run_embs: Dict[Tuple[str, int], np.ndarray],
    between_sim: pd.DataFrame,
    n_boot: int,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Bootstrap errors for within-agent mean similarity, reproducibility index,
    centroid compactness, and (global) between-agent similarity + silhouette.
    """
    rng = np.random.default_rng(seed)
    rows = []

    agents = sorted({a for (a, _) in run_embs.keys()})

    # Per-agent within similarity + reproducibility index + compactness
    for agent in agents:
        # intra-agent pairwise cosine values
        if not pairwise_sim.empty:
            mask = (pairwise_sim["agent_a"] == agent) & (pairwise_sim["agent_b"] == agent)
            sims = pairwise_sim[mask]["cosine_sim"].dropna().to_numpy(dtype=float)
        else:
            sims = np.array([])

        if len(sims) > 0:
            boot_means = np.array([
                np.mean(rng.choice(sims, len(sims), replace=True)) for _ in range(n_boot)
            ])
            mean_sim = float(np.mean(sims))
            sim_ci = (float(np.percentile(boot_means, 2.5)),
                      float(np.percentile(boot_means, 97.5)))
            sim_sd = float(np.std(boot_means, ddof=1)) if len(boot_means) > 1 else 0.0
            # reproducibility index per bootstrap = 1 - CV
            repro_boot = []
            for _ in range(n_boot):
                s = rng.choice(sims, len(sims), replace=True)
                m = np.mean(s)
                cv = np.std(s, ddof=1) / m if m != 0 and len(s) > 1 else 0.0
                repro_boot.append(max(0.0, 1 - cv))
            repro_mean = float(np.mean(repro_boot))
            repro_ci = (float(np.percentile(repro_boot, 2.5)),
                        float(np.percentile(repro_boot, 97.5)))
        else:
            mean_sim = float("nan"); sim_ci = (float("nan"), float("nan")); sim_sd = float("nan")
            repro_mean = float("nan"); repro_ci = (float("nan"), float("nan"))

        # Centroid compactness: mean distance of runs to agent centroid
        vecs = [v for (a, _), v in run_embs.items() if a == agent]
        if len(vecs) >= 2:
            mat = np.stack(vecs)
            centroid = mat.mean(axis=0)
            dists = np.linalg.norm(mat - centroid, axis=1)
            comp_boot = np.array([
                np.mean(rng.choice(dists, len(dists), replace=True)) for _ in range(n_boot)
            ])
            compactness = float(np.mean(dists))
            comp_ci = (float(np.percentile(comp_boot, 2.5)),
                       float(np.percentile(comp_boot, 97.5)))
        else:
            compactness = float("nan"); comp_ci = (float("nan"), float("nan"))

        rows.append({
            "scope": "within_agent",
            "agent": agent,
            "mean_within_similarity": round(mean_sim, 4) if not np.isnan(mean_sim) else float("nan"),
            "within_sim_ci95_lo": round(sim_ci[0], 4) if not np.isnan(sim_ci[0]) else float("nan"),
            "within_sim_ci95_hi": round(sim_ci[1], 4) if not np.isnan(sim_ci[1]) else float("nan"),
            "within_sim_bootstrap_sd": round(sim_sd, 4) if not np.isnan(sim_sd) else float("nan"),
            "reproducibility_index": round(repro_mean, 4) if not np.isnan(repro_mean) else float("nan"),
            "repro_ci95_lo": round(repro_ci[0], 4) if not np.isnan(repro_ci[0]) else float("nan"),
            "repro_ci95_hi": round(repro_ci[1], 4) if not np.isnan(repro_ci[1]) else float("nan"),
            "centroid_compactness": round(compactness, 4) if not np.isnan(compactness) else float("nan"),
            "compactness_ci95_lo": round(comp_ci[0], 4) if not np.isnan(comp_ci[0]) else float("nan"),
            "compactness_ci95_hi": round(comp_ci[1], 4) if not np.isnan(comp_ci[1]) else float("nan"),
            "n_bootstrap": n_boot,
        })

    # Global between-agent similarity
    if not between_sim.empty and "centroid_cosine_sim" in between_sim.columns:
        bvals = between_sim["centroid_cosine_sim"].dropna().to_numpy(dtype=float)
        if len(bvals) > 0:
            rows.append({
                "scope": "between_agent",
                "agent": "ALL_PAIRS",
                "mean_within_similarity": round(float(np.mean(bvals)), 4),
                "within_sim_ci95_lo": float("nan"),
                "within_sim_ci95_hi": float("nan"),
                "within_sim_bootstrap_sd": float("nan"),
                "reproducibility_index": float("nan"),
                "repro_ci95_lo": float("nan"),
                "repro_ci95_hi": float("nan"),
                "centroid_compactness": float("nan"),
                "compactness_ci95_lo": float("nan"),
                "compactness_ci95_hi": float("nan"),
                "n_bootstrap": n_boot,
            })

    # Silhouette (global) if available
    sil = _silhouette(run_embs)
    if not np.isnan(sil):
        rows.append({
            "scope": "global_silhouette",
            "agent": "ALL",
            "mean_within_similarity": round(sil, 4),
            "within_sim_ci95_lo": float("nan"),
            "within_sim_ci95_hi": float("nan"),
            "within_sim_bootstrap_sd": float("nan"),
            "reproducibility_index": float("nan"),
            "repro_ci95_lo": float("nan"),
            "repro_ci95_hi": float("nan"),
            "centroid_compactness": float("nan"),
            "compactness_ci95_lo": float("nan"),
            "compactness_ci95_hi": float("nan"),
            "n_bootstrap": n_boot,
        })

    return pd.DataFrame(rows)


def _silhouette(run_embs: Dict[Tuple[str, int], np.ndarray]) -> float:
    try:
        from sklearn.metrics import silhouette_score
        keys = sorted(run_embs.keys())
        labels = [k[0] for k in keys]
        if len(set(labels)) < 2:
            return float("nan")
        mat = np.stack([run_embs[k] for k in keys])
        return float(silhouette_score(mat, labels, metric="cosine"))
    except Exception as exc:
        logger.debug("Silhouette not available: %s", exc)
        return float("nan")


# ---------------------------------------------------------------------------
# 5. Monte Carlo error summary
# ---------------------------------------------------------------------------

def monte_carlo_error_summary(
    run_embs: Dict[Tuple[str, int], np.ndarray],
    boot_by_agent: Dict[str, np.ndarray],
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Errors for the Monte Carlo satellite analysis.
    Distances computed in the original embedding space; centroid CI also
    reported in 2-D PCA space for interpretability.
    """
    if not run_embs or not boot_by_agent:
        return pd.DataFrame()

    agents = sorted({a for (a, _) in run_embs.keys()})
    real_centroids = {}
    for agent in agents:
        vecs = [v for (a, _), v in run_embs.items() if a == agent]
        if vecs:
            real_centroids[agent] = np.stack(vecs).mean(axis=0)

    # PCA for centroid CI in 2-D
    pca = None
    try:
        from sklearn.decomposition import PCA
        all_boot = np.vstack([boot_by_agent[a] for a in boot_by_agent])
        n_comp = min(2, all_boot.shape[1], all_boot.shape[0] - 1)
        if n_comp >= 1:
            pca = PCA(n_components=n_comp, random_state=seed).fit(all_boot)
    except Exception as exc:
        logger.debug("MC PCA failed: %s", exc)

    rows = []
    for agent in agents:
        if agent not in boot_by_agent or agent not in real_centroids:
            continue
        boot = boot_by_agent[agent]
        centroid = real_centroids[agent]

        # Distance of satellites to real centroid
        dists = np.linalg.norm(boot - centroid, axis=1)
        mean_dist = float(np.mean(dists))
        sd_dist = float(np.std(dists, ddof=1)) if len(dists) > 1 else 0.0
        d_lo = float(np.percentile(dists, 2.5))
        d_hi = float(np.percentile(dists, 97.5))

        # Centroid mean/sd in PCA space
        if pca is not None:
            boot_pca = pca.transform(boot)
            centroid_pca_mean = boot_pca.mean(axis=0)
            centroid_pca_sd = boot_pca.std(axis=0)
            pc1_lo, pc1_hi = np.percentile(boot_pca[:, 0], [2.5, 97.5])
        else:
            centroid_pca_mean = np.array([np.nan])
            centroid_pca_sd = np.array([np.nan])
            pc1_lo = pc1_hi = float("nan")

        # Nearest-centroid classification uncertainty (cloud overlap)
        other = {a: c for a, c in real_centroids.items() if a != agent}
        if other:
            misassigned = 0
            for pt in boot:
                own_d = np.linalg.norm(pt - centroid)
                nearest_other = min(np.linalg.norm(pt - c) for c in other.values())
                if nearest_other < own_d:
                    misassigned += 1
            overlap = misassigned / len(boot)
        else:
            overlap = float("nan")

        rows.append({
            "agent": agent,
            "centroid_pca1_mean": round(float(centroid_pca_mean[0]), 4),
            "centroid_pca1_sd": round(float(centroid_pca_sd[0]), 4),
            "centroid_pca1_ci95_lo": round(float(pc1_lo), 4) if not np.isnan(pc1_lo) else float("nan"),
            "centroid_pca1_ci95_hi": round(float(pc1_hi), 4) if not np.isnan(pc1_hi) else float("nan"),
            "mean_satellite_distance": round(mean_dist, 4),
            "sd_satellite_distance": round(sd_dist, 4),
            "distance_ci95_lo": round(d_lo, 4),
            "distance_ci95_hi": round(d_hi, 4),
            "nearest_centroid_overlap": round(overlap, 4) if not np.isnan(overlap) else float("nan"),
            "n_satellites": len(boot),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Distance distribution data (for figure)
# ---------------------------------------------------------------------------

def mc_distance_distributions(
    run_embs: Dict[Tuple[str, int], np.ndarray],
    boot_by_agent: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Return {agent: array of satellite→centroid distances} for plotting."""
    out = {}
    agents = sorted({a for (a, _) in run_embs.keys()})
    for agent in agents:
        vecs = [v for (a, _), v in run_embs.items() if a == agent]
        if not vecs or agent not in boot_by_agent:
            continue
        centroid = np.stack(vecs).mean(axis=0)
        out[agent] = np.linalg.norm(boot_by_agent[agent] - centroid, axis=1)
    return out


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def flag_outlier_runs(
    run_scores: pd.DataFrame,
    score_col: str = "AgentScore",
    iqr_k: float = 1.5,
    zscore_thresh: float = 2.5,
) -> pd.DataFrame:
    """
    Flag outlier runs within each agent using two complementary methods:

    1. IQR method (Tukey fences): outlier if value < Q1 - k*IQR or > Q3 + k*IQR.
    2. Z-score method: outlier if |z| > zscore_thresh (within-agent standardisation).

    A run is flagged as an outlier if EITHER method flags it.

    Returns a DataFrame with all original columns plus:
      outlier_iqr      : bool
      outlier_zscore   : bool
      outlier          : bool  (union of both)
      zscore           : float  (within-agent)
      iqr_lower_fence  : float
      iqr_upper_fence  : float
      outlier_reason   : str   (e.g. "IQR+Z", "IQR", "Z", "")
    """
    if run_scores is None or run_scores.empty or score_col not in run_scores.columns:
        return pd.DataFrame()

    rows = []
    for agent in sorted(run_scores["agent"].unique()):
        sub = run_scores[run_scores["agent"] == agent].copy()
        vals = sub[score_col].to_numpy(dtype=float)
        n = len(vals)

        # IQR fences
        if n >= 4:
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            lo_fence = q1 - iqr_k * iqr
            hi_fence = q3 + iqr_k * iqr
        else:
            lo_fence = hi_fence = float("nan")

        # Z-scores (within-agent)
        mean_v = float(np.mean(vals))
        std_v  = float(np.std(vals, ddof=1)) if n > 1 else 0.0

        for _, row in sub.iterrows():
            v = float(row[score_col])
            z = (v - mean_v) / std_v if std_v > 0 else 0.0
            iqr_out = (v < lo_fence or v > hi_fence) if not np.isnan(lo_fence) else False
            z_out   = abs(z) > zscore_thresh

            reasons = []
            if iqr_out:
                reasons.append("IQR")
            if z_out:
                reasons.append("Z")

            rows.append({
                **row.to_dict(),
                "outlier_iqr":       iqr_out,
                "outlier_zscore":    z_out,
                "outlier":           iqr_out or z_out,
                "zscore":            round(z, 4),
                "iqr_lower_fence":   round(lo_fence, 4) if not np.isnan(lo_fence) else float("nan"),
                "iqr_upper_fence":   round(hi_fence, 4) if not np.isnan(hi_fence) else float("nan"),
                "outlier_reason":    "+".join(reasons),
            })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    n_out = int(out["outlier"].sum())
    if n_out:
        flagged = out[out["outlier"]][["agent", "run", score_col, "outlier_reason"]]
        logger.info(
            "Outlier detection (%s): %d run(s) flagged — %s",
            score_col, n_out,
            "; ".join(f"{r.agent} R{r.run} ({r.outlier_reason})"
                      for r in flagged.itertuples()),
        )
    else:
        logger.info("Outlier detection (%s): no outliers flagged.", score_col)

    return out
