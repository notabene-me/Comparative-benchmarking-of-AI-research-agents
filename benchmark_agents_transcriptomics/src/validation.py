"""
validation.py — ranking validation via bootstrap / Monte Carlo,
and descriptive reliability statistics (Kendall W, Spearman, Cronbach α).
"""

import logging
from itertools import combinations, permutations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

SEED = 42


# ---------------------------------------------------------------------------
# Bootstrap ranking
# ---------------------------------------------------------------------------

def bootstrap_ranking(
    run_scores: pd.DataFrame,
    n_mc: int = 10_000,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Resample runs within each agent n_mc times.
    Returns:
        ranking_uncertainty  : probability of each rank per agent
        pairwise_win         : P(agent_i beats agent_j)
        validation_summary   : absolute / relative error per agent
    """
    rng = np.random.default_rng(seed)
    agents = sorted(run_scores["agent"].unique())
    n_agents = len(agents)

    # Store mean scores per bootstrap iteration
    boot_means: Dict[str, List[float]] = {a: [] for a in agents}

    for _ in range(n_mc):
        for agent in agents:
            sc = run_scores[run_scores["agent"] == agent]["AgentScore"].values
            n = len(sc)
            if n == 0:
                boot_means[agent].append(0.0)
            else:
                boot_means[agent].append(float(np.mean(rng.choice(sc, size=n, replace=True))))

    boot_arr = np.column_stack([boot_means[a] for a in agents])  # (n_mc, n_agents)

    # Ranking per iteration (1 = best)
    ranks_arr = np.argsort(-boot_arr, axis=1).argsort(axis=1) + 1  # shape (n_mc, n_agents)

    # P(rank r) for each agent
    rank_records = []
    for ai, agent in enumerate(agents):
        for r in range(1, n_agents + 1):
            prob = float(np.mean(ranks_arr[:, ai] == r))
            rank_records.append({"agent": agent, "rank": r, "probability": prob})
    ranking_uncertainty = pd.DataFrame(rank_records)

    # Pairwise win probabilities
    pairwise_rows = []
    for a1, a2 in combinations(agents, 2):
        i1 = agents.index(a1)
        i2 = agents.index(a2)
        p12 = float(np.mean(boot_arr[:, i1] > boot_arr[:, i2]))
        p21 = 1 - p12
        pairwise_rows.append({"agent_i": a1, "agent_j": a2, "p_i_beats_j": p12})
        pairwise_rows.append({"agent_i": a2, "agent_j": a1, "p_i_beats_j": p21})
    pairwise_win = pd.DataFrame(pairwise_rows)

    # Absolute / relative error per agent
    val_rows = []
    for ai, agent in enumerate(agents):
        obs_mean = float(run_scores[run_scores["agent"] == agent]["AgentScore"].mean())
        boot_vec = boot_arr[:, ai]
        abs_err  = float(np.std(boot_vec))
        rel_err  = abs_err / obs_mean * 100 if obs_mean != 0 else float("nan")
        ci_lo    = float(np.percentile(boot_vec, 2.5))
        ci_hi    = float(np.percentile(boot_vec, 97.5))
        val_rows.append({
            "agent":          agent,
            "observed_mean":  obs_mean,
            "bootstrap_sd":   abs_err,
            "absolute_error": abs_err,
            "relative_error_pct": rel_err,
            "ci95_lo":        ci_lo,
            "ci95_hi":        ci_hi,
        })
    validation_summary = pd.DataFrame(val_rows)

    return ranking_uncertainty, pairwise_win, validation_summary


# ---------------------------------------------------------------------------
# Reliability statistics
# ---------------------------------------------------------------------------

def kendall_w(rank_matrix: np.ndarray) -> float:
    """
    Kendall's W (coefficient of concordance) for a matrix of
    shape (n_judges, n_subjects).
    Returns W in [0, 1].
    """
    n_judges, n_subjects = rank_matrix.shape
    if n_judges < 2 or n_subjects < 2:
        return float("nan")
    # Sum of squared deviations of rank sums from grand mean
    rank_sums = rank_matrix.sum(axis=0)
    mean_rank_sum = rank_sums.mean()
    SS = np.sum((rank_sums - mean_rank_sum) ** 2)
    W = 12 * SS / (n_judges ** 2 * (n_subjects ** 3 - n_subjects))
    return float(W)


def cronbach_alpha(score_matrix: np.ndarray) -> float:
    """
    Cronbach's alpha for a (n_items, n_subjects) score matrix.
    Returns alpha.
    """
    n_items, n_subjects = score_matrix.shape
    if n_items < 2 or n_subjects < 2:
        return float("nan")
    item_vars = np.var(score_matrix, axis=1, ddof=1)
    total_var = np.var(score_matrix.sum(axis=0), ddof=1)
    if total_var == 0:
        return float("nan")
    alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)
    return float(alpha)


def compute_reliability(run_scores: pd.DataFrame) -> dict:
    """
    Compute descriptive reliability statistics.

    Returns a dict with:
        kendall_w        : ranking consistency across agents
        spearman_matrix  : Spearman correlation of run score vectors
        cronbach_alpha   : if ≥ 3 agents
    """
    agents = sorted(run_scores["agent"].unique())
    n_agents = len(agents)
    results = {}

    # Build run-score vectors per agent (padded / truncated to equal length)
    max_runs = int(run_scores.groupby("agent")["run"].count().max())
    score_mat = np.full((n_agents, max_runs), np.nan)
    for ai, agent in enumerate(agents):
        sc = run_scores[run_scores["agent"] == agent]["AgentScore"].values
        score_mat[ai, :len(sc)] = sc

    # Rank matrix (rank runs within each agent)
    rank_mat = np.zeros_like(score_mat)
    for ai in range(n_agents):
        valid = ~np.isnan(score_mat[ai])
        rank_mat[ai, valid] = stats.rankdata(-score_mat[ai, valid])

    results["kendall_w"] = kendall_w(rank_mat[:, :max_runs])

    # Spearman correlations between agent run-score vectors
    spearman_rows = []
    for a1, a2 in combinations(agents, 2):
        sc1 = run_scores[run_scores["agent"] == a1]["AgentScore"].values
        sc2 = run_scores[run_scores["agent"] == a2]["AgentScore"].values
        n = min(len(sc1), len(sc2))
        if n < 3:
            rho, pval = float("nan"), float("nan")
        else:
            res = stats.spearmanr(sc1[:n], sc2[:n])
            rho  = float(res.statistic)
            pval = float(res.pvalue)
        spearman_rows.append({
            "agent_a": a1, "agent_b": a2,
            "spearman_rho": rho, "p_value": pval,
        })
    results["spearman_df"] = pd.DataFrame(spearman_rows)

    # Cronbach alpha (if ≥ 3 agents, treat agents as items)
    if n_agents >= 3:
        valid_mat = np.where(np.isnan(score_mat), 0, score_mat)
        results["cronbach_alpha"] = cronbach_alpha(valid_mat)
    else:
        results["cronbach_alpha"] = float("nan")

    return results


# ---------------------------------------------------------------------------
# Validation claims table
# ---------------------------------------------------------------------------

def build_validation_table(
    within_repro: pd.DataFrame,
    between_sim: pd.DataFrame,
    ranking_unc: pd.DataFrame,
    reliability: dict,
) -> pd.DataFrame:
    rows = []

    # 1. Within-agent reproducibility
    if not within_repro.empty:
        mean_repro = within_repro["reproducibility"].mean()
        rows.append({
            "Claim": "Within-agent runs are reproducible (CV < 10%)",
            "Validation_method": "Bootstrap pairwise cosine similarity CV",
            "Key_quantitative_result":
                f"Mean reproducibility index = {mean_repro:.3f}; "
                f"Mean CV = {within_repro['cv_pct'].mean():.1f}%",
            "Verdict": "Supported descriptively" if mean_repro > 0.8 else "Partially supported",
        })

    # 2. Between-agent differences
    if not between_sim.empty:
        cross_mean = between_sim["centroid_cosine_sim"].mean()
        rows.append({
            "Claim": "Agents produce semantically distinct outputs",
            "Validation_method": "Centroid cosine similarity between agent pairs",
            "Key_quantitative_result":
                f"Mean between-agent cosine similarity = {cross_mean:.3f}",
            "Verdict": "Supported descriptively" if cross_mean < 0.9 else "Not supported",
        })

    # 3. Ranking uncertainty
    if not ranking_unc.empty:
        top_probs = ranking_unc[ranking_unc["rank"] == 1]
        rows.append({
            "Claim": "Ranking is stable across bootstrap resampling",
            "Validation_method": "Bootstrap ranking probability (n=10000)",
            "Key_quantitative_result":
                "; ".join(f"{r['agent']}: P(rank1)={r['probability']:.2f}"
                          for _, r in top_probs.iterrows()),
            "Verdict": "Descriptive-only",
        })

    # 4. Reliability
    kw = reliability.get("kendall_w", float("nan"))
    rows.append({
        "Claim": "Run scores are internally consistent",
        "Validation_method": "Kendall's W across agent run vectors",
        "Key_quantitative_result": f"Kendall W = {kw:.3f}" if not np.isnan(kw) else "Insufficient data",
        "Verdict": "Descriptive-only (n_runs limited)",
    })

    # 5. Proxy scores
    rows.append({
        "Claim": "Proxy scores approximate expert judgment",
        "Validation_method": "Structural feature keywords (no ground-truth labels)",
        "Key_quantitative_result": "N/A — proxy scores are keyword-count-based",
        "Verdict": "Requires external validation",
    })

    # 6. Monte Carlo
    rows.append({
        "Claim": "MC satellite points characterise uncertainty, not real observations",
        "Validation_method": "Design: bootstrap centroids and Gaussian samples",
        "Key_quantitative_result": "By construction",
        "Verdict": "Supported descriptively",
    })

    return pd.DataFrame(rows)
