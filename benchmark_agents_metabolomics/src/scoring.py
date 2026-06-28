"""
scoring.py — AgentScore calculation.

Formula:
    AgentScore = P × (0.50 + 0.30 × D/100 + 0.20 × R/100)

where R = result accuracy, P = process quality, D = interpretation depth.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .features import (
    compute_proxy_depth,
    compute_proxy_process,
    compute_proxy_result,
    proxy_process_components,
    proxy_result_components,
    proxy_depth_components,
    all_proxy_components,
    VOLUME_RELATED_COMPONENTS,
)

logger = logging.getLogger(__name__)

SEED = 42
N_BOOTSTRAP = 10_000


def agent_score(R: float, P: float, D: float) -> float:
    """
    AgentScore = P × (0.50 + 0.30 × D/100 + 0.20 × R/100)
    """
    return P * (0.50 + 0.30 * D / 100.0 + 0.20 * R / 100.0)


# ---------------------------------------------------------------------------
# Load manual scores
# ---------------------------------------------------------------------------

def load_manual_scores(path: str | Path) -> Optional[pd.DataFrame]:
    """
    Load manual scores from a CSV with columns: agent, run, R, P, D
    Returns None if path is not provided or file cannot be read.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning("Manual scores file not found: %s", p)
        return None
    try:
        df = pd.read_csv(p)
        required = {"agent", "run", "R", "P", "D"}
        missing = required - set(df.columns)
        if missing:
            logger.error("Manual scores CSV missing columns: %s", missing)
            return None
        df["agent"] = df["agent"].str.strip()
        logger.info("Loaded manual scores: %d rows from %s", len(df), p)
        return df
    except Exception as exc:
        logger.error("Failed to read manual scores: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Compute scores
# ---------------------------------------------------------------------------

def compute_run_scores(
    run_features: pd.DataFrame,
    manual_scores: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame with one row per (agent, run) containing
    R, P, D, AgentScore, and score_type (manual / proxy-estimated).
    """
    if run_features.empty:
        return pd.DataFrame()

    rows = []
    for _, feat_row in run_features.iterrows():
        agent = feat_row["agent"]
        run   = feat_row["run"]

        manual_row = None
        if manual_scores is not None:
            mask = (
                (manual_scores["agent"].str.lower() == str(agent).lower()) &
                (manual_scores["run"] == run)
            )
            if mask.any():
                manual_row = manual_scores[mask].iloc[0]

        if manual_row is not None:
            R = float(manual_row["R"])
            P = float(manual_row["P"])
            D = float(manual_row["D"])
            score_type = "manual"
        else:
            row_dict = feat_row.to_dict()
            P = compute_proxy_process(row_dict)
            R = compute_proxy_result(row_dict)
            D = compute_proxy_depth(row_dict)
            score_type = "proxy-estimated"

        rows.append({
            "agent":       agent,
            "run":         run,
            "R":           R,
            "P":           P,
            "D":           D,
            "AgentScore":  agent_score(R, P, D),
            "score_type":  score_type,
        })

    return pd.DataFrame(rows)


def decompose_proxy_components(run_features: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per (agent, run) with every binary proxy component
    plus the P/R/D subtotals (0–100). This makes the proxy score fully
    auditable.
    """
    if run_features.empty:
        return pd.DataFrame()

    rows = []
    for _, feat_row in run_features.iterrows():
        rd = feat_row.to_dict()
        comps = all_proxy_components(rd)
        row = {"agent": rd["agent"], "run": rd["run"]}
        row.update(comps)
        row["P_proxy"] = compute_proxy_process(rd)
        row["R_proxy"] = compute_proxy_result(rd)
        row["D_proxy"] = compute_proxy_depth(rd)
        row["AgentScore_proxy"] = agent_score(row["R_proxy"], row["P_proxy"], row["D_proxy"])
        rows.append(row)

    return pd.DataFrame(rows)


def compute_run_scores_excluding_volume(run_features: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute proxy AgentScore after dropping output-volume / artifact-count
    related components (definition G in the ranking sensitivity analysis).
    """
    if run_features.empty:
        return pd.DataFrame()

    rows = []
    for _, feat_row in run_features.iterrows():
        rd = feat_row.to_dict()
        P = compute_proxy_process(rd, exclude=VOLUME_RELATED_COMPONENTS)
        R = compute_proxy_result(rd, exclude=VOLUME_RELATED_COMPONENTS)
        D = compute_proxy_depth(rd, exclude=VOLUME_RELATED_COMPONENTS)
        rows.append({
            "agent": rd["agent"], "run": rd["run"],
            "R": R, "P": P, "D": D,
            "AgentScore_no_volume": agent_score(R, P, D),
        })
    return pd.DataFrame(rows)


def build_manual_vs_proxy(
    run_features: pd.DataFrame,
    manual_scores: Optional[pd.DataFrame],
    domain_scores: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compare manual, proxy and domain-aware scores per run, with agent-level
    rankings under each and a rank-disagreement flag.

    Returns empty DataFrame if manual scores are not provided.
    """
    if manual_scores is None or run_features.empty:
        return pd.DataFrame()

    # Proxy per run
    proxy_rows = []
    for _, fr in run_features.iterrows():
        rd = fr.to_dict()
        P = compute_proxy_process(rd)
        R = compute_proxy_result(rd)
        D = compute_proxy_depth(rd)
        proxy_rows.append({
            "agent": rd["agent"], "run": rd["run"],
            "proxy_R": R, "proxy_P": P, "proxy_D": D,
            "proxy_AgentScore": agent_score(R, P, D),
        })
    proxy_df = pd.DataFrame(proxy_rows)

    man = manual_scores.copy()
    man["manual_AgentScore"] = man.apply(
        lambda r: agent_score(float(r["R"]), float(r["P"]), float(r["D"])), axis=1)
    man = man.rename(columns={"R": "manual_R", "P": "manual_P", "D": "manual_D"})

    merged = man.merge(proxy_df, on=["agent", "run"], how="left")

    if domain_scores is not None and not domain_scores.empty:
        merged = merged.merge(
            domain_scores[["agent", "run", "DomainScore"]],
            on=["agent", "run"], how="left")
        merged = merged.rename(columns={"DomainScore": "domain_score"})
    else:
        merged["domain_score"] = float("nan")

    merged["diff_manual_minus_proxy"] = merged["manual_AgentScore"] - merged["proxy_AgentScore"]
    merged["diff_manual_minus_domain"] = merged["manual_AgentScore"] - merged["domain_score"]

    # Agent-level ranks
    def _agent_rank(df, col):
        means = df.groupby("agent")[col].mean()
        return means.rank(ascending=False, method="min").astype(int).to_dict()

    rank_manual = _agent_rank(merged, "manual_AgentScore")
    rank_proxy  = _agent_rank(merged, "proxy_AgentScore")
    rank_domain = _agent_rank(merged, "domain_score") if merged["domain_score"].notna().any() else {}

    merged["rank_by_manual"] = merged["agent"].map(rank_manual)
    merged["rank_by_proxy"]  = merged["agent"].map(rank_proxy)
    merged["rank_by_domain"] = merged["agent"].map(rank_domain) if rank_domain else float("nan")
    merged["rank_disagreement"] = (
        (merged["rank_by_manual"] != merged["rank_by_proxy"]).astype(int)
    )
    return merged


def compute_agent_summary(run_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate run-level scores to agent level.
    Includes mean, SD, SEM, 95% bootstrap CI, ranking.
    """
    if run_scores.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(SEED)
    records = []

    for agent in sorted(run_scores["agent"].unique()):
        adf = run_scores[run_scores["agent"] == agent]
        scores = adf["AgentScore"].values
        n = len(scores)
        mean_s = float(np.mean(scores))
        sd_s   = float(np.std(scores, ddof=1)) if n > 1 else 0.0
        sem_s  = sd_s / np.sqrt(n) if n > 0 else 0.0

        # Bootstrap CI
        boot_means = np.array([
            np.mean(rng.choice(scores, size=n, replace=True))
            for _ in range(N_BOOTSTRAP)
        ])
        ci_lo = float(np.percentile(boot_means, 2.5))
        ci_hi = float(np.percentile(boot_means, 97.5))

        records.append({
            "agent":       agent,
            "n_runs":      n,
            "mean_score":  mean_s,
            "sd_score":    sd_s,
            "sem_score":   sem_s,
            "ci95_lo":     ci_lo,
            "ci95_hi":     ci_hi,
            "score_type":  adf["score_type"].mode().iloc[0] if not adf.empty else "unknown",
        })

    df = pd.DataFrame(records)
    df["rank"] = df["mean_score"].rank(ascending=False).astype(int)
    return df.sort_values("rank")
