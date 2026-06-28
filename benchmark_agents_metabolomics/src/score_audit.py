"""
score_audit.py — auditability layer for the scoring system.

Provides:
  - output-volume metrics per run
  - Spearman correlations between AgentScore and output-volume metrics
  - an exploratory output-volume-penalised (residualised) score
  - ranking sensitivity analysis across several scoring definitions
  - scoring audit summary
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

SEED = 42
N_BOOT_RHO = 2000  # bootstrap iterations for correlation CIs

VOLUME_METRICS = [
    "file_count",
    "text_length",
    "table_count",
    "figure_count",
    "chunk_count",
    "artifact_diversity",
    "total_file_size",
]


# ---------------------------------------------------------------------------
# 1. Output-volume metrics
# ---------------------------------------------------------------------------

def _artifact_diversity(type_counts: Dict[str, int]) -> float:
    """Shannon entropy over file-type distribution (normalised to [0, 1])."""
    counts = np.array([c for c in type_counts.values() if c > 0], dtype=float)
    if counts.sum() == 0 or len(counts) <= 1:
        return 0.0
    p = counts / counts.sum()
    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(len(counts))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def compute_output_volume_metrics(
    inventory: pd.DataFrame,
    run_features: pd.DataFrame,
    chunk_map: Dict[str, list],
    file_meta: List[dict],
) -> pd.DataFrame:
    """
    Per (agent, run) output-volume metrics.
    """
    if inventory.empty:
        return pd.DataFrame()

    # chunk counts per (agent, run) using file_meta to map file_id → agent/run
    meta_df = pd.DataFrame(file_meta)
    chunk_counts = {fid: len(chunks) for fid, chunks in chunk_map.items()}
    meta_df["n_chunks"] = meta_df["file_id"].map(chunk_counts).fillna(0).astype(int)
    chunks_by_run = meta_df.groupby(["agent", "run"])["n_chunks"].sum()

    # text_length from run_features (n_chars)
    text_len = {}
    if not run_features.empty and "n_chars" in run_features.columns:
        for _, r in run_features.iterrows():
            text_len[(r["agent"], r["run"])] = r["n_chars"]

    rows = []
    for (agent, run), grp in inventory.groupby(["agent", "run"]):
        type_counts = grp["file_type"].value_counts().to_dict()
        n_text = int(type_counts.get("text", 0))
        n_table = int(type_counts.get("table", 0))
        n_image = int(type_counts.get("image", 0))
        n_code = int(type_counts.get("code", 0))
        n_doc = int(type_counts.get("document", 0))
        total_files = len(grp)
        total_size = float(grp["size_bytes"].sum())
        mean_size = float(grp["size_bytes"].mean()) if total_files else 0.0

        rows.append({
            "agent": agent,
            "run": run,
            "file_count": total_files,
            "n_text_files": n_text,
            "table_count": n_table,
            "figure_count": n_image,
            "n_code_files": n_code,
            "n_document_files": n_doc,
            "text_length": int(text_len.get((agent, run), 0)),
            "chunk_count": int(chunks_by_run.get((agent, run), 0)),
            "artifact_diversity": round(_artifact_diversity(type_counts), 4),
            "mean_file_size": round(mean_size, 2),
            "total_file_size": round(total_size, 2),
        })

    return pd.DataFrame(rows).sort_values(["agent", "run"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Score vs volume correlations
# ---------------------------------------------------------------------------

def _bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray,
                           n_boot: int = N_BOOT_RHO, seed: int = SEED
                           ) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 4:
        return float("nan"), float("nan")
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        rho = stats.spearmanr(x[idx], y[idx]).statistic
        if not np.isnan(rho):
            rhos.append(rho)
    if not rhos:
        return float("nan"), float("nan")
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def compute_score_volume_correlation(
    run_scores: pd.DataFrame,
    volume_metrics: pd.DataFrame,
    score_col: str = "AgentScore",
) -> pd.DataFrame:
    """
    Spearman correlation between the score and each output-volume metric,
    with bootstrap 95% CI for rho.
    """
    if run_scores.empty or volume_metrics.empty:
        return pd.DataFrame()

    merged = run_scores.merge(volume_metrics, on=["agent", "run"], how="inner")
    rows = []
    for metric in VOLUME_METRICS:
        if metric not in merged.columns:
            continue
        x = merged[score_col].to_numpy(dtype=float)
        y = merged[metric].to_numpy(dtype=float)
        n = len(x)
        if n < 3 or np.all(y == y[0]):
            rho, pval = float("nan"), float("nan")
            lo, hi = float("nan"), float("nan")
        else:
            res = stats.spearmanr(x, y)
            rho, pval = float(res.statistic), float(res.pvalue)
            lo, hi = _bootstrap_spearman_ci(x, y)
        rows.append({
            "metric": metric,
            "spearman_rho": rho,
            "p_value": pval,
            "rho_ci95_lo": lo,
            "rho_ci95_hi": hi,
            "n_runs": n,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Output-volume-penalised (residualised) score — EXPLORATORY
# ---------------------------------------------------------------------------

def compute_volume_penalized_score(
    run_scores: pd.DataFrame,
    volume_metrics: pd.DataFrame,
    score_col: str = "AgentScore",
) -> pd.DataFrame:
    """
    Residualise AgentScore against output-volume metrics using a robust
    (Huber) linear regression, then re-centre to the original mean.

    EXPLORATORY ONLY — not a primary score. Avoids over-correction by
    re-adding the global mean so values stay on a comparable scale.
    """
    if run_scores.empty or volume_metrics.empty:
        return pd.DataFrame()

    merged = run_scores.merge(volume_metrics, on=["agent", "run"], how="inner")
    feat_cols = [m for m in VOLUME_METRICS if m in merged.columns]
    if not feat_cols:
        return pd.DataFrame()

    X = merged[feat_cols].to_numpy(dtype=float)
    y = merged[score_col].to_numpy(dtype=float)

    # Standardise predictors
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    Xs = (X - X_mean) / X_std

    resid = y.copy()
    try:
        from sklearn.linear_model import HuberRegressor
        if len(y) > len(feat_cols) + 1 and np.std(y) > 0:
            model = HuberRegressor(max_iter=500)
            model.fit(Xs, y)
            pred = model.predict(Xs)
            resid = y - pred + float(np.mean(y))  # re-centre to original mean
    except Exception as exc:
        logger.warning("Volume-penalised regression failed (%s); using raw score.", exc)
        resid = y.copy()

    out = merged[["agent", "run"]].copy()
    out["AgentScore_raw"] = y
    out["AgentScore_volume_penalized"] = np.round(resid, 3)
    out["note"] = "exploratory residualised score"
    return out


# ---------------------------------------------------------------------------
# 4. Ranking sensitivity analysis (definitions A–G)
# ---------------------------------------------------------------------------

def _agent_means(df: pd.DataFrame, value_col: str) -> Dict[str, float]:
    if df.empty or value_col not in df.columns:
        return {}
    return df.groupby("agent")[value_col].mean().to_dict()


def compute_ranking_sensitivity(
    run_scores: pd.DataFrame,
    domain_scores: pd.DataFrame,
    within_repro: pd.DataFrame,
    volume_penalized: pd.DataFrame,
    no_volume_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank agents under several scoring definitions:
        A. Proxy AgentScore
        B. Domain-aware metabolomics score
        C. Reproducibility index only
        D. Interpretation depth (D) only
        E. Process quality (P) only
        F. Output-volume-penalised AgentScore
        G. AgentScore excluding output-volume features
    Returns long-format DataFrame: definition, agent, mean_value, rank.
    """
    definitions: Dict[str, Dict[str, float]] = {}

    definitions["A_proxy_agentscore"] = _agent_means(run_scores, "AgentScore")
    definitions["B_domain_score"]     = _agent_means(domain_scores, "DomainScore")

    if not within_repro.empty and "reproducibility" in within_repro.columns:
        definitions["C_reproducibility"] = dict(
            zip(within_repro["agent"], within_repro["reproducibility"])
        )
    definitions["D_interpretation_depth"] = _agent_means(run_scores, "D")
    definitions["E_process_quality"]      = _agent_means(run_scores, "P")

    if not volume_penalized.empty:
        definitions["F_volume_penalized"] = _agent_means(
            volume_penalized, "AgentScore_volume_penalized")
    if not no_volume_scores.empty:
        definitions["G_no_volume_features"] = _agent_means(
            no_volume_scores, "AgentScore_no_volume")

    rows = []
    for defn, means in definitions.items():
        if not means:
            continue
        s = pd.Series(means)
        ranks = s.rank(ascending=False, method="min").astype(int)
        for agent in s.index:
            rows.append({
                "definition": defn,
                "agent": agent,
                "mean_value": round(float(s[agent]), 3),
                "rank": int(ranks[agent]),
            })

    return pd.DataFrame(rows)


def build_ranking_sensitivity_matrix(sens_long: pd.DataFrame) -> pd.DataFrame:
    """Wide matrix: agents × definitions of ranks (for heatmap)."""
    if sens_long.empty:
        return pd.DataFrame()
    return sens_long.pivot(index="agent", columns="definition", values="rank")


# ---------------------------------------------------------------------------
# 5. Scoring audit summary
# ---------------------------------------------------------------------------

def build_score_audit(
    run_scores: pd.DataFrame,
    domain_scores: pd.DataFrame,
    score_volume_corr: pd.DataFrame,
    volume_penalized: pd.DataFrame,
    ranking_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    """High-level audit table summarising key diagnostics."""
    rows = []

    # Max |rho| with output volume
    if not score_volume_corr.empty:
        valid = score_volume_corr.dropna(subset=["spearman_rho"])
        if not valid.empty:
            top = valid.iloc[valid["spearman_rho"].abs().argmax()]
            rows.append({
                "diagnostic": "max_abs_score_volume_correlation",
                "value": round(float(top["spearman_rho"]), 3),
                "detail": f"metric={top['metric']} (Spearman)",
            })

    # Proxy vs domain agreement (Spearman of agent means)
    if not run_scores.empty and not domain_scores.empty:
        pa = run_scores.groupby("agent")["AgentScore"].mean()
        da = domain_scores.groupby("agent")["DomainScore"].mean()
        common = pa.index.intersection(da.index)
        if len(common) >= 3:
            rho = stats.spearmanr(pa[common], da[common]).statistic
            rows.append({
                "diagnostic": "proxy_vs_domain_rank_correlation",
                "value": round(float(rho), 3),
                "detail": "Spearman of agent-mean scores",
            })

    # Top-ranked agent stability across definitions
    if not ranking_sensitivity.empty:
        top_by_def = (ranking_sensitivity[ranking_sensitivity["rank"] == 1]
                      .groupby("definition")["agent"].first())
        n_defs = top_by_def.nunique()
        most_common = top_by_def.mode().iloc[0] if not top_by_def.empty else "n/a"
        rows.append({
            "diagnostic": "top_agent_stability",
            "value": f"{(top_by_def == most_common).mean():.2f}",
            "detail": f"most_common_top={most_common}; distinct_winners={n_defs}",
        })

    # Volume-penalised vs raw rank change
    if not volume_penalized.empty:
        raw = volume_penalized.groupby("agent")["AgentScore_raw"].mean().rank(ascending=False)
        pen = volume_penalized.groupby("agent")["AgentScore_volume_penalized"].mean().rank(ascending=False)
        changed = int((raw != pen).sum())
        rows.append({
            "diagnostic": "agents_changing_rank_after_volume_penalty",
            "value": changed,
            "detail": "exploratory residualised score",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Scoring validation summary (question / method / result / interpretation)
# ---------------------------------------------------------------------------

def build_scoring_validation_summary(
    run_scores: pd.DataFrame,
    domain_scores: pd.DataFrame,
    score_volume_corr: pd.DataFrame,
    ranking_sensitivity: pd.DataFrame,
    bootstrap_error: pd.DataFrame,
    pairwise_diff: pd.DataFrame,
    ranking_error: pd.DataFrame,
    repro_error: pd.DataFrame,
    fairness_summary: pd.DataFrame,
    score_type: str,
) -> pd.DataFrame:
    """
    Build tables/scoring_validation_summary.csv with columns:
        Question | Method | Key result | Interpretation | Limitation
    """
    rows = []

    def add(q, method, result, interp, limitation):
        rows.append({
            "Question": q, "Method": method, "Key_result": result,
            "Interpretation": interp, "Limitation": limitation,
        })

    # Q1 ranking stability across definitions
    if not ranking_sensitivity.empty:
        top = (ranking_sensitivity[ranking_sensitivity["rank"] == 1]
               .groupby("definition")["agent"].first())
        winners = top.value_counts()
        most = winners.index[0] if len(winners) else "n/a"
        stable = winners.iloc[0] / winners.sum() if len(winners) else 0.0
        add("Is the ranking stable across scoring definitions?",
            "7 scoring definitions (A–G), rank per definition",
            f"{most} top in {winners.iloc[0]}/{winners.sum()} definitions",
            "Stable" if stable >= 0.8 else "Framework-dependent",
            "Only a few scoring definitions; rule-based")

    # Q2 score vs volume
    if not score_volume_corr.empty:
        valid = score_volume_corr.dropna(subset=["spearman_rho"])
        if not valid.empty:
            top = valid.iloc[valid["spearman_rho"].abs().argmax()]
            add("Is AgentScore correlated with output volume?",
                "Spearman correlation (score vs volume metrics)",
                f"max |ρ|={abs(top['spearman_rho']):.2f} ({top['metric']})",
                "Possible volume sensitivity" if abs(top["spearman_rho"]) > 0.5
                else "Weak volume sensitivity",
                "n_runs small; correlation unstable")

    # Q3 proxy vs domain agreement
    if not run_scores.empty and not domain_scores.empty:
        pa = run_scores.groupby("agent")["AgentScore"].mean()
        da = domain_scores.groupby("agent")["DomainScore"].mean()
        common = pa.index.intersection(da.index)
        if len(common) >= 3:
            rho = stats.spearmanr(pa[common], da[common]).statistic
            add("Do proxy and domain-aware scores agree?",
                "Spearman of agent-mean proxy vs domain scores",
                f"ρ={rho:.2f}",
                "Agree" if rho >= 0.5 else "Partial/disagree",
                "Both rule-based; not expert review")

    # Q4 fairness / information-access sensitivity
    if fairness_summary is not None and not fairness_summary.empty:
        add("Are rankings sensitive to interaction/information-access status?",
            "Stratified AgentScore by metadata-provided status (run_registry)",
            "See fairness_sensitivity_summary.csv",
            "Information-seeking treated as agentic capability",
            "Observational; confounded with agent identity")
    else:
        add("Are rankings sensitive to interaction/information-access status?",
            "run_registry.csv not provided",
            "Not assessed",
            "Descriptive-only",
            "No registry available")

    # Q5 top agent stability under bootstrap
    if not ranking_error.empty:
        top1 = ranking_error.sort_values("expected_rank").iloc[0]
        add("Does the top-ranked agent remain stable after output-volume diagnostics / bootstrap?",
            "Bootstrap ranking (rank-switch probability)",
            f"{top1['agent']} switch prob={top1['rank_switch_probability']:.2f}",
            "Stable" if top1["rank_switch_probability"] < 0.25 else "Unstable",
            "Only 8 real runs per agent")

    # Q6 absolute / relative error per agent
    if not bootstrap_error.empty:
        ae = "; ".join(f"{r['agent']}: ±{r['absolute_error']:.2f}"
                       for _, r in bootstrap_error.iterrows())
        re = "; ".join(f"{r['agent']}: {r['relative_error_pct']:.1f}%"
                       for _, r in bootstrap_error.iterrows())
        add("What is the absolute error of each agent-level score?",
            "Bootstrap SD of the agent mean AgentScore",
            ae, "Descriptive uncertainty", "Bootstrap over 8 runs only")
        add("What is the relative error of each agent-level score?",
            "absolute_error / mean_score × 100",
            re, "Descriptive uncertainty", "Unstable when mean is small")

    # Q7 pairwise differences vs uncertainty
    if not pairwise_diff.empty:
        n_sep = int((pairwise_diff["diff_ci95_lo"] * pairwise_diff["diff_ci95_hi"] > 0).sum())
        add("Are pairwise score differences larger than their bootstrap uncertainty?",
            "Bootstrap 95% CI of pairwise difference excludes 0",
            f"{n_sep}/{len(pairwise_diff)} pairs separated",
            "Some/most pairs overlap" if n_sep < len(pairwise_diff) else "Pairs separated",
            "Practical equivalence band |Δ|<2")

    # Q8 reproducibility robustness
    if not repro_error.empty:
        wa = repro_error[repro_error["scope"] == "within_agent"]
        if not wa.empty:
            add("Are reproducibility estimates robust to run-level resampling?",
                "Bootstrap CI of within-agent similarity",
                "; ".join(f"{r['agent']}: {r['mean_within_similarity']:.2f}"
                          f"[{r['within_sim_ci95_lo']:.2f},{r['within_sim_ci95_hi']:.2f}]"
                          for _, r in wa.iterrows()),
                "Descriptive reproducibility", "Few pairwise comparisons")

    # Q9 uncertainty from volume sensitivity
    add("How much uncertainty is introduced by output-volume sensitivity?",
        "Compare ranking with/without volume-penalised score (defs A vs F/G)",
        "See ranking_sensitivity_analysis.csv",
        "Exploratory residualisation",
        "Residualisation may over/under-correct")

    # Score type note
    add("Are scores manual or proxy-estimated?",
        "Score source flag",
        f"score_type = {score_type}",
        "Proxy = exploratory until expert-validated" if score_type != "manual"
        else "Manual scores used as primary",
        "Proxy scores are not ground truth")

    return pd.DataFrame(rows)
