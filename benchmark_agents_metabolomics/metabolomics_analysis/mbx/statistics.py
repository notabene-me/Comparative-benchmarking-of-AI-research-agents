"""Univariate differential-abundance statistics.

For each metabolite we compute, comparing case vs. control:
    - log2 fold change (case / control), on the linear normalized scale
    - parametric p-value (paired or Welch's t-test, on the log scale)
    - non-parametric p-value (Wilcoxon signed-rank or Mann-Whitney U)
    - Benjamini-Hochberg FDR (q-value) for the chosen primary test
    - Cohen's d effect size
    - univariate ROC AUC (case vs control discrimination)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import Config
from .io_utils import Dataset
from .preprocessing import Processed


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Return BH-adjusted q-values for a vector of p-values (NaN-safe)."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    pm = p[mask]
    n = pm.size
    if n == 0:
        return q
    order = np.argsort(pm)
    ranked = pm[order]
    adj = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity from the largest p downward
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    q[mask] = out
    return q


def _auc(case: np.ndarray, control: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (rank-based)."""
    case = case[np.isfinite(case)]
    control = control[np.isfinite(control)]
    n1, n0 = case.size, control.size
    if n1 == 0 or n0 == 0:
        return np.nan
    combined = np.concatenate([case, control])
    ranks = stats.rankdata(combined)
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    auc = u1 / (n1 * n0)
    return max(auc, 1 - auc)   # report discrimination irrespective of direction


def _cohens_d(case: np.ndarray, control: np.ndarray, paired: bool) -> float:
    case = case[np.isfinite(case)]
    control = control[np.isfinite(control)]
    if paired and case.size == control.size and case.size > 1:
        diff = case - control
        sd = diff.std(ddof=1)
        return float(diff.mean() / sd) if sd > 0 else np.nan
    if case.size < 2 or control.size < 2:
        return np.nan
    n1, n0 = case.size, control.size
    sp = np.sqrt(((n1 - 1) * case.var(ddof=1) + (n0 - 1) * control.var(ddof=1)) / (n1 + n0 - 2))
    return float((case.mean() - control.mean()) / sp) if sp > 0 else np.nan


def differential_abundance(ds: Dataset, proc: Processed) -> pd.DataFrame:
    cfg: Config = ds.cfg
    ctrl_label, case_label = cfg.group_labels
    ctrl_cols = ds.group_cols[ctrl_label]
    case_cols = ds.group_cols[case_label]

    lin = proc.normalized          # linear scale for fold change
    log = proc.transformed         # log scale for t-tests

    rows = []
    for feat in lin.index:
        ctrl_lin = lin.loc[feat, ctrl_cols].to_numpy(dtype=float)
        case_lin = lin.loc[feat, case_cols].to_numpy(dtype=float)
        ctrl_log = log.loc[feat, ctrl_cols].to_numpy(dtype=float)
        case_log = log.loc[feat, case_cols].to_numpy(dtype=float)

        ctrl_ok = ctrl_lin[np.isfinite(ctrl_lin)]
        case_ok = case_lin[np.isfinite(case_lin)]
        if ctrl_ok.size < cfg.min_samples_per_group or case_ok.size < cfg.min_samples_per_group:
            continue

        mean_ctrl = np.nanmean(ctrl_lin)
        mean_case = np.nanmean(case_lin)
        log2fc = np.log2(mean_case / mean_ctrl) if mean_ctrl > 0 and mean_case > 0 else np.nan

        # Parametric + non-parametric tests on the log scale.
        p_param = np.nan
        p_np = np.nan
        try:
            if cfg.paired and len(ds.pairs) >= cfg.min_samples_per_group:
                c_idx = [ctrl_cols.index(c) for c, _ in ds.pairs]
                k_idx = [case_cols.index(k) for _, k in ds.pairs]
                a = case_log[k_idx]
                b = ctrl_log[c_idx]
                ok = np.isfinite(a) & np.isfinite(b)
                a, b = a[ok], b[ok]
                if a.size >= cfg.min_samples_per_group:
                    p_param = stats.ttest_rel(a, b).pvalue
                    if np.any(a - b != 0):
                        p_np = stats.wilcoxon(a, b, zero_method="wilcox").pvalue
            else:
                p_param = stats.ttest_ind(case_log, ctrl_log, equal_var=False,
                                          nan_policy="omit").pvalue
                p_np = stats.mannwhitneyu(case_ok, ctrl_ok, alternative="two-sided").pvalue
        except ValueError:
            pass

        rows.append({
            "metabolite": feat,
            "mean_control": mean_ctrl,
            "mean_case": mean_case,
            "log2fc": log2fc,
            "fold_change": (mean_case / mean_ctrl) if mean_ctrl > 0 else np.nan,
            "p_param": p_param,
            "p_nonparam": p_np,
            "cohens_d": _cohens_d(case_log, ctrl_log, cfg.paired),
            "auc": _auc(case_lin, ctrl_lin),
            "n_control": int(ctrl_ok.size),
            "n_case": int(case_ok.size),
        })

    res = pd.DataFrame(rows).set_index("metabolite")

    # Primary p = parametric; fall back to non-parametric where missing.
    primary = res["p_param"].fillna(res["p_nonparam"])
    res["p_value"] = primary
    res["q_value"] = benjamini_hochberg(primary.to_numpy())
    res["q_nonparam"] = benjamini_hochberg(res["p_nonparam"].to_numpy())

    res["neg_log10_p"] = -np.log10(res["p_value"].clip(lower=1e-300))
    res["significant"] = (res["q_value"] < cfg.alpha)
    res["regulation"] = np.where(
        res["significant"] & (res["log2fc"] > 0), "up",
        np.where(res["significant"] & (res["log2fc"] < 0), "down", "ns"),
    )
    res["candidate_biomarker"] = (
        res["significant"] & (res["log2fc"].abs() >= cfg.log2fc_threshold)
    )

    res = res.sort_values(["q_value", "p_value"])
    return res


def summarize(res: pd.DataFrame, cfg: Config) -> dict:
    sig = res[res["significant"]]
    return {
        "n_tested": int(len(res)),
        "n_significant": int(len(sig)),
        "n_up": int((sig["regulation"] == "up").sum()),
        "n_down": int((sig["regulation"] == "down").sum()),
        "n_candidate_biomarkers": int(res["candidate_biomarker"].sum()),
        "alpha_fdr": cfg.alpha,
        "log2fc_threshold": cfg.log2fc_threshold,
        "top_up": sig[sig["regulation"] == "up"].head(10).index.tolist(),
        "top_down": sig[sig["regulation"] == "down"].head(10).index.tolist(),
    }
