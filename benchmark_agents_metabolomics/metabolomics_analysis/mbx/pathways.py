"""Pathway over-representation analysis (ORA) and gene mapping.

ORA tests whether the set of significant metabolites is enriched for members of
each pathway relative to the background of all *detected* (analyzed) metabolites,
using Fisher's exact test (hypergeometric). q-values are BH-adjusted.

Gene mapping links each affected pathway to its representative human enzyme /
transporter genes, and each significant metabolite to the pathways and genes it
participates in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import knowledge_base as kb
from .config import Config
from .statistics import benjamini_hochberg


def annotate_metabolites(metabolites: list[str]) -> pd.DataFrame:
    """Map each metabolite to its pathways (one row per metabolite)."""
    rows = []
    for m in metabolites:
        pws = kb.map_metabolite(m)
        rows.append({
            "metabolite": m,
            "n_pathways": len(pws),
            "pathways": "; ".join(pws),
            "mapped": bool(pws),
        })
    return pd.DataFrame(rows).set_index("metabolite")


def _pathway_membership(detected: list[str]) -> dict[str, set[str]]:
    """pathway -> set of detected metabolites that belong to it."""
    membership: dict[str, set[str]] = {pw: set() for pw in kb.all_pathways()}
    for m in detected:
        for pw in kb.map_metabolite(m):
            membership.setdefault(pw, set()).add(m)
    return membership


def enrichment_analysis(stats_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Over-representation analysis of significant metabolites across pathways."""
    detected = list(stats_df.index)
    sig = set(stats_df[stats_df["significant"]].index)

    membership = _pathway_membership(detected)
    # Background = metabolites mapped to *any* pathway (standard ORA convention).
    background = set().union(*membership.values()) if membership else set()
    N = len(background)
    n_sig_in_bg = len(sig & background)

    rows = []
    for pw, members in membership.items():
        members = members & background
        K = len(members)                      # pathway size in background
        if K < cfg.min_pathway_size:
            continue
        hits = members & sig                  # significant members
        k = len(hits)
        # Fisher exact on the 2x2 contingency table.
        a = k
        b = K - k
        c = n_sig_in_bg - k
        d = N - K - c
        table = [[a, b], [c, d]]
        try:
            _, p = stats.fisher_exact(table, alternative="greater")
        except ValueError:
            p = np.nan
        expected = K * n_sig_in_bg / N if N else np.nan
        fold = (k / expected) if expected and expected > 0 else np.nan
        member_stats = stats_df.loc[sorted(hits)] if hits else stats_df.iloc[0:0]
        mean_log2fc = member_stats["log2fc"].mean() if len(member_stats) else np.nan
        rows.append({
            "pathway": pw,
            "class": kb.CLASS_OF_PATHWAY.get(pw, "Other"),
            "pathway_size": K,
            "n_significant": k,
            "expected": expected,
            "enrichment_ratio": fold,
            "p_value": p,
            "mean_log2fc_sig": mean_log2fc,
            "n_genes": len(kb.PATHWAY_GENES.get(pw, [])),
            "hit_metabolites": "; ".join(sorted(hits)),
            "genes": "; ".join(kb.PATHWAY_GENES.get(pw, [])),
        })

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["q_value"] = benjamini_hochberg(res["p_value"].to_numpy())
    res["neg_log10_p"] = -np.log10(res["p_value"].clip(lower=1e-300))
    res["enriched"] = res["q_value"] < cfg.enrichment_alpha
    res = res.sort_values(["p_value", "enrichment_ratio"], ascending=[True, False])
    return res.set_index("pathway")


def gene_table(stats_df: pd.DataFrame, enrich_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per-gene table aggregating evidence from significant metabolites.

    A gene is implicated if it belongs to a pathway that contains >=1 significant
    metabolite. We score it by the number of significant metabolites in its
    pathways and the strongest pathway enrichment it participates in.
    """
    sig = stats_df[stats_df["significant"]]
    sig_set = set(sig.index)
    membership = _pathway_membership(list(stats_df.index))

    gene_rows: dict[str, dict] = {}
    for pw, genes in kb.PATHWAY_GENES.items():
        members = membership.get(pw, set())
        sig_members = members & sig_set
        if not sig_members:
            continue
        pw_q = np.nan
        if not enrich_df.empty and pw in enrich_df.index:
            pw_q = enrich_df.loc[pw, "q_value"]
        for g in genes:
            rec = gene_rows.setdefault(g, {
                "gene": g, "pathways": set(), "sig_metabolites": set(),
                "best_pathway_q": np.nan,
            })
            rec["pathways"].add(pw)
            rec["sig_metabolites"].update(sig_members)
            if np.isnan(rec["best_pathway_q"]) or (not np.isnan(pw_q) and pw_q < rec["best_pathway_q"]):
                rec["best_pathway_q"] = pw_q

    rows = []
    for g, rec in gene_rows.items():
        rows.append({
            "gene": g,
            "n_pathways": len(rec["pathways"]),
            "pathways": "; ".join(sorted(rec["pathways"])),
            "n_sig_metabolites": len(rec["sig_metabolites"]),
            "sig_metabolites": "; ".join(sorted(rec["sig_metabolites"])),
            "best_pathway_q": rec["best_pathway_q"],
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["n_sig_metabolites", "best_pathway_q"],
                          ascending=[False, True])
    return out.set_index("gene")
