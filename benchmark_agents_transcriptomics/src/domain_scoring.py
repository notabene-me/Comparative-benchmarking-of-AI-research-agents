"""
domain_scoring.py — domain-aware transcriptomics scoring.

Each criterion is graded 0 / 1 / 2:
    0 = absent or incorrect
    1 = partially present
    2 = clearly present and methodologically appropriate

The normalised sum (0–100) is reported ALONGSIDE AgentScore as an
independent, rule-based sensitivity analysis. It is NOT a replacement
for AgentScore and NOT an expert review.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _grade(absent_cond: bool, partial_cond: bool, clear_cond: bool) -> int:
    if clear_cond:
        return 2
    if partial_cond:
        return 1
    return 0


def _c_rnaseq_platform(r: dict) -> int:
    signals = (r.get("kw_RNA-seq", 0) + r.get("kw_bulk_RNA-seq", 0)
               + r.get("kw_transcriptomics", 0) + r.get("kw_STAR", 0)
               + r.get("kw_salmon", 0) + r.get("kw_kallisto", 0))
    return _grade(signals == 0, signals == 1, signals >= 2)


def _c_qc_strategy(r: dict) -> int:
    qc = r.get("kw_QC", 0) + r.get("kw_FastQC", 0) + r.get("kw_MultiQC", 0)
    return _grade(qc == 0, qc == 1, qc >= 2)


def _c_normalization(r: dict) -> int:
    norm = r.get("kw_normalization", 0)
    scal = r.get("kw_scaling", 0)
    return _grade(norm == 0 and scal == 0, (norm > 0) ^ (scal > 0), norm > 0 and scal > 0)


def _c_batch_correction(r: dict) -> int:
    bc = r.get("kw_batch_correction", 0)
    return _grade(bc == 0, bc == 1, bc >= 2)


def _c_de_analysis(r: dict) -> int:
    de = (r.get("kw_DESeq2", 0) + r.get("kw_edgeR", 0) + r.get("kw_limma", 0)
          + r.get("kw_differential_expression", 0))
    return _grade(de == 0, de == 1, de >= 2)


def _c_multiple_testing(r: dict) -> int:
    bh = r.get("kw_Benjamini-Hochberg", 0)
    fdr = r.get("kw_FDR", 0) + r.get("n_fdr", 0)
    return _grade(fdr == 0 and bh == 0, fdr > 0 and bh == 0, bh > 0)


def _c_effect_sizes(r: dict) -> int:
    fc = r.get("kw_fold_change", 0)
    log2 = r.get("kw_log2FC", 0)
    return _grade(fc == 0 and log2 == 0, (fc > 0) ^ (log2 > 0), fc > 0 and log2 > 0)


def _c_pca_exploratory(r: dict) -> int:
    pca = r.get("kw_PCA", 0)
    plsda = r.get("kw_PLS-DA", 0) + r.get("kw_OPLS-DA", 0)
    return _grade(pca == 0 and plsda == 0, pca > 0 and plsda == 0, pca > 0 and plsda > 0)


def _c_pca_caution(r: dict) -> int:
    has_pca = r.get("kw_PCA", 0) > 0
    caution = r.get("aud_caution", 0) + r.get("aud_uncertainty", 0)
    if not has_pca:
        return 0
    return _grade(caution == 0, caution == 1, caution >= 2)


def _c_significant_genes(r: dict) -> int:
    sig = r.get("aud_significant", 0)
    genes = r.get("n_gene_mentions", 0)
    return _grade(sig == 0 and genes == 0, sig > 0 and genes <= 5, sig > 0 and genes > 5)


def _c_annotation_uncertainty(r: dict) -> int:
    ann = r.get("aud_annotation", 0)
    return _grade(ann == 0, ann == 1, ann >= 2)


def _c_avoid_overclaiming(r: dict) -> int:
    caution = r.get("aud_caution", 0)
    ann = r.get("aud_annotation", 0)
    return _grade(caution == 0 and ann == 0, (caution > 0) ^ (ann > 0), caution > 0 and ann > 0)


def _c_pathway_go(r: dict) -> int:
    pw = (r.get("kw_pathway_analysis", 0) + r.get("kw_KEGG", 0)
          + r.get("kw_GO_enrichment", 0) + r.get("kw_GSEA", 0)
          + r.get("kw_GSVA", 0) + r.get("kw_Reactome", 0)
          + r.get("n_pathway_mentions", 0))
    func = (r.get("bio_transcription_factor", 0) + r.get("bio_immune", 0)
            + r.get("bio_cell_cycle", 0) + r.get("bio_signaling", 0))
    return _grade(pw == 0 and func == 0, (pw > 0) ^ (func > 0), pw > 0 and func > 0)


def _c_biological_mechanism(r: dict) -> int:
    mech = r.get("aud_mechanism", 0)
    plaus = r.get("aud_plausibility", 0)
    return _grade(mech == 0 and plaus == 0, (mech > 0) ^ (plaus > 0), mech > 0 and plaus > 0)


def _c_limitations(r: dict) -> int:
    lim = r.get("aud_limitations", 0)
    return _grade(lim == 0, lim == 1, lim >= 2)


def _c_reproducible_code_tables(r: dict) -> int:
    code = r.get("n_code_blocks", 0)
    tables = r.get("n_tables", 0)
    return _grade(code == 0 and tables == 0, (code > 0) ^ (tables > 0), code > 0 and tables > 0)


DOMAIN_CRITERIA: List = [
    ("dm_rnaseq_platform",          _c_rnaseq_platform),
    ("dm_qc_strategy",              _c_qc_strategy),
    ("dm_normalization",            _c_normalization),
    ("dm_batch_correction",         _c_batch_correction),
    ("dm_de_analysis",              _c_de_analysis),
    ("dm_multiple_testing",         _c_multiple_testing),
    ("dm_effect_sizes",             _c_effect_sizes),
    ("dm_pca_exploratory",          _c_pca_exploratory),
    ("dm_pca_caution",              _c_pca_caution),
    ("dm_significant_genes",        _c_significant_genes),
    ("dm_annotation_uncertainty",   _c_annotation_uncertainty),
    ("dm_avoid_overclaiming",       _c_avoid_overclaiming),
    ("dm_pathway_go",               _c_pathway_go),
    ("dm_biological_mechanism",     _c_biological_mechanism),
    ("dm_limitations",              _c_limitations),
    ("dm_reproducible_code_tables", _c_reproducible_code_tables),
]

MAX_POINTS = len(DOMAIN_CRITERIA) * 2  # 16 × 2 = 32


def compute_domain_scores(run_features: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (agent, run) with domain criteria and DomainScore."""
    if run_features.empty:
        return pd.DataFrame()

    rows = []
    for _, feat_row in run_features.iterrows():
        rd = feat_row.to_dict()
        row = {"agent": rd["agent"], "run": rd["run"]}
        total = 0
        for name, fn in DOMAIN_CRITERIA:
            val = int(fn(rd))
            row[name] = val
            total += val
        row["domain_raw_sum"] = total
        row["domain_max"] = MAX_POINTS
        row["DomainScore"] = round(total / MAX_POINTS * 100, 2)
        rows.append(row)

    return pd.DataFrame(rows)


def domain_agent_summary(domain_scores: pd.DataFrame) -> pd.DataFrame:
    """Mean DomainScore per agent with SD and rank."""
    if domain_scores.empty:
        return pd.DataFrame()
    agg = (domain_scores.groupby("agent")["DomainScore"]
           .agg(["mean", "std", "count"]).reset_index()
           .rename(columns={"mean": "mean_domain_score",
                            "std": "sd_domain_score",
                            "count": "n_runs"}))
    agg["sd_domain_score"] = agg["sd_domain_score"].fillna(0.0)
    agg["rank"] = agg["mean_domain_score"].rank(ascending=False).astype(int)
    return agg.sort_values("rank")
