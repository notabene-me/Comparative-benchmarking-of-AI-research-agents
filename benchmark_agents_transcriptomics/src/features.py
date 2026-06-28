"""
features.py — extract structural and methodological features from file texts.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------

METHODOLOGICAL_KEYWORDS: Dict[str, List[str]] = {
    "PCA":                  [r"\bPCA\b", r"principal component analysis"],
    "PLS-DA":               [r"\bPLS[-\s]?DA\b"],
    "OPLS-DA":              [r"\bOPLS[-\s]?DA\b"],
    "t-test":               [r"\bt[-\s]?test\b"],
    "Mann-Whitney":         [r"\bMann[-\s]Whitney\b"],
    "Wilcoxon":             [r"\bWilcoxon\b"],
    "ANOVA":                [r"\bANOVA\b"],
    "FDR":                  [r"\bFDR\b"],
    "Benjamini-Hochberg":   [r"\bBenjamini[-\s]Hochberg\b", r"\bBH\s+correction\b"],
    "fold_change":          [r"\bfold\s+change\b", r"\bFC\b"],
    "log2FC":               [r"\blog2?FC\b", r"\blog[_\s]?fold\s+change\b"],
    "volcano_plot":         [r"\bvolcano\s+plot\b"],
    "heatmap":              [r"\bheatmap\b"],
    "QC":                   [r"\bQC\b", r"\bquality\s+control\b"],
    "FastQC":               [r"\bFastQC\b"],
    "MultiQC":              [r"\bMultiQC\b"],
    "batch_correction":     [r"\bbatch\s+correction\b", r"\bComBat\b", r"\bRUV\b"],
    "normalization":        [r"\bnormali[sz]ation\b", r"\bnormali[sz]e[d]?\b", r"\bTMM\b"],
    "scaling":              [r"\bscaling\b", r"\bunit\s+variance\b", r"\bvst\b", r"\brlog\b"],
    "imputation":           [r"\bimput(ation|ed|e)\b"],
    "pathway_analysis":     [r"\bpathway\s+analysis\b", r"\bpathway\s+enrichment\b"],
    "GO_enrichment":        [r"\bGO\s+enrichment\b", r"\bgene\s+ontology\s+enrichment\b"],
    "GSEA":                 [r"\bGSEA\b", r"\bgene\s+set\s+enrichment\b"],
    "GSVA":                 [r"\bGSVA\b"],
    "KEGG":                 [r"\bKEGG\b"],
    "Reactome":             [r"\bReactome\b"],
    "MSigDB":               [r"\bMSigDB\b", r"\bmolecular\s+signatures\s+database\b"],
    "clusterProfiler":      [r"\bclusterProfiler\b"],
    "Ensembl":              [r"\bEnsembl\b"],
    "GENCODE":              [r"\bGENCODE\b"],
    "DESeq2":               [r"\bDESeq2\b"],
    "edgeR":                [r"\bedgeR\b"],
    "limma":                [r"\blimma\b", r"\bvoom\b"],
    "RNA-seq":              [r"\bRNA[-\s]?seq\b", r"\bRNA\s+sequencing\b"],
    "bulk_RNA-seq":         [r"\bbulk\s+RNA[-\s]?seq\b"],
    "transcriptomics":      [r"\btranscriptomics\b", r"\btranscriptome\b"],
    "STAR":                 [r"\bSTAR\b"],
    "HISAT2":               [r"\bHISAT2\b"],
    "salmon":               [r"\bsalmon\b"],
    "kallisto":             [r"\bkallisto\b"],
    "featureCounts":        [r"\bfeatureCounts\b", r"\bfeature\s+counts\b"],
    "HTSeq":                [r"\bHTSeq\b"],
    "TPM":                  [r"\bTPM\b"],
    "FPKM":                 [r"\bFPKM\b"],
    "RPKM":                 [r"\bRPKM\b"],
    "counts":               [r"\b(raw\s+)?counts\b", r"\bcount\s+matrix\b"],
    "differential_expression": [r"\bdifferential\s+expression\b", r"\bDEG[s]?\b",
                               r"\bdifferentially\s+expressed\s+gene[s]?\b"],
}

BIOLOGICAL_TERMS: Dict[str, List[str]] = {
    "gene":                 [r"\bgene[s]?\b"],
    "transcript":           [r"\btranscript[s]?\b", r"\bisoform[s]?\b"],
    "pathway":              [r"\bpathway[s]?\b"],
    "GO":                   [r"\bgene\s+ontology\b", r"\bGO\s+term[s]?\b"],
    "transcription_factor": [r"\btranscription\s+factor[s]?\b"],
    "immune":               [r"\bimmune\b", r"\bimmunolog"],
    "inflammation":         [r"\binflammation\b", r"\binflammatory\b"],
    "cell_cycle":           [r"\bcell\s+cycle\b"],
    "apoptosis":            [r"\bapoptosis\b", r"\bapoptotic\b"],
    "metabolism":           [r"\bmetaboli(sm|c)\b"],
    "signaling":            [r"\bsignal(ing| transduction)\b"],
    "oxidative_stress":     [r"\boxidative\s+stress\b"],
    "biomarker":            [r"\bbiomarker[s]?\b"],
    "lncRNA":               [r"\blncRNA[s]?\b", r"\blong\s+non[-\s]coding\s+RNA\b"],
    "miRNA":                [r"\bmiRNA[s]?\b", r"\bmicroRNA[s]?\b"],
    "splicing":             [r"\bsplic(ing|ed|e)\b", r"\balternative\s+splicing\b"],
    "epigenetic":           [r"\bepigenetic\b", r"\bDNA\s+methylation\b", r"\bhistone\b"],
    "feature_selection":    [r"\bfeature\s+selection\b"],
}

# Audit keywords — used for proxy decomposition and domain-aware scoring.
# Kept separate from METHODOLOGICAL_KEYWORDS so they do NOT alter the
# methodological Jaccard sets used for similarity.
AUDIT_KEYWORDS: Dict[str, List[str]] = {
    "preprocessing":   [r"\bpre[-\s]?process", r"\bdata\s+cleaning\b", r"\btrim(ming)?\b",
                        r"\badapt(er)?\s+remov", r"\balignment\b", r"\bmapping\b",
                        r"\bquantification\b", r"\bcount\s+matrix\b"],
    "significant":     [r"\bsignificant", r"\bsignif\.", r"\bdifferentially\s+expressed\b",
                        r"\bDEG[s]?\b"],
    "limitations":     [r"\blimitation", r"\bcaveat", r"\bshortcoming"],
    "mechanism":       [r"\bmechanis", r"\bmode\s+of\s+action\b"],
    "plausibility":    [r"\bplausib", r"\bbiologically\s+(meaningful|relevant)\b"],
    "annotation":      [r"\bannotation", r"\bputativ", r"\btentative", r"\blevel\s+[1-3]\b",
                        r"\bconfidence\s+level\b", r"\bidentification\s+confidence\b"],
    "consistency":     [r"\bconsisten", r"\breproducib", r"\bacross\s+runs\b",
                        r"\bacross\s+replicates\b"],
    "uncertainty":     [r"\buncertain", r"\bconfidence\s+interval", r"\bvariabilit"],
    "exploratory":     [r"\bexploratory\b", r"\bunsupervised\b"],
    "caution":         [r"\bcaution", r"\bshould\s+be\s+interpreted", r"\bnot\s+conclusive\b",
                        r"\bpreliminary\b", r"\bovercla"],
}

P_VALUE_PAT  = re.compile(r"\bp[-\s]?val", re.IGNORECASE)
FDR_PAT      = re.compile(r"\bFDR\b", re.IGNORECASE)
TABLE_PAT    = re.compile(r"\|[-\s|]+\||\t.*\t|<table", re.IGNORECASE)
IMG_PAT      = re.compile(r"!\[.*?\]\(|<img\s|\.png\b|\.jpg\b|\.jpeg\b|\.svg\b", re.IGNORECASE)
NUM_PAT      = re.compile(r"\b\d+(?:[.,]\d+)*\b")
CODE_PAT     = re.compile(r"```|~~~|<code>", re.IGNORECASE)
# Matches the sheet-header lines emitted by _read_excel: "=== Sheet: <name> ==="
EXCEL_SHEET_PAT = re.compile(r"^=== Sheet:", re.MULTILINE)


# ---------------------------------------------------------------------------
# Core feature extractor
# ---------------------------------------------------------------------------

def extract_file_features(file_id: str, text: str) -> dict:
    """Extract structural and methodological features from a single file's text."""
    n_chars  = len(text)
    words    = text.split()
    n_words  = len(words)
    n_lines  = text.count("\n") + 1 if text else 0
    # For Excel workbooks the parser emits one "=== Sheet: X ===" header per
    # sheet; each sheet counts as a table.  For other formats count markdown/
    # HTML table markers as before.
    n_sheets = len(EXCEL_SHEET_PAT.findall(text))
    n_tables = n_sheets if n_sheets > 0 else len(TABLE_PAT.findall(text))
    n_images = len(IMG_PAT.findall(text))
    n_nums   = len(NUM_PAT.findall(text))
    n_pvals  = len(P_VALUE_PAT.findall(text))
    n_fdr    = len(FDR_PAT.findall(text))
    n_code   = len(CODE_PAT.findall(text))

    features: dict = {
        "file_id":   file_id,
        "n_chars":   n_chars,
        "n_words":   n_words,
        "n_lines":   n_lines,
        "n_sheets":  n_sheets,   # >0 only for Excel workbooks
        "n_tables":  n_tables,
        "n_images":  n_images,
        "n_numeric": n_nums,
        "n_pvalues": n_pvals,
        "n_fdr":     n_fdr,
        "n_code_blocks": n_code,
    }

    # Methodological keyword counts
    text_lower = text  # already mixed case — use case-insensitive regex
    for kw, patterns in METHODOLOGICAL_KEYWORDS.items():
        cnt = sum(len(re.findall(p, text, re.IGNORECASE)) for p in patterns)
        features[f"kw_{kw}"] = cnt

    # Biological term counts
    n_genes = 0
    n_pathways = 0
    for term, patterns in BIOLOGICAL_TERMS.items():
        cnt = sum(len(re.findall(p, text, re.IGNORECASE)) for p in patterns)
        features[f"bio_{term}"] = cnt
        if term == "gene":
            n_genes = cnt
        if term == "pathway":
            n_pathways = cnt

    features["n_gene_mentions"] = n_genes
    features["n_pathway_mentions"] = n_pathways

    # Audit keyword counts (aud_*)
    for term, patterns in AUDIT_KEYWORDS.items():
        cnt = sum(len(re.findall(p, text, re.IGNORECASE)) for p in patterns)
        features[f"aud_{term}"] = cnt

    return features


def extract_all_file_features(
    file_texts: Dict[str, str],
) -> pd.DataFrame:
    """Return a DataFrame with one row per file."""
    rows = []
    for fid, text in file_texts.items():
        rows.append(extract_file_features(fid, text))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def aggregate_run_features(
    file_features: pd.DataFrame,
    file_meta: List[dict],
) -> pd.DataFrame:
    """
    Sum / mean file-level features per (agent, run).

    Numeric columns are summed; n_* columns summed; kw_* summed.
    """
    if file_features.empty:
        return pd.DataFrame()

    meta_df = pd.DataFrame(file_meta)[["file_id", "agent", "run"]]
    merged = meta_df.merge(file_features, on="file_id", how="left")

    numeric_cols = [c for c in merged.columns
                    if c not in {"file_id", "agent", "run"}]
    agg = merged.groupby(["agent", "run"])[numeric_cols].sum().reset_index()
    agg["n_files"] = merged.groupby(["agent", "run"])["file_id"].count().values
    return agg


def aggregate_agent_features(run_features: pd.DataFrame) -> pd.DataFrame:
    if run_features.empty:
        return pd.DataFrame()
    numeric_cols = [c for c in run_features.columns
                    if c not in {"agent", "run"}]
    agg = run_features.groupby("agent")[numeric_cols].mean().reset_index()
    agg.rename(columns={c: f"mean_{c}" for c in numeric_cols}, inplace=True)
    return agg


# ---------------------------------------------------------------------------
# Keyword-set helpers (for Jaccard similarity)
# ---------------------------------------------------------------------------

def keyword_set(features_row: dict) -> set:
    """Return the set of methodological keywords with count > 0."""
    return {kw for kw in METHODOLOGICAL_KEYWORDS
            if features_row.get(f"kw_{kw}", 0) > 0}


def bio_term_set(features_row: dict) -> set:
    """Return the set of biological terms with count > 0."""
    return {t for t in BIOLOGICAL_TERMS
            if features_row.get(f"bio_{t}", 0) > 0}


# ---------------------------------------------------------------------------
# Proxy score helpers
# ---------------------------------------------------------------------------

# Each component function returns an ordered dict {component_name: 0/1}.
# The proxy subtotal is the mean of components × 100, so the decomposition
# is exact and fully auditable.

# Components flagged as "volume/artifact-related" — used by the
# output-volume-excluded scoring variant in score_audit.py.
VOLUME_RELATED_COMPONENTS = {
    "P_reproducible_tables",
    "R_result_tables_provided",
}


def proxy_process_components(row: dict) -> dict:
    """Binary components of Process quality (P)."""
    return {
        "P_preprocessing_described":
            int(row.get("aud_preprocessing", 0) > 0 or row.get("kw_imputation", 0) > 0),
        "P_normalization_described":
            int(row.get("kw_normalization", 0) > 0 or row.get("kw_scaling", 0) > 0),
        "P_qc_described":
            int(row.get("kw_QC", 0) > 0 or row.get("kw_pooled_QC", 0) > 0),
        "P_missing_value_handling":
            int(row.get("kw_imputation", 0) > 0),
        "P_statistical_testing":
            int(row.get("kw_t-test", 0) > 0 or row.get("kw_Mann-Whitney", 0) > 0
                or row.get("kw_Wilcoxon", 0) > 0 or row.get("kw_ANOVA", 0) > 0),
        "P_multiple_testing_correction":
            int(row.get("kw_FDR", 0) > 0 or row.get("kw_Benjamini-Hochberg", 0) > 0
                or row.get("n_fdr", 0) > 0),
        "P_code_workflow_transparency":
            int(row.get("n_code_blocks", 0) > 0),
        "P_reproducible_tables":
            int(row.get("n_tables", 0) > 0),
    }


def proxy_result_components(row: dict) -> dict:
    """Binary components of Result accuracy (R)."""
    return {
        "R_numerical_results_present":
            int(row.get("n_numeric", 0) > 10),
        "R_pvalues_present":
            int(row.get("n_pvalues", 0) > 0),
        "R_fdr_qvalues_present":
            int(row.get("n_fdr", 0) > 0 or row.get("kw_FDR", 0) > 0),
        "R_fold_changes_present":
            int(row.get("kw_fold_change", 0) > 0 or row.get("kw_log2FC", 0) > 0),
        "R_significant_features_reported":
            int(row.get("aud_significant", 0) > 0),
        "R_result_tables_provided":
            int(row.get("n_tables", 0) > 0),
        "R_consistency_run_level":
            int(row.get("aud_consistency", 0) > 0),
    }


def proxy_depth_components(row: dict) -> dict:
    """Binary components of Interpretation depth (D)."""
    return {
        "D_gene_interpretation":
            int(row.get("n_gene_mentions", 0) > 5),
        "D_pathway_interpretation":
            int(row.get("n_pathway_mentions", 0) > 0 or row.get("kw_pathway_analysis", 0) > 0
                or row.get("kw_KEGG", 0) > 0 or row.get("kw_GO_enrichment", 0) > 0
                or row.get("kw_GSEA", 0) > 0 or row.get("kw_GSVA", 0) > 0
                or row.get("kw_Reactome", 0) > 0),
        "D_gene_function_interpretation":
            int(row.get("bio_transcription_factor", 0) > 0 or row.get("bio_immune", 0) > 0
                or row.get("bio_cell_cycle", 0) > 0 or row.get("bio_signaling", 0) > 0
                or row.get("bio_metabolism", 0) > 0 or row.get("bio_splicing", 0) > 0),
        "D_biological_mechanism":
            int(row.get("aud_mechanism", 0) > 0),
        "D_uncertainty_limitations":
            int(row.get("aud_limitations", 0) > 0 or row.get("aud_uncertainty", 0) > 0),
        "D_annotation_confidence_caution":
            int(row.get("aud_annotation", 0) > 0),
        "D_biological_plausibility":
            int(row.get("aud_plausibility", 0) > 0),
    }


def _mean_components(components: dict, exclude: Optional[set] = None) -> float:
    exclude = exclude or set()
    vals = [v for k, v in components.items() if k not in exclude]
    if not vals:
        return 0.0
    return round(sum(vals) / len(vals) * 100, 2)


def compute_proxy_process(row: dict, exclude: Optional[set] = None) -> float:
    """Proxy Process quality (P), 0–100 — mean of binary components."""
    return _mean_components(proxy_process_components(row), exclude)


def compute_proxy_result(row: dict, exclude: Optional[set] = None) -> float:
    """Proxy Result accuracy (R), 0–100 — mean of binary components."""
    return _mean_components(proxy_result_components(row), exclude)


def compute_proxy_depth(row: dict, exclude: Optional[set] = None) -> float:
    """Proxy Interpretation depth (D), 0–100 — mean of binary components."""
    return _mean_components(proxy_depth_components(row), exclude)


def all_proxy_components(row: dict) -> dict:
    """Merged dict of all P/R/D components for a feature row."""
    out = {}
    out.update(proxy_process_components(row))
    out.update(proxy_result_components(row))
    out.update(proxy_depth_components(row))
    return out
