"""
consensus_divergence.py — consensus, commonality, and divergence analysis.

Analyses what analytical elements are common across agents (consensus core)
versus agent-specific (divergence), and scores each run and agent on how
closely they align with the cross-agent consensus.

All uncertainty estimates are DESCRIPTIVE over the 8 real runs per agent.
They are NOT population-level confidence intervals.

Outputs (all saved by main.py):
  tables/
    consensus_method_terms.csv
    consensus_biological_terms.csv
    consensus_reporting_terms.csv
    shared_terms_summary.csv
    agent_specific_terms.csv
    consensus_core_terms.csv
    consensus_alignment_scores.csv
    consensus_alignment_error_summary.csv
    agent_distinctiveness_scores.csv
    agent_distinctiveness_error_summary.csv
    commonality_difference_summary.csv
    consensus_validation_summary.csv
"""

import logging
import re
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEED = 42

# ---------------------------------------------------------------------------
# Term dictionaries
# ---------------------------------------------------------------------------

METHOD_TERMS: Dict[str, List[str]] = {
    "PCA":                   [r"\bPCA\b", r"principal component analysis"],
    "PLS-DA":                [r"\bPLS[-\s]?DA\b"],
    "OPLS-DA":               [r"\bOPLS[-\s]?DA\b"],
    "t-test":                [r"\bt[-\s]?test\b"],
    "Mann-Whitney":          [r"\bMann[-\s]Whitney\b"],
    "Wilcoxon":              [r"\bWilcoxon\b"],
    "ANOVA":                 [r"\bANOVA\b"],
    "FDR":                   [r"\bFDR\b"],
    "Benjamini-Hochberg":    [r"\bBenjamini[-\s]Hochberg\b", r"\bBH\s+correction\b"],
    "fold_change":           [r"\bfold\s+change\b"],
    "log2FC":                [r"\blog2?FC\b", r"\blog[_\s]?fold\s+change\b"],
    "volcano_plot":          [r"\bvolcano\s+plot\b"],
    "heatmap":               [r"\bheatmap\b"],
    "QC":                    [r"\bQC\b", r"\bquality\s+control\b"],
    "FastQC":                [r"\bFastQC\b"],
    "MultiQC":               [r"\bMultiQC\b"],
    "batch_correction":      [r"\bbatch\s+correction\b", r"\bComBat\b", r"\bRUV\b"],
    "normalization":         [r"\bnormali[sz]ation\b", r"\bTMM\b"],
    "scaling":               [r"\bscaling\b", r"\bvst\b", r"\brlog\b"],
    "imputation":            [r"\bimput(ation|ed|e)\b"],
    "pathway_analysis":      [r"\bpathway\s+(analysis|enrichment)\b"],
    "GO_enrichment":         [r"\bGO\s+enrichment\b", r"\bgene\s+ontology\s+enrichment\b"],
    "GSEA":                  [r"\bGSEA\b", r"\bgene\s+set\s+enrichment\b"],
    "GSVA":                  [r"\bGSVA\b"],
    "KEGG":                  [r"\bKEGG\b"],
    "Reactome":              [r"\bReactome\b"],
    "MSigDB":                [r"\bMSigDB\b", r"\bmolecular\s+signatures\s+database\b"],
    "clusterProfiler":       [r"\bclusterProfiler\b"],
    "Ensembl":               [r"\bEnsembl\b"],
    "GENCODE":               [r"\bGENCODE\b"],
    "DESeq2":                [r"\bDESeq2\b"],
    "edgeR":                 [r"\bedgeR\b"],
    "limma":                 [r"\blimma\b", r"\bvoom\b"],
    "RNA-seq":               [r"\bRNA[-\s]?seq\b", r"\bRNA\s+sequencing\b"],
    "bulk_RNA-seq":          [r"\bbulk\s+RNA[-\s]?seq\b"],
    "transcriptomics":       [r"\btranscriptomics\b", r"\btranscriptome\b"],
    "STAR":                  [r"\bSTAR\b"],
    "HISAT2":                [r"\bHISAT2\b"],
    "salmon":                [r"\bsalmon\b"],
    "kallisto":              [r"\bkallisto\b"],
    "featureCounts":         [r"\bfeatureCounts\b", r"\bfeature\s+counts\b"],
    "TPM":                   [r"\bTPM\b"],
    "FPKM":                  [r"\bFPKM\b"],
    "RPKM":                  [r"\bRPKM\b"],
    "counts":                [r"\b(raw\s+)?counts\b", r"\bcount\s+matrix\b"],
    "differential_expression": [r"\bdifferential\s+expression\b", r"\bDEG[s]?\b",
                                r"\bdifferentially\s+expressed\s+gene[s]?\b"],
}

BIO_TERMS: Dict[str, List[str]] = {
    "genes":                   [r"\bgene[s]?\b"],
    "transcripts":             [r"\btranscript[s]?\b", r"\bisoform[s]?\b"],
    "pathway_enrichment":      [r"\bpathway\s+enrichment\b"],
    "GO":                      [r"\bgene\s+ontology\b", r"\bGO\s+term[s]?\b"],
    "transcription_factors":   [r"\btranscription\s+factor[s]?\b"],
    "immune":                  [r"\bimmune\b", r"\bimmunolog"],
    "inflammation":            [r"\binflammation\b", r"\binflammatory\b"],
    "cell_cycle":              [r"\bcell\s+cycle\b"],
    "apoptosis":               [r"\bapoptosis\b", r"\bapoptotic\b"],
    "metabolism":              [r"\bmetaboli(sm|c)\b"],
    "signaling":               [r"\bsignal(ing| transduction)\b"],
    "oxidative_stress":        [r"\boxidative\s+stress\b"],
    "lncRNA":                  [r"\blncRNA[s]?\b", r"\blong\s+non[-\s]coding\s+RNA\b"],
    "miRNA":                   [r"\bmiRNA[s]?\b", r"\bmicroRNA[s]?\b"],
    "splicing":                [r"\bsplic(ing|ed|e)\b", r"\balternative\s+splicing\b"],
    "epigenetic":              [r"\bepigenetic\b", r"\bDNA\s+methylation\b", r"\bhistone\b"],
    "biomarker":               [r"\bbiomarker[s]?\b"],
}

REPORTING_TERMS: Dict[str, List[str]] = {
    "p-value":            [r"\bp[-\s]?value[s]?\b", r"\bp\s*<\s*0\.\d+"],
    "adjusted_p-value":   [r"\badjusted\s+p[-\s]?value[s]?\b", r"\bq[-\s]?value[s]?\b"],
    "confidence_interval":[r"\bconfidence\s+interval[s]?\b", r"\b95\s*%\s*CI\b"],
    "effect_size":        [r"\beffect\s+size[s]?\b"],
    "sample_size":        [r"\bsample\s+size[s]?\b", r"\bn\s*=\s*\d+"],
    "limitations":        [r"\blimitation[s]?\b", r"\bcaveat[s]?\b"],
    "reproducibility":    [r"\breproducib(ility|le)\b"],
    "validation":         [r"\bvalidat(ion|ed|e)\b"],
    "significant":        [r"\bstatistically\s+significant\b", r"\bsignificant\s+difference"],
    "table":              [r"\btable\s+\d", r"\bsupplementary\s+table"],
    "figure":             [r"\bfigure\s+\d", r"\bfig\.\s*\d"],
}

# Compiled pattern cache
_COMPILED: Dict[str, list] = {}


def _get_compiled(term_dict: Dict[str, List[str]]) -> Dict[str, list]:
    key = id(term_dict)
    if key not in _COMPILED:
        _COMPILED[key] = {
            term: [re.compile(p, re.IGNORECASE) for p in patterns]
            for term, patterns in term_dict.items()
        }
    return _COMPILED[key]


def _count_term(text: str, patterns: list) -> int:
    return sum(len(p.findall(text)) for p in patterns)


def _any_term(text: str, patterns: list) -> int:
    return int(any(p.search(text) for p in patterns))


# ---------------------------------------------------------------------------
# Part 2. Extract run-level term matrices
# ---------------------------------------------------------------------------

def extract_run_term_matrix(
    file_texts: Dict[str, str],
    file_meta: List[dict],
    term_dict: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    For each (agent, run), aggregate term presence (binary) and count across all
    files in that run.

    Returns a DataFrame with columns: agent, run, <term>_count, <term>_present
    """
    compiled = _get_compiled(term_dict)
    terms = list(term_dict.keys())

    # Build file-level rows
    rows = []
    for meta in file_meta:
        fid  = meta["file_id"]
        text = file_texts.get(fid, "")
        row  = {"agent": meta["agent"], "run": meta["run"]}
        for term in terms:
            row[f"{term}_count"] = _count_term(text, compiled[term])
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    count_cols = [f"{t}_count" for t in terms]

    # Aggregate to run level: sum counts, binary presence
    agg = df.groupby(["agent", "run"])[count_cols].sum().reset_index()
    for term in terms:
        agg[f"{term}_present"] = (agg[f"{term}_count"] > 0).astype(int)

    return agg


# ---------------------------------------------------------------------------
# Part 3. Shared vs unique terms
# ---------------------------------------------------------------------------

def compute_shared_vs_unique(
    method_mat: pd.DataFrame,
    bio_mat: pd.DataFrame,
    reporting_mat: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each term, compute:
      - n_agents_present: number of agents with ≥1 run containing the term
      - n_runs_present: total runs containing the term
      - mean_freq_by_agent: dict-str of mean count per agent
      - shared_all_agents: True if term present in ≥1 run of every agent
      - shared_2plus_agents: True if term present in ≥1 run of ≥2 agents
      - agent_specific: True if term present only in exactly 1 agent
      - dominant_agent: agent with highest mean freq (or 'none')

    Returns (shared_terms_summary, agent_specific_terms).
    """
    rows = []
    agents = None

    def _process(mat: pd.DataFrame, category: str) -> None:
        nonlocal agents
        if mat is None or mat.empty:
            return
        if agents is None:
            agents = sorted(mat["agent"].unique())
        present_cols = [c for c in mat.columns if c.endswith("_present")]
        count_cols   = [c for c in mat.columns if c.endswith("_count")]
        term_names   = [c.replace("_present", "") for c in present_cols]

        for term, pcol, ccol in zip(term_names, present_cols, count_cols):
            n_runs_total = len(mat)
            n_runs_present = int(mat[pcol].sum())
            agents_with_term = []
            mean_freq = {}
            for agent in agents:
                sub = mat[mat["agent"] == agent]
                if sub[pcol].any():
                    agents_with_term.append(agent)
                mf = float(sub[ccol].mean()) if len(sub) > 0 else 0.0
                mean_freq[agent] = round(mf, 3)

            n_agents = len(agents_with_term)
            shared_all  = (n_agents == len(agents)) if agents else False
            shared_2plus = (n_agents >= 2)
            specific = (n_agents == 1)
            dominant = max(mean_freq, key=mean_freq.get) if mean_freq else "none"

            rows.append({
                "category": category,
                "term": term,
                "n_agents_present": n_agents,
                "n_runs_present": n_runs_present,
                "n_runs_total": n_runs_total,
                "run_prevalence_pct": round(100 * n_runs_present / n_runs_total, 1)
                    if n_runs_total > 0 else 0.0,
                "shared_all_agents": shared_all,
                "shared_2plus_agents": shared_2plus,
                "agent_specific": specific,
                "dominant_agent": dominant if specific else
                    (dominant if not shared_all else "shared"),
                "mean_freq": str(mean_freq),
            })

    _process(method_mat,    "method")
    _process(bio_mat,       "biological")
    _process(reporting_mat, "reporting")

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    shared_df = pd.DataFrame(rows)
    specific_df = shared_df[shared_df["agent_specific"]].reset_index(drop=True)
    return shared_df, specific_df


# ---------------------------------------------------------------------------
# Part 4. Consensus core
# ---------------------------------------------------------------------------

def compute_consensus_core(
    method_mat: pd.DataFrame,
    bio_mat: pd.DataFrame,
    reporting_mat: pd.DataFrame,
    robust_min_agent_frac: float = 0.5,
    robust_min_agents: int = 2,
) -> pd.DataFrame:
    """
    Strict consensus: term present in ≥1 run of EVERY agent.
    Robust consensus: term present in ≥50% runs of ≥2 agents.

    Returns consensus_core_terms DataFrame.
    """
    rows = []

    def _process(mat: pd.DataFrame, category: str) -> None:
        if mat is None or mat.empty:
            return
        agents = sorted(mat["agent"].unique())
        present_cols = [c for c in mat.columns if c.endswith("_present")]
        count_cols   = [c for c in mat.columns if c.endswith("_count")]
        term_names   = [c.replace("_present", "") for c in present_cols]

        for term, pcol, ccol in zip(term_names, present_cols, count_cols):
            # Strict: every agent has ≥1 run with term
            strict = all(
                mat[mat["agent"] == a][pcol].any()
                for a in agents
            )

            # Robust: ≥robust_min_agents agents have ≥robust_min_agent_frac of their runs with term
            agents_robust = 0
            for a in agents:
                sub = mat[mat["agent"] == a]
                if len(sub) == 0:
                    continue
                frac = sub[pcol].mean()
                if frac >= robust_min_agent_frac:
                    agents_robust += 1
            robust = (agents_robust >= robust_min_agents)

            if strict or robust:
                rows.append({
                    "category": category,
                    "term": term,
                    "strict_consensus": strict,
                    "robust_consensus": robust,
                    "n_agents_strict": sum(
                        1 for a in agents if mat[mat["agent"] == a][pcol].any()
                    ),
                    "n_agents_robust": agents_robust,
                    "total_agents": len(agents),
                })

    _process(method_mat,    "method")
    _process(bio_mat,       "biological")
    _process(reporting_mat, "reporting")

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Part 5. Consensus alignment score
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_consensus_alignment(
    run_embs: Dict[Tuple[str, int], np.ndarray],
    method_mat: pd.DataFrame,
    bio_mat: pd.DataFrame,
    reporting_mat: pd.DataFrame,
    consensus_core: pd.DataFrame,
    n_boot: int = 500,
    seed: int = SEED,
    w_emb: float = 0.40,
    w_strict: float = 0.30,
    w_robust: float = 0.30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each run compute a consensus alignment score (0–100):

      alignment = w_emb * cosine_sim(run_emb, global_centroid) [0,1]
                + w_strict * strict_overlap [0,1]
                + w_robust * robust_overlap [0,1]

    Scaled to 0–100.

    Returns (run_alignment_df, agent_alignment_error_df).
    """
    if not run_embs:
        return pd.DataFrame(), pd.DataFrame()

    # Global centroid
    all_vecs = np.stack(list(run_embs.values()))
    global_centroid = all_vecs.mean(axis=0)

    # Strict / robust consensus term lists
    strict_method  = set(consensus_core[
        (consensus_core["strict_consensus"]) & (consensus_core["category"] == "method")
    ]["term"]) if not consensus_core.empty else set()
    robust_method  = set(consensus_core[
        (consensus_core["robust_consensus"]) & (consensus_core["category"] == "method")
    ]["term"]) if not consensus_core.empty else set()
    strict_bio     = set(consensus_core[
        (consensus_core["strict_consensus"]) & (consensus_core["category"] == "biological")
    ]["term"]) if not consensus_core.empty else set()
    robust_bio     = set(consensus_core[
        (consensus_core["robust_consensus"]) & (consensus_core["category"] == "biological")
    ]["term"]) if not consensus_core.empty else set()

    strict_all = strict_method | strict_bio
    robust_all = robust_method | robust_bio

    def _term_overlap(agent: str, run: int, term_set: set, *mats) -> float:
        if not term_set:
            return 0.0
        hits = 0
        total = len(term_set)
        for mat in mats:
            if mat is None or mat.empty:
                continue
            sub = mat[(mat["agent"] == agent) & (mat["run"] == run)]
            if sub.empty:
                continue
            for term in term_set:
                pcol = f"{term}_present"
                if pcol in sub.columns and sub[pcol].iloc[0] > 0:
                    hits += 1
        return hits / total if total > 0 else 0.0

    run_rows = []
    for (agent, run), vec in sorted(run_embs.items()):
        cos_sim = _cosine_sim(vec, global_centroid)
        # Normalise cosine similarity from [-1,1] to [0,1]
        cos_norm = (cos_sim + 1.0) / 2.0

        strict_ov = _term_overlap(agent, run, strict_all,
                                   method_mat, bio_mat, reporting_mat)
        robust_ov = _term_overlap(agent, run, robust_all,
                                   method_mat, bio_mat, reporting_mat)

        raw = w_emb * cos_norm + w_strict * strict_ov + w_robust * robust_ov
        score = round(raw * 100.0, 2)

        run_rows.append({
            "agent": agent, "run": run,
            "cosine_sim_to_global": round(cos_sim, 4),
            "strict_term_overlap": round(strict_ov, 4),
            "robust_term_overlap": round(robust_ov, 4),
            "consensus_alignment_score": score,
        })

    run_df = pd.DataFrame(run_rows)

    # Bootstrap error per agent
    rng = np.random.default_rng(seed)
    agent_rows = []
    for agent in sorted(run_df["agent"].unique()):
        vals = run_df[run_df["agent"] == agent]["consensus_alignment_score"].to_numpy()
        n = len(vals)
        if n == 0:
            continue
        mean  = float(np.mean(vals))
        sd    = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        sem   = sd / np.sqrt(n) if n > 0 else 0.0
        boot  = np.array([np.mean(rng.choice(vals, n, replace=True))
                          for _ in range(n_boot)])
        ci_lo = float(np.percentile(boot, 2.5))
        ci_hi = float(np.percentile(boot, 97.5))
        abs_err = float(np.std(boot, ddof=1)) if len(boot) > 1 else 0.0
        rel_err = abs_err / mean * 100 if mean != 0 else float("nan")
        agent_rows.append({
            "agent": agent,
            "mean_consensus_alignment": round(mean, 3),
            "sd":  round(sd, 3),
            "sem": round(sem, 3),
            "ci95_lo": round(ci_lo, 3),
            "ci95_hi": round(ci_hi, 3),
            "absolute_error": round(abs_err, 3),
            "relative_error_pct": round(rel_err, 3) if not np.isnan(rel_err) else float("nan"),
            "n_runs": n,
        })

    error_df = pd.DataFrame(agent_rows)
    return run_df, error_df


# ---------------------------------------------------------------------------
# Part 6. Distinctiveness / uniqueness score
# ---------------------------------------------------------------------------

def compute_distinctiveness(
    run_embs: Dict[Tuple[str, int], np.ndarray],
    method_mat: pd.DataFrame,
    bio_mat: pd.DataFrame,
    reporting_mat: pd.DataFrame,
    shared_terms: pd.DataFrame,
    run_scores: Optional[pd.DataFrame] = None,
    n_boot: int = 500,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each run compute a distinctiveness score (0–100):

      Components:
        dist_global  = 1 - (cosine_sim(run_emb, global_centroid)+1)/2    [0,1]
        agent_specific_frac = n_agent_specific_terms_present / total_terms [0,1]
        nonconsensus_frac   = 1 - strict_overlap                           [0,1]

      distinctiveness = 0.40 * dist_global + 0.30 * agent_specific_frac
                      + 0.30 * nonconsensus_frac,  scaled to 0–100.

    Returns (run_distinct_df, agent_distinct_error_df).
    """
    if not run_embs:
        return pd.DataFrame(), pd.DataFrame()

    all_vecs = np.stack(list(run_embs.values()))
    global_centroid = all_vecs.mean(axis=0)

    # Agent-specific terms from shared_terms
    agent_specific_terms: Dict[str, set] = {}
    if shared_terms is not None and not shared_terms.empty and "dominant_agent" in shared_terms.columns:
        for _, row in shared_terms[shared_terms["agent_specific"]].iterrows():
            a = row["dominant_agent"]
            if a not in agent_specific_terms:
                agent_specific_terms[a] = set()
            agent_specific_terms[a].add(row["term"])

    # Strict consensus terms (for non-consensus fraction)
    strict_terms: set = set()
    for mat_name, mat in [("method", method_mat), ("biological", bio_mat),
                           ("reporting", reporting_mat)]:
        if mat is None or mat.empty:
            continue
        agents = sorted(mat["agent"].unique())
        present_cols = [c for c in mat.columns if c.endswith("_present")]
        for pcol in present_cols:
            term = pcol.replace("_present", "")
            if all(mat[mat["agent"] == a][pcol].any() for a in agents):
                strict_terms.add(term)

    # Total terms across all categories
    all_terms: List[str] = []
    for mat in [method_mat, bio_mat, reporting_mat]:
        if mat is not None and not mat.empty:
            all_terms += [c.replace("_present", "")
                          for c in mat.columns if c.endswith("_present")]

    run_rows = []
    for (agent, run), vec in sorted(run_embs.items()):
        cos_sim = _cosine_sim(vec, global_centroid)
        dist_global = 1.0 - (cos_sim + 1.0) / 2.0

        # Count agent-specific terms present in this run
        spec_terms = agent_specific_terms.get(agent, set())
        spec_count = 0
        for mat in [method_mat, bio_mat, reporting_mat]:
            if mat is None or mat.empty:
                continue
            sub = mat[(mat["agent"] == agent) & (mat["run"] == run)]
            if sub.empty:
                continue
            for term in spec_terms:
                pcol = f"{term}_present"
                if pcol in sub.columns and sub[pcol].iloc[0] > 0:
                    spec_count += 1
        agent_spec_frac = spec_count / len(spec_terms) if spec_terms else 0.0

        # Non-consensus fraction
        present_count = 0
        strict_present = 0
        for mat in [method_mat, bio_mat, reporting_mat]:
            if mat is None or mat.empty:
                continue
            sub = mat[(mat["agent"] == agent) & (mat["run"] == run)]
            if sub.empty:
                continue
            for term in all_terms:
                pcol = f"{term}_present"
                if pcol in sub.columns:
                    if sub[pcol].iloc[0] > 0:
                        present_count += 1
                        if term in strict_terms:
                            strict_present += 1
        nonconsensus_frac = (1 - strict_present / present_count
                             if present_count > 0 else 0.5)

        raw = 0.40 * dist_global + 0.30 * agent_spec_frac + 0.30 * nonconsensus_frac
        score = round(raw * 100.0, 2)

        run_rows.append({
            "agent": agent, "run": run,
            "cosine_dist_to_global": round(dist_global, 4),
            "agent_specific_terms_present": spec_count,
            "agent_specific_term_frac": round(agent_spec_frac, 4),
            "nonconsensus_frac": round(nonconsensus_frac, 4),
            "distinctiveness_score": score,
        })

    run_df = pd.DataFrame(run_rows)

    # Merge AgentScore if available
    if run_scores is not None and not run_scores.empty and "AgentScore" in run_scores.columns:
        run_df = run_df.merge(
            run_scores[["agent", "run", "AgentScore"]], on=["agent", "run"], how="left"
        )

    # Bootstrap error per agent
    rng = np.random.default_rng(seed)
    agent_rows = []
    for agent in sorted(run_df["agent"].unique()):
        vals = run_df[run_df["agent"] == agent]["distinctiveness_score"].to_numpy()
        n = len(vals)
        if n == 0:
            continue
        mean  = float(np.mean(vals))
        sd    = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        sem   = sd / np.sqrt(n) if n > 0 else 0.0
        boot  = np.array([np.mean(rng.choice(vals, n, replace=True))
                          for _ in range(n_boot)])
        ci_lo = float(np.percentile(boot, 2.5))
        ci_hi = float(np.percentile(boot, 97.5))
        abs_err = float(np.std(boot, ddof=1)) if len(boot) > 1 else 0.0
        rel_err = abs_err / mean * 100 if mean != 0 else float("nan")
        agent_rows.append({
            "agent": agent,
            "mean_distinctiveness": round(mean, 3),
            "sd": round(sd, 3),
            "sem": round(sem, 3),
            "ci95_lo": round(ci_lo, 3),
            "ci95_hi": round(ci_hi, 3),
            "absolute_error": round(abs_err, 3),
            "relative_error_pct": round(rel_err, 3) if not np.isnan(rel_err) else float("nan"),
            "n_runs": n,
        })

    error_df = pd.DataFrame(agent_rows)
    return run_df, error_df


# ---------------------------------------------------------------------------
# Part 7. Commonality–difference summary table
# ---------------------------------------------------------------------------

def build_commonality_difference_summary(
    shared_terms: pd.DataFrame,
    consensus_core: pd.DataFrame,
    alignment_error: pd.DataFrame,
    distinct_error: pd.DataFrame,
    run_scores: Optional[pd.DataFrame] = None,
    within_repro: Optional[pd.DataFrame] = None,
    volume_metrics: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Fixed 8-row descriptive table comparing common vs agent-specific features
    across key analytical dimensions.
    """
    rows = []

    def _top_agent(df: pd.DataFrame, col: str, ascending: bool = False) -> str:
        if df is None or df.empty or col not in df.columns:
            return "N/A"
        idx = df[col].idxmax() if not ascending else df[col].idxmin()
        return str(df.loc[idx, "agent"])

    def _strict_terms(category: str) -> str:
        if consensus_core is None or consensus_core.empty:
            return "N/A"
        sub = consensus_core[(consensus_core["strict_consensus"]) &
                             (consensus_core["category"] == category)]
        if sub.empty:
            return "none"
        return ", ".join(sub["term"].tolist()[:8])

    def _agent_specific_terms(category: str) -> str:
        if shared_terms is None or shared_terms.empty:
            return "N/A"
        sub = shared_terms[(shared_terms["agent_specific"]) &
                           (shared_terms["category"] == category)]
        if sub.empty:
            return "none"
        return ", ".join(sub.apply(
            lambda r: f"{r['dominant_agent']}:{r['term']}", axis=1
        ).tolist()[:8])

    top_align  = _top_agent(alignment_error, "mean_consensus_alignment")
    top_dist   = _top_agent(distinct_error,  "mean_distinctiveness")
    top_score  = _top_agent(run_scores.groupby("agent")["AgentScore"].mean().reset_index()
                            .rename(columns={"AgentScore": "mean_AgentScore"}),
                            "mean_AgentScore") if (
        run_scores is not None and not run_scores.empty) else "N/A"

    rows.append({
        "dimension": "Methodological workflow",
        "common_across_agents": _strict_terms("method"),
        "agent_specific_differences": _agent_specific_terms("method"),
        "interpretation": (
            "Shared method terms indicate cross-agent agreement on analytical workflow. "
            "Agent-specific terms reflect distinctive methodological choices."
        ),
        "evidence_table": "consensus_core_terms.csv",
        "evidence_figure": "consensus_method_overlap_heatmap.png",
    })
    rows.append({
        "dimension": "Statistical reporting",
        "common_across_agents": _strict_terms("reporting"),
        "agent_specific_differences": _agent_specific_terms("reporting"),
        "interpretation": (
            "Terms shared in all agents reflect common reporting standards. "
            "Agent-specific reporting terms may reflect depth or style differences."
        ),
        "evidence_table": "shared_terms_summary.csv",
        "evidence_figure": "agent_specific_terms_barplot.png",
    })
    rows.append({
        "dimension": "Biological interpretation",
        "common_across_agents": _strict_terms("biological"),
        "agent_specific_differences": _agent_specific_terms("biological"),
        "interpretation": (
            "Shared biological themes reflect convergent transcriptomics interpretation. "
            "Agent-specific biological terms may indicate unique scientific depth or noise."
        ),
        "evidence_table": "consensus_core_terms.csv",
        "evidence_figure": "consensus_biological_theme_heatmap.png",
    })
    rows.append({
        "dimension": "Pathway / gene-function interpretation",
        "common_across_agents": "pathway_analysis, GO enrichment, GSEA (if shared)",
        "agent_specific_differences": "See consensus_core_terms.csv biological category",
        "interpretation": (
            "Pathway coverage is compared across agents. Convergence on pathway databases "
            "(GO, KEGG, Reactome, MSigDB) indicates common analytical toolkit."
        ),
        "evidence_table": "consensus_core_terms.csv",
        "evidence_figure": "consensus_biological_theme_heatmap.png",
    })
    rows.append({
        "dimension": "Consensus alignment",
        "common_across_agents": f"Highest consensus alignment: {top_align}",
        "agent_specific_differences": f"Most distinctive agent: {top_dist}",
        "interpretation": (
            "Consensus alignment quantifies closeness to the cross-agent common structure. "
            "Distinctiveness quantifies unique contributions. Neither implies correctness."
        ),
        "evidence_table": "consensus_alignment_error_summary.csv",
        "evidence_figure": "consensus_alignment_errorbar.png",
    })

    repro_str = "N/A"
    if within_repro is not None and not within_repro.empty and "reproducibility" in within_repro.columns:
        best = within_repro.sort_values("reproducibility", ascending=False).iloc[0]
        repro_str = f"Highest: {best['agent']} (RI={best['reproducibility']:.3f})"
    rows.append({
        "dimension": "Reproducibility",
        "common_across_agents": "Measured by within-agent cosine similarity across 8 runs",
        "agent_specific_differences": repro_str,
        "interpretation": (
            "Reproducibility reflects consistency of output structure across repeated runs. "
            "High reproducibility is desirable but not sufficient for high quality."
        ),
        "evidence_table": "within_agent_reproducibility.csv",
        "evidence_figure": "reproducibility_errorbar.png",
    })

    vol_diff = "See output_volume_metrics.csv"
    if volume_metrics is not None and not volume_metrics.empty and "file_count" in volume_metrics.columns:
        top_vol = (
            volume_metrics.groupby("agent")["file_count"].mean()
            .sort_values(ascending=False)
        )
        if not top_vol.empty:
            vol_diff = f"{top_vol.index[0]} produced highest mean file counts"

    rows.append({
        "dimension": "Output volume / artifact production",
        "common_across_agents": "Compared in output_volume_metrics.csv",
        "agent_specific_differences": vol_diff,
        "interpretation": (
            "Output volume is an observable quantity but not a direct quality signal. "
            "Volume-score correlation is reported separately."
        ),
        "evidence_table": "output_volume_metrics.csv",
        "evidence_figure": "output_volume_by_agent_boxplot.png",
    })

    rows.append({
        "dimension": "Uncertainty / error profile",
        "common_across_agents": "Bootstrap SD of mean score per agent",
        "agent_specific_differences": "See bootstrap_error_summary.csv",
        "interpretation": (
            "Error bars reflect within-agent run-to-run variability. "
            "Larger uncertainty does not imply lower quality."
        ),
        "evidence_table": "bootstrap_error_summary.csv",
        "evidence_figure": "agent_score_errorbar.png",
    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 9. Consensus validation summary
# ---------------------------------------------------------------------------

def build_consensus_validation_summary(
    shared_terms: pd.DataFrame,
    consensus_core: pd.DataFrame,
    alignment_error: pd.DataFrame,
    distinct_error: pd.DataFrame,
    run_scores: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Q&A-style table answering the 7 key consensus validation questions.
    """
    rows = []

    def _safe(df, cond, col, default="N/A"):
        try:
            sub = df[cond]
            if sub.empty:
                return default
            return str(sub[col].tolist())
        except Exception:
            return default

    # Q1: Which analytical terms were shared by all agents?
    if consensus_core is not None and not consensus_core.empty:
        strict_all = consensus_core[consensus_core["strict_consensus"]]
        by_cat = strict_all.groupby("category")["term"].apply(list).to_dict()
        result_q1 = "; ".join(f"{k}: {v}" for k, v in by_cat.items())
    else:
        result_q1 = "N/A"
    rows.append({
        "question": "Which analytical terms were shared by all agents?",
        "method": "Strict consensus: term present in ≥1 run of every agent",
        "key_result": result_q1,
        "interpretation": "Strict consensus terms form the verifiable common analytical core.",
        "limitation": "Text-based detection; absent text ≠ absent method.",
    })

    # Q2: Which biological themes were shared by all agents?
    if consensus_core is not None and not consensus_core.empty:
        strict_bio = consensus_core[
            (consensus_core["strict_consensus"]) &
            (consensus_core["category"] == "biological")
        ]["term"].tolist()
        result_q2 = ", ".join(strict_bio) if strict_bio else "none"
    else:
        result_q2 = "N/A"
    rows.append({
        "question": "Which biological themes were shared by all agents?",
        "method": "Strict consensus for biological category",
        "key_result": result_q2,
        "interpretation": "Shared biological terms reflect convergent scientific interpretation.",
        "limitation": "Keyword matching; may miss semantic paraphrases.",
    })

    # Q3: Which agent was closest to the cross-agent consensus?
    if alignment_error is not None and not alignment_error.empty:
        top = alignment_error.loc[
            alignment_error["mean_consensus_alignment"].idxmax()
        ]
        result_q3 = (f"{top['agent']} (mean={top['mean_consensus_alignment']:.1f}, "
                     f"CI=[{top['ci95_lo']:.1f},{top['ci95_hi']:.1f}])")
    else:
        result_q3 = "N/A"
    rows.append({
        "question": "Which agent was closest to the cross-agent consensus?",
        "method": "Highest mean consensus alignment score (0–100)",
        "key_result": result_q3,
        "interpretation": "High consensus alignment does not imply correctness; only structural agreement.",
        "limitation": "Only 8 runs per agent; small sample.",
    })

    # Q4: Which agent was most distinctive?
    if distinct_error is not None and not distinct_error.empty:
        top = distinct_error.loc[
            distinct_error["mean_distinctiveness"].idxmax()
        ]
        result_q4 = (f"{top['agent']} (mean={top['mean_distinctiveness']:.1f}, "
                     f"CI=[{top['ci95_lo']:.1f},{top['ci95_hi']:.1f}])")
    else:
        result_q4 = "N/A"
    rows.append({
        "question": "Which agent was most distinctive?",
        "method": "Highest mean distinctiveness score (0–100)",
        "key_result": result_q4,
        "interpretation": "Distinctiveness may reflect unique scientific insight or idiosyncratic style.",
        "limitation": "Only 8 runs per agent; small sample.",
    })

    # Q5: Did the top-ranked agent show highest consensus alignment?
    if (run_scores is not None and not run_scores.empty and
            alignment_error is not None and not alignment_error.empty):
        agent_means = run_scores.groupby("agent")["AgentScore"].mean()
        top_scored = str(agent_means.idxmax())
        top_aligned = str(alignment_error.loc[
            alignment_error["mean_consensus_alignment"].idxmax(), "agent"])
        match = "yes" if top_scored == top_aligned else "no"
        result_q5 = (f"Top-ranked by score: {top_scored}; "
                     f"Highest alignment: {top_aligned}; Match: {match}")
    else:
        result_q5 = "N/A"
    rows.append({
        "question": "Did the top-ranked agent also show highest consensus alignment?",
        "method": "Compare AgentScore rank 1 vs highest consensus alignment score",
        "key_result": result_q5,
        "interpretation": (
            "If the top-ranked agent is also closest to consensus, it suggests "
            "the ranking reflects shared analytical quality standards. "
            "If not, the top agent may be penalised for distinctive contributions."
        ),
        "limitation": "Proxy-estimated AgentScore; small n.",
    })

    # Q6: Did biological interpretation depth correspond to distinctiveness?
    if (run_scores is not None and not run_scores.empty and
            distinct_error is not None and not distinct_error.empty and
            "D" in run_scores.columns):
        agent_d = run_scores.groupby("agent")["D"].mean()
        most_distinct = str(distinct_error.loc[
            distinct_error["mean_distinctiveness"].idxmax(), "agent"])
        top_depth = str(agent_d.idxmax())
        match = "yes" if most_distinct == top_depth else "no"
        result_q6 = (f"Deepest interpretation (D): {top_depth}; "
                     f"Most distinctive: {most_distinct}; Match: {match}")
    else:
        result_q6 = "N/A"
    rows.append({
        "question": "Did biological interpretation depth correspond to distinctiveness?",
        "method": "Compare interpretation depth (D score) vs distinctiveness score",
        "key_result": result_q6,
        "interpretation": (
            "A match suggests distinctive agents produce deeper biological interpretations. "
            "A mismatch suggests distinctiveness may reflect output style rather than depth."
        ),
        "limitation": "D score is proxy-estimated from keyword features.",
    })

    # Q7: Are differences larger than bootstrap uncertainty?
    if alignment_error is not None and not alignment_error.empty and len(alignment_error) >= 2:
        srt = alignment_error.sort_values("mean_consensus_alignment", ascending=False)
        top1 = srt.iloc[0]
        top2 = srt.iloc[1]
        diff = top1["mean_consensus_alignment"] - top2["mean_consensus_alignment"]
        pooled_err = ((top1["absolute_error"] ** 2 + top2["absolute_error"] ** 2) ** 0.5)
        result_q7 = (f"Top two agents differ by {diff:.1f} score points; "
                     f"pooled bootstrap error = {pooled_err:.1f}; "
                     f"difference {'larger' if diff > pooled_err else 'within'} uncertainty")
    else:
        result_q7 = "N/A"
    rows.append({
        "question": "Are consensus alignment differences larger than bootstrap uncertainty?",
        "method": "Compare pairwise differences in mean alignment vs bootstrap SD",
        "key_result": result_q7,
        "interpretation": (
            "If differences exceed uncertainty, the alignment ranking is more reliable. "
            "If within uncertainty, consensus alignment ranking is tentative."
        ),
        "limitation": "Only 8 runs per agent; bootstrap intervals may be wide.",
    })

    return pd.DataFrame(rows)
