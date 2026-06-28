"""
report.py — generate benchmark_report.md and benchmark_report.html.
"""

import datetime
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _df_to_md(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_No data available._\n"
    if len(df) > max_rows:
        df = pd.concat([df.head(max_rows // 2), df.tail(max_rows // 2)])
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def _fig_md(name: str, caption: str) -> str:
    return f"![{caption}](../figures/{name})\n\n*{caption}*\n"


PROXY_SAFEGUARD = (
    "Proxy-estimated scores are not ground-truth assessments of scientific "
    "correctness. They quantify detectable output properties and methodological "
    "signals. Therefore, any ranking based on proxy scores should be interpreted "
    "as exploratory until validated against manual expert scoring."
)

DOMAIN_SAFEGUARD = (
    "Domain-aware scoring provides a transcriptomics-specific sensitivity analysis "
    "but remains rule-based and should not be treated as an independent expert "
    "review."
)

ERROR_SAFEGUARD = (
    "Error bars and confidence intervals quantify uncertainty induced by the "
    "observed set of repeated agent runs. They do not represent population-level "
    "confidence intervals for all possible agent behaviours, because only eight "
    "real runs per agent were available."
)


def _write_audit_section(audit: dict, h, p, ul, table, fig, run_scores) -> None:
    """Write the scoring-audit + error-estimation report section."""

    def _df(key):
        v = audit.get(key)
        return v if v is not None else pd.DataFrame()

    score_type = "proxy-estimated"
    if not run_scores.empty and "score_type" in run_scores.columns:
        score_type = run_scores["score_type"].mode().iloc[0]

    h(2, "Scoring audit, output-volume diagnostics, and domain-aware interpretation")

    # Proxy limitations
    h(3, "Proxy score limitations")
    p(PROXY_SAFEGUARD)
    ul([
        "Proxy scores may be sensitive to output length, number of files, "
        "number of figures/tables, stylistic completeness, and explicit "
        "methodological keyword availability.",
        f"Current score source: **{score_type}**.",
    ])

    # Output-volume audit
    h(3, "Output-volume audit")
    p("Per-run output-volume metrics quantify how much material each agent "
      "produced, independent of quality.")
    table(_df("output_volume_metrics"))
    corr = _df("score_volume_correlation")
    if not corr.empty:
        p("Spearman correlation between AgentScore and output-volume metrics "
          "(small n; non-linear relationships possible):")
        table(corr)
        strong = corr.dropna(subset=["spearman_rho"])
        if not strong.empty:
            top = strong.iloc[strong["spearman_rho"].abs().argmax()]
            if abs(top["spearman_rho"]) > 0.5:
                p(f"**Caution:** AgentScore shows a non-trivial association with "
                  f"`{top['metric']}` (Spearman ρ={top['spearman_rho']:.2f}). "
                  f"Ranking **may be influenced by output volume / artifact richness**.")
            else:
                p("AgentScore shows only weak associations with output-volume "
                  "metrics; ranking is unlikely to be dominated by output volume, "
                  "though n is small.")
    for name, cap in [
        ("score_vs_file_count.png", "AgentScore vs file count (Spearman ρ in caption)"),
        ("score_vs_text_length.png", "AgentScore vs total text length"),
        ("score_vs_table_count.png", "AgentScore vs table count"),
        ("score_vs_figure_count.png", "AgentScore vs figure/image count"),
        ("score_vs_artifact_diversity.png", "AgentScore vs artifact diversity index"),
        ("output_volume_by_agent_boxplot.png", "Output-volume metrics by agent"),
        ("output_volume_vs_score_panel.png", "Panel: AgentScore vs each volume metric"),
    ]:
        fig(name, cap)

    # Proxy decomposition
    h(3, "Proxy score decomposition")
    p("Each proxy R/P/D score is the mean of explicit binary components, "
      "making the score fully auditable.")
    table(_df("proxy_score_components"))
    fig("score_component_heatmap.png", "Proxy score components (runs × components)")

    # Domain-aware
    h(3, "Domain-aware transcriptomics score")
    p(DOMAIN_SAFEGUARD)
    p("Each criterion is graded 0 (absent) / 1 (partial) / 2 (clear) and the sum "
      "is normalised to 0–100. Reported **alongside** AgentScore as an independent "
      "sensitivity analysis — it does **not** replace AgentScore.")
    table(_df("domain_transcriptomics_score_components"))
    fig("domain_score_component_heatmap.png", "Domain-aware criteria (runs × criteria, 0/1/2)")
    fig("domain_score_radar.png", "Domain-aware criteria mean grade per agent")

    # Proxy vs domain comparison
    h(3, "Comparison of proxy AgentScore and domain-aware score")
    audit_tbl = _df("score_audit")
    if not audit_tbl.empty:
        table(audit_tbl)
    fig("proxy_vs_domain_score_scatter.png",
        "Proxy AgentScore vs domain-aware score (per run)")

    # Manual vs proxy
    mvp = _df("manual_vs_proxy_scores")
    if not mvp.empty:
        h(3, "Manual vs proxy/domain scores")
        p("Manual expert scores were provided and used as the primary score source.")
        table(mvp)
        fig("proxy_vs_manual_score_scatter.png", "Manual vs proxy AgentScore")
    else:
        h(3, "Manual scoring")
        p("No manual scores were provided. **All ranking results are "
          "proxy-estimated and exploratory.**")

    # Ranking sensitivity
    h(3, "Ranking sensitivity analysis")
    p("Agent rankings were recomputed under seven scoring definitions: "
      "(A) proxy AgentScore, (B) domain-aware score, (C) reproducibility index, "
      "(D) interpretation depth, (E) process quality, (F) output-volume-penalised "
      "AgentScore (exploratory), and (G) AgentScore excluding output-volume features.")
    rs = _df("ranking_sensitivity_analysis")
    if not rs.empty:
        table(rs)
        # Stability statement
        top = rs[rs["rank"] == 1].groupby("definition")["agent"].first()
        if not top.empty:
            winners = top.value_counts()
            most = winners.index[0]
            if winners.iloc[0] == winners.sum():
                p(f"**{most}** is top-ranked under **all** scoring definitions — "
                  f"the top-ranked agent is **stable** across scoring frameworks.")
            else:
                p(f"The top-ranked agent **depends on the scoring framework** "
                  f"({most} wins {winners.iloc[0]}/{winners.sum()} definitions).")
    fig("ranking_sensitivity_heatmap.png", "Rank per agent across scoring definitions")
    fig("ranking_sensitivity_barplot.png",
        "Mean score by agent for each definition (95% bootstrap CI)")

    # Fairness
    fair = _df("fairness_sensitivity_summary")
    h(3, "Fairness / information-access note")
    if not fair.empty:
        ul([
            "Some agents may operate in an interactive mode.",
            "If an agent explicitly requested additional clinical or sample-level "
            "metadata, the requested information was provided after the request.",
            "Agents that did not request additional information did not receive it.",
            "Information-seeking behaviour is treated as part of agentic performance.",
            "Rankings should be interpreted with this design feature in mind.",
        ])
        table(fair)
    else:
        p("No `run_registry.csv` was found, so the fairness / information-access "
          "audit was skipped. If present, it would record interactive mode and "
          "metadata-request status per run. Information-seeking behaviour is "
          "treated as part of agentic performance, not a protocol violation.")

    # --- Outlier subsection ---
    h(3, "Outlier detection")
    outlier_tbl = _df("outlier_runs")
    p(
        "Each run was tested for outlier status within its agent using two complementary methods: "
        "(1) **IQR method** (Tukey fences, k = 1.5): a run is flagged if its AgentScore falls "
        "below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR; "
        "(2) **Z-score method**: a run is flagged if |z| > 2.5 within its agent's distribution. "
        "A run is reported as an outlier if either method flags it. "
        "Outliers are annotated in all 2D projection plots (PCA, MDS, t-SNE) with a red ring."
    )
    ul([
        "Outliers are not removed from any score calculation — they are flagged for inspection only.",
        "With only 8 runs per agent, IQR fences are wide and outlier flagging is conservative.",
        "An outlier in score space does not necessarily correspond to an outlier in embedding space.",
    ])
    if not outlier_tbl.empty:
        flagged = outlier_tbl[outlier_tbl["outlier"]] if "outlier" in outlier_tbl.columns else pd.DataFrame()
        if not flagged.empty:
            p(f"**{len(flagged)} run(s) flagged as outliers:**")
            table(flagged[["agent", "run", "AgentScore", "zscore",
                            "iqr_lower_fence", "iqr_upper_fence",
                            "outlier_reason"]].rename(
                columns={"iqr_lower_fence": "IQR_lo", "iqr_upper_fence": "IQR_hi"}
            ))
        else:
            p("**No runs were flagged as outliers** under the current thresholds "
              "(IQR k=1.5, |Z|>2.5).")
    else:
        p("_Outlier analysis not available._")
    fig("outlier_runs.png",
        "Per-run AgentScore with outlier flags (red X), IQR fences (dashed), "
        "and agent colours")

    # --- Error estimation subsection ---
    h(3, "Error estimation and bootstrap uncertainty")
    p("**Absolute error** is the bootstrap standard deviation of the agent-level "
      "mean score. **Relative error** is absolute_error / mean_score × 100%. "
      "**Bootstrap confidence intervals** are the 2.5th/97.5th percentiles of the "
      "bootstrap distribution of the mean. **Ranking uncertainty** is estimated by "
      "resampling runs within each agent and recomputing the ranking. A robust "
      "alternative error, 1.4826 × MAD / √n, is reported alongside the bootstrap error.")
    p(ERROR_SAFEGUARD)
    ul([
        "All uncertainty estimates are **descriptive** because only 8 real runs "
        "per agent are available.",
        "Monte Carlo satellites are **not** additional real observations — they "
        "are uncertainty probes around the observed runs.",
    ])
    table(_df("bootstrap_error_summary"))
    pde = _df("pairwise_score_difference_error")
    if not pde.empty:
        p("Pairwise agent score differences with bootstrap uncertainty:")
        table(pde)
    re = _df("ranking_error_summary")
    if not re.empty:
        p("Ranking uncertainty (expected rank, rank entropy, rank-switch probability):")
        table(re)
    for name, cap in [
        ("agent_score_errorbar.png", "Agent mean score ± 95% bootstrap CI"),
        ("agent_score_bootstrap_ci.png", "Bootstrap distribution of the mean AgentScore"),
        ("pairwise_score_difference_ci.png",
         "Pairwise score differences ± 95% bootstrap CI (grey = practical equivalence)"),
        ("rank_uncertainty_interval.png", "Expected rank ± 95% bootstrap interval"),
        ("reproducibility_errorbar.png", "Within-agent reproducibility ± 95% bootstrap CI"),
        ("centroid_uncertainty_pca2d.png", "Bootstrap centroid 95% ellipses (2-D PCA)"),
        ("monte_carlo_distance_distribution.png",
         "Distribution of satellite distances to the real centroid"),
    ]:
        fig(name, cap)

    # Interpretation safeguards block
    h(3, "Interpretation safeguards")
    ul([PROXY_SAFEGUARD, DOMAIN_SAFEGUARD, ERROR_SAFEGUARD])


# ---------------------------------------------------------------------------
# Consensus & divergence section
# ---------------------------------------------------------------------------

CONSENSUS_SAFEGUARD = (
    "Consensus does not necessarily mean correctness. "
    "Distinctiveness does not necessarily mean error. "
    "A distinctive agent may contribute unique biological interpretation or may deviate "
    "from reproducible consensus. Consensus and ranking should be interpreted together. "
    "The analysis is descriptive because there are only 8 real runs per agent."
)


def _write_consensus_section(consensus: dict, h, p, ul, table, fig) -> None:
    """Write the consensus and divergence analysis report section."""

    def _df(key):
        v = consensus.get(key)
        return v if v is not None and not (hasattr(v, "empty") and v.empty) else pd.DataFrame()

    h(2, "Consensus and divergence analysis")
    p(
        "This section analyses what analytical elements were common across all agents "
        "(consensus core) and what was agent-specific (divergence). "
        "Consensus alignment measures how closely each agent's output aligns with the "
        "cross-agent common structure. Distinctiveness measures how unique each agent's "
        "approach was. Both are scored 0–100 and are **descriptive only**."
    )
    p(CONSENSUS_SAFEGUARD)

    # --- 3D PCA ---
    h(3, "3D PCA visualisation of run embeddings")
    p(
        "The interactive 3D PCA plot shows 24 real run embeddings as labelled points, "
        "coloured by agent. Diamonds mark agent centroids. PCA is used for visualisation "
        "only — axes show explained variance fractions."
    )
    p("Interactive 3D PCA is available in `figures/pca_3d.html`.")
    p("Interactive Monte Carlo satellite plot is available in `figures/monte_carlo_satellites_3d.html`.")
    p("Consensus alignment in PCA space is available in `figures/consensus_alignment_pca3d.html`.")

    # --- Shared analytical core ---
    h(3, "Shared analytical core")
    core = _df("consensus_core_terms")
    if not core.empty:
        p(
            "The **strict consensus core** contains terms present in at least one run "
            "of every agent. The **robust consensus core** contains terms present in ≥50% "
            "of runs of at least two agents."
        )
        by_cat = {}
        for _, row in core[core["strict_consensus"]].iterrows():
            by_cat.setdefault(row["category"], []).append(row["term"])
        for cat, terms in sorted(by_cat.items()):
            p(f"**Strict consensus {cat} terms ({len(terms)}):** {', '.join(terms)}")
        table(core)
    else:
        p("_Consensus core not computed._")

    fig("consensus_method_overlap_heatmap.png",
        "Methodological term presence per agent (fraction of runs with term)")
    fig("consensus_biological_theme_heatmap.png",
        "Biological theme presence per agent (fraction of runs with term)")

    # --- Shared vs unique terms ---
    h(3, "Shared versus agent-specific terms")
    p(
        "Terms shared by all agents reflect verifiable common analytical choices. "
        "Terms present in only one agent are agent-specific and may reflect either "
        "unique scientific contributions or idiosyncratic style."
    )
    shared = _df("shared_terms_summary")
    if not shared.empty:
        n_shared_all   = int(shared["shared_all_agents"].sum())
        n_shared_2plus = int(shared["shared_2plus_agents"].sum())
        n_specific     = int(shared["agent_specific"].sum())
        p(
            f"Across all term categories: **{n_shared_all}** terms shared by all agents, "
            f"**{n_shared_2plus}** shared by ≥2 agents, **{n_specific}** agent-specific."
        )
        table(shared)
    spec = _df("agent_specific_terms")
    if not spec.empty:
        p("Agent-specific terms (present in only one agent):")
        table(spec)

    fig("shared_vs_unique_features_barplot.png",
        "Shared vs unique terms by category (stacked bar)")
    fig("agent_specific_terms_barplot.png",
        "Agent-specific terms by agent and category")

    # --- Commonality vs difference summary ---
    h(3, "Commonality versus difference summary")
    cd = _df("commonality_difference_summary")
    if not cd.empty:
        table(cd)
    else:
        p("_No commonality/difference summary available._")

    # --- Consensus alignment ---
    h(3, "Consensus alignment versus distinctiveness")
    p(
        "**Consensus alignment** (0–100) is a composite score based on: "
        "(a) cosine similarity of the run embedding to the global centroid (40%), "
        "(b) overlap with strict consensus terms (30%), and "
        "(c) overlap with robust consensus terms (30%). "
        "A high score means the run closely mirrors the cross-agent common structure."
    )
    p(
        "**Distinctiveness** (0–100) captures: "
        "(a) embedding distance from the global centroid (40%), "
        "(b) presence of agent-specific terms (30%), and "
        "(c) proportion of non-consensus terms (30%). "
        "A high score means the run diverges from the common structure."
    )
    p(CONSENSUS_SAFEGUARD)

    align_err = _df("consensus_alignment_error_summary")
    if not align_err.empty:
        p("Consensus alignment by agent (mean ± 95% bootstrap CI):")
        table(align_err)
    dist_err = _df("agent_distinctiveness_error_summary")
    if not dist_err.empty:
        p("Distinctiveness by agent (mean ± 95% bootstrap CI):")
        table(dist_err)

    for name, cap in [
        ("consensus_alignment_barplot.png",
         "Consensus alignment score by agent (mean ± 95% bootstrap CI)"),
        ("consensus_alignment_errorbar.png",
         "Per-run consensus alignment scores with agent mean overlay"),
        ("consensus_alignment_pca3d.html",
         "Consensus alignment score in 3D PCA space (interactive)"),
        ("agent_distinctiveness_barplot.png",
         "Distinctiveness score by agent (mean ± 95% bootstrap CI)"),
        ("agent_distinctiveness_vs_agentscore.png",
         "Distinctiveness vs AgentScore — note: high distinctiveness ≠ higher quality"),
    ]:
        fig(name, cap)

    # --- Agent-specific divergence ---
    h(3, "Agent-specific divergence")
    p(
        "Agents with high distinctiveness scores produced outputs that differed "
        "structurally and terminologically from the cross-agent consensus. "
        "This may reflect unique scientific insight, different tool usage, "
        "or idiosyncratic reporting style."
    )
    p(
        "Distinctiveness in the embedding space (centroid distance) captures "
        "overall output style, while agent-specific term counts capture vocabulary divergence. "
        "Both are reported to give a multidimensional view of divergence."
    )

    # Consensus validation
    h(3, "Consensus validation summary")
    cv = _df("consensus_validation_summary")
    if not cv.empty:
        table(cv, max_rows=20)

    # Final safeguards
    h(3, "Interpretation safeguards (consensus analysis)")
    ul([
        CONSENSUS_SAFEGUARD,
        "Consensus alignment is not a quality metric — it measures structural similarity "
        "to the cross-agent mean, which may or may not reflect ground truth.",
        "Agent-specific terms are detected by keyword matching and may miss semantically "
        "equivalent phrases.",
        "The analysis is based on 8 real runs per agent — results are descriptive.",
    ])


def generate_report(
    inventory: pd.DataFrame,
    file_features: pd.DataFrame,
    run_features: pd.DataFrame,
    agent_features: pd.DataFrame,
    pairwise_sim: pd.DataFrame,
    within_repro: pd.DataFrame,
    between_sim: pd.DataFrame,
    pca_df: pd.DataFrame,
    mc_summary: pd.DataFrame,
    run_scores: pd.DataFrame,
    agent_summary: pd.DataFrame,
    ranking_uncertainty: pd.DataFrame,
    pairwise_win: pd.DataFrame,
    validation_summary: pd.DataFrame,
    validation_table: pd.DataFrame,
    reliability: dict,
    output_dir: Path,
    args_dict: dict,
    audit: Optional[dict] = None,
    consensus: Optional[dict] = None,
) -> None:
    """Write benchmark_report.md and benchmark_report.html to output_dir/report/."""
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    embedding_model = args_dict.get("embedding_model", "N/A")
    n_mc = args_dict.get("n_mc", "N/A")

    # -----------------------------------------------------------------------
    # Build Markdown
    # -----------------------------------------------------------------------
    lines = []

    def h(level: int, text: str) -> None:
        lines.append("#" * level + " " + text)
        lines.append("")

    def p(text: str) -> None:
        lines.append(text)
        lines.append("")

    def ul(items) -> None:
        for item in items:
            lines.append(f"- {item}")
        lines.append("")

    def table(df: pd.DataFrame, max_rows: int = 30) -> None:
        lines.append(_df_to_md(df, max_rows))
        lines.append("")

    def fig(name: str, caption: str) -> None:
        lines.append(_fig_md(name, caption))

    # --- Title ---
    h(1, "AI Agent Benchmarking Report")
    p(f"**Date:** {date_str}  |  **Embedding model:** `{embedding_model}`  |  **MC iterations:** {n_mc}")
    lines.append("---")
    lines.append("")

    # --- 1. Executive Summary ---
    h(2, "1. Executive Summary")
    n_agents = inventory["agent"].nunique() if not inventory.empty else 0
    n_files  = len(inventory) if not inventory.empty else 0
    agent_list = (
        ", ".join(sorted(inventory["agent"].unique()))
        if not inventory.empty else "N/A"
    )
    p(
        f"This report presents a multi-level quantitative benchmark of {n_agents} AI research agents "
        f"({agent_list}) evaluated on a bulk RNA-seq / transcriptomics analysis task. "
        f"Each agent produced 8 independent runs, yielding a total of {n_files} output files. "
        f"Outputs were compared at file, run, and agent level using semantic embeddings, "
        f"structural feature extraction, and methodological keyword analysis. "
        f"Ranking uncertainty was estimated via {n_mc} bootstrap iterations."
    )

    # --- 2. Study Design ---
    h(2, "2. Study Design")
    ul([
        "Each AI agent analysed the **same** transcriptomics dataset.",
        "Each agent produced **8 independent outputs**.",
        "Different agents produced **different numbers and types of files**.",
        "**Biomni** had an interactive dialogue function.",
        "Biomni requested additional clinical metadata and received it.",
        "Agents that did not ask additional questions did not receive this additional information.",
        "Therefore, **information-seeking behaviour is treated as part of agentic performance**, not as an unfair protocol violation.",
        "Synthetic Monte Carlo points are **not real observations** — they are uncertainty probes.",
        "Reliability statistics are **descriptive** because the number of real runs is limited.",
    ])

    # --- 3. Input File Inventory ---
    h(2, "3. Input File Inventory")
    if not inventory.empty:
        p(f"Total files: **{len(inventory)}** across **{inventory['agent'].nunique()}** agents "
          f"and **{inventory[['agent','run']].drop_duplicates().shape[0]}** runs.")
        summary = (
            inventory.groupby(["agent", "run", "file_type"])
            .size()
            .reset_index(name="n_files")
        )
        table(summary)
    else:
        p("_No files found in input directory._")

    # --- 4. Feature Extraction Strategy ---
    h(2, "4. Feature Extraction Strategy")
    p("Each text-like file was split into overlapping chunks (800 tokens, 150-token overlap). "
      "Structural features (character count, word count, table count, etc.) and methodological "
      "keyword frequencies were extracted from each file.")
    if not file_features.empty:
        cols = ["file_id", "n_chars", "n_words", "n_sheets", "n_tables",
                "n_pvalues", "n_fdr",
                "kw_normalization", "kw_PCA", "kw_pathway_analysis"]
        cols = [c for c in cols if c in file_features.columns]
        sample = file_features[cols].head(15)
        table(sample)
        # Explain why early rows may show large values
        if "n_chars" in file_features.columns and len(file_features) > 1:
            first_row = file_features.iloc[0]
            max_chars = file_features["n_chars"].max()
            if first_row.get("n_chars", 0) >= 0.5 * max_chars:
                p(
                    "**Note on the first row:** The first file listed has unusually large "
                    "`n_chars`, `n_words`, and `n_tables` values. This is expected because "
                    "files are sorted alphabetically by agent and run, and the first file "
                    "often corresponds to a large spreadsheet (e.g. `.xlsx` or `.csv` "
                    "results table) that contains many rows of gene expression data — producing "
                    "a large character and word count when converted to text. This does not "
                    "indicate an error; it reflects genuine variation in file types and sizes "
                    "across agents."
                )
            else:
                p(
                    "**Note:** Variation in `n_chars`, `n_words`, and `n_tables` across files "
                    "reflects genuine differences in file types (spreadsheets, manuscripts, "
                    "plain-text summaries). Large values in any row typically correspond to "
                    "data tables with many gene entries. This is expected and not an error."
                )

    # --- 5. Semantic Similarity Strategy ---
    h(2, "5. Semantic Similarity Strategy")
    p(
        f"Chunk-level embeddings were computed using **`{embedding_model}`** "
        f"(or TF-IDF+SVD fallback if unavailable). "
        f"File embeddings = mean of chunk embeddings. "
        f"Run embeddings = mean of file embeddings. "
        f"Agent centroids = mean of run embeddings. "
        f"Pairwise similarity was computed as cosine similarity between run-level vectors."
    )

    # --- 6. Within-Agent Reproducibility ---
    h(2, "6. Within-Agent Reproducibility")
    if not within_repro.empty:
        table(within_repro)
        fig("within_agent_similarity_boxplot.png",
            "Within-agent cosine similarity distribution across 8 runs")
    else:
        p("_Insufficient data for within-agent reproducibility._")

    # --- 7. Between-Agent Comparison ---
    h(2, "7. Between-Agent Comparison")
    if not between_sim.empty:
        table(between_sim)
        fig("run_similarity_heatmap.png",
            "Pairwise run cosine similarity heatmap (all 24 runs)")
    else:
        p("_Insufficient data for between-agent comparison._")

    # --- 8. Dimensionality Reduction ---
    h(2, "8. Dimensionality Reduction")
    p("Run-level embedding vectors were projected into 2D and 3D using PCA, MDS, t-SNE, and UMAP.")
    p(
        "**Confidence ellipses** are drawn on all 2D plots. Each ellipse shows the "
        "covariance-based 95% confidence region (n_std = 2) for one agent's 8 runs. "
        "This is computed from the 2×2 covariance matrix of the projected coordinates "
        "using eigen-decomposition. With only 8 runs per agent the ellipses are "
        "**descriptive** (not formal statistical confidence regions). "
        "**Outlier runs** (if any) are circled in red."
    )
    fig("pca_2d.png", "PCA 2D — Run Embeddings with 95% confidence ellipses and outlier annotations")
    fig("mds_2d.png", "MDS 2D — Run Embeddings with 95% confidence ellipses and outlier annotations")
    fig("tsne_2d.png", "t-SNE 2D — Run Embeddings with 95% confidence ellipses and outlier annotations")
    fig("umap_2d.png", "UMAP 2D — Run Embeddings with 95% confidence ellipses and outlier annotations")
    p("Interactive 3D PCA is available in `figures/pca_3d.html`.")

    # --- 9. Monte Carlo ---
    h(2, "9. Monte Carlo Satellite Simulation")
    p(
        f"**Synthetic satellite points are not real observations.** "
        f"They are uncertainty probes generated by bootstrap centroid resampling and "
        f"regularised multivariate Gaussian sampling around the {8} real run vectors "
        f"of each agent. {n_mc} iterations per agent per method."
    )
    if not mc_summary.empty:
        table(mc_summary)
    p("Interactive 3D satellite plot available in `figures/monte_carlo_satellites_3d.html`.")

    # --- 10. AgentScore ---
    h(2, "10. AgentScore Calculation")
    p("**Formula:** `AgentScore = P × (0.50 + 0.30 × D/100 + 0.20 × R/100)`")
    ul([
        "R = Result accuracy (0–100)",
        "P = Process quality (0–100)",
        "D = Interpretation depth (0–100)",
    ])
    score_type = run_scores["score_type"].mode().iloc[0] if not run_scores.empty else "N/A"
    p(f"Score type: **{score_type}**")
    if not agent_summary.empty:
        table(agent_summary)
        fig("score_distribution_boxplot.png", "AgentScore distribution across 8 runs")

    # --- 11. Ranking Uncertainty ---
    h(2, "11. Ranking Uncertainty")
    if not ranking_uncertainty.empty:
        rank1 = ranking_uncertainty[ranking_uncertainty["rank"] == 1]
        table(rank1)
        fig("ranking_probability_barplot.png", "Probability of each rank by agent (bootstrap)")
    if not validation_summary.empty:
        table(validation_summary)
    if not pairwise_win.empty:
        table(pairwise_win)

    # -----------------------------------------------------------------------
    # NEW SECTION — Scoring audit, output-volume diagnostics, domain-aware
    # -----------------------------------------------------------------------
    if audit:
        _write_audit_section(audit, h, p, ul, table, fig, run_scores)

    # -----------------------------------------------------------------------
    # NEW SECTION — Consensus and divergence analysis
    # -----------------------------------------------------------------------
    if consensus:
        _write_consensus_section(consensus, h, p, ul, table, fig)

    # --- 12. Limitations ---
    h(2, "12. Limitations")
    ul([
        "Only 8 real runs per agent — statistical power is limited.",
        "Proxy scores are keyword-count-based and do not replace expert assessment.",
        "Semantic similarity depends on the embedding model used.",
        "Biomni received additional clinical metadata; this is modelled as an agentic capability, not a confounder.",
        "Synthetic Monte Carlo points probe uncertainty space but do not add real data.",
        "Reliability statistics (Kendall W, Cronbach α) are descriptive effect sizes only.",
    ])

    # --- 13. Results and Discussion ---
    h(2, "13. Results and Discussion")
    h(3, "Cross-Agent Reproducibility and Ranking of AI Research Agents on a Bulk RNA-seq / Transcriptomics Task")
    p(
        f"{n_agents} AI agents — {agent_list} — each independently analysed the same "
        "transcriptomics dataset and produced 8 runs of output. "
        "Because the number and types of output files differ across agents, "
        "direct file-by-file comparison was not applicable. "
        "Instead, outputs were compared at run level via aggregated semantic embeddings."
    )
    p(
        "Biomni possessed an interactive dialogue function and proactively requested additional "
        "clinical metadata, which was provided. Agents that did not ask additional questions "
        "did not receive this information. This information-seeking behaviour is explicitly "
        "treated as part of agentic performance, not as a protocol violation."
    )
    if not within_repro.empty:
        best_repro = within_repro.sort_values("reproducibility", ascending=False).iloc[0]
        p(
            f"Within-agent reproducibility was highest for **{best_repro['agent']}** "
            f"(reproducibility index = {best_repro['reproducibility']:.3f}). "
            f"All reliability statistics are descriptive because the number of real runs is limited."
        )
    if not agent_summary.empty:
        top = agent_summary.iloc[0]
        p(
            f"**{top['agent']}** achieved the highest mean AgentScore "
            f"({top['mean_score']:.1f} ± {top['sd_score']:.1f}, "
            f"95% CI [{top['ci95_lo']:.1f}, {top['ci95_hi']:.1f}])."
        )

    kw = reliability.get("kendall_w", float("nan"))
    alpha = reliability.get("cronbach_alpha", float("nan"))
    kw_str    = f"{kw:.3f}" if not np.isnan(kw) else "N/A"
    alpha_str = f"{alpha:.3f}" if not np.isnan(alpha) else "N/A"
    p(
        f"Kendall's W = {kw_str}; Cronbach's α = {alpha_str}. "
        "These are interpreted as descriptive effect sizes only."
    )

    # --- 14. Validation Summary ---
    h(2, "14. Validation Summary")
    if not validation_table.empty:
        table(validation_table)
    else:
        p("_No validation summary available._")

    if audit:
        svs = audit.get("scoring_validation_summary")
        if svs is not None and not svs.empty:
            h(3, "Scoring validation summary")
            p("Question / method / key result / interpretation / limitation for "
              "the scoring-audit and error-estimation layer:")
            table(svs, max_rows=40)

    # -----------------------------------------------------------------------
    # Write Markdown
    # -----------------------------------------------------------------------
    md_path = report_dir / "benchmark_report.md"
    md_content = "\n".join(lines)
    md_path.write_text(md_content, encoding="utf-8")
    logger.info("Saved Markdown report → %s", md_path)

    # -----------------------------------------------------------------------
    # Convert to HTML
    # -----------------------------------------------------------------------
    html_path = report_dir / "benchmark_report.html"
    try:
        import markdown as _md
        html_body = _md.markdown(md_content, extensions=["tables", "fenced_code"])
    except ImportError:
        # Fallback: wrap in <pre>
        html_body = f"<pre>{md_content}</pre>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Agent Benchmarking Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1.5rem; color: #222; }}
    h1,h2,h3 {{ border-bottom: 1px solid #ddd; padding-bottom: .3em; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: .9em; }}
    th, td {{ border: 1px solid #ccc; padding: .4em .7em; text-align: left; }}
    th {{ background: #f4f4f4; }}
    img {{ max-width: 100%; }}
    code {{ background: #f6f8fa; padding: .15em .4em; border-radius: 3px; }}
    pre code {{ display: block; padding: 1em; overflow-x: auto; }}
  </style>
</head>
<body>
{html_body}
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    logger.info("Saved HTML report → %s", html_path)
