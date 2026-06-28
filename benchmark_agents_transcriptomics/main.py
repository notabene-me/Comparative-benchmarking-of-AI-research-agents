#!/usr/bin/env python3
"""
main.py — CLI entry point for the AI Agent Benchmark pipeline.

Usage:
    python main.py \\
        --input_dir ./agent_outputs \\
        --output_dir ./benchmark_results \\
        --n_mc 10000 \\
        --embedding_model sentence-transformers/all-MiniLM-L6-v2

    python main.py --input_dir ./agent_outputs --output_dir ./benchmark_results \\
        --scores ./manual_scores.csv

    python main.py --input_dir ./agent_outputs --output_dir ./benchmark_results \\
        --no_embeddings

    # Scan only — no embeddings, PCA, MC or scoring:
    python main.py --validate_only \\
        --input_dir ./agent_outputs --output_dir ./benchmark_results

    # Generate synthetic demo data then exit:
    python main.py --generate_demo --input_dir ./agent_outputs
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# Imports (after logging is configured)
# ---------------------------------------------------------------------------

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable


from src.scanner      import scan, summarise
from src.parser       import parse_file
from src.inventory    import build_inventory, save_inventory, print_summary
from src.chunking     import chunk_records
from src.embeddings   import (
    load_backend, embed_chunks,
    aggregate_file_embeddings, aggregate_run_embeddings,
    aggregate_agent_embeddings, build_run_matrix,
)
from src.features     import (
    extract_all_file_features,
    aggregate_run_features, aggregate_agent_features,
)
from src.similarity   import (
    pairwise_run_similarity,
    within_agent_reproducibility,
    between_agent_similarity,
)
from src.dimensionality import run_pca, run_mds, run_tsne, run_umap
from src.scoring      import (
    load_manual_scores, compute_run_scores, compute_agent_summary,
    decompose_proxy_components, compute_run_scores_excluding_volume,
    build_manual_vs_proxy,
)
from src.domain_scoring import compute_domain_scores, domain_agent_summary
from src.score_audit  import (
    compute_output_volume_metrics, compute_score_volume_correlation,
    compute_volume_penalized_score, compute_ranking_sensitivity,
    build_ranking_sensitivity_matrix, build_score_audit,
    build_scoring_validation_summary,
)
from src.fairness     import load_run_registry, fairness_sensitivity_summary
from src.error_analysis import (
    score_error_summary, bootstrap_error_summary, pairwise_score_difference_error,
    ranking_error_summary, reproducibility_error_summary, monte_carlo_error_summary,
    mc_distance_distributions, flag_outlier_runs,
)
from src.consensus_divergence import (
    METHOD_TERMS as CONS_METHOD_TERMS,
    BIO_TERMS    as CONS_BIO_TERMS,
    REPORTING_TERMS as CONS_REPORTING_TERMS,
    extract_run_term_matrix,
    compute_shared_vs_unique,
    compute_consensus_core,
    compute_consensus_alignment,
    compute_distinctiveness,
    build_commonality_difference_summary,
    build_consensus_validation_summary,
)
from src.monte_carlo  import run_monte_carlo
from src.visualization import (
    plot_pca_2d, plot_pca_3d, plot_mds_2d, plot_tsne_2d, plot_umap_2d,
    plot_run_similarity_heatmap, plot_within_agent_boxplot,
    plot_mc_satellites_3d, plot_score_distribution, plot_ranking_probability,
    plot_score_component_heatmap, plot_domain_score_component_heatmap,
    plot_score_vs_metric, plot_output_volume_by_agent_boxplot,
    plot_output_volume_vs_score_panel, plot_proxy_vs_domain_scatter,
    plot_domain_score_radar, plot_proxy_vs_manual_scatter,
    plot_ranking_sensitivity_heatmap, plot_ranking_sensitivity_barplot,
    plot_agent_score_errorbar, plot_agent_score_bootstrap_ci,
    plot_pairwise_score_difference_ci, plot_rank_uncertainty_interval,
    plot_reproducibility_errorbar, plot_centroid_uncertainty_pca2d,
    plot_monte_carlo_distance_distribution,
    plot_consensus_method_overlap_heatmap, plot_consensus_biological_theme_heatmap,
    plot_agent_specific_terms_barplot, plot_shared_vs_unique_features_barplot,
    plot_consensus_alignment_barplot, plot_consensus_alignment_errorbar,
    plot_consensus_alignment_pca3d,
    plot_agent_distinctiveness_barplot, plot_agent_distinctiveness_vs_agentscore,
    plot_outlier_runs,
)
from src.validation   import bootstrap_ranking, compute_reliability, build_validation_table
from src.report       import generate_report
from src.platform_utils import (
    configure_matplotlib_backend,
    ensure_benchmark_output_layout,
    ensure_directory,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="AI Agent Benchmark — bulk RNA-seq / transcriptomics comparison pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_dir",  default="./agent_outputs_transcriptomics",
                   help="Root directory containing agent output files.")
    p.add_argument("--output_dir", default="./benchmark_results_transcriptomics",
                   help="Root directory for all outputs.")
    p.add_argument("--n_mc",       type=int, default=10_000,
                   help="Number of Monte Carlo / bootstrap iterations.")
    p.add_argument("--embedding_model",
                   default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Sentence-transformers model name (or HuggingFace path).")
    p.add_argument("--no_embeddings", action="store_true",
                   help="Skip embedding computation; use TF-IDF fallback.")
    p.add_argument("--scores",     default=None,
                   help="Path to manual scores CSV (agent, run, R, P, D).")
    p.add_argument("--chunk_size", type=int, default=800,
                   help="Chunk size in tokens.")
    p.add_argument("--chunk_overlap", type=int, default=150,
                   help="Chunk overlap in tokens.")
    p.add_argument("--log_level",  default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity.")
    p.add_argument("--validate_only", action="store_true",
                   help=(
                       "Scan files, detect agents/runs, save inventory and "
                       "design_validation.csv, print summary — then stop. "
                       "No embeddings, PCA, MC or scoring."
                   ))
    p.add_argument("--generate_demo", action="store_true",
                   help="Create synthetic demo agent outputs in input_dir and exit.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_csv(df, path: Path, name: str) -> None:
    if df is not None and not df.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Saved %s → %s", name, path)


# ---------------------------------------------------------------------------
# Validate-only mode
# ---------------------------------------------------------------------------

EXPECTED_AGENTS = 4
EXPECTED_RUNS   = 8


def run_validate_only(args) -> None:
    """
    Scan input_dir, detect agents and runs, save:
      - file_inventory.csv
      - agent_run_file_counts.csv
      - design_validation.csv
      - report/initial_validation_report.md
    Then print a short summary and exit.
    """
    import datetime

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    tables_dir, _, report_dir = ensure_benchmark_output_layout(output_dir)

    logger.info("=== VALIDATE-ONLY: scanning %s ===", input_dir)

    # --- Scan ---
    from src.scanner import scan as _scan, KNOWN_AGENTS, AGENT_ALIASES
    all_records = []

    # We scan broadly — including files that couldn't be classified —
    # by temporarily reading raw walker output.
    import os
    from pathlib import Path as _Path
    from src.scanner import (
        FileRecord, _detect_agent_from_name, _detect_run_number,
        _normalise_agent, _resolve_agent_dir,
    )

    unknown_agent_files = []
    unknown_run_files   = []
    good_records        = []

    if not input_dir.exists():
        logger.warning("Input directory does not exist: %s", input_dir)
    else:
        for dirpath, dirnames, filenames in os.walk(input_dir):
            dirnames.sort()
            dp  = _Path(dirpath)
            rel = dp.relative_to(input_dir)
            parts = list(rel.parts)

            for fname in sorted(filenames):
                if fname.startswith("."):
                    continue
                fp = dp / fname

                # Try to detect agent
                agent = None
                run   = None

                if parts:
                    agent = _resolve_agent_dir(parts[0])
                if len(parts) >= 2 and agent:
                    run = _detect_run_number(parts[1])
                    if run is None:
                        run = _detect_run_number(parts[0])
                elif len(parts) == 1 and agent:
                    run = _detect_run_number(parts[0])

                if agent is None:
                    agent = _detect_agent_from_name(fname)
                if run is None:
                    run = _detect_run_number(fname)

                if agent is None:
                    for p in parts:
                        a = _resolve_agent_dir(p) if p == parts[0] else _detect_agent_from_name(p)
                        if a:
                            agent = a
                            break
                if run is None:
                    for p in parts:
                        r = _detect_run_number(p)
                        if r is not None:
                            run = r
                            break

                if agent is None:
                    unknown_agent_files.append(str(fp))
                    continue
                agent = _normalise_agent(agent)

                if run is None:
                    unknown_run_files.append(str(fp))
                    continue

                good_records.append(FileRecord(agent=agent, run=run, filepath=fp))

    # --- Build inventory ---
    from src.inventory import build_inventory as _build_inv, save_inventory as _save_inv, print_summary as _print_sum
    inventory = _build_inv(good_records)
    _save_inv(inventory, output_dir)
    _print_sum(inventory)

    # --- Agent × run file counts ---
    if not inventory.empty:
        counts = (
            inventory.groupby(["agent", "run"])
            .size()
            .reset_index(name="n_files")
        )
    else:
        counts = pd.DataFrame(columns=["agent", "run", "n_files"])

    _save_csv(counts, tables_dir / "agent_run_file_counts.csv", "agent_run_file_counts")

    # --- Design validation ---
    detected_agents = sorted(inventory["agent"].unique().tolist()) if not inventory.empty else []
    n_agents        = len(detected_agents)

    design_rows = []
    for agent in detected_agents:
        adf  = inventory[inventory["agent"] == agent]
        runs = sorted(adf["run"].unique().tolist())
        n_runs = len(runs)
        missing_runs = sorted(set(range(1, EXPECTED_RUNS + 1)) - set(runs))
        design_rows.append({
            "agent":         agent,
            "detected_runs": n_runs,
            "expected_runs": EXPECTED_RUNS,
            "run_numbers":   str(runs),
            "missing_runs":  str(missing_runs) if missing_runs else "none",
            "total_files":   len(adf),
            "status":        "OK" if n_runs == EXPECTED_RUNS else f"MISSING {len(missing_runs)} run(s)",
        })

    # Global design check
    expected_real_runs = EXPECTED_AGENTS * EXPECTED_RUNS
    observed_real_runs = len(inventory[["agent", "run"]].drop_duplicates()) if not inventory.empty else 0

    design_df = pd.DataFrame(design_rows)
    _save_csv(design_df, tables_dir / "design_validation.csv", "design_validation")

    # --- Terminal summary ---
    total_files = len(inventory) + len(unknown_agent_files) + len(unknown_run_files)
    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Total files scanned       : {total_files}")
    print(f"  Files with known agent    : {len(inventory) + len(unknown_run_files)}")
    print(f"  Unrecognised agent files  : {len(unknown_agent_files)}")
    print(f"  Unrecognised run files    : {len(unknown_run_files)}")
    print(f"  Detected agents ({n_agents})       : {', '.join(detected_agents) or 'none'}")
    print(f"  Expected design           : {EXPECTED_AGENTS} agents × {EXPECTED_RUNS} runs = {expected_real_runs} runs")
    print(f"  Observed real runs        : {observed_real_runs}")
    print()
    for row in design_rows:
        print(f"  {row['agent']:12s}  runs detected={row['detected_runs']}/{row['expected_runs']}  "
              f"files={row['total_files']}  {row['status']}")
    print("=" * 60 + "\n")

    # --- Markdown report ---
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    md_lines = [
        "# Initial Validation Report",
        "",
        f"**Generated:** {date_str}  ",
        f"**Input directory:** `{input_dir}`  ",
        "",
        "## Summary",
        "",
        f"| Item | Value |",
        f"|------|-------|",
        f"| Total files scanned | {total_files} |",
        f"| Files classified | {len(inventory)} |",
        f"| Unrecognised agent | {len(unknown_agent_files)} |",
        f"| Unrecognised run | {len(unknown_run_files)} |",
        f"| Detected agents | {', '.join(detected_agents) or 'none'} |",
        f"| Expected design | {EXPECTED_AGENTS} agents × {EXPECTED_RUNS} runs = {expected_real_runs} runs |",
        f"| Observed real runs | {observed_real_runs} |",
        "",
        "## Per-Agent Design Check",
        "",
    ]

    if not design_df.empty:
        try:
            md_lines.append(design_df.to_markdown(index=False))
        except Exception:
            md_lines.append(design_df.to_string(index=False))
    else:
        md_lines.append("_No agents detected._")

    md_lines += [
        "",
        "## File Inventory (first 30 rows)",
        "",
    ]
    if not inventory.empty:
        try:
            md_lines.append(inventory.head(30).to_markdown(index=False))
        except Exception:
            md_lines.append(inventory.head(30).to_string(index=False))
    else:
        md_lines.append("_Empty._")

    if unknown_agent_files:
        md_lines += ["", "## Unrecognised Agent Files", ""]
        for f in unknown_agent_files[:20]:
            md_lines.append(f"- `{f}`")
        if len(unknown_agent_files) > 20:
            md_lines.append(f"- … and {len(unknown_agent_files) - 20} more")

    if unknown_run_files:
        md_lines += ["", "## Unrecognised Run Files", ""]
        for f in unknown_run_files[:20]:
            md_lines.append(f"- `{f}`")

    md_path = report_dir / "initial_validation_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Saved validation report → %s", md_path)
    logger.info("=== Validate-only complete. ===")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args) -> None:
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    tables_dir, figures_dir, _ = ensure_benchmark_output_layout(output_dir)

    # -----------------------------------------------------------------------
    # 1. Scan & inventory
    # -----------------------------------------------------------------------
    logger.info("=== STEP 1: Scanning input directory ===")
    records = scan(input_dir)

    if not records:
        logger.error(
            "No agent files found in '%s'. "
            "Expected layout: <input_dir>/<AgentName>/run_N/file.ext\n"
            "Use --generate_demo to create synthetic data.",
            input_dir,
        )
        sys.exit(1)

    inventory = build_inventory(records)
    save_inventory(inventory, output_dir)
    print_summary(inventory)

    # -----------------------------------------------------------------------
    # 2. Parse files
    # -----------------------------------------------------------------------
    logger.info("=== STEP 2: Parsing files ===")
    file_texts: dict = {}
    file_meta:  list = []

    for rec in tqdm(records, desc="Parsing files", disable=not _HAS_TQDM):
        fid  = f"{rec.agent}__run{rec.run}__{rec.filepath.name}"
        text = parse_file(rec.filepath)
        file_texts[fid] = text
        file_meta.append({
            "file_id":  fid,
            "agent":    rec.agent,
            "run":      rec.run,
            "filepath": str(rec.filepath),
            "filename": rec.filepath.name,
            "file_type": rec.file_type,
        })

    logger.info("Parsed %d files.", len(file_texts))

    # -----------------------------------------------------------------------
    # 3. Feature extraction
    # -----------------------------------------------------------------------
    logger.info("=== STEP 3: Feature extraction ===")
    file_features  = extract_all_file_features(file_texts)
    run_features   = aggregate_run_features(file_features, file_meta)
    agent_features = aggregate_agent_features(run_features)

    # Merge meta into file_features
    meta_df = pd.DataFrame(file_meta)[["file_id", "agent", "run"]]
    if not file_features.empty:
        file_features = meta_df.merge(file_features, on="file_id", how="left")

    _save_csv(file_features,  tables_dir / "file_level_features.csv",  "file features")
    _save_csv(run_features,   tables_dir / "run_level_features.csv",   "run features")
    _save_csv(agent_features, tables_dir / "agent_level_features.csv", "agent features")

    # -----------------------------------------------------------------------
    # 4. Chunking
    # -----------------------------------------------------------------------
    logger.info("=== STEP 4: Chunking ===")
    chunk_map = chunk_records(file_texts, args.chunk_size, args.chunk_overlap)
    total_chunks = sum(len(v) for v in chunk_map.values())
    logger.info("Total chunks: %d", total_chunks)

    # -----------------------------------------------------------------------
    # 5. Embeddings
    # -----------------------------------------------------------------------
    logger.info("=== STEP 5: Embeddings ===")
    model_name = args.embedding_model.replace("sentence-transformers/", "")
    backend    = load_backend(model_name, force_fallback=args.no_embeddings)

    chunk_embs = embed_chunks(chunk_map, backend)
    file_embs  = aggregate_file_embeddings(chunk_map, chunk_embs)
    run_embs   = aggregate_run_embeddings(file_embs, file_meta)
    agent_embs = aggregate_agent_embeddings(run_embs)

    logger.info(
        "Embeddings: %d chunks, %d files, %d runs, %d agents.",
        len(chunk_embs), len(file_embs), len(run_embs), len(agent_embs),
    )

    # -----------------------------------------------------------------------
    # 6. Similarity
    # -----------------------------------------------------------------------
    logger.info("=== STEP 6: Similarity ===")
    pairwise_sim = pairwise_run_similarity(run_embs, run_features)
    within_repro = within_agent_reproducibility(pairwise_sim)
    between_sim  = between_agent_similarity(run_embs, agent_embs)

    _save_csv(pairwise_sim, tables_dir / "pairwise_run_similarity.csv",   "pairwise similarity")
    _save_csv(within_repro, tables_dir / "within_agent_reproducibility.csv", "within-agent repro")
    _save_csv(between_sim,  tables_dir / "between_agent_similarity.csv",  "between-agent sim")

    # -----------------------------------------------------------------------
    # 7. Dimensionality reduction
    # -----------------------------------------------------------------------
    logger.info("=== STEP 7: Dimensionality reduction ===")
    run_mat, run_keys = build_run_matrix(run_embs)

    pca_df = mds_df = tsne_df = umap_df = None
    _pca_model = None
    if run_mat.shape[0] >= 2:
        pca_df, _pca_model = run_pca(run_mat, run_keys, n_components=3)
        mds_df             = run_mds(run_mat, run_keys, n_components=2)
        tsne_df            = run_tsne(run_mat, run_keys, n_components=2)
        umap_df            = run_umap(run_mat, run_keys, n_components=2)
    else:
        logger.warning("Too few run embeddings for dimensionality reduction (need ≥ 2).")

    # -----------------------------------------------------------------------
    # 8. Figures — dimensionality
    # -----------------------------------------------------------------------
    logger.info("=== STEP 8: Visualisation ===")
    # outlier_df is populated in Step 12b; use None here and re-call plots after
    if pca_df is not None:
        plot_pca_2d(pca_df, figures_dir)
        _var_ratio = (list(_pca_model.explained_variance_ratio_)
                      if _pca_model is not None else None)
        plot_pca_3d(pca_df, figures_dir, var_ratio=_var_ratio)
    if mds_df is not None:
        plot_mds_2d(mds_df, figures_dir)
    plot_tsne_2d(tsne_df, figures_dir)
    plot_umap_2d(umap_df, figures_dir)
    plot_run_similarity_heatmap(pairwise_sim, figures_dir)
    plot_within_agent_boxplot(pairwise_sim, figures_dir)

    # -----------------------------------------------------------------------
    # 9. Monte Carlo
    # -----------------------------------------------------------------------
    logger.info("=== STEP 9: Monte Carlo satellite generation ===")
    mc_centroid_df = mc_summary_df = None
    boot_by_agent  = gauss_by_agent = {}

    if run_embs:
        mc_centroid_df, mc_summary_df, boot_by_agent, gauss_by_agent = run_monte_carlo(
            run_embs, n_mc=args.n_mc, seed=SEED,
        )
        _save_csv(mc_centroid_df, tables_dir / "monte_carlo_centroids.csv", "MC centroids")
        _save_csv(mc_summary_df,  tables_dir / "monte_carlo_summary.csv",   "MC summary")
        plot_mc_satellites_3d(run_embs, boot_by_agent, gauss_by_agent, figures_dir)

    # -----------------------------------------------------------------------
    # 10. Scoring
    # -----------------------------------------------------------------------
    logger.info("=== STEP 10: AgentScore calculation ===")
    manual_scores = load_manual_scores(args.scores)
    run_scores    = compute_run_scores(run_features, manual_scores)
    agent_summary = compute_agent_summary(run_scores)

    # Proxy decomposition + domain-aware score (independent sensitivity analysis)
    proxy_components = decompose_proxy_components(run_features)
    domain_scores    = compute_domain_scores(run_features)
    domain_summary   = domain_agent_summary(domain_scores)

    _save_csv(run_scores,       tables_dir / "agent_scores.csv",                       "run scores")
    _save_csv(agent_summary,    tables_dir / "ranking_summary.csv",                    "ranking summary")
    _save_csv(proxy_components, tables_dir / "proxy_score_components.csv",             "proxy components")
    _save_csv(domain_scores,    tables_dir / "domain_transcriptomics_score_components.csv","domain components")
    plot_score_distribution(run_scores, figures_dir, agent_summary=agent_summary)

    # -----------------------------------------------------------------------
    # 11. Validation / ranking uncertainty
    # -----------------------------------------------------------------------
    logger.info("=== STEP 11: Ranking uncertainty & reliability ===")
    ranking_uncertainty = pairwise_win = validation_summary_df = None
    reliability = {"kendall_w": float("nan"), "cronbach_alpha": float("nan"),
                   "spearman_df": pd.DataFrame()}

    if not run_scores.empty:
        ranking_uncertainty, pairwise_win, validation_summary_df = bootstrap_ranking(
            run_scores, n_mc=args.n_mc, seed=SEED,
        )
        reliability = compute_reliability(run_scores)

        _save_csv(ranking_uncertainty,  tables_dir / "ranking_uncertainty.csv",      "ranking uncertainty")
        _save_csv(pairwise_win,         tables_dir / "pairwise_win_probabilities.csv","pairwise win probs")
        _save_csv(validation_summary_df,tables_dir / "validation_summary.csv",       "validation summary")

        plot_ranking_probability(ranking_uncertainty, figures_dir)

    def _or_empty(df):
        return df if df is not None else pd.DataFrame()

    # -----------------------------------------------------------------------
    # 12. Scoring audit + output-volume diagnostics + domain-aware sensitivity
    # -----------------------------------------------------------------------
    outlier_df = pd.DataFrame()  # initialised early; populated in Step 12b
    logger.info("=== STEP 12: Scoring audit & output-volume diagnostics ===")

    volume_metrics = compute_output_volume_metrics(
        inventory, run_features, chunk_map, file_meta)
    score_vol_corr = compute_score_volume_correlation(run_scores, volume_metrics)
    volume_penalized = compute_volume_penalized_score(run_scores, volume_metrics)
    no_volume_scores = compute_run_scores_excluding_volume(run_features)

    ranking_sensitivity = compute_ranking_sensitivity(
        run_scores, domain_scores, _or_empty(within_repro),
        volume_penalized, no_volume_scores)
    sens_matrix = build_ranking_sensitivity_matrix(ranking_sensitivity)

    score_audit_tbl = build_score_audit(
        run_scores, domain_scores, score_vol_corr,
        volume_penalized, ranking_sensitivity)

    # Manual vs proxy (only when manual scores supplied)
    manual_vs_proxy = build_manual_vs_proxy(run_features, manual_scores, domain_scores)

    # Fairness / information-access audit
    run_registry = load_run_registry(input_dir)
    fairness_summary = fairness_sensitivity_summary(run_scores, run_registry)

    _save_csv(volume_metrics,     tables_dir / "output_volume_metrics.csv",          "output volume metrics")
    _save_csv(score_vol_corr,     tables_dir / "score_volume_correlation.csv",       "score-volume corr")
    _save_csv(ranking_sensitivity,tables_dir / "ranking_sensitivity_analysis.csv",   "ranking sensitivity")
    _save_csv(score_audit_tbl,    tables_dir / "score_audit.csv",                    "score audit")
    if not fairness_summary.empty:
        _save_csv(fairness_summary, tables_dir / "fairness_sensitivity_summary.csv", "fairness summary")
    if not manual_vs_proxy.empty:
        _save_csv(manual_vs_proxy, tables_dir / "manual_vs_proxy_scores.csv",        "manual vs proxy")

    # -----------------------------------------------------------------------
    # 12b. Error estimation & uncertainty
    # -----------------------------------------------------------------------
    logger.info("=== STEP 12b: Error estimation & uncertainty ===")

    # --- Outlier detection (before bootstrap error summaries) ---
    outlier_df = flag_outlier_runs(run_scores, score_col="AgentScore")
    if not outlier_df.empty:
        _save_csv(outlier_df, tables_dir / "outlier_runs.csv", "outlier runs")
        plot_outlier_runs(outlier_df, figures_dir)
        # Re-draw 2D embedding plots with outlier annotations
        if pca_df is not None:
            plot_pca_2d(pca_df, figures_dir, outlier_df=outlier_df)
        if mds_df is not None:
            plot_mds_2d(mds_df, figures_dir, outlier_df=outlier_df)
        plot_tsne_2d(tsne_df, figures_dir, outlier_df=outlier_df)
        plot_umap_2d(umap_df, figures_dir, outlier_df=outlier_df)

    n_boot = max(200, min(args.n_mc, 10_000))

    score_defs = {
        "A_proxy_agentscore": (run_scores, "AgentScore"),
        "B_domain_score":     (domain_scores, "DomainScore"),
        "D_interpretation_depth": (run_scores, "D"),
        "E_process_quality":  (run_scores, "P"),
        "F_volume_penalized": (volume_penalized, "AgentScore_volume_penalized"),
        "G_no_volume_features": (no_volume_scores, "AgentScore_no_volume"),
    }
    score_err  = score_error_summary(score_defs, n_boot=n_boot, seed=SEED)
    boot_err   = bootstrap_error_summary(run_scores, n_boot=n_boot, seed=SEED)
    pair_diff  = pairwise_score_difference_error(run_scores, n_boot=n_boot, seed=SEED)
    rank_err   = ranking_error_summary(run_scores, n_boot=n_boot, seed=SEED)
    repro_err  = reproducibility_error_summary(
        _or_empty(pairwise_sim), run_embs, _or_empty(between_sim), n_boot=n_boot, seed=SEED)
    mc_err     = monte_carlo_error_summary(run_embs, boot_by_agent, seed=SEED)

    _save_csv(score_err,  tables_dir / "score_error_summary.csv",          "score error")
    _save_csv(boot_err,   tables_dir / "bootstrap_error_summary.csv",      "bootstrap error")
    _save_csv(repro_err,  tables_dir / "reproducibility_error_summary.csv","reproducibility error")
    _save_csv(rank_err,   tables_dir / "ranking_error_summary.csv",        "ranking error")
    _save_csv(mc_err,     tables_dir / "monte_carlo_error_summary.csv",    "MC error")
    _save_csv(pair_diff,  tables_dir / "pairwise_score_difference_error.csv","pairwise diff error")

    # Scoring validation summary
    score_type = (run_scores["score_type"].mode().iloc[0]
                  if not run_scores.empty and "score_type" in run_scores.columns
                  else "proxy-estimated")
    scoring_val_summary = build_scoring_validation_summary(
        run_scores, domain_scores, score_vol_corr, ranking_sensitivity,
        boot_err, pair_diff, rank_err, repro_err, fairness_summary, score_type)
    _save_csv(scoring_val_summary, tables_dir / "scoring_validation_summary.csv",
              "scoring validation summary")

    # -----------------------------------------------------------------------
    # 12c. Audit & error figures
    # -----------------------------------------------------------------------
    logger.info("=== STEP 12c: Audit & uncertainty figures ===")
    # Scoring-audit figures
    plot_score_component_heatmap(proxy_components, figures_dir)
    plot_domain_score_component_heatmap(domain_scores, figures_dir)
    plot_score_vs_metric(run_scores, volume_metrics, "file_count",
                         "score_vs_file_count.png", figures_dir, score_vol_corr)
    plot_score_vs_metric(run_scores, volume_metrics, "text_length",
                         "score_vs_text_length.png", figures_dir, score_vol_corr)
    plot_score_vs_metric(run_scores, volume_metrics, "table_count",
                         "score_vs_table_count.png", figures_dir, score_vol_corr)
    plot_score_vs_metric(run_scores, volume_metrics, "figure_count",
                         "score_vs_figure_count.png", figures_dir, score_vol_corr)
    plot_score_vs_metric(run_scores, volume_metrics, "artifact_diversity",
                         "score_vs_artifact_diversity.png", figures_dir, score_vol_corr)
    plot_output_volume_by_agent_boxplot(volume_metrics, figures_dir)
    plot_output_volume_vs_score_panel(run_scores, volume_metrics, figures_dir, score_vol_corr)

    # Proxy vs domain caption with Spearman
    dom_caption = ""
    if not score_audit_tbl.empty:
        row = score_audit_tbl[score_audit_tbl["diagnostic"] == "proxy_vs_domain_rank_correlation"]
        if not row.empty:
            dom_caption = f"Agent-mean Spearman ρ = {row['value'].iloc[0]}"
    plot_proxy_vs_domain_scatter(run_scores, domain_scores, figures_dir, dom_caption)
    plot_domain_score_radar(domain_scores, figures_dir)
    if not manual_vs_proxy.empty:
        plot_proxy_vs_manual_scatter(manual_vs_proxy, figures_dir)
    plot_ranking_sensitivity_heatmap(sens_matrix, figures_dir, rank_err)
    plot_ranking_sensitivity_barplot(score_err, figures_dir)

    # Error / uncertainty figures
    plot_agent_score_errorbar(boot_err, figures_dir)
    plot_agent_score_bootstrap_ci(run_scores, boot_err, figures_dir)
    plot_pairwise_score_difference_ci(pair_diff, figures_dir)
    plot_rank_uncertainty_interval(rank_err, figures_dir)
    plot_reproducibility_errorbar(repro_err, figures_dir)
    plot_centroid_uncertainty_pca2d(run_embs, boot_by_agent, figures_dir)
    dist_dists = mc_distance_distributions(run_embs, boot_by_agent)
    plot_monte_carlo_distance_distribution(dist_dists, figures_dir)

    # -----------------------------------------------------------------------
    # 12d. Consensus and divergence analysis
    # -----------------------------------------------------------------------
    logger.info("=== STEP 12d: Consensus and divergence analysis ===")
    consensus_bundle = {}
    try:
        n_cons_boot = max(200, min(args.n_mc, 2000))

        method_mat    = extract_run_term_matrix(file_texts, file_meta, CONS_METHOD_TERMS)
        bio_mat       = extract_run_term_matrix(file_texts, file_meta, CONS_BIO_TERMS)
        reporting_mat = extract_run_term_matrix(file_texts, file_meta, CONS_REPORTING_TERMS)

        _save_csv(method_mat,    tables_dir / "consensus_method_terms.csv",    "method terms")
        _save_csv(bio_mat,       tables_dir / "consensus_biological_terms.csv","biological terms")
        _save_csv(reporting_mat, tables_dir / "consensus_reporting_terms.csv", "reporting terms")

        shared_terms, agent_specific = compute_shared_vs_unique(
            method_mat, bio_mat, reporting_mat)
        _save_csv(shared_terms,   tables_dir / "shared_terms_summary.csv",   "shared terms")
        _save_csv(agent_specific, tables_dir / "agent_specific_terms.csv",   "agent-specific terms")

        consensus_core = compute_consensus_core(method_mat, bio_mat, reporting_mat)
        _save_csv(consensus_core, tables_dir / "consensus_core_terms.csv", "consensus core")

        align_run, align_err = compute_consensus_alignment(
            run_embs, method_mat, bio_mat, reporting_mat,
            consensus_core, n_boot=n_cons_boot, seed=SEED)
        _save_csv(align_run, tables_dir / "consensus_alignment_scores.csv",
                  "consensus alignment scores")
        _save_csv(align_err, tables_dir / "consensus_alignment_error_summary.csv",
                  "consensus alignment error")

        distinct_run, distinct_err = compute_distinctiveness(
            run_embs, method_mat, bio_mat, reporting_mat,
            shared_terms, run_scores, n_boot=n_cons_boot, seed=SEED)
        _save_csv(distinct_run, tables_dir / "agent_distinctiveness_scores.csv",
                  "distinctiveness scores")
        _save_csv(distinct_err, tables_dir / "agent_distinctiveness_error_summary.csv",
                  "distinctiveness error")

        cd_summary = build_commonality_difference_summary(
            shared_terms, consensus_core, align_err, distinct_err,
            run_scores, _or_empty(within_repro), volume_metrics)
        _save_csv(cd_summary, tables_dir / "commonality_difference_summary.csv",
                  "commonality/difference summary")

        cons_val = build_consensus_validation_summary(
            shared_terms, consensus_core, align_err, distinct_err, run_scores)
        _save_csv(cons_val, tables_dir / "consensus_validation_summary.csv",
                  "consensus validation")

        # Consensus figures
        logger.info("=== STEP 12e: Consensus figures ===")
        plot_consensus_method_overlap_heatmap(method_mat, figures_dir)
        plot_consensus_biological_theme_heatmap(bio_mat, figures_dir)
        plot_agent_specific_terms_barplot(agent_specific, figures_dir)
        plot_shared_vs_unique_features_barplot(shared_terms, figures_dir)
        plot_consensus_alignment_barplot(align_err, figures_dir)
        plot_consensus_alignment_errorbar(align_run, align_err, figures_dir)
        plot_consensus_alignment_pca3d(run_embs, align_run, figures_dir)
        plot_agent_distinctiveness_barplot(distinct_err, figures_dir)
        plot_agent_distinctiveness_vs_agentscore(distinct_run, figures_dir)

        consensus_bundle = {
            "consensus_method_terms":              method_mat,
            "consensus_biological_terms":          bio_mat,
            "consensus_reporting_terms":           reporting_mat,
            "shared_terms_summary":                shared_terms,
            "agent_specific_terms":                agent_specific,
            "consensus_core_terms":                consensus_core,
            "consensus_alignment_scores":          align_run,
            "consensus_alignment_error_summary":   align_err,
            "agent_distinctiveness_scores":        distinct_run,
            "agent_distinctiveness_error_summary": distinct_err,
            "commonality_difference_summary":      cd_summary,
            "consensus_validation_summary":        cons_val,
        }
        logger.info("Consensus and divergence analysis complete.")
    except Exception as exc:
        logger.warning("Consensus analysis failed (non-blocking): %s", exc)

    # -----------------------------------------------------------------------
    # 12g. Validation table
    # -----------------------------------------------------------------------
    validation_table = build_validation_table(
        _or_empty(within_repro),
        _or_empty(between_sim),
        _or_empty(ranking_uncertainty),
        reliability,
    )

    audit_bundle = {
        "output_volume_metrics":               volume_metrics,
        "score_volume_correlation":            score_vol_corr,
        "proxy_score_components":              proxy_components,
        "domain_transcriptomics_score_components": domain_scores,
        "score_audit":                         score_audit_tbl,
        "manual_vs_proxy_scores":              manual_vs_proxy,
        "ranking_sensitivity_analysis":        ranking_sensitivity,
        "fairness_sensitivity_summary":        fairness_summary,
        "bootstrap_error_summary":             boot_err,
        "pairwise_score_difference_error":     pair_diff,
        "ranking_error_summary":               rank_err,
        "scoring_validation_summary":          scoring_val_summary,
        "outlier_runs":                        outlier_df if not outlier_df.empty else pd.DataFrame(),
    }

    # -----------------------------------------------------------------------
    # 13. Report
    # -----------------------------------------------------------------------
    logger.info("=== STEP 13: Generating report ===")

    generate_report(
        inventory           = inventory,
        file_features       = _or_empty(file_features),
        run_features        = _or_empty(run_features),
        agent_features      = _or_empty(agent_features),
        pairwise_sim        = _or_empty(pairwise_sim),
        within_repro        = _or_empty(within_repro),
        between_sim         = _or_empty(between_sim),
        pca_df              = _or_empty(pca_df),
        mc_summary          = _or_empty(mc_summary_df),
        run_scores          = _or_empty(run_scores),
        agent_summary       = _or_empty(agent_summary),
        ranking_uncertainty = _or_empty(ranking_uncertainty),
        pairwise_win        = _or_empty(pairwise_win),
        validation_summary  = _or_empty(validation_summary_df),
        validation_table    = _or_empty(validation_table),
        reliability         = reliability,
        output_dir          = output_dir,
        args_dict           = vars(args),
        audit               = audit_bundle,
        consensus           = consensus_bundle if consensus_bundle else None,
    )

    logger.info("=== Pipeline complete. Results in: %s ===", output_dir)


# ---------------------------------------------------------------------------
# Demo data generator
# ---------------------------------------------------------------------------

def _create_demo_data(input_dir: Path) -> None:
    """
    Create minimal synthetic agent output files so the pipeline can run
    end-to-end without real data.
    """
    import random
    random.seed(SEED)

    agents = ["ChatGPT", "Biomni", "KDense", "Finch"]
    n_runs = 8

    TRANSCRIPTOMICS_PHRASES = [
        "Bulk RNA-seq analysis was performed on Illumina paired-end reads.",
        "Raw reads were quality-checked with FastQC and aggregated with MultiQC.",
        "Reads were aligned to the reference genome using STAR.",
        "Gene-level counts were obtained using featureCounts.",
        "Differential expression was tested using DESeq2 with Benjamini-Hochberg FDR correction.",
        "Principal component analysis (PCA) revealed clear sample clustering by condition.",
        "Batch effects were corrected using ComBat prior to downstream analysis.",
        "Normalization was performed using DESeq2 size factors and variance-stabilizing transformation.",
        "A total of 18,432 genes passed filtering and were included in DE analysis.",
        "Volcano plot visualization revealed 842 significantly differentially expressed genes.",
        "GO enrichment and KEGG pathway analysis were conducted on upregulated DEGs.",
        "GSEA was run against MSigDB Hallmark gene sets.",
        "Log2 fold change (log2FC) thresholds of |log2FC| > 1 and FDR < 0.05 were applied.",
        "Heatmap of the top 50 DEGs showed distinct expression profiles between groups.",
        "Immune response and cell cycle pathways were significantly enriched.",
        "Transcription factor activity was discussed in biological interpretation.",
        "Alternative splicing events were noted as a limitation of bulk RNA-seq.",
        "Results tables include gene symbol, baseMean, log2FC, p-value, and padj.",
        "clusterProfiler was used for GO and Reactome enrichment visualization.",
        "Biomarker gene candidates were validated across all 8 independent runs.",
    ]

    BIOMNI_EXTRA = [
        "I also requested additional clinical metadata including age, BMI, and medication use.",
        "After receiving the clinical covariates, I incorporated them into the statistical model.",
        "ANCOVA was used to adjust for age and BMI as confounders.",
        "The information-seeking approach allowed more nuanced interpretation.",
    ]

    for agent in agents:
        for run_num in range(1, n_runs + 1):
            run_dir = input_dir / agent / f"run_{run_num}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Main analysis report
            n_phrases = random.randint(8, 15)
            phrases = random.choices(TRANSCRIPTOMICS_PHRASES, k=n_phrases)
            if agent == "Biomni" and run_num <= 3:
                phrases += random.choices(BIOMNI_EXTRA, k=2)

            report_text = f"# {agent} Transcriptomics Analysis — Run {run_num}\n\n"
            report_text += "\n".join(f"- {ph}" for ph in phrases)
            report_text += f"\n\n## Results Table\n\nGene | p-value | padj | log2FC\n"
            report_text += "---|---|---|---\n"
            for i in range(random.randint(5, 20)):
                pval = round(random.uniform(0.0001, 0.05), 5)
                fdr  = round(pval * 1.2, 5)
                fc   = round(random.uniform(-3, 3), 3)
                report_text += f"Gene_{i+1} | {pval} | {fdr} | {fc}\n"

            (run_dir / "analysis_report.txt").write_text(report_text, encoding="utf-8")

            # Small CSV
            import csv, io
            csv_buf = io.StringIO()
            writer = csv.writer(csv_buf)
            writer.writerow(["gene", "pvalue", "padj", "log2fc", "significant"])
            for i in range(random.randint(10, 30)):
                pval = round(random.uniform(0.0001, 0.2), 5)
                fdr  = round(pval * 1.5, 5)
                fc   = round(random.uniform(-4, 4), 3)
                sig  = "yes" if fdr < 0.05 else "no"
                writer.writerow([f"GENE{i+1:05d}", pval, fdr, fc, sig])
            (run_dir / "results.csv").write_text(csv_buf.getvalue(), encoding="utf-8")

    logger.info("Created demo agent outputs in: %s", input_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    configure_matplotlib_backend()
    args = parse_args(argv)
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.generate_demo:
        _create_demo_data(Path(args.input_dir))
        logger.info("Demo data created. Run without --generate_demo to start the pipeline.")
        return

    if args.validate_only:
        run_validate_only(args)
        return

    run_pipeline(args)


if __name__ == "__main__":
    main()
