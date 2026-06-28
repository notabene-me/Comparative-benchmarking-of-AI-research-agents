#!/usr/bin/env python3
"""End-to-end metabolomics analysis pipeline.

Usage
-----
    python run_analysis.py --input "path/to/table.xlsx" --output results

The pipeline runs: load -> preprocess -> univariate stats -> multivariate ->
pathway ORA + gene mapping -> figures -> biological interpretation report, and
writes all tables/figures/reports under the output directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# Make matplotlib happy in restricted/headless environments.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplcache"))

from mbx import RANDOM_SEED
from mbx.config import Config
from mbx.io_utils import load_dataset, ensure_dir
from mbx.preprocessing import preprocess
from mbx.statistics import differential_abundance, summarize
from mbx.multivariate import run_multivariate
from mbx.pathways import annotate_metabolites, enrichment_analysis, gene_table
from mbx import plotting
from mbx.interpretation import build_report


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description="Untargeted metabolomics analysis pipeline")
    p.add_argument("--input", required=True, help="intensity table (.xlsx/.csv)")
    p.add_argument("--output", default="results", help="output directory")
    p.add_argument("--feature-col", default="Molecule")
    p.add_argument("--sheet", default=0)
    p.add_argument("--control-prefix", default="Contr")
    p.add_argument("--case-prefix", default="Fibr")
    p.add_argument("--control-label", default="Control")
    p.add_argument("--case-label", default="Fibrosis")
    p.add_argument("--qc-prefix", default="QC")
    p.add_argument("--unpaired", action="store_true", help="treat groups as independent")
    p.add_argument("--qc-cv", type=float, default=0.30, help="QC CV cutoff (set <0 to disable)")
    p.add_argument("--normalization", default="pqn", choices=["pqn", "median", "sum", "none"])
    p.add_argument("--transform", default="log2", choices=["log2", "log10", "ln", "none"])
    p.add_argument("--scaling", default="pareto", choices=["pareto", "auto", "range", "none"])
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--log2fc", type=float, default=1.0)
    a = p.parse_args(argv)

    sheet = a.sheet
    try:
        sheet = int(sheet)
    except (TypeError, ValueError):
        pass

    return Config(
        input_path=a.input,
        sheet=sheet,
        feature_col=a.feature_col,
        output_dir=a.output,
        qc_prefix=a.qc_prefix,
        group_prefixes=(a.control_prefix, a.case_prefix),
        group_labels=(a.control_label, a.case_label),
        paired=not a.unpaired,
        qc_cv_threshold=None if a.qc_cv < 0 else a.qc_cv,
        normalization=None if a.normalization == "none" else a.normalization,
        transform=None if a.transform == "none" else a.transform,
        scaling=None if a.scaling == "none" else a.scaling,
        alpha=a.alpha,
        log2fc_threshold=a.log2fc,
        seed=RANDOM_SEED,
    )


def main(argv=None) -> int:
    cfg = parse_args(argv)
    np.random.seed(cfg.seed)

    out = ensure_dir(cfg.output_dir)
    tdir = ensure_dir(os.path.join(out, "tables"))
    fdir = ensure_dir(os.path.join(out, "figures"))
    rdir = ensure_dir(os.path.join(out, "report"))
    cfg.to_json(os.path.join(out, "config.json"))

    print(f"[1/6] Loading {cfg.input_path}")
    ds = load_dataset(cfg)
    print(f"      {ds.data.shape[0]} features | "
          f"{len(ds.qc_cols)} QC | "
          f"{', '.join(f'{k}={len(v)}' for k,v in ds.group_cols.items())} | "
          f"{len(ds.pairs)} pairs")

    print("[2/6] Preprocessing")
    proc = preprocess(ds)
    for s in proc.steps:
        print("      -", s)
    proc.raw_bio.to_csv(os.path.join(tdir, "filtered_raw_intensities.csv"))
    proc.normalized.to_csv(os.path.join(tdir, "normalized_intensities.csv"))
    proc.transformed.to_csv(os.path.join(tdir, "transformed_intensities.csv"))
    proc.feature_qc.to_csv(os.path.join(tdir, "feature_qc_metrics.csv"))

    print("[3/6] Differential abundance")
    stats_df = differential_abundance(ds, proc)
    stats_summary = summarize(stats_df, cfg)
    stats_df.to_csv(os.path.join(tdir, "differential_abundance.csv"))
    stats_df[stats_df["significant"]].to_csv(os.path.join(tdir, "significant_metabolites.csv"))
    print(f"      {stats_summary['n_significant']} significant "
          f"({stats_summary['n_up']} up / {stats_summary['n_down']} down)")

    print("[4/6] Multivariate analysis")
    try:
        mv = run_multivariate(ds, proc)
        mv.pca_scores.to_csv(os.path.join(tdir, "pca_scores.csv"))
        mv.pls_scores.to_csv(os.path.join(tdir, "plsda_scores.csv"))
        mv.vip.to_frame().to_csv(os.path.join(tdir, "plsda_vip_scores.csv"))
        print(f"      PLS-DA R2={mv.pls_r2:.2f} Q2={mv.pls_q2:.2f}; "
              f"PC1+PC2={mv.pca_explained[:2].sum()*100:.1f}% variance")
    except Exception as e:                       # pragma: no cover - robustness
        print(f"      multivariate analysis failed: {e}")
        mv = None

    print("[5/6] Pathway over-representation + gene mapping")
    annot = annotate_metabolites(list(stats_df.index))
    annot.to_csv(os.path.join(tdir, "metabolite_pathway_annotation.csv"))
    enrich_df = enrichment_analysis(stats_df, cfg)
    enrich_df.to_csv(os.path.join(tdir, "pathway_enrichment.csv"))
    gene_df = gene_table(stats_df, enrich_df, cfg)
    gene_df.to_csv(os.path.join(tdir, "implicated_genes.csv"))
    n_enriched = int(enrich_df["enriched"].sum()) if not enrich_df.empty else 0
    print(f"      {annot['mapped'].sum()}/{len(annot)} metabolites mapped; "
          f"{n_enriched} pathways enriched; {len(gene_df)} genes implicated")

    print("[6/6] Figures + report")
    figs = {}
    try:
        figs["qc_cv"] = plotting.qc_cv_histogram(proc.feature_qc, cfg.qc_cv_threshold, fdir)
        figs["boxplot"] = plotting.sample_boxplot(proc.normalized, ds.sample_groups, cfg.transform, fdir)
        figs["volcano"] = plotting.volcano(stats_df, cfg, fdir)
        figs["heatmap"] = plotting.heatmap_top(proc.transformed, stats_df, ds.sample_groups, fdir)
        figs["enrichment"] = plotting.enrichment_barplot(enrich_df, fdir)
        figs["roc"] = plotting.roc_top(proc.normalized, stats_df, ds, fdir)
        if mv is not None:
            figs["pca"] = plotting.pca_plot(mv.pca_scores, mv.pca_explained, mv.labels, fdir)
            figs["plsda"] = plotting.pls_plot(mv.pls_scores, ds.sample_groups, mv.pls_q2, mv.pls_r2, fdir)
    except Exception as e:                        # pragma: no cover
        print(f"      figure generation issue: {e}")

    report = build_report(stats_df, enrich_df, gene_df, mv, proc, ds, stats_summary)
    with open(os.path.join(rdir, "biological_interpretation.md"), "w") as fh:
        fh.write(report)

    summary = {
        "config": cfg.input_path,
        "n_features_in": int(ds.data.shape[0]),
        "n_features_analyzed": int(stats_df.shape[0]),
        "preprocessing_steps": proc.steps,
        "statistics": stats_summary,
        "multivariate": None if mv is None else {
            "pls_r2": mv.pls_r2, "pls_q2": mv.pls_q2,
            "pca_explained_top2": mv.pca_explained[:2].tolist(),
        },
        "n_pathways_enriched": n_enriched,
        "n_genes_implicated": int(len(gene_df)),
        "figures": {k: v for k, v in figs.items() if v},
    }
    with open(os.path.join(out, "run_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nDone. Results in: {out}")
    print(f"  - report/biological_interpretation.md")
    print(f"  - tables/  ({len(os.listdir(tdir))} CSVs)")
    print(f"  - figures/ ({len([f for f in os.listdir(fdir)])} PNGs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
