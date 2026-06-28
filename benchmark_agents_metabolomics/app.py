#!/usr/bin/env python3
"""
app.py — Streamlit web dashboard for the AI Agent Benchmark pipeline.

Run with:
    streamlit run app.py

Or headlessly (just run the pipeline):
    python app.py --input_dir ./agent_outputs_metabolomics --output_dir ./benchmark_results_metabolomics
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("benchmark.app")


def _is_streamlit() -> bool:
    try:
        import streamlit as st  # noqa: F401
        return "STREAMLIT" in os.environ or "streamlit" in sys.argv[0]
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def run_streamlit_app() -> None:
    import streamlit as st
    import pandas as pd

    st.set_page_config(
        page_title="AI Agent Benchmark",
        page_icon="🧬",
        layout="wide",
    )

    st.title("🧬 AI Agent Benchmark — Untargeted Metabolomics")
    st.markdown(
        "Compare how **ChatGPT**, **Biomni**, and **K-Dense** process the same "
        "untargeted metabolomics dataset across 8 independent runs."
    )

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        input_dir  = st.text_input("Input directory",  value="./agent_outputs_metabolomics")
        output_dir = st.text_input("Output directory", value="./benchmark_results_metabolomics")
        n_mc       = st.number_input("MC iterations",  value=1000, min_value=100, max_value=100_000)
        emb_model  = st.text_input("Embedding model",
                                   value="sentence-transformers/all-MiniLM-L6-v2")
        no_emb     = st.checkbox("No embeddings (TF-IDF fallback)", value=False)
        scores_file = st.text_input("Manual scores CSV (optional)", value="")
        run_btn    = st.button("▶ Run Pipeline", type="primary")

    if run_btn:
        from main import run_pipeline, parse_args
        argv = [
            "--input_dir",  input_dir,
            "--output_dir", output_dir,
            "--n_mc",       str(n_mc),
            "--embedding_model", emb_model,
        ]
        if no_emb:
            argv.append("--no_embeddings")
        if scores_file:
            argv += ["--scores", scores_file]

        with st.spinner("Running pipeline…"):
            logging.basicConfig(level=logging.INFO)
            try:
                args = parse_args(argv)
                run_pipeline(args)
                st.success("Pipeline complete!")
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                logger.exception("Pipeline error")

    # Display results if output dir exists
    out = Path(output_dir)
    tables = out / "tables"
    figures = out / "figures"

    if tables.exists():
        st.header("Results")
        tab_labels = ["Inventory", "Run Features", "Similarity",
                      "Scores", "Ranking", "Validation"]
        tabs = st.tabs(tab_labels)

        def _load(name: str) -> pd.DataFrame:
            p = tables / name
            if p.exists():
                return pd.read_csv(p)
            return pd.DataFrame()

        with tabs[0]:
            st.subheader("File Inventory")
            st.dataframe(_load("file_inventory.csv"), use_container_width=True)

        with tabs[1]:
            st.subheader("Run-Level Features")
            st.dataframe(_load("run_level_features.csv"), use_container_width=True)

        with tabs[2]:
            st.subheader("Pairwise Run Similarity")
            st.dataframe(_load("pairwise_run_similarity.csv"), use_container_width=True)
            st.subheader("Within-Agent Reproducibility")
            st.dataframe(_load("within_agent_reproducibility.csv"), use_container_width=True)
            st.subheader("Between-Agent Similarity")
            st.dataframe(_load("between_agent_similarity.csv"), use_container_width=True)

        with tabs[3]:
            st.subheader("Agent Scores (per run)")
            st.dataframe(_load("agent_scores.csv"), use_container_width=True)
            st.subheader("Ranking Summary")
            st.dataframe(_load("ranking_summary.csv"), use_container_width=True)

        with tabs[4]:
            st.subheader("Ranking Uncertainty")
            st.dataframe(_load("ranking_uncertainty.csv"), use_container_width=True)
            st.subheader("Pairwise Win Probabilities")
            st.dataframe(_load("pairwise_win_probabilities.csv"), use_container_width=True)

        with tabs[5]:
            st.subheader("Validation Summary")
            st.dataframe(_load("validation_summary.csv"), use_container_width=True)

    # Show figures
    if figures.exists():
        st.header("Figures")
        fig_cols = st.columns(2)
        pngs = sorted(figures.glob("*.png"))
        for i, fp in enumerate(pngs):
            fig_cols[i % 2].image(str(fp), caption=fp.stem, use_container_width=True)

        htmls = sorted(figures.glob("*.html"))
        for hp in htmls:
            st.subheader(hp.stem.replace("_", " ").title())
            st.markdown(f"[Open interactive plot]({hp})", unsafe_allow_html=True)

    # Link to report
    report_path = out / "report" / "benchmark_report.html"
    if report_path.exists():
        st.header("Report")
        with open(report_path, encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)


# ---------------------------------------------------------------------------
# CLI fallback
# ---------------------------------------------------------------------------

def run_cli() -> None:
    """Run pipeline from CLI without Streamlit."""
    from main import main as pipeline_main
    pipeline_main()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Check if launched via Streamlit
    try:
        import streamlit as _st
        if any("streamlit" in arg for arg in sys.argv):
            run_streamlit_app()
        else:
            run_cli()
    except ImportError:
        run_cli()
