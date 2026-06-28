"""
visualization.py — generate all figures.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEED = 42

# Colour palette per agent (consistent across all plots)
AGENT_COLOURS = {
    "ChatGPT": "#4e79a7",
    "Biomni":  "#f28e2b",
    "KDense":  "#59a14f",
    "Finch":   "#e15759",
}
DEFAULT_COLOUR = "#8c564b"
_EXTRA_COLOURS = ["#76b7b2", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]

MARKER_MAP = {
    "ChatGPT": "o",
    "Biomni":  "s",
    "KDense":  "^",
    "Finch":   "D",
}


def _agent_colour(agent: str) -> str:
    if agent in AGENT_COLOURS:
        return AGENT_COLOURS[agent]
    return _EXTRA_COLOURS[hash(agent) % len(_EXTRA_COLOURS)]


def _agent_marker(agent: str) -> str:
    if agent in MARKER_MAP:
        return MARKER_MAP[agent]
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]
    return markers[hash(agent) % len(markers)]


# ---------------------------------------------------------------------------
# 2-D scatter helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Confidence ellipse helper (reused across all 2D scatter plots)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SVG companion helper — save figure alongside every PNG
# ---------------------------------------------------------------------------

def _save_svg(fig, png_path: Path) -> None:
    """Save *fig* as an SVG file with the same stem as *png_path*."""
    svg_path = png_path.with_suffix(".svg")
    try:
        fig.savefig(svg_path, format="svg")
        logger.info("Saved SVG  %s", svg_path)
    except Exception as exc:
        logger.warning("Could not save SVG for %s: %s", svg_path, exc)


def _draw_confidence_ellipse(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    colour: str,
    n_std: float = 2.0,
    alpha: float = 0.15,
    label: Optional[str] = None,
) -> None:
    """
    Draw a covariance-based confidence ellipse for 2-D point cloud (x, y).

    n_std=2 corresponds approximately to a 95% confidence region for
    bivariate normal data.  With only 8 points per agent the ellipse is
    descriptive, not a formal confidence region.
    """
    from matplotlib.patches import Ellipse as _Ell
    import matplotlib.transforms as _transforms

    if len(x) < 3:
        return
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return
    # Eigen-decomposition
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h  = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ell = _Ell(
        xy=(np.mean(x), np.mean(y)),
        width=w, height=h, angle=angle,
        facecolor=colour, alpha=alpha,
        edgecolor=colour, linewidth=1.2,
        label=label,
    )
    ax.add_patch(ell)


def _scatter2d(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    outlier_df: Optional[pd.DataFrame] = None,
    show_ellipses: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        for agent in sorted(df["agent"].unique()):
            sub = df[df["agent"] == agent]
            x = sub[x_col].to_numpy(dtype=float)
            y = sub[y_col].to_numpy(dtype=float)

            # 95 % confidence ellipse
            if show_ellipses and len(x) >= 3:
                _draw_confidence_ellipse(ax, x, y, _agent_colour(agent), n_std=2.0)

            ax.scatter(
                x, y,
                label=agent,
                color=_agent_colour(agent),
                marker=_agent_marker(agent),
                s=80, edgecolors="white", linewidths=0.5, alpha=0.9,
            )

            # Annotate outliers if provided
            if outlier_df is not None and not outlier_df.empty and "outlier" in outlier_df.columns:
                out_sub = outlier_df[(outlier_df["agent"] == agent) & (outlier_df["outlier"])]
                if "run" in sub.columns and not out_sub.empty:
                    for _, orow in out_sub.iterrows():
                        run_mask = sub["run"] == orow["run"]
                        if run_mask.any():
                            ox = float(sub.loc[run_mask, x_col].iloc[0])
                            oy = float(sub.loc[run_mask, y_col].iloc[0])
                            ax.scatter([ox], [oy], s=160, facecolors="none",
                                       edgecolors="red", linewidths=1.8, zorder=6)
                            ax.annotate(f"R{int(orow['run'])}*",
                                        (ox, oy), fontsize=7, color="red",
                                        xytext=(4, 4), textcoords="offset points")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out_path)
        _save_svg(fig, out_path)
    except Exception as exc:
        logger.warning("Could not create %s: %s", out_path, exc)


# ---------------------------------------------------------------------------
# PCA 2D
# ---------------------------------------------------------------------------

def plot_pca_2d(
    pca_df: pd.DataFrame, out_dir: Path,
    outlier_df: Optional[pd.DataFrame] = None,
) -> None:
    out = out_dir / "pca_2d.png"
    if "PC1" not in pca_df.columns or "PC2" not in pca_df.columns:
        logger.warning("PCA 2D: missing PC1/PC2 columns.")
        return
    _scatter2d(pca_df, "PC1", "PC2",
               "PCA 2D — Run Embeddings (ellipses = 95% CI, ★ = outlier)",
               "PC1", "PC2", out, outlier_df=outlier_df)


# ---------------------------------------------------------------------------
# PCA 3D (interactive Plotly HTML)
# ---------------------------------------------------------------------------

def plot_pca_3d(
    pca_df: pd.DataFrame,
    out_dir: Path,
    var_ratio: Optional[List[float]] = None,
) -> None:
    """Interactive 3D PCA scatter with centroids, run labels, and explained variance."""
    out = out_dir / "pca_3d.html"
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        pc3 = "PC3" if "PC3" in pca_df.columns else "PC2"

        # Axis labels with explained variance
        def _ax(prefix: str, idx: int) -> str:
            if var_ratio is not None and idx < len(var_ratio):
                return f"{prefix} ({var_ratio[idx] * 100:.1f}%)"
            return prefix

        agents = sorted(pca_df["agent"].unique())

        # Real run points (large)
        for agent in agents:
            sub = pca_df[pca_df["agent"] == agent]
            fig.add_trace(go.Scatter3d(
                x=sub["PC1"], y=sub["PC2"], z=sub[pc3],
                mode="markers+text",
                marker=dict(size=9, color=_agent_colour(agent), opacity=0.90,
                            line=dict(width=1, color="white")),
                text=[f"{agent} R{r}" for r in sub["run"]],
                textfont=dict(size=9),
                name=agent,
            ))

        # Agent centroids (diamonds)
        for agent in agents:
            sub = pca_df[pca_df["agent"] == agent]
            cx = sub["PC1"].mean(); cy = sub["PC2"].mean()
            cz = sub[pc3].mean()
            fig.add_trace(go.Scatter3d(
                x=[cx], y=[cy], z=[cz],
                mode="markers+text",
                marker=dict(size=14, color=_agent_colour(agent),
                            symbol="diamond", opacity=1.0,
                            line=dict(width=2, color="black")),
                text=[f"{agent} centroid"],
                name=f"{agent} centroid",
                showlegend=True,
            ))

        n_runs = len(pca_df)
        fig.update_layout(
            title=("PCA 3D — Run Embeddings<br>"
                   f"<sup>{n_runs} real runs · PCA used for visualisation only</sup>"),
            scene=dict(
                xaxis_title=_ax("PC1", 0),
                yaxis_title=_ax("PC2", 1),
                zaxis_title=_ax(pc3, 2),
            ),
            legend_title="Agent",
        )
        fig.write_html(str(out))
        logger.info("Saved %s", out)
    except Exception as exc:
        logger.warning("Could not create PCA 3D: %s", exc)


# ---------------------------------------------------------------------------
# MDS 2D
# ---------------------------------------------------------------------------

def plot_mds_2d(
    mds_df: pd.DataFrame, out_dir: Path,
    outlier_df: Optional[pd.DataFrame] = None,
) -> None:
    if mds_df is None or mds_df.empty:
        return
    out = out_dir / "mds_2d.png"
    _scatter2d(mds_df, "MDS1", "MDS2",
               "MDS 2D — Run Embeddings (ellipses = 95% CI, ★ = outlier)",
               "MDS1", "MDS2", out, outlier_df=outlier_df)


# ---------------------------------------------------------------------------
# t-SNE 2D
# ---------------------------------------------------------------------------

def plot_tsne_2d(
    tsne_df: Optional[pd.DataFrame], out_dir: Path,
    outlier_df: Optional[pd.DataFrame] = None,
) -> None:
    if tsne_df is None or tsne_df.empty:
        logger.info("t-SNE plot skipped (no data).")
        return
    out = out_dir / "tsne_2d.png"
    _scatter2d(tsne_df, "TSNE1", "TSNE2",
               "t-SNE 2D — Run Embeddings (ellipses = 95% CI, ★ = outlier)",
               "TSNE1", "TSNE2", out, outlier_df=outlier_df)


# ---------------------------------------------------------------------------
# UMAP 2D
# ---------------------------------------------------------------------------

def plot_umap_2d(
    umap_df: Optional[pd.DataFrame], out_dir: Path,
    outlier_df: Optional[pd.DataFrame] = None,
) -> None:
    if umap_df is None or umap_df.empty:
        logger.info("UMAP plot skipped (no data).")
        return
    out = out_dir / "umap_2d.png"
    _scatter2d(umap_df, "UMAP1", "UMAP2",
               "UMAP 2D — Run Embeddings (ellipses = 95% CI, ★ = outlier)",
               "UMAP1", "UMAP2", out, outlier_df=outlier_df)


# ---------------------------------------------------------------------------
# Run similarity heatmap
# ---------------------------------------------------------------------------

def plot_run_similarity_heatmap(pairwise: pd.DataFrame, out_dir: Path) -> None:
    if pairwise is None or pairwise.empty:
        return
    out = out_dir / "run_similarity_heatmap.png"
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Build symmetric matrix
        pairs_a = list(zip(pairwise["agent_a"], pairwise["run_a"]))
        pairs_b = list(zip(pairwise["agent_b"], pairwise["run_b"]))
        all_keys = sorted(set(pairs_a + pairs_b), key=lambda x: (x[0], x[1]))
        labels = [f"{a}\nR{r}" for a, r in all_keys]
        n = len(all_keys)
        idx = {k: i for i, k in enumerate(all_keys)}

        mat = np.full((n, n), np.nan)
        np.fill_diagonal(mat, 1.0)

        for _, row in pairwise.iterrows():
            i = idx.get((row["agent_a"], row["run_a"]))
            j = idx.get((row["agent_b"], row["run_b"]))
            if i is not None and j is not None:
                mat[i, j] = row["cosine_sim"]
                mat[j, i] = row["cosine_sim"]

        fig, ax = plt.subplots(figsize=(max(8, n * 0.45), max(6, n * 0.4)))
        sns.heatmap(
            mat, xticklabels=labels, yticklabels=labels,
            cmap="RdYlGn", vmin=-1, vmax=1, center=0,
            annot=(n <= 24), fmt=".2f", ax=ax,
            linewidths=0.3, linecolor="white",
        )
        ax.set_title("Pairwise Run Cosine Similarity")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create similarity heatmap: %s", exc)


# ---------------------------------------------------------------------------
# Within-agent similarity boxplot
# ---------------------------------------------------------------------------

def plot_within_agent_boxplot(pairwise: pd.DataFrame, out_dir: Path) -> None:
    if pairwise is None or pairwise.empty:
        return
    out = out_dir / "within_agent_similarity_boxplot.png"
    try:
        import matplotlib.pyplot as plt

        data_by_agent = {}
        agents = sorted(set(pairwise["agent_a"]) | set(pairwise["agent_b"]))
        for agent in agents:
            mask = (pairwise["agent_a"] == agent) & (pairwise["agent_b"] == agent)
            data_by_agent[agent] = pairwise[mask]["cosine_sim"].dropna().values

        if not any(len(v) > 0 for v in data_by_agent.values()):
            return

        fig, ax = plt.subplots(figsize=(6, 4))
        positions = range(1, len(agents) + 1)
        data_list = [data_by_agent[a] for a in agents]
        bps = ax.boxplot(
            [d if len(d) > 0 else [float("nan")] for d in data_list],
            positions=list(positions),
            patch_artist=True,
            widths=0.5,
        )
        for patch, agent in zip(bps["boxes"], agents):
            patch.set_facecolor(_agent_colour(agent))
            patch.set_alpha(0.7)

        # Overlay mean + 95% bootstrap CI + n pairs annotation
        rng = np.random.default_rng(42)
        for pos, agent in zip(positions, agents):
            vals = data_by_agent[agent]
            if len(vals) == 0:
                continue
            mean_v = float(np.mean(vals))
            boot = np.array([np.mean(rng.choice(vals, len(vals), replace=True))
                             for _ in range(2000)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            ax.scatter([pos], [mean_v], color="black", marker="D", s=35, zorder=5)
            ax.errorbar([pos], [mean_v], yerr=[[mean_v - lo], [hi - mean_v]],
                        fmt="none", ecolor="black", capsize=5, zorder=4)
            ax.annotate(f"n={len(vals)}", (pos, ax.get_ylim()[0]),
                        ha="center", va="bottom", fontsize=7, color="dimgray")

        ax.set_xticks(list(positions))
        ax.set_xticklabels(agents, fontsize=9)
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Within-Agent Reproducibility (mean ◆ ± 95% bootstrap CI)")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create within-agent boxplot: %s", exc)


# ---------------------------------------------------------------------------
# Monte Carlo 3D satellites
# ---------------------------------------------------------------------------

def plot_mc_satellites_3d(
    run_embs: Dict[Tuple[str, int], np.ndarray],
    boot_by_agent: Dict[str, np.ndarray],
    gauss_by_agent: Dict[str, np.ndarray],
    out_dir: Path,
    max_satellites: int = 2000,
) -> None:
    out = out_dir / "monte_carlo_satellites_3d.html"
    try:
        import plotly.graph_objects as go
        from sklearn.decomposition import PCA as _PCA

        # Collect all vecs to fit a common PCA
        all_vecs = []
        real_keys: List[Tuple[str, int]] = []
        for k, v in sorted(run_embs.items()):
            all_vecs.append(v)
            real_keys.append(k)

        for agent, arr in boot_by_agent.items():
            sample_idx = np.linspace(0, len(arr) - 1, min(max_satellites, len(arr)), dtype=int)
            all_vecs.extend(arr[sample_idx])

        mat = np.stack(all_vecs, axis=0).astype(np.float64)
        n_comp = min(3, mat.shape[1], mat.shape[0] - 1)
        if n_comp < 1:
            n_comp = 1
        pca = _PCA(n_components=n_comp, random_state=SEED)
        mat_red = pca.fit_transform(mat)

        n_real = len(real_keys)
        real_red = mat_red[:n_real]

        fig = go.Figure()

        # Real run points (large)
        for agent in sorted(set(k[0] for k in real_keys)):
            idxs = [i for i, k in enumerate(real_keys) if k[0] == agent]
            sub = real_red[idxs]
            z_col = sub[:, 2] if n_comp >= 3 else np.zeros(len(idxs))
            fig.add_trace(go.Scatter3d(
                x=sub[:, 0], y=sub[:, 1], z=z_col,
                mode="markers+text",
                marker=dict(size=10, color=_agent_colour(agent),
                            symbol="circle", opacity=1.0,
                            line=dict(width=1, color="black")),
                text=[f"{agent} R{real_keys[i][1]}" for i in idxs],
                name=f"{agent} (real)",
            ))

        # Satellite points (small, transparent)
        offset = n_real
        for agent in sorted(boot_by_agent.keys()):
            arr = boot_by_agent[agent]
            sample_idx = np.linspace(0, len(arr) - 1, min(max_satellites, len(arr)), dtype=int)
            n_samp = len(sample_idx)
            sub = mat_red[offset:offset + n_samp]
            z_col = sub[:, 2] if n_comp >= 3 else np.zeros(n_samp)
            fig.add_trace(go.Scatter3d(
                x=sub[:, 0], y=sub[:, 1], z=z_col,
                mode="markers",
                marker=dict(size=2, color=_agent_colour(agent),
                            opacity=0.08),
                name=f"{agent} (MC bootstrap)",
                showlegend=True,
            ))
            offset += n_samp

        # Agent centroids (real-run means) as large diamonds
        for agent in sorted(set(k[0] for k in real_keys)):
            idxs = [i for i, k in enumerate(real_keys) if k[0] == agent]
            sub = real_red[idxs]
            cz = sub[:, 2].mean() if n_comp >= 3 else 0.0
            fig.add_trace(go.Scatter3d(
                x=[sub[:, 0].mean()], y=[sub[:, 1].mean()], z=[cz],
                mode="markers",
                marker=dict(size=14, color=_agent_colour(agent),
                            symbol="diamond", opacity=1.0,
                            line=dict(width=2, color="black")),
                name=f"{agent} (centroid)",
            ))

        fig.update_layout(
            title=("Monte Carlo Satellites — Run Embeddings (3D PCA)<br>"
                   "<sup>Synthetic points are uncertainty probes, NOT real "
                   "observations</sup>"),
            scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
            legend_title="Agent",
        )
        fig.write_html(str(out))
        logger.info("Saved %s", out)
    except Exception as exc:
        logger.warning("Could not create MC satellites 3D: %s", exc)


# ---------------------------------------------------------------------------
# Score distribution boxplot
# ---------------------------------------------------------------------------

def plot_score_distribution(run_scores: pd.DataFrame, out_dir: Path,
                            agent_summary: Optional[pd.DataFrame] = None) -> None:
    """Boxplot of AgentScore per agent, overlaid with mean and 95% bootstrap CI."""
    if run_scores is None or run_scores.empty:
        return
    out = out_dir / "score_distribution_boxplot.png"
    try:
        import matplotlib.pyplot as plt

        agents = sorted(run_scores["agent"].unique())
        data = [run_scores[run_scores["agent"] == a]["AgentScore"].values for a in agents]

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        bps = ax.boxplot(
            [d if len(d) > 0 else [float("nan")] for d in data],
            patch_artist=True, widths=0.5,
        )
        for patch, agent in zip(bps["boxes"], agents):
            patch.set_facecolor(_agent_colour(agent))
            patch.set_alpha(0.6)

        # Overlay mean + 95% bootstrap CI
        ci_label_added = False
        for i, agent in enumerate(agents, start=1):
            vals = run_scores[run_scores["agent"] == agent]["AgentScore"].values
            mean_v = np.mean(vals) if len(vals) else np.nan
            ax.scatter([i], [mean_v], color="black", marker="D", s=40, zorder=5,
                       label="mean" if not ci_label_added else None)
            if agent_summary is not None and not agent_summary.empty:
                arow = agent_summary[agent_summary["agent"] == agent]
                if not arow.empty and {"ci95_lo", "ci95_hi"}.issubset(arow.columns):
                    lo = float(arow["ci95_lo"].iloc[0])
                    hi = float(arow["ci95_hi"].iloc[0])
                    ax.errorbar([i], [mean_v], yerr=[[mean_v - lo], [hi - mean_v]],
                                fmt="none", ecolor="black", capsize=5, zorder=4,
                                label="95% bootstrap CI" if not ci_label_added else None)
                    ci_label_added = True

        ax.set_xticks(range(1, len(agents) + 1))
        ax.set_xticklabels(agents, fontsize=9)
        ax.set_ylabel("AgentScore")
        ax.set_title("AgentScore Distribution (mean ◆ ± 95% bootstrap CI)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create score distribution: %s", exc)


# ---------------------------------------------------------------------------
# Ranking probability barplot
# ---------------------------------------------------------------------------

def plot_ranking_probability(ranking_df: pd.DataFrame, out_dir: Path) -> None:
    if ranking_df is None or ranking_df.empty:
        return
    out = out_dir / "ranking_probability_barplot.png"
    try:
        import matplotlib.pyplot as plt

        agents = sorted(ranking_df["agent"].unique())
        n_agents = len(agents)
        # Show all ranks (up to n_agents), not just top 3
        ranks  = sorted(ranking_df["rank"].unique())[:n_agents]
        x = np.arange(n_agents)
        width = 0.8 / max(len(ranks), 1)

        fig, ax = plt.subplots(figsize=(max(7.5, n_agents * 1.8), 4.5))
        rank_colours = ["#2196F3", "#FF9800", "#4CAF50", "#e15759",
                        "#9c27b0", "#00bcd4", "#ff5722", "#607d8b"]
        for i, rank in enumerate(ranks):
            probs = [
                ranking_df[(ranking_df["agent"] == a) & (ranking_df["rank"] == rank)]["probability"].values[0]
                if len(ranking_df[(ranking_df["agent"] == a) & (ranking_df["rank"] == rank)]) > 0
                else 0.0
                for a in agents
            ]
            colour = rank_colours[i % len(rank_colours)]
            ax.bar(x + i * width, probs, width, label=f"Rank {rank}",
                   color=colour, alpha=0.85)

        ax.set_xticks(x + width * (len(ranks) - 1) / 2)
        ax.set_xticklabels(agents, fontsize=9)
        ax.set_ylabel("Rank probability")
        ax.set_ylim(0, 1.05)
        ax.set_title("Ranking Probability by Agent (Bootstrap resampling)\n"
                     "All ranks shown")
        ax.legend(title="Rank")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create ranking probability plot: %s", exc)


def plot_outlier_runs(
    outlier_df: pd.DataFrame,
    out_dir: Path,
    score_col: str = "AgentScore",
) -> None:
    """
    Strip-chart of per-run scores with outlier runs highlighted in red.
    Includes IQR fence lines and Z-score annotations.
    """
    out = out_dir / "outlier_runs.png"
    if outlier_df is None or outlier_df.empty or score_col not in outlier_df.columns:
        logger.warning("outlier_runs plot skipped: no data.")
        return
    try:
        import matplotlib.pyplot as plt

        agents = sorted(outlier_df["agent"].unique())
        fig, ax = plt.subplots(figsize=(max(6, len(agents) * 1.8), 5))

        for i, agent in enumerate(agents):
            sub = outlier_df[outlier_df["agent"] == agent]
            for _, row in sub.iterrows():
                v    = float(row[score_col])
                is_out = bool(row.get("outlier", False))
                colour = "red" if is_out else _agent_colour(agent)
                marker = "X" if is_out else _agent_marker(agent)
                size   = 90  if is_out else 55
                ax.scatter([i], [v], color=colour, marker=marker, s=size,
                           zorder=5, edgecolors="black" if is_out else "white",
                           linewidths=0.8)
                if is_out and "run" in row:
                    reason = row.get("outlier_reason", "")
                    ax.annotate(
                        f"R{int(row['run'])} ({reason})",
                        (i, v), fontsize=7, color="red",
                        xytext=(6, 0), textcoords="offset points",
                        va="center",
                    )

            # IQR fence lines
            if "iqr_lower_fence" in sub.columns:
                lo = sub["iqr_lower_fence"].iloc[0]
                hi = sub["iqr_upper_fence"].iloc[0]
                if np.isfinite(lo) and np.isfinite(hi):
                    ax.hlines([lo, hi], i - 0.3, i + 0.3,
                               colors="gray", linestyles="--", linewidths=1, alpha=0.7)

        ax.set_xticks(range(len(agents)))
        ax.set_xticklabels(agents, fontsize=9)
        ax.set_ylabel(score_col)
        ax.set_title("Outlier Detection per Agent\n"
                     "(red X = outlier · dashed = IQR fences · IQR k=1.5, Z>2.5)")
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create outlier_runs plot: %s", exc)


# ===========================================================================
# SCORING-AUDIT FIGURES
# ===========================================================================

def _sorted_runs(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["agent", "run"]).reset_index(drop=True)


def plot_score_component_heatmap(proxy_components: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap: runs (rows) × proxy components (cols), grouped by agent."""
    out = out_dir / "score_component_heatmap.png"
    if proxy_components is None or proxy_components.empty:
        logger.warning("score_component_heatmap skipped: no proxy components.")
        return
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = _sorted_runs(proxy_components)
        comp_cols = [c for c in df.columns
                     if c.startswith(("P_", "R_", "D_"))
                     and not c.endswith("_proxy")]
        if not comp_cols:
            logger.warning("score_component_heatmap skipped: no component columns.")
            return
        labels = [f"{r.agent} R{r.run}" for r in df.itertuples()]
        mat = df[comp_cols].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(max(10, len(comp_cols) * 0.45),
                                        max(6, len(labels) * 0.32)))
        sns.heatmap(mat, xticklabels=comp_cols, yticklabels=labels,
                    cmap="YlGnBu", vmin=0, vmax=1, cbar_kws={"label": "component present"},
                    linewidths=0.3, linecolor="white", ax=ax)
        ax.set_title("Proxy Score Components (runs × components)")
        ax.set_xlabel("Component")
        ax.set_ylabel("Run")
        plt.xticks(rotation=90, fontsize=7)
        plt.yticks(fontsize=7)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create score_component_heatmap: %s", exc)


def plot_domain_score_component_heatmap(domain_scores: pd.DataFrame, out_dir: Path) -> None:
    out = out_dir / "domain_score_component_heatmap.png"
    if domain_scores is None or domain_scores.empty:
        logger.warning("domain_score_component_heatmap skipped: no domain scores.")
        return
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = _sorted_runs(domain_scores)
        comp_cols = [c for c in df.columns if c.startswith("dm_")]
        if not comp_cols:
            return
        labels = [f"{r.agent} R{r.run}" for r in df.itertuples()]
        mat = df[comp_cols].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(max(10, len(comp_cols) * 0.5),
                                        max(6, len(labels) * 0.32)))
        sns.heatmap(mat, xticklabels=comp_cols, yticklabels=labels,
                    cmap="viridis", vmin=0, vmax=2,
                    cbar_kws={"label": "0 absent · 1 partial · 2 clear"},
                    linewidths=0.3, linecolor="white", ax=ax)
        ax.set_title("Domain-Aware Proteomics Score Components (runs × criteria)")
        ax.set_xlabel("Criterion")
        ax.set_ylabel("Run")
        plt.xticks(rotation=90, fontsize=7)
        plt.yticks(fontsize=7)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create domain_score_component_heatmap: %s", exc)


def plot_score_vs_metric(
    run_scores: pd.DataFrame,
    volume_metrics: pd.DataFrame,
    metric: str,
    out_name: str,
    out_dir: Path,
    corr_df: Optional[pd.DataFrame] = None,
    score_col: str = "AgentScore",
) -> None:
    """Scatter of AgentScore vs a volume metric, with Spearman rho (+CI) caption."""
    out = out_dir / out_name
    if run_scores is None or run_scores.empty or volume_metrics is None or volume_metrics.empty:
        logger.warning("%s skipped: missing data.", out_name)
        return
    if metric not in volume_metrics.columns:
        logger.warning("%s skipped: metric %s absent.", out_name, metric)
        return
    try:
        import matplotlib.pyplot as plt

        merged = run_scores.merge(volume_metrics, on=["agent", "run"], how="inner")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for agent in sorted(merged["agent"].unique()):
            sub = merged[merged["agent"] == agent]
            ax.scatter(sub[metric], sub[score_col], label=agent,
                       color=_agent_colour(agent), marker=_agent_marker(agent),
                       s=60, alpha=0.85, edgecolors="white", linewidths=0.5)

        # Descriptive smoother (linear) — clearly labelled
        x = merged[metric].to_numpy(dtype=float)
        y = merged[score_col].to_numpy(dtype=float)
        if len(x) >= 2 and np.std(x) > 0:
            coef = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, np.polyval(coef, xs), "--", color="gray", alpha=0.7,
                    label="descriptive linear fit")

        caption = ""
        if corr_df is not None and not corr_df.empty:
            crow = corr_df[corr_df["metric"] == metric]
            if not crow.empty:
                rho = crow["spearman_rho"].iloc[0]
                lo = crow["rho_ci95_lo"].iloc[0]
                hi = crow["rho_ci95_hi"].iloc[0]
                caption = f"Spearman ρ={rho:.2f} [95% CI {lo:.2f}, {hi:.2f}]"

        ax.set_xlabel(metric.replace("_", " "))
        ax.set_ylabel(score_col)
        ax.set_title(f"AgentScore vs {metric.replace('_', ' ')}")
        if caption:
            ax.text(0.02, 0.98, caption, transform=ax.transAxes, fontsize=8,
                    va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        ax.legend(loc="lower right", fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create %s: %s", out_name, exc)


def plot_output_volume_by_agent_boxplot(volume_metrics: pd.DataFrame, out_dir: Path) -> None:
    out = out_dir / "output_volume_by_agent_boxplot.png"
    if volume_metrics is None or volume_metrics.empty:
        logger.warning("output_volume_by_agent_boxplot skipped: no data.")
        return
    try:
        import matplotlib.pyplot as plt

        metrics = ["file_count", "text_length", "table_count",
                   "figure_count", "chunk_count", "artifact_diversity"]
        metrics = [m for m in metrics if m in volume_metrics.columns]
        agents = sorted(volume_metrics["agent"].unique())
        n = len(metrics)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
        axes = np.atleast_1d(axes).ravel()

        for ax, metric in zip(axes, metrics):
            data = [volume_metrics[volume_metrics["agent"] == a][metric].values for a in agents]
            bps = ax.boxplot([d if len(d) else [np.nan] for d in data],
                             patch_artist=True, widths=0.5)
            for patch, agent in zip(bps["boxes"], agents):
                patch.set_facecolor(_agent_colour(agent))
                patch.set_alpha(0.7)
            ax.set_xticks(range(1, len(agents) + 1))
            ax.set_xticklabels(agents, fontsize=8, rotation=15)
            ax.set_title(metric.replace("_", " "), fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.3)

        for ax in axes[len(metrics):]:
            ax.set_visible(False)

        fig.suptitle("Output Volume Metrics by Agent")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create output_volume_by_agent_boxplot: %s", exc)


def plot_output_volume_vs_score_panel(
    run_scores: pd.DataFrame, volume_metrics: pd.DataFrame, out_dir: Path,
    corr_df: Optional[pd.DataFrame] = None,
) -> None:
    out = out_dir / "output_volume_vs_score_panel.png"
    if run_scores is None or run_scores.empty or volume_metrics is None or volume_metrics.empty:
        logger.warning("output_volume_vs_score_panel skipped: missing data.")
        return
    try:
        import matplotlib.pyplot as plt

        merged = run_scores.merge(volume_metrics, on=["agent", "run"], how="inner")
        metrics = ["file_count", "text_length", "table_count",
                   "figure_count", "chunk_count", "total_file_size"]
        metrics = [m for m in metrics if m in merged.columns]
        ncols = 3
        nrows = int(np.ceil(len(metrics) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
        axes = np.atleast_1d(axes).ravel()

        for ax, metric in zip(axes, metrics):
            for agent in sorted(merged["agent"].unique()):
                sub = merged[merged["agent"] == agent]
                ax.scatter(sub[metric], sub["AgentScore"], label=agent,
                           color=_agent_colour(agent), marker=_agent_marker(agent),
                           s=35, alpha=0.8)
            title = metric.replace("_", " ")
            if corr_df is not None and not corr_df.empty:
                crow = corr_df[corr_df["metric"] == metric]
                if not crow.empty:
                    title += f"  (ρ={crow['spearman_rho'].iloc[0]:.2f})"
            ax.set_title(title, fontsize=8)
            ax.set_xlabel(metric.replace("_", " "), fontsize=7)
            ax.set_ylabel("AgentScore", fontsize=7)
            ax.grid(True, linestyle="--", alpha=0.3)

        for ax in axes[len(metrics):]:
            ax.set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=8)
        fig.suptitle("AgentScore vs Output Volume (descriptive)")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create output_volume_vs_score_panel: %s", exc)


def plot_proxy_vs_domain_scatter(
    run_scores: pd.DataFrame, domain_scores: pd.DataFrame, out_dir: Path,
    caption: str = "",
) -> None:
    out = out_dir / "proxy_vs_domain_score_scatter.png"
    if run_scores is None or run_scores.empty or domain_scores is None or domain_scores.empty:
        logger.warning("proxy_vs_domain_score_scatter skipped: missing data.")
        return
    try:
        import matplotlib.pyplot as plt

        merged = run_scores.merge(domain_scores[["agent", "run", "DomainScore"]],
                                  on=["agent", "run"], how="inner")
        fig, ax = plt.subplots(figsize=(6, 5.5))
        for agent in sorted(merged["agent"].unique()):
            sub = merged[merged["agent"] == agent]
            x = sub["AgentScore"].to_numpy(dtype=float)
            y = sub["DomainScore"].to_numpy(dtype=float)
            if len(x) >= 3:
                _draw_confidence_ellipse(ax, x, y, _agent_colour(agent))
            ax.scatter(x, y, label=agent,
                       color=_agent_colour(agent), marker=_agent_marker(agent),
                       s=60, alpha=0.85, edgecolors="white", linewidths=0.5)
            for r in sub.itertuples():
                ax.annotate(f"R{r.run}", (r.AgentScore, r.DomainScore),
                            fontsize=6, alpha=0.6)
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "--", color="gray", alpha=0.5, label="y = x")
        ax.set_xlabel("Proxy AgentScore")
        ax.set_ylabel("Domain-aware proteomics score")
        ax.set_title("Proxy AgentScore vs Domain-Aware Score")
        if caption:
            ax.text(0.02, 0.98, caption, transform=ax.transAxes, fontsize=8,
                    va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        ax.legend(loc="lower right", fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create proxy_vs_domain_score_scatter: %s", exc)


def plot_domain_score_radar(domain_scores: pd.DataFrame, out_dir: Path) -> None:
    out = out_dir / "domain_score_radar.png"
    if domain_scores is None or domain_scores.empty:
        logger.warning("domain_score_radar skipped: no domain scores.")
        return
    try:
        import matplotlib.pyplot as plt

        comp_cols = [c for c in domain_scores.columns if c.startswith("dm_")]
        if not comp_cols:
            return
        agents = sorted(domain_scores["agent"].unique())
        means = {a: domain_scores[domain_scores["agent"] == a][comp_cols].mean().values
                 for a in agents}

        labels = [c.replace("dm_", "") for c in comp_cols]
        N = len(labels)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        for agent in agents:
            vals = list(means[agent]) + [means[agent][0]]
            ax.plot(angles, vals, label=agent, color=_agent_colour(agent), linewidth=2)
            ax.fill(angles, vals, color=_agent_colour(agent), alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(0, 2)
        ax.set_title("Domain-Aware Criteria — Mean Grade by Agent (0–2)", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        radar_csv = pd.DataFrame({a: means[a] for a in agents}, index=labels)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create domain_score_radar: %s", exc)


def plot_proxy_vs_manual_scatter(manual_vs_proxy: pd.DataFrame, out_dir: Path) -> None:
    out = out_dir / "proxy_vs_manual_score_scatter.png"
    if manual_vs_proxy is None or manual_vs_proxy.empty:
        logger.info("proxy_vs_manual_score_scatter skipped: no manual scores.")
        return
    if not {"manual_AgentScore", "proxy_AgentScore"}.issubset(manual_vs_proxy.columns):
        return
    try:
        import matplotlib.pyplot as plt

        df = manual_vs_proxy
        fig, ax = plt.subplots(figsize=(6, 5.5))
        for agent in sorted(df["agent"].unique()):
            sub = df[df["agent"] == agent]
            ax.scatter(sub["proxy_AgentScore"], sub["manual_AgentScore"], label=agent,
                       color=_agent_colour(agent), marker=_agent_marker(agent),
                       s=60, alpha=0.85, edgecolors="white", linewidths=0.5)
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "--", color="gray", alpha=0.5, label="y = x")
        ax.set_xlabel("Proxy AgentScore")
        ax.set_ylabel("Manual AgentScore")
        ax.set_title("Manual vs Proxy AgentScore")
        ax.legend(loc="lower right", fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create proxy_vs_manual_score_scatter: %s", exc)


def plot_ranking_sensitivity_heatmap(
    sens_matrix: pd.DataFrame, out_dir: Path,
    ranking_error: Optional[pd.DataFrame] = None,
) -> None:
    out = out_dir / "ranking_sensitivity_heatmap.png"
    if sens_matrix is None or sens_matrix.empty:
        logger.warning("ranking_sensitivity_heatmap skipped: no data.")
        return
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        mat = sens_matrix.copy()
        fig, ax = plt.subplots(figsize=(max(8, mat.shape[1] * 1.2),
                                        max(4, mat.shape[0] * 0.8)))
        sns.heatmap(mat, annot=True, fmt=".0f", cmap="RdYlGn_r",
                    cbar_kws={"label": "rank (1 = best)"}, ax=ax,
                    linewidths=0.5, linecolor="white")
        ax.set_title("Ranking Sensitivity Across Scoring Definitions")
        ax.set_xlabel("Scoring definition")
        ax.set_ylabel("Agent")
        plt.xticks(rotation=30, ha="right", fontsize=8)

        # Annotate expected rank / switch prob if available
        if ranking_error is not None and not ranking_error.empty:
            note = "; ".join(
                f"{r['agent']}: E[rank]={r['expected_rank']:.1f}, "
                f"switch={r['rank_switch_probability']:.2f}"
                for _, r in ranking_error.iterrows()
            )
            fig.text(0.5, -0.02, note, ha="center", fontsize=7, wrap=True)

        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create ranking_sensitivity_heatmap: %s", exc)


def plot_ranking_sensitivity_barplot(
    score_error: pd.DataFrame, out_dir: Path,
) -> None:
    """Grouped bars of mean score by agent for each scoring definition, with 95% CI."""
    out = out_dir / "ranking_sensitivity_barplot.png"
    if score_error is None or score_error.empty:
        logger.warning("ranking_sensitivity_barplot skipped: no data.")
        return
    try:
        import matplotlib.pyplot as plt

        defs = sorted(score_error["definition"].unique())
        agents = sorted(score_error["agent"].unique())
        ncols = 2
        nrows = int(np.ceil(len(defs) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows))
        axes = np.atleast_1d(axes).ravel()

        for ax, defn in zip(axes, defs):
            sub = score_error[score_error["definition"] == defn]
            xs = np.arange(len(agents))
            means, los, his, cols = [], [], [], []
            for a in agents:
                arow = sub[sub["agent"] == a]
                if arow.empty:
                    means.append(0); los.append(0); his.append(0)
                else:
                    m = float(arow["mean"].iloc[0])
                    lo = float(arow["ci95_lo"].iloc[0])
                    hi = float(arow["ci95_hi"].iloc[0])
                    means.append(m); los.append(max(0, m - lo)); his.append(max(0, hi - m))
                cols.append(_agent_colour(a))
            ax.bar(xs, means, yerr=[los, his], capsize=4, color=cols, alpha=0.8)
            ax.set_xticks(xs)
            ax.set_xticklabels(agents, fontsize=8, rotation=15)
            ax.set_title(defn, fontsize=8)
            ax.set_ylabel("mean ± 95% CI", fontsize=7)
            ax.grid(True, linestyle="--", alpha=0.3)

        for ax in axes[len(defs):]:
            ax.set_visible(False)

        fig.suptitle("Mean Score by Agent across Scoring Definitions (95% bootstrap CI)")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create ranking_sensitivity_barplot: %s", exc)


# ===========================================================================
# ERROR / UNCERTAINTY FIGURES
# ===========================================================================

def plot_agent_score_errorbar(bootstrap_error: pd.DataFrame, out_dir: Path) -> None:
    out = out_dir / "agent_score_errorbar.png"
    if bootstrap_error is None or bootstrap_error.empty:
        logger.warning("agent_score_errorbar skipped: no data.")
        return
    try:
        import matplotlib.pyplot as plt

        df = bootstrap_error.sort_values("mean", ascending=False)
        agents = df["agent"].tolist()
        xs = np.arange(len(agents))
        means = df["mean"].to_numpy()
        los = (df["mean"] - df["ci95_lo"]).clip(lower=0).to_numpy()
        his = (df["ci95_hi"] - df["mean"]).clip(lower=0).to_numpy()

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.bar(xs, means, color=[_agent_colour(a) for a in agents], alpha=0.8)
        ax.errorbar(xs, means, yerr=[los, his], fmt="none", ecolor="black",
                    capsize=6, label="95% bootstrap CI")
        ax.set_xticks(xs)
        ax.set_xticklabels(agents, fontsize=9)
        ax.set_ylabel("Mean AgentScore")
        ax.set_title("Agent Mean Score (error bars = 95% bootstrap CI of the mean)")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create agent_score_errorbar: %s", exc)


def plot_agent_score_bootstrap_ci(
    run_scores: pd.DataFrame, bootstrap_error: pd.DataFrame, out_dir: Path,
    n_boot: int = 2000,
) -> None:
    """Violin/strip of bootstrap mean distribution per agent with CI band."""
    out = out_dir / "agent_score_bootstrap_ci.png"
    if run_scores is None or run_scores.empty:
        logger.warning("agent_score_bootstrap_ci skipped: no data.")
        return
    try:
        import matplotlib.pyplot as plt

        agents = sorted(run_scores["agent"].unique())
        rng = np.random.default_rng(SEED)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        boot_data = []
        for a in agents:
            vals = run_scores[run_scores["agent"] == a]["AgentScore"].to_numpy(dtype=float)
            if len(vals) == 0:
                boot_data.append(np.array([np.nan]))
                continue
            bm = np.array([np.mean(rng.choice(vals, len(vals), replace=True))
                           for _ in range(n_boot)])
            boot_data.append(bm)

        parts = ax.violinplot(boot_data, showmeans=True, showextrema=False)
        for pc, a in zip(parts["bodies"], agents):
            pc.set_facecolor(_agent_colour(a))
            pc.set_alpha(0.6)
        ax.set_xticks(range(1, len(agents) + 1))
        ax.set_xticklabels(agents, fontsize=9)
        ax.set_ylabel("Bootstrap mean AgentScore")
        ax.set_title("Bootstrap Distribution of the Mean AgentScore")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        # Save boot_data per agent as CSV
        boot_csv = pd.DataFrame({a: d for a, d in zip(agents, boot_data)})
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create agent_score_bootstrap_ci: %s", exc)


def plot_pairwise_score_difference_ci(pairwise_diff: pd.DataFrame, out_dir: Path) -> None:
    out = out_dir / "pairwise_score_difference_ci.png"
    if pairwise_diff is None or pairwise_diff.empty:
        logger.warning("pairwise_score_difference_ci skipped: no data.")
        return
    try:
        import matplotlib.pyplot as plt

        df = pairwise_diff.copy()
        labels = [f"{r.agent_a} − {r.agent_b}" for r in df.itertuples()]
        ys = np.arange(len(labels))
        means = df["mean_difference"].to_numpy()
        los = (df["mean_difference"] - df["diff_ci95_lo"]).to_numpy()
        his = (df["diff_ci95_hi"] - df["mean_difference"]).to_numpy()

        fig, ax = plt.subplots(figsize=(7, 0.8 * len(labels) + 2))
        ax.errorbar(means, ys, xerr=[los, his], fmt="o", color="navy",
                    capsize=5, label="95% bootstrap CI")
        ax.axvline(0, color="red", linestyle="--", alpha=0.6, label="no difference")
        ax.axvspan(-2, 2, color="gray", alpha=0.15, label="practical equivalence (|Δ|<2)")
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Mean score difference")
        ax.set_title("Pairwise Agent Score Differences (95% bootstrap CI)")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create pairwise_score_difference_ci: %s", exc)


def plot_rank_uncertainty_interval(ranking_error: pd.DataFrame, out_dir: Path) -> None:
    out = out_dir / "rank_uncertainty_interval.png"
    if ranking_error is None or ranking_error.empty:
        logger.warning("rank_uncertainty_interval skipped: no data.")
        return
    try:
        import matplotlib.pyplot as plt

        df = ranking_error.sort_values("expected_rank")
        agents = df["agent"].tolist()
        ys = np.arange(len(agents))
        exp = df["expected_rank"].to_numpy()
        lo = (df["expected_rank"] - df["rank_ci95_lo"]).clip(lower=0).to_numpy()
        hi = (df["rank_ci95_hi"] - df["expected_rank"]).clip(lower=0).to_numpy()

        fig, ax = plt.subplots(figsize=(6.5, 0.9 * len(agents) + 2))
        ax.errorbar(exp, ys, xerr=[lo, hi], fmt="o", capsize=6,
                    color="darkgreen", label="95% bootstrap rank interval")
        for y, r in zip(ys, df.itertuples()):
            ax.annotate(f"obs={r.observed_rank}, switch={r.rank_switch_probability:.2f}",
                        (r.expected_rank, y), textcoords="offset points",
                        xytext=(8, 0), fontsize=7, va="center")
        ax.set_yticks(ys)
        ax.set_yticklabels(agents, fontsize=9)
        ax.set_xlabel("Rank (1 = best)")
        ax.set_title("Rank Uncertainty (expected rank ± 95% bootstrap interval)")
        ax.invert_xaxis()
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create rank_uncertainty_interval: %s", exc)


def plot_reproducibility_errorbar(repro_error: pd.DataFrame, out_dir: Path) -> None:
    out = out_dir / "reproducibility_errorbar.png"
    if repro_error is None or repro_error.empty:
        logger.warning("reproducibility_errorbar skipped: no data.")
        return
    try:
        import matplotlib.pyplot as plt

        df = repro_error[repro_error["scope"] == "within_agent"].copy()
        if df.empty:
            logger.warning("reproducibility_errorbar skipped: no within-agent rows.")
            return
        agents = df["agent"].tolist()
        xs = np.arange(len(agents))
        means = df["mean_within_similarity"].to_numpy(dtype=float)
        lo = (df["mean_within_similarity"] - df["within_sim_ci95_lo"]).clip(lower=0).to_numpy()
        hi = (df["within_sim_ci95_hi"] - df["mean_within_similarity"]).clip(lower=0).to_numpy()

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.bar(xs, means, color=[_agent_colour(a) for a in agents], alpha=0.8)
        ax.errorbar(xs, means, yerr=[lo, hi], fmt="none", ecolor="black",
                    capsize=6, label="95% bootstrap CI")
        ax.set_xticks(xs)
        ax.set_xticklabels(agents, fontsize=9)
        ax.set_ylabel("Mean within-agent cosine similarity")
        ax.set_title("Within-Agent Reproducibility (95% bootstrap CI)")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create reproducibility_errorbar: %s", exc)


def plot_centroid_uncertainty_pca2d(
    run_embs: Dict[Tuple[str, int], np.ndarray],
    boot_by_agent: Dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    out = out_dir / "centroid_uncertainty_pca2d.png"
    if not run_embs or not boot_by_agent:
        logger.warning("centroid_uncertainty_pca2d skipped: no data.")
        return
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from sklearn.decomposition import PCA

        keys = sorted(run_embs.keys())
        real_mat = np.stack([run_embs[k] for k in keys])
        all_boot = np.vstack([boot_by_agent[a] for a in boot_by_agent])
        combined = np.vstack([real_mat, all_boot])
        n_comp = min(2, combined.shape[1], combined.shape[0] - 1)
        if n_comp < 2:
            logger.warning("centroid_uncertainty_pca2d skipped: <2 PCA dims.")
            return
        pca = PCA(n_components=2, random_state=SEED).fit(combined)
        real_p = pca.transform(real_mat)

        fig, ax = plt.subplots(figsize=(7, 6))
        for agent in sorted(boot_by_agent.keys()):
            boot_p = pca.transform(boot_by_agent[agent])
            mu = boot_p.mean(axis=0)
            cov = np.cov(boot_p, rowvar=False)
            # 95% ellipse
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(vals * 5.991)  # chi2 2df 95%
            ell = Ellipse(xy=mu, width=width, height=height, angle=theta,
                          facecolor=_agent_colour(agent), alpha=0.18,
                          edgecolor=_agent_colour(agent), lw=1.5)
            ax.add_patch(ell)
            ax.scatter([mu[0]], [mu[1]], color=_agent_colour(agent),
                       marker="X", s=120, edgecolors="black", zorder=5)

        for agent in sorted(set(k[0] for k in keys)):
            idxs = [i for i, k in enumerate(keys) if k[0] == agent]
            ax.scatter(real_p[idxs, 0], real_p[idxs, 1], label=agent,
                       color=_agent_colour(agent), marker=_agent_marker(agent),
                       s=55, edgecolors="white", linewidths=0.5, zorder=4)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Centroid Uncertainty (2-D PCA)\n"
                     "X = bootstrap centroid · shaded = 95% bootstrap ellipse")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        rows = []
        for k, idx in [(k, i) for i, k in enumerate(keys)]:
            rows.append({"agent": k[0], "run": k[1], "PC1": float(real_p[idx, 0]),
                         "PC2": float(real_p[idx, 1]), "point_type": "real"})
        for agent in sorted(boot_by_agent.keys()):
            boot_p = pca.transform(boot_by_agent[agent])
            mu = boot_p.mean(axis=0)
            rows.append({"agent": agent, "run": "boot_centroid",
                         "PC1": float(mu[0]), "PC2": float(mu[1]),
                         "point_type": "bootstrap_centroid"})
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create centroid_uncertainty_pca2d: %s", exc)


def plot_monte_carlo_distance_distribution(
    distance_dists: Dict[str, np.ndarray], out_dir: Path,
) -> None:
    out = out_dir / "monte_carlo_distance_distribution.png"
    if not distance_dists:
        logger.warning("monte_carlo_distance_distribution skipped: no data.")
        return
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for agent, dists in distance_dists.items():
            if len(dists) == 0:
                continue
            ax.hist(dists, bins=40, density=True, alpha=0.45,
                    color=_agent_colour(agent), label=agent)
            ax.axvline(np.median(dists), color=_agent_colour(agent),
                       linestyle="--", alpha=0.8)
        ax.set_xlabel("Distance of satellite to real centroid")
        ax.set_ylabel("Density")
        ax.set_title("Monte Carlo Satellite Distance Distribution\n"
                     "(dashed = median; satellites are uncertainty probes)")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        # Flatten distance arrays per agent into a long-form CSV
        mc_rows = [{"agent": a, "distance": float(d)}
                   for a, dists in distance_dists.items() for d in dists]
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create monte_carlo_distance_distribution: %s", exc)


# ===========================================================================
# CONSENSUS / DIVERGENCE FIGURES
# ===========================================================================

def plot_consensus_method_overlap_heatmap(
    method_mat: pd.DataFrame, out_dir: Path,
) -> None:
    """Heatmap: agents (rows) × method terms (columns), value = mean presence fraction."""
    out = out_dir / "consensus_method_overlap_heatmap.png"
    if method_mat is None or method_mat.empty:
        logger.warning("consensus_method_overlap_heatmap skipped: no data.")
        return
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        present_cols = [c for c in method_mat.columns if c.endswith("_present")]
        if not present_cols:
            logger.warning("consensus_method_overlap_heatmap: no present cols.")
            return
        terms = [c.replace("_present", "") for c in present_cols]
        agents = sorted(method_mat["agent"].unique())

        mat = np.zeros((len(agents), len(terms)))
        for i, agent in enumerate(agents):
            sub = method_mat[method_mat["agent"] == agent]
            for j, pcol in enumerate(present_cols):
                mat[i, j] = float(sub[pcol].mean()) if len(sub) > 0 else 0.0

        fig, ax = plt.subplots(figsize=(max(12, len(terms) * 0.5),
                                        max(4, len(agents) * 0.8)))
        sns.heatmap(mat, xticklabels=terms, yticklabels=agents,
                    cmap="Blues", vmin=0, vmax=1,
                    cbar_kws={"label": "fraction of runs with term"},
                    annot=True, fmt=".2f", linewidths=0.3, linecolor="white", ax=ax)
        ax.set_title("Methodological Term Presence — Mean Fraction per Agent")
        ax.set_xlabel("Method term")
        ax.set_ylabel("Agent")
        plt.xticks(rotation=60, ha="right", fontsize=7)
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create consensus_method_overlap_heatmap: %s", exc)


def plot_consensus_biological_theme_heatmap(
    bio_mat: pd.DataFrame, out_dir: Path,
) -> None:
    """Heatmap: agents (rows) × biological terms (columns), value = mean presence fraction."""
    out = out_dir / "consensus_biological_theme_heatmap.png"
    if bio_mat is None or bio_mat.empty:
        logger.warning("consensus_biological_theme_heatmap skipped: no data.")
        return
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        present_cols = [c for c in bio_mat.columns if c.endswith("_present")]
        if not present_cols:
            return
        terms = [c.replace("_present", "") for c in present_cols]
        agents = sorted(bio_mat["agent"].unique())

        mat = np.zeros((len(agents), len(terms)))
        for i, agent in enumerate(agents):
            sub = bio_mat[bio_mat["agent"] == agent]
            for j, pcol in enumerate(present_cols):
                mat[i, j] = float(sub[pcol].mean()) if len(sub) > 0 else 0.0

        fig, ax = plt.subplots(figsize=(max(12, len(terms) * 0.5),
                                        max(4, len(agents) * 0.8)))
        sns.heatmap(mat, xticklabels=terms, yticklabels=agents,
                    cmap="Greens", vmin=0, vmax=1,
                    cbar_kws={"label": "fraction of runs with term"},
                    annot=True, fmt=".2f", linewidths=0.3, linecolor="white", ax=ax)
        ax.set_title("Biological Theme Presence — Mean Fraction per Agent")
        ax.set_xlabel("Biological term")
        ax.set_ylabel("Agent")
        plt.xticks(rotation=60, ha="right", fontsize=7)
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create consensus_biological_theme_heatmap: %s", exc)


def plot_agent_specific_terms_barplot(
    agent_specific: pd.DataFrame, out_dir: Path,
) -> None:
    """Horizontal bar chart of agent-specific term counts by agent and category."""
    out = out_dir / "agent_specific_terms_barplot.png"
    if agent_specific is None or agent_specific.empty:
        logger.warning("agent_specific_terms_barplot skipped: no data.")
        return
    try:
        import matplotlib.pyplot as plt

        agents = sorted(agent_specific["dominant_agent"].unique())
        categories = sorted(agent_specific["category"].unique())

        counts = {}
        for agent in agents:
            counts[agent] = {}
            for cat in categories:
                n = len(agent_specific[
                    (agent_specific["dominant_agent"] == agent) &
                    (agent_specific["category"] == cat)
                ])
                counts[agent][cat] = n

        x = np.arange(len(agents))
        width = 0.25
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, cat in enumerate(categories):
            vals = [counts[a].get(cat, 0) for a in agents]
            ax.bar(x + i * width, vals, width, label=cat, alpha=0.85)

        ax.set_xticks(x + width * (len(categories) - 1) / 2)
        ax.set_xticklabels(agents, fontsize=9)
        ax.set_ylabel("Number of agent-specific terms")
        ax.set_title("Agent-Specific Terms by Category")
        ax.legend(title="Category")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create agent_specific_terms_barplot: %s", exc)


def plot_shared_vs_unique_features_barplot(
    shared_terms: pd.DataFrame, out_dir: Path,
) -> None:
    """Stacked bar: shared (all agents) vs shared (2+ agents) vs unique per category."""
    out = out_dir / "shared_vs_unique_features_barplot.png"
    if shared_terms is None or shared_terms.empty:
        logger.warning("shared_vs_unique_features_barplot skipped: no data.")
        return
    try:
        import matplotlib.pyplot as plt

        categories = sorted(shared_terms["category"].unique())
        x = np.arange(len(categories))
        width = 0.6

        shared_all   = [len(shared_terms[(shared_terms["category"] == c) &
                                          (shared_terms["shared_all_agents"])]) for c in categories]
        shared_2plus = [len(shared_terms[(shared_terms["category"] == c) &
                                          (shared_terms["shared_2plus_agents"]) &
                                          (~shared_terms["shared_all_agents"])]) for c in categories]
        unique       = [len(shared_terms[(shared_terms["category"] == c) &
                                          (shared_terms["agent_specific"])]) for c in categories]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        p1 = ax.bar(x, shared_all,   width, label="Shared all agents", color="#2196F3", alpha=0.85)
        p2 = ax.bar(x, shared_2plus, width, bottom=shared_all,
                    label="Shared ≥2 agents", color="#FF9800", alpha=0.85)
        p3 = ax.bar(x, unique, width,
                    bottom=[a + b for a, b in zip(shared_all, shared_2plus)],
                    label="Agent-specific", color="#F44336", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylabel("Number of terms")
        ax.set_title("Shared vs Unique Terms by Category")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create shared_vs_unique_features_barplot: %s", exc)


def plot_consensus_alignment_barplot(
    alignment_error: pd.DataFrame, out_dir: Path,
) -> None:
    """Bar chart of mean consensus alignment score per agent with 95% CI."""
    out = out_dir / "consensus_alignment_barplot.png"
    if alignment_error is None or alignment_error.empty:
        logger.warning("consensus_alignment_barplot skipped: no data.")
        return
    try:
        import matplotlib.pyplot as plt

        df = alignment_error.sort_values("mean_consensus_alignment", ascending=False)
        agents = df["agent"].tolist()
        xs = np.arange(len(agents))
        means = df["mean_consensus_alignment"].to_numpy()
        los = (df["mean_consensus_alignment"] - df["ci95_lo"]).clip(lower=0).to_numpy()
        his = (df["ci95_hi"] - df["mean_consensus_alignment"]).clip(lower=0).to_numpy()

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.bar(xs, means, color=[_agent_colour(a) for a in agents], alpha=0.8)
        ax.errorbar(xs, means, yerr=[los, his], fmt="none", ecolor="black",
                    capsize=6, label="95% bootstrap CI")
        ax.set_xticks(xs)
        ax.set_xticklabels(agents, fontsize=9)
        ax.set_ylabel("Mean consensus alignment score (0–100)")
        ax.set_ylim(0, 105)
        ax.set_title("Consensus Alignment Score by Agent")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create consensus_alignment_barplot: %s", exc)


def plot_consensus_alignment_errorbar(
    alignment_run: pd.DataFrame,
    alignment_error: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Strip chart of per-run alignment scores with agent-level mean ± CI overlay."""
    out = out_dir / "consensus_alignment_errorbar.png"
    if alignment_run is None or alignment_run.empty:
        logger.warning("consensus_alignment_errorbar skipped: no data.")
        return
    try:
        import matplotlib.pyplot as plt

        agents = sorted(alignment_run["agent"].unique())
        xs = np.arange(len(agents))

        fig, ax = plt.subplots(figsize=(6.5, 5))
        for i, agent in enumerate(agents):
            sub = alignment_run[alignment_run["agent"] == agent]
            jitter = np.random.default_rng(SEED).uniform(-0.15, 0.15, len(sub))
            ax.scatter(np.full(len(sub), i) + jitter,
                       sub["consensus_alignment_score"].values,
                       color=_agent_colour(agent), alpha=0.6, s=35,
                       marker=_agent_marker(agent))

        if alignment_error is not None and not alignment_error.empty:
            for i, agent in enumerate(agents):
                row = alignment_error[alignment_error["agent"] == agent]
                if row.empty:
                    continue
                m  = float(row["mean_consensus_alignment"].iloc[0])
                lo = float(row["ci95_lo"].iloc[0])
                hi = float(row["ci95_hi"].iloc[0])
                ax.scatter([i], [m], color="black", marker="D", s=60, zorder=5)
                ax.errorbar([i], [m], yerr=[[m - lo], [hi - m]],
                            fmt="none", ecolor="black", capsize=6, zorder=4)

        ax.set_xticks(xs)
        ax.set_xticklabels(agents, fontsize=9)
        ax.set_ylabel("Consensus alignment score (0–100)")
        ax.set_title("Consensus Alignment: Per-run scores and agent mean ± 95% CI")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create consensus_alignment_errorbar: %s", exc)


def plot_consensus_alignment_pca3d(
    run_embs: Dict[Tuple[str, int], np.ndarray],
    alignment_run: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Interactive 3D PCA coloured by consensus alignment score.
    Saves consensus_alignment_pca3d.html.
    If static PNG is unavailable, logs a warning and keeps the HTML only.
    """
    out = out_dir / "consensus_alignment_pca3d.html"
    if not run_embs or alignment_run is None or alignment_run.empty:
        logger.warning("consensus_alignment_pca3d skipped: missing data.")
        return
    try:
        import plotly.graph_objects as go
        from sklearn.decomposition import PCA as _PCA

        keys = sorted(run_embs.keys())
        mat  = np.stack([run_embs[k] for k in keys]).astype(np.float64)
        n_comp = min(3, mat.shape[1], mat.shape[0] - 1)
        pca = _PCA(n_components=n_comp, random_state=SEED)
        coords = pca.fit_transform(mat)
        evr = pca.explained_variance_ratio_

        def _ax(idx):
            return f"PC{idx+1} ({evr[idx]*100:.1f}%)" if idx < len(evr) else f"PC{idx+1}"

        # Merge alignment scores
        scores_map = {}
        for _, row in alignment_run.iterrows():
            scores_map[(row["agent"], row["run"])] = row["consensus_alignment_score"]

        agents_list = [k[0] for k in keys]
        runs_list   = [k[1] for k in keys]
        scores_list = [scores_map.get(k, float("nan")) for k in keys]
        texts_list  = [f"{a} R{r}" for a, r in keys]

        z_col = coords[:, 2] if n_comp >= 3 else np.zeros(len(keys))

        fig = go.Figure()

        # Plot per agent (so legend shows agent names)
        agents = sorted(set(agents_list))
        for agent in agents:
            idxs = [i for i, a in enumerate(agents_list) if a == agent]
            fig.add_trace(go.Scatter3d(
                x=coords[idxs, 0], y=coords[idxs, 1], z=z_col[idxs],
                mode="markers+text",
                marker=dict(
                    size=10,
                    color=[scores_list[i] for i in idxs],
                    colorscale="Viridis",
                    cmin=0, cmax=100,
                    opacity=0.90,
                    line=dict(width=1, color="white"),
                    showscale=(agent == agents[0]),
                    colorbar=dict(title="Alignment\nscore", thickness=12, len=0.5)
                    if agent == agents[0] else None,
                ),
                text=[texts_list[i] for i in idxs],
                name=agent,
            ))

        # Centroids
        for agent in agents:
            idxs = [i for i, a in enumerate(agents_list) if a == agent]
            cx = coords[idxs, 0].mean(); cy = coords[idxs, 1].mean()
            cz = z_col[idxs].mean()
            fig.add_trace(go.Scatter3d(
                x=[cx], y=[cy], z=[cz],
                mode="markers",
                marker=dict(size=14, color=_agent_colour(agent),
                            symbol="diamond", opacity=1.0,
                            line=dict(width=2, color="black")),
                name=f"{agent} centroid",
            ))

        fig.update_layout(
            title=("Consensus Alignment in PCA Space<br>"
                   "<sup>Colour = consensus alignment score (0–100) · "
                   "diamonds = agent centroids</sup>"),
            scene=dict(
                xaxis_title=_ax(0), yaxis_title=_ax(1),
                zaxis_title=_ax(2) if n_comp >= 3 else "PC3",
            ),
            legend_title="Agent",
        )
        fig.write_html(str(out))
        logger.info("Saved %s", out)
    except Exception as exc:
        logger.warning("Could not create consensus_alignment_pca3d: %s", exc)


def plot_agent_distinctiveness_barplot(
    distinct_error: pd.DataFrame, out_dir: Path,
) -> None:
    """Bar chart of mean distinctiveness score per agent with 95% CI."""
    out = out_dir / "agent_distinctiveness_barplot.png"
    if distinct_error is None or distinct_error.empty:
        logger.warning("agent_distinctiveness_barplot skipped: no data.")
        return
    try:
        import matplotlib.pyplot as plt

        df = distinct_error.sort_values("mean_distinctiveness", ascending=False)
        agents = df["agent"].tolist()
        xs = np.arange(len(agents))
        means = df["mean_distinctiveness"].to_numpy()
        los = (df["mean_distinctiveness"] - df["ci95_lo"]).clip(lower=0).to_numpy()
        his = (df["ci95_hi"] - df["mean_distinctiveness"]).clip(lower=0).to_numpy()

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.bar(xs, means, color=[_agent_colour(a) for a in agents], alpha=0.8)
        ax.errorbar(xs, means, yerr=[los, his], fmt="none", ecolor="black",
                    capsize=6, label="95% bootstrap CI")
        ax.set_xticks(xs)
        ax.set_xticklabels(agents, fontsize=9)
        ax.set_ylabel("Mean distinctiveness score (0–100)")
        ax.set_ylim(0, 105)
        ax.set_title("Agent Distinctiveness Score")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create agent_distinctiveness_barplot: %s", exc)


def plot_agent_distinctiveness_vs_agentscore(
    distinct_run: pd.DataFrame, out_dir: Path,
) -> None:
    """Scatter: distinctiveness score vs AgentScore per run, coloured by agent."""
    out = out_dir / "agent_distinctiveness_vs_agentscore.png"
    if distinct_run is None or distinct_run.empty:
        logger.warning("agent_distinctiveness_vs_agentscore skipped: no data.")
        return
    if "AgentScore" not in distinct_run.columns:
        logger.warning("agent_distinctiveness_vs_agentscore: AgentScore missing.")
        return
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 5))
        for agent in sorted(distinct_run["agent"].unique()):
            sub = distinct_run[distinct_run["agent"] == agent]
            ax.scatter(sub["AgentScore"], sub["distinctiveness_score"],
                       label=agent, color=_agent_colour(agent),
                       marker=_agent_marker(agent),
                       s=60, alpha=0.85, edgecolors="white", linewidths=0.5)
            for r in sub.itertuples():
                ax.annotate(f"R{r.run}", (r.AgentScore, r.distinctiveness_score),
                            fontsize=6, alpha=0.6)

        ax.set_xlabel("AgentScore")
        ax.set_ylabel("Distinctiveness score (0–100)")
        ax.set_title("Distinctiveness vs AgentScore\n"
                     "(high distinctiveness ≠ high quality)")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)
        _save_svg(fig, out)
    except Exception as exc:
        logger.warning("Could not create agent_distinctiveness_vs_agentscore: %s", exc)
