"""Publication-style figures. All functions save to `outdir` and return the path."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
_PALETTE = {"Control": "#4C72B0", "Fibrosis": "#C44E52", "QC": "#7f7f7f"}


def _save(fig, outdir: str, name: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def qc_cv_histogram(feature_qc: pd.DataFrame, threshold, outdir: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    cv = feature_qc["qc_cv"].dropna() * 100
    ax.hist(cv, bins=40, color="#55A868", edgecolor="white")
    if threshold is not None:
        ax.axvline(threshold * 100, color="red", ls="--", label=f"cutoff {threshold:.0%}")
        ax.legend()
    ax.set_xlabel("QC coefficient of variation (%)")
    ax.set_ylabel("Number of features")
    ax.set_title("Analytical reproducibility (QC CV)")
    return _save(fig, outdir, "qc_cv_histogram.png")


def sample_boxplot(normalized: pd.DataFrame, sample_groups: pd.Series, transform, outdir: str) -> str:
    data = np.log2(normalized.clip(lower=0) + 1) if transform else normalized
    order = list(data.columns)
    colors = [_PALETTE.get(sample_groups.get(c, "QC"), "#999999") for c in order]
    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.28), 5))
    bp = ax.boxplot([data[c].dropna() for c in order], patch_artist=True,
                    showfliers=False)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=90, fontsize=6)
    ax.set_ylabel("log2 normalized intensity")
    ax.set_title("Per-sample intensity distributions (post-normalization)")
    return _save(fig, outdir, "sample_boxplots.png")


def pca_plot(scores: pd.DataFrame, explained, labels: pd.Series, outdir: str) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for grp in labels.dropna().unique():
        idx = labels[labels == grp].index
        ax.scatter(scores.loc[idx, "PC1"], scores.loc[idx, "PC2"],
                   label=grp, s=70, alpha=0.8,
                   color=_PALETTE.get(grp, None), edgecolor="k", linewidth=0.4)
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_title("PCA score plot")
    ax.legend(frameon=True)
    return _save(fig, outdir, "pca_scores.png")


def pls_plot(scores: pd.DataFrame, labels: pd.Series, q2, r2, outdir: str) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    lv2 = "LV2" if "LV2" in scores.columns else "LV1"
    for grp in labels.reindex(scores.index).dropna().unique():
        idx = labels[labels == grp].index.intersection(scores.index)
        y = scores.loc[idx, lv2] if lv2 != "LV1" else np.zeros(len(idx))
        ax.scatter(scores.loc[idx, "LV1"], y, label=grp, s=80, alpha=0.85,
                   color=_PALETTE.get(grp, None), edgecolor="k", linewidth=0.4)
    ax.set_xlabel("LV1")
    ax.set_ylabel(lv2 if lv2 != "LV1" else "")
    ax.set_title(f"PLS-DA  (R2={r2:.2f}, Q2={q2:.2f})")
    ax.legend(frameon=True)
    return _save(fig, outdir, "plsda_scores.png")


def volcano(stats_df: pd.DataFrame, cfg, outdir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    x = stats_df["log2fc"]
    y = stats_df["neg_log10_p"]
    colors = np.where(
        stats_df["regulation"] == "up", "#C44E52",
        np.where(stats_df["regulation"] == "down", "#4C72B0", "#bbbbbb"),
    )
    ax.scatter(x, y, c=colors, s=28, alpha=0.7, edgecolor="none")
    ax.axvline(cfg.log2fc_threshold, color="grey", ls=":")
    ax.axvline(-cfg.log2fc_threshold, color="grey", ls=":")
    # annotate top hits
    top = stats_df[stats_df["significant"]].sort_values("q_value").head(12)
    for name, row in top.iterrows():
        ax.annotate(str(name)[:18], (row["log2fc"], row["neg_log10_p"]),
                    fontsize=7, alpha=0.9)
    ax.set_xlabel("log2 fold change (Fibrosis / Control)")
    ax.set_ylabel("-log10 p-value")
    ax.set_title("Volcano plot")
    return _save(fig, outdir, "volcano.png")


def heatmap_top(transformed: pd.DataFrame, stats_df: pd.DataFrame,
                sample_groups: pd.Series, outdir: str, top_n: int = 30) -> str:
    top = stats_df[stats_df["significant"]].sort_values("q_value").head(top_n).index
    top = [t for t in top if t in transformed.index]
    if not top:
        return ""
    cols = [c for c in transformed.columns if c in sample_groups.index]
    sub = transformed.loc[top, cols]
    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1, ddof=1).replace(0, np.nan), axis=0)
    col_colors = sample_groups.reindex(cols).map(_PALETTE)
    g = sns.clustermap(z.fillna(0), cmap="RdBu_r", center=0,
                       col_colors=col_colors.values, figsize=(12, 10),
                       yticklabels=True, xticklabels=False,
                       cbar_kws={"label": "row z-score"})
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=7)
    path = os.path.join(outdir, "heatmap_top.png")
    g.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    return path


def enrichment_barplot(enrich_df: pd.DataFrame, outdir: str, top_n: int = 15) -> str:
    if enrich_df.empty:
        return ""
    sub = enrich_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, len(sub) * 0.45)))
    colors = ["#C44E52" if e else "#88a" for e in sub["enriched"]]
    ax.barh(sub.index, sub["neg_log10_p"], color=colors)
    ax.set_xlabel("-log10 p-value (Fisher exact)")
    ax.set_title("Pathway over-representation")
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.text(row["neg_log10_p"], i, f"  {int(row['n_significant'])}/{int(row['pathway_size'])}",
                va="center", fontsize=8)
    return _save(fig, outdir, "pathway_enrichment.png")


def roc_top(normalized: pd.DataFrame, stats_df: pd.DataFrame, ds, outdir: str, top_n: int = 6) -> str:
    from sklearn.metrics import roc_curve, auc as sk_auc
    ctrl_label, case_label = ds.cfg.group_labels
    ctrl = ds.group_cols[ctrl_label]
    case = ds.group_cols[case_label]
    top = stats_df.sort_values("q_value").head(top_n).index
    fig, ax = plt.subplots(figsize=(7, 7))
    for name in top:
        if name not in normalized.index:
            continue
        vals = np.concatenate([normalized.loc[name, case].values,
                               normalized.loc[name, ctrl].values])
        y = np.concatenate([np.ones(len(case)), np.zeros(len(ctrl))])
        ok = np.isfinite(vals)
        if ok.sum() < 4:
            continue
        fpr, tpr, _ = roc_curve(y[ok], vals[ok])
        a = sk_auc(fpr, tpr)
        if a < 0.5:
            fpr, tpr, _ = roc_curve(y[ok], -vals[ok]); a = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{str(name)[:18]} (AUC={a:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC: top candidate biomarkers")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, outdir, "roc_top.png")
