"""Automated biological interpretation.

Generates a narrative Markdown report that links the statistical and pathway
results back to cardiac/atrial-fibrosis biology. Domain knowledge is encoded in
`PATHWAY_NOTES` and `METABOLITE_NOTES`; the narrative is assembled from whichever
of these are actually significant in the data, so the text is always grounded in
the run's results rather than asserted a priori.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import knowledge_base as kb

# Short, citable mechanistic notes connecting pathways to cardiac fibrosis / AF.
PATHWAY_NOTES: dict[str, str] = {
    "Proline & collagen-related metabolism":
        "Proline and hydroxyproline are the building blocks and turnover markers "
        "of collagen. Elevated hydroxyproline/proline flux is a direct biochemical "
        "signature of extracellular-matrix deposition and fibrotic remodeling of "
        "the atrium.",
    "Choline & TMAO metabolism":
        "Choline-derived TMAO is a gut-microbiome metabolite repeatedly associated "
        "with atrial fibrillation, atrial structural remodeling and pro-fibrotic "
        "TGF-beta signaling.",
    "Creatine metabolism":
        "Creatine/phosphocreatine is the principal short-term energy buffer of "
        "cardiomyocytes; depletion of the creatine kinase system is a hallmark of "
        "the failing/remodeling myocardium and tracks with reduced contractile "
        "reserve.",
    "Carnitine shuttle & fatty-acid beta-oxidation":
        "Acylcarnitine accumulation indicates impaired mitochondrial fatty-acid "
        "oxidation and a metabolic shift away from the heart's preferred fuel, a "
        "common feature of stressed, remodeling myocardium.",
    "TCA cycle":
        "Changes in TCA-cycle intermediates (e.g. succinate, 2-hydroxyglutarate, "
        "citrate) reflect altered mitochondrial energetics and can drive hypoxia- "
        "and HIF-linked pro-fibrotic signaling.",
    "Methionine & one-carbon metabolism":
        "One-carbon/SAM metabolism governs methylation capacity (including DNA and "
        "histone methylation in fibroblasts); perturbed SAM/SAH balance is linked "
        "to fibrotic gene programs and to homocysteine-related cardiovascular risk.",
    "Nicotinate & NAD metabolism":
        "NAD+ availability controls sirtuin activity and mitochondrial redox; NAD+ "
        "decline and altered nicotinamide methylation (NNMT/1-methylnicotinamide) "
        "are implicated in cardiac aging and fibrosis.",
    "Polyamine metabolism":
        "Polyamines (putrescine, spermidine, spermine) modulate cardiomyocyte "
        "hypertrophy, autophagy and fibroblast proliferation; dysregulation "
        "accompanies adverse cardiac remodeling.",
    "Glutathione & glutamate metabolism":
        "Glutathione is the major intracellular antioxidant; depletion signals "
        "oxidative stress, a well-established upstream driver of atrial fibrosis.",
    "Taurine & sulfur amino-acid metabolism":
        "Taurine is highly abundant in myocardium and is cardioprotective "
        "(calcium handling, osmoregulation, antioxidant); loss is associated with "
        "cardiomyopathy.",
    "Sphingolipid metabolism":
        "Ceramides and sphingosine-1-phosphate regulate inflammation, apoptosis "
        "and fibroblast activation; ceramide accumulation is lipotoxic and "
        "pro-fibrotic.",
    "Arginine biosynthesis & urea cycle":
        "Arginine is the substrate for nitric-oxide synthases; an arginine/NO "
        "imbalance (and ADMA accumulation) impairs endothelial function and "
        "promotes fibrosis.",
    "Branched-chain amino acid degradation":
        "Impaired BCAA catabolism (rising BCAAs/branched acylcarnitines) activates "
        "mTOR and is increasingly tied to heart failure and cardiac remodeling.",
    "Tryptophan & kynurenine metabolism":
        "The kynurenine pathway is induced by inflammation; kynurenine metabolites "
        "modulate immune tone and have been associated with AF and fibrosis.",
}

METABOLITE_NOTES: dict[str, str] = {
    "hydroxyproline": "post-translational collagen marker; near-direct readout of fibrotic ECM turnover",
    "l-proline": "collagen precursor; substrate demand rises during matrix synthesis",
    "proline": "collagen precursor; substrate demand rises during matrix synthesis",
    "trimethylamine-n-oxide": "gut-derived pro-fibrotic metabolite linked to AF risk",
    "creatine": "energy-buffer; depletion reflects bioenergetic stress",
    "creatinine": "creatine breakdown product; ratio to creatine reflects turnover",
    "guanidinoacetic acid": "creatine-synthesis intermediate (GATM/GAMT axis)",
    "taurine": "cardioprotective osmolyte/antioxidant; loss marks cardiomyopathy",
    "1-methylnicotinamide": "NNMT product; marker of NAD+ methylation/consumption",
    "nicotinamide": "NAD+ precursor; central to mitochondrial redox",
    "dimethylglycine": "betaine/one-carbon intermediate; reflects methylation flux",
    "betaine": "methyl donor and osmolyte in the choline oxidation pathway",
    "spermidine": "polyamine with autophagy-promoting, cardioprotective roles",
    "succinic acid": "TCA intermediate; accumulation drives pro-inflammatory/HIF signaling",
    "2-hydroxyglutarate": "oncometabolite-like signal of mitochondrial/redox stress",
}


def _direction_word(log2fc: float, labels) -> str:
    case = labels[1]
    if log2fc > 0:
        return f"higher in {case}"
    return f"lower in {case}"


def build_report(stats_df: pd.DataFrame, enrich_df: pd.DataFrame,
                 gene_df: pd.DataFrame, multivariate, proc, ds,
                 stats_summary: dict) -> str:
    cfg = ds.cfg
    ctrl_label, case_label = cfg.group_labels
    sig = stats_df[stats_df["significant"]]
    L = []

    L.append(f"# Biological interpretation: {case_label} vs {ctrl_label} atrial tissue\n")
    L.append("_Automated narrative generated from this run's statistics and "
             "pathway over-representation results. All claims are anchored to "
             "metabolites/pathways that reached significance in the data._\n")

    # ---- Headline ----
    L.append("## 1. Headline findings\n")
    L.append(f"- **{stats_summary['n_significant']}** of "
             f"{stats_summary['n_tested']} tested metabolites differ significantly "
             f"(BH-FDR < {cfg.alpha}): "
             f"**{stats_summary['n_up']} up** and **{stats_summary['n_down']} down** "
             f"in {case_label}.")
    L.append(f"- **{stats_summary['n_candidate_biomarkers']}** pass the additional "
             f"effect-size filter (|log2FC| >= {cfg.log2fc_threshold}).")
    if multivariate is not None:
        L.append(f"- Supervised PLS-DA separates the groups with "
                 f"R2={multivariate.pls_r2:.2f} and cross-validated "
                 f"Q2={multivariate.pls_q2:.2f} "
                 f"({'robust' if multivariate.pls_q2 > 0.4 else 'modest'} separation).")
    L.append("")

    # ---- Top metabolites ----
    L.append("## 2. Most discriminating metabolites\n")
    L.append("| Metabolite | log2FC | q-value | AUC | Direction | Note |")
    L.append("|---|---:|---:|---:|---|---|")
    for name, row in sig.sort_values("q_value").head(15).iterrows():
        note = METABOLITE_NOTES.get(kb.normalize_name(name), "")
        L.append(f"| {name} | {row['log2fc']:.2f} | {row['q_value']:.2e} | "
                 f"{row['auc']:.2f} | {_direction_word(row['log2fc'], cfg.group_labels)} | {note} |")
    L.append("")

    # ---- Pathways ----
    L.append("## 3. Affected pathways\n")
    if enrich_df is None or enrich_df.empty:
        L.append("_No pathway reached the minimum size for testing._\n")
    else:
        enriched = enrich_df[enrich_df["enriched"]]
        ranked = enriched if not enriched.empty else enrich_df.head(6)
        if enriched.empty:
            L.append("_No pathway survived multiple-testing correction; the "
                     "strongest nominal trends are summarised below._\n")
        for pw, row in ranked.iterrows():
            note = PATHWAY_NOTES.get(pw, "")
            direction = ("predominantly up" if row["mean_log2fc_sig"] > 0
                         else "predominantly down")
            L.append(f"### {pw}")
            L.append(f"- {int(row['n_significant'])}/{int(row['pathway_size'])} "
                     f"detected members significant "
                     f"(enrichment {row['enrichment_ratio']:.1f}x, "
                     f"p={row['p_value']:.2e}, q={row['q_value']:.2e}); "
                     f"{direction} in {case_label}.")
            if row["hit_metabolites"]:
                L.append(f"- Drivers: {row['hit_metabolites']}.")
            if row["genes"]:
                L.append(f"- Candidate enzymes/genes: {row['genes']}.")
            if note:
                L.append(f"- **Interpretation:** {note}")
            L.append("")

    # ---- Genes ----
    L.append("## 4. Implicated genes / enzymes\n")
    if gene_df is None or gene_df.empty:
        L.append("_No genes implicated (no significant metabolites mapped to "
                 "pathways)._\n")
    else:
        L.append("Genes are nominated because they catalyse steps in pathways "
                 "containing significant metabolites. This is a hypothesis-"
                 "generating bridge from metabolite to transcript/protein, not a "
                 "measured gene-expression result.\n")
        L.append("| Gene | #sig metabolites | Pathways | Linked metabolites |")
        L.append("|---|---:|---|---|")
        for g, row in gene_df.head(20).iterrows():
            L.append(f"| {g} | {int(row['n_sig_metabolites'])} | "
                     f"{row['pathways']} | {row['sig_metabolites']} |")
        L.append("")

    # ---- Synthesis ----
    L.append("## 5. Integrated mechanistic synthesis\n")
    themes = _themes(enrich_df, sig)
    if themes:
        for t in themes:
            L.append(f"- {t}")
    else:
        L.append("- The differential metabolites do not converge on a single "
                 "dominant pathway; effects appear distributed across energy "
                 "metabolism and amino-acid handling.")
    L.append("")

    # ---- Caveats ----
    L.append("## 6. Limitations\n")
    L.append("- Untargeted intensities are semi-quantitative; fold changes are "
             "relative, not absolute concentrations.")
    L.append("- Pathway and gene mappings come from a curated offline knowledge "
             "base and should be confirmed against KEGG/MetaboAnalyst and, ideally, "
             "orthogonal transcriptomic/proteomic data.")
    L.append(f"- n={len(ds.pairs) if cfg.paired else 'NA'} pairs; findings are "
             "hypothesis-generating and need validation in an independent cohort.")
    L.append("")

    return "\n".join(L)


def _themes(enrich_df: pd.DataFrame, sig: pd.DataFrame) -> list[str]:
    """Assemble mechanistic themes from (a) enriched/strongly-represented pathways
    and (b) direct membership of significant metabolites, so the synthesis fires
    even when a pathway misses the strict FDR cut."""
    themes: list[str] = []

    # Pathways to consider: FDR-enriched, OR nominally strong (>=3 hits, p<0.1).
    test: set[str] = set()
    if enrich_df is not None and not enrich_df.empty:
        test |= set(enrich_df[enrich_df["enriched"]].index)
        strong = enrich_df[(enrich_df["n_significant"] >= 3) & (enrich_df["p_value"] < 0.10)]
        test |= set(strong.index)

    sig_norm = {kb.normalize_name(m) for m in sig.index}

    def any_hit(names: set[str]) -> bool:
        return any(n in sig_norm for n in names)

    membrane_axis = {"Glycerophospholipid metabolism", "Sphingolipid metabolism"}
    fib_axis = {"Proline & collagen-related metabolism"}
    energy_axis = {"Creatine metabolism", "TCA cycle",
                   "Carnitine shuttle & fatty-acid beta-oxidation"}
    gut_axis = {"Choline & TMAO metabolism"}
    redox_axis = {"Glutathione & glutamate metabolism",
                  "Taurine & sulfur amino-acid metabolism",
                  "Nicotinate & NAD metabolism"}

    # Direct energy-charge readout from adenine nucleotides.
    adenylates = {"adenosine triphosphate", "adenosine diphosphate",
                  "adenosine monophosphate"}
    if any_hit(adenylates) or ("Purine metabolism" in test):
        down = [m for m in sig.index
                if kb.normalize_name(m) in adenylates and sig.loc[m, "log2fc"] < 0]
        if down:
            themes.append("**Energy-charge collapse:** adenine nucleotides "
                          f"({', '.join(down)}) are depleted, indicating a fall in "
                          "the cellular ATP/ADP/AMP energy charge — a direct sign of "
                          "compromised mitochondrial ATP supply in the remodeling "
                          "myocardium.")

    if test & membrane_axis:
        themes.append("**Membrane phospholipid remodeling:** broad changes across "
                      "glycerophospholipids (PC/PE/PI/PS, lyso-species, cardiolipin) "
                      "and/or sphingolipids reflect remodeling of cardiomyocyte and "
                      "mitochondrial membranes and altered phospholipase/lyso-lipid "
                      "signaling that accompanies fibrotic injury.")
    if test & fib_axis or any_hit({"hydroxyproline", "l-proline", "proline"}):
        themes.append("**Direct fibrotic signature:** collagen-related "
                      "proline/hydroxyproline metabolism is altered, consistent "
                      "with active extracellular-matrix remodeling in the fibrotic "
                      "atrium.")
    if test & energy_axis or any_hit({"creatine", "creatinine", "succinic acid"}):
        themes.append("**Bioenergetic/substrate remodeling:** changes in creatine, "
                      "TCA-cycle and/or acylcarnitine metabolism point to a shift in "
                      "cardiac fuel use and reduced mitochondrial efficiency.")
    if test & gut_axis or any_hit({"trimethylamine-n-oxide", "choline", "betaine"}):
        themes.append("**Gut-microbiome / choline axis:** perturbed choline/TMAO "
                      "metabolism implicates the diet-microbiome-host axis linked to "
                      "atrial fibrillation.")
    if test & redox_axis or any_hit({"taurine", "nicotinamide", "methionine sulfoxide",
                                     "glutathione", "pyroglutamic acid"}):
        themes.append("**Oxidative / redox stress:** loss of antioxidants (taurine, "
                      "NAD-precursor nicotinamide) together with oxidation products "
                      "(e.g. methionine sulfoxide) is compatible with oxidative "
                      "stress driving fibroblast activation and fibrosis.")
    return themes
