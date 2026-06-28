"""Central configuration for the metabolomics pipeline.

All tunable analysis parameters live here so that a run is fully reproducible
from a single object. Defaults follow common untargeted-metabolomics practice
(Dunn et al. 2011; Broadhurst et al. 2018).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class Config:
    # ---- Input / output ----------------------------------------------------
    input_path: str = ""
    sheet: int | str = 0
    feature_col: str = "Molecule"
    output_dir: str = "results"

    # Sample-group detection. Columns are assigned to a group if their name
    # contains one of these (case-insensitive) prefixes.
    qc_prefix: str = "QC"
    group_prefixes: tuple[str, str] = ("Contr", "Fibr")   # (control, case)
    group_labels: tuple[str, str] = ("Control", "Fibrosis")

    # Paired design: case/control samples are matched by their trailing index
    # (Contr-1 <-> Fibr-1, ...). Set False for an independent-groups design.
    paired: bool = True

    # ---- Preprocessing -----------------------------------------------------
    # Drop a feature if it is missing in more than this fraction of *biological*
    # samples (QC samples are not counted toward this filter).
    max_missing_frac: float = 0.50
    # Remove features whose coefficient of variation across QC samples exceeds
    # this value (analytical reproducibility filter). Set None to skip.
    qc_cv_threshold: Optional[float] = 0.30
    # Missing-value imputation strategy: "half_min" (1/2 of per-feature minimum),
    # "min", or "knn".
    impute_method: str = "half_min"
    # Sample normalization: "pqn" (probabilistic quotient), "median", "sum"
    # (total ion current) or None.
    normalization: str = "pqn"
    # Transform applied after normalization: "log2", "log10", "ln" or None.
    transform: str = "log2"
    # Scaling used *only* for multivariate analysis: "pareto", "auto" (z-score),
    # "range" or None.
    scaling: str = "pareto"

    # ---- Statistics --------------------------------------------------------
    alpha: float = 0.05                  # FDR significance threshold
    log2fc_threshold: float = 1.0        # |log2 fold change| for "biologically relevant"
    fdr_method: str = "bh"               # Benjamini-Hochberg
    min_samples_per_group: int = 3       # skip a feature with fewer finite values

    # ---- Multivariate ------------------------------------------------------
    pls_components: int = 2
    vip_threshold: float = 1.0

    # ---- Pathway analysis --------------------------------------------------
    # A pathway must contain at least this many *detected* metabolites to be
    # tested for enrichment.
    min_pathway_size: int = 3
    enrichment_alpha: float = 0.10

    seed: int = 42

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2, default=str)
