"""mbx - a self-contained untargeted-metabolomics analysis toolkit.

Pipeline stages:
    1. preprocessing  - QC filtering, missing-value handling, normalization, transform, scaling
    2. statistics     - paired/unpaired differential abundance, FDR, fold change, ROC
    3. multivariate   - PCA and PLS-DA (with VIP scores)
    4. pathways       - over-representation analysis + metabolite/gene mapping
    5. interpretation - automated biological narrative
    6. plotting       - publication-style figures

The package was written for a HILIC untargeted-metabolomics study comparing
fibrotic vs. control atrial tissue (paired design), but is dataset-agnostic.
"""

__version__ = "1.0.0"

RANDOM_SEED = 42
