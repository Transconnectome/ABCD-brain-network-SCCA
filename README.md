# ABCD-brain-network-SCCA

This repository contains code and synthetic data for the research paper titled "Multivariate patterns between brain network properties, polygenic scores, phenotypes, and environment in preadolescents" by Seo et al. (2023). The paper is available as a medRxiv preprint at: https://www.medrxiv.org/content/10.1101/2023.07.24.23293075v1

## SCCA Code

This section describes the codes for Sparse Canonical Correlation Analysis (Witten et al., 2009) used in this paper and and its visualization. The code is organized into different scripts:

- `scca_pipeline_example.sh`: A pipeline for SCCA computation and visualization.

**Codes for computation:**

- `regress_out_train_test_split.py`: Code for regressing out potential confounding effects of covariates before SCCA.
- `SCCA_hyper_param.py`: Code for hyper-parameter tuning in SCCA.
- `SCCA_permutation_test.py`: Code for performing permutation tests.
- `SCCA_bootstrap.py`: Code for performing SCCA with bootstrap samples.
- `SCCA_bootstrap_result_summary.py`: Code for summarizing bootstrap results.

**Codes for visualization:**

- `hyper_param_result_visualizer.py`: Visualization code for hyper-parameter tuning results.
- `non_BNM_vis.py`: Visualization code for SCCA result (loading) of polygenic scores, phenotypic outcomes, and environmental factors.
- `brain_visualize.R`: Visualization code for SCCA result of nodal brain network measures.
- `bootstrap_vis_global_rich_club_BNM.py`: Visualization code for SCCA result of global brain network measures.

## Data
