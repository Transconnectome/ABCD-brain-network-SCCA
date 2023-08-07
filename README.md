# ABCD-brain-network-SCCA
This repository contains code and synthetic data for Multivariate patterns between brain network properties, polygenic scores, phenotypes, and environment in preadolescents, Seo et al., (2023), medRxiv preprint, https://www.medrxiv.org/content/10.1101/2023.07.24.23293075v1

# SCCA Code
+ scca_pipeline_example.sh : pipeline for scca computation & visualization 
+ codes for computation
  + regress_out_train_test_split.py : regressing out potential confounding effects of covariates
  + SCCA_hyper_param.py : hyper-parameter tuning
  + SCCA_permutation_test.py : permutation test
  + SCCA_bootstrap.py : SCCA with bootstrap samples
  + SCCA_bootstrap_result_summary.py : summarize bootstrap results
+ codes for visualization
  + hyper_param_result_visualizer.py : visualization code for hyper-parameter tuning result
  + non_BNM_vis.py : visualization code for polygenic scores, phenotypic outcomes, environmental factors
  + brain_visualize.R : visualization code for nodal brain network measures
  + bootstrap_vis_global_rich_club_BNM.py : visualization code for global brain network measures

# Data
