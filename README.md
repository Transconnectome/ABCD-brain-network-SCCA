# ABCD-brain-network-SCCA

This repository contains code and synthetic data for the research paper titled **"Multivariate patterns between brain network properties, polygenic scores, phenotypes, and environment in preadolescents"** by Seo et al. (2023). The paper is available as a medRxiv preprint at: https://www.medrxiv.org/content/10.1101/2023.07.24.23293075v1

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
- `bootstrap_vis_global_BNM.py`: Visualization code for SCCA result of global brain network measures.

## Data
- `DTI_node_coordinate.csv`, `ENV_Pheno_separate.csv`, `ROI_list.csv` : Data required to visualize SCCA results
- `cov_synthetic.csv` : Covariates data
- `pgs_synthetic.csv`, `env_synthetic.csv`, `pheno_synthetic.csv`, `BNM_1-4_synthetic.csv` : Main data

Since the NDA prohibits the distribution of ABCD data, we are sharing synthetic data in place of real data. The original data is EXCLUSIVELY accessible through the NDA (https://nda.nih.gov/abcd/). This synthetic data was produced using conditional GAN for tabular data, or CTGAN, which learns the distribution of real data and generates synthetic data that emulates the probabilistic distribution (Xu et al., 2019). The synthetic data showed an overall quality score of 91.49%, indicating an overall similarity of column and column pair shapes between real and synthetic data is 0.9149. However, it is important to note that this synthetic data does not guarantee identical analysis results. It may lead to different statistical significance and effect sizes. Hyperparameters were optimized using Optuna (Akiba et al., 2019).

## Reference
Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019, July). Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 2623-2631).

Witten, D. M., Tibshirani, R., & Hastie, T. (2009). A penalized matrix decomposition, with applications to sparse principal components and canonical correlation analysis. Biostatistics, 10(3), 515-534. https://doi.org/10.1093/biostatistics/kxp008 

Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). Modeling tabular data using conditional gan. Advances in Neural Information Processing Systems, 32.
