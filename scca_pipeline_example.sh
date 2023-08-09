#!/bin/bash
#SBATCH --job-name SCCA
#SBATCH --nodes 1
#SBATCH --nodelist node4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10GB
#SBATCH -o /scratch/connectome/seojw/"BNM analysis"/logs/output_log/SCCA_%A.o
#SBATCH -e /scratch/connectome/seojw/"BNM analysis"/logs/error_log/SCCA_%A.e

cd /scratch/connectome/seojw/"BNM analysis"/code/python_code
source activate bct_cca

python3 SCCA_hyper_param.py  --domain $1 --dataset $2 --exp_tag $3 \
&& python3 hyper_param_result_visualizer.py --domain $1 --dataset $2 --exp_tag $3 \
&& (python3 SCCA_permutation_test.py --domain $1 --dataset $2 --exp_tag $3 --n_perm 5000 \
& (python3 SCCA_bootstrap.py --domain $1 --dataset $2 --exp_tag $3 --n_boot 5000 --n_start 1 \
&& python3 SCCA_bootstrap_result_summary.py --domain $1 --dataset $2 --exp_tag $3 --n_boot 5000)) \
&& (python3 non_BNM_vis.py --domain $1 --exp_tag $3) \
&& (python3 bootstrap_vis_global_rich_club_BNM.py --domain $1 --exp_tag $3) \
&& (python3 SCCA_permutation_test.py --domain $1 --dataset testset --exp_tag $3 --n_perm 5000) \
&& (Rscript /scratch/connectome/seojw/"BNM analysis"/code/R_code/brain_visualize.R $1 $3)
