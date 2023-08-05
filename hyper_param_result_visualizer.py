import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--domain')
parser.add_argument('--dataset')
parser.add_argument('--exp_tag')

args = parser.parse_args()

domain = args.domain
exp_tag = args.exp_tag

param_result_dir = f'/storage/connectome/seojw/data/SCCA_param_tune_result/{domain}_BNM/first_comp_cov/{exp_tag}/param_result_{domain}_{exp_tag}.csv'

cv_result = pd.read_csv(param_result_dir, index_col=0)

col_list = cv_result.max() == cv_result.max().max()
row_list = cv_result.max(axis=1) == cv_result.max().max()
box_c1 = cv_result.loc[row_list, col_list].index[-1]
box_c2 = cv_result.loc[row_list, col_list].columns[0]
box_c1, box_c2
x1 = round((float(box_c1.split('_')[-1]) - 0.1) / 0.05)
x2 = round((float(box_c2.split('_')[-1]) - 0.1) / 0.05)



fig, ax = plt.subplots(figsize=(8,8)) 
sns.heatmap(cv_result, cbar_kws={'label': 'covariance'})
ax.figure.axes[-1].yaxis.label.set_size(12)
ax.add_patch(patches.Rectangle((x2, x1), 1.0, 1.0, edgecolor='skyblue',fill=False,lw=2))
plt.title(f'{domain}_{exp_tag}', fontsize=15)
plt.savefig(f'/storage/connectome/seojw/data/SCCA_param_tune_result/{domain}_BNM/first_comp_cov/{exp_tag}/param_result_{domain}_{exp_tag}.pdf')
plt.savefig(f'/storage/connectome/seojw/data/SCCA_param_tune_result/{domain}_BNM/first_comp_cov/{exp_tag}/param_result_{domain}_{exp_tag}.png')
