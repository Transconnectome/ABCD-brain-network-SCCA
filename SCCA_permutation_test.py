# import packages
import sys
import os
from cca_zoo.models import SCCA_PMD

import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from statsmodels.stats.multitest import fdrcorrection
import seaborn as sns

import argparse

import matplotlib.pyplot as plt

def PMD_permutation_test(cca_x, cca_y, penalty, n_comp, n_perm, dataset, saving_dir, file_name):
    null_cov_pd = pd.DataFrame()
    null_x_var_pd = pd.DataFrame()
    null_y_var_pd = pd.DataFrame()
    null_cor_pd = pd.DataFrame()
    
    
    if dataset == 'trainset':
        for i in range(1, n_perm+1):

            init = time.time()

            cca_x_perm = np.random.permutation(cca_x)
            cca_perm = SCCA_PMD(c=penalty, latent_dims=n_comp).fit([cca_x_perm, cca_y])

            null_cov = np.zeros(n_comp)
            null_x_var = np.zeros(n_comp)
            null_y_var = np.zeros(n_comp)
            for comp_num in range(n_comp):
                cov_mat = np.cov(cca_perm.transform([cca_x_perm, cca_y])[0][:, comp_num], cca_perm.transform([cca_x_perm, cca_y])[1][:, comp_num])
                null_cov[comp_num] = cov_mat[0, 1]
                null_x_var[comp_num] = cov_mat[0, 0]
                null_y_var[comp_num] = cov_mat[1,1]

            null_cor = cca_perm.score([cca_x_perm, cca_y])

            null_cov_pd = pd.concat([null_cov_pd, pd.DataFrame(null_cov, index=[f'cov_comp={comp_num}' for comp_num in range(1, n_comp+1)])], axis=1)
            null_x_var_pd = pd.concat([null_x_var_pd, pd.DataFrame(null_x_var, index=[f'x_var_comp={comp_num}' for comp_num in range(1, n_comp+1)])], axis=1)
            null_y_var_pd = pd.concat([null_y_var_pd, pd.DataFrame(null_y_var, index=[f'y_var_comp={comp_num}' for comp_num in range(1, n_comp+1)])], axis=1)
            null_cor_pd = pd.concat([null_cor_pd, pd.DataFrame(null_cor, index=[f'corr_comp={comp_num}' for comp_num in range(1, n_comp+1)])], axis=1)
    elif dataset == 'testset':

        cca_x_train_file_name = file_dir + f'regressed_{domain}_BNM_CCA_trainset_{domain}_{exp_tag}.csv'
        cca_y_train_file_name = file_dir + f'regressed_{domain}_BNM_CCA_trainset_BNM_{exp_tag}.csv'
        cca_x_train = pd.read_csv(cca_x_train_file_name, index_col=0)
        cca_y_train = pd.read_csv(cca_y_train_file_name, index_col=0)

        # make sure the input data are zscaled before run SCCA
        scaler = StandardScaler()
        cca_x_train = pd.DataFrame(scaler.fit_transform(cca_x_train), index=cca_x_train.index, columns=cca_x_train.columns)
        cca_y_train = pd.DataFrame(scaler.fit_transform(cca_y_train), index=cca_y_train.index, columns=cca_y_train.columns)
        
        cca_train_res = SCCA_PMD(c=penalty, latent_dims=n_comp).fit([cca_x_train, cca_y_train])
        for i in range(1, n_perm+1):
            cca_x_perm = np.random.permutation(cca_x)
            cca_perm_res_x, cca_perm_res_y = cca_train_res.transform([cca_x_perm, cca_y])
            
            null_cov = np.zeros(n_comp)
            null_x_var = np.zeros(n_comp)
            null_y_var = np.zeros(n_comp)
            for comp_num in range(n_comp):
                cov_mat = np.cov(cca_perm_res_x[:, comp_num], cca_perm_res_y[:, comp_num])
                null_cov[comp_num] = cov_mat[0, 1]
                null_x_var[comp_num] = cov_mat[0, 0]
                null_y_var[comp_num] = cov_mat[1,1]

            null_cor = cca_train_res.score([cca_x_perm, cca_y])

            null_cov_pd = pd.concat([null_cov_pd, pd.DataFrame(null_cov, index=[f'cov_comp={comp_num}' for comp_num in range(1, n_comp+1)])], axis=1)
            null_x_var_pd = pd.concat([null_x_var_pd, pd.DataFrame(null_x_var, index=[f'x_var_comp={comp_num}' for comp_num in range(1, n_comp+1)])], axis=1)
            null_y_var_pd = pd.concat([null_y_var_pd, pd.DataFrame(null_y_var, index=[f'y_var_comp={comp_num}' for comp_num in range(1, n_comp+1)])], axis=1)
            null_cor_pd = pd.concat([null_cor_pd, pd.DataFrame(null_cor, index=[f'corr_comp={comp_num}' for comp_num in range(1, n_comp+1)])], axis=1)
            
            
    null_cor_pd = null_cor_pd.T
    null_cor_pd.index = [f'{i}_th_null' for i in range(1, n_perm+1)]
    null_cov_pd = null_cov_pd.T
    null_cov_pd.index = [f'{i}_th_null' for i in range(1, n_perm+1)]
    null_x_var_pd = null_x_var_pd.T
    null_x_var_pd.index = [f'{i}_th_null' for i in range(1, n_perm+1)]
    null_y_var_pd = null_y_var_pd.T
    null_y_var_pd.index = [f'{i}_th_null' for i in range(1, n_perm+1)]

    null_info_pd = pd.concat([null_cor_pd, null_cov_pd, null_x_var_pd, null_y_var_pd], axis=1)

    null_cor_mean = null_cor_pd.mean(axis=0)
    null_cor_std = null_cor_pd.std(axis=0)
    null_cov_mean = null_cov_pd.mean(axis=0)
    null_cov_std = null_cov_pd.std(axis=0)

    if dataset == 'trainset':
        cca_res = SCCA_PMD(c=penalty, latent_dims=n_comp).fit([cca_x, cca_y])
        corr = cca_res.score([cca_x, cca_y])
        cov = np.zeros(n_comp)
        for comp_num in range(n_comp):
            cov_mat = np.cov(cca_res.transform([cca_x, cca_y])[0][:, comp_num], cca_res.transform([cca_x, cca_y])[1][:, comp_num])
            cov[comp_num] = cov_mat[0, 1]
    elif dataset == 'testset':
        corr = cca_train_res.score([cca_x, cca_y])
        cov = np.zeros(n_comp)
        for comp_num in range(n_comp):
            cov_mat = np.cov(cca_train_res.transform([cca_x, cca_y])[0][:, comp_num], cca_train_res.transform([cca_x, cca_y])[1][:, comp_num])
            cov[comp_num] = cov_mat[0, 1]

    pval_cov = (null_cov_pd > cov).sum() / n_perm
    pval_cov.index = [f'comp{i}' for i in range(1, 6)]
    fdr_pval_cov = fdrcorrection(np.array(pval_cov))[1]
    fdr_pval_cov = pd.DataFrame(fdr_pval_cov, index=[f'comp{i}' for i in range(1, 6)])
    zstat_cov = (cov - null_cov_mean) / null_cov_std
    zstat_cov.index = [f'comp{i}' for i in range(1, 6)]
    cov_pd = pd.DataFrame(cov, index=[f'comp{i}' for i in range(1, 6)])

    summary_df = pd.concat([pval_cov, fdr_pval_cov, cov_pd, zstat_cov], axis=1)
    summary_df.columns = ['pval_cov', 'fdr_pval_cov', 'cov', 'zstat_cov']

    print(summary_df)

    for i in range(n_comp):

        if dataset == 'trainset':

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(1,1,1)
            ax.hist(null_cov_pd.iloc[:, i], label=f'pval={pval_cov[i]:.3f}, zstat={zstat_cov[i]:.2f}')
            ax.axvline(x=cov[i], ls=':', color='r')
            plt.title('null_cov_dist_'+f'n_perm={n_perm}_'+file_name+f'_comp{i}')
            plt.xlabel('covariance', fontsize=15)
            plt.ylabel('# of permutation set', fontsize=15)
            plt.legend()

            fig.savefig(saving_dir + 'null_cov_dist_'+f'n_perm={n_perm}_'+file_name+f'_comp{i}.pdf')
            fig.savefig(saving_dir + 'null_cov_dist_'+f'n_perm={n_perm}_'+file_name+f'_comp{i}.png')

        elif dataset == 'testset':

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(1,1,1)
            ax.hist(null_cov_pd.iloc[:, i], label=f'pval={pval_cov[i]:.3f}, zstat={zstat_cov[i]:.2f}')
            ax.axvline(x=cov[i], ls=':', color='r')
            plt.title('null_cov_dist_'+f'n_perm={n_perm}_'+file_name+f'_comp{i}_rep')
            plt.xlabel('covariance', fontsize=15)
            plt.ylabel('# of permutation set', fontsize=15)
            plt.legend()

            fig.savefig(saving_dir + 'null_cov_dist_'+f'n_perm={n_perm}_'+file_name+f'_comp{i}_rep.pdf')
            fig.savefig(saving_dir + 'null_cov_dist_'+f'n_perm={n_perm}_'+file_name+f'_comp{i}_rep.png')

    return null_info_pd, summary_df

#######################################################################################################################

#####################################################################################
######### main body #################################################################
#####################################################################################

parser = argparse.ArgumentParser()

parser.add_argument('--domain')
parser.add_argument('--dataset')
parser.add_argument('--exp_tag')
parser.add_argument('--n_perm', required=True, help="type n_perm")
  
args = parser.parse_args()

domain = args.domain
dataset = args.dataset
exp_tag = args.exp_tag
n_perm = int(args.n_perm)

# import dataset
file_dir = f'/storage/connectome/seojw/data/SCCA_dataset/{domain}_BNM/'

cca_x_file_name = file_dir + f'regressed_{domain}_BNM_CCA_{dataset}_{domain}_{exp_tag}.csv'
cca_y_file_name = file_dir + f'regressed_{domain}_BNM_CCA_{dataset}_BNM_{exp_tag}.csv'
cca_x = pd.read_csv(cca_x_file_name, index_col=0)
cca_y = pd.read_csv(cca_y_file_name, index_col=0)

# make sure the input data are zscaled before run SCCA
scaler = StandardScaler()
cca_x = pd.DataFrame(scaler.fit_transform(cca_x), index=cca_x.index, columns=cca_x.columns)
cca_y = pd.DataFrame(scaler.fit_transform(cca_y), index=cca_y.index, columns=cca_y.columns)

# import hyper parameter tuning result
param_result_dir = f'/storage/connectome/seojw/data/SCCA_param_tune_result/{domain}_BNM/first_comp_cov/{exp_tag}/param_result_{domain}_{exp_tag}.csv'
tune_result = pd.read_csv(param_result_dir, index_col=0)

# optimal parameter load
col_list = tune_result.max() == tune_result.max().max()
row_list = tune_result.max(axis=1) == tune_result.max().max()
box_c1 = tune_result.loc[row_list, col_list].index[-1]
box_c2 = tune_result.loc[row_list, col_list].columns[0]
c1 = float(box_c1.split('_')[-1])
c2 = float(box_c2.split('_')[-1])


saving_dir = f'/storage/connectome/seojw/data/SCCA_permutation_test_result/{domain}_BNM/first_comp_cov/{dataset}/{exp_tag}/'
os.makedirs(saving_dir, exist_ok=True)

saving_file_name = f'{domain}_{dataset}_{exp_tag}_{n_perm}'

null_info_pd, summary_df = PMD_permutation_test(cca_x, cca_y, [c1, c2], 5, n_perm, dataset, saving_dir, saving_file_name)

if dataset == 'trainset':
    null_info_pd.to_csv(saving_dir + 'null_info_pd_'+saving_file_name+'.csv')
    summary_df.to_csv(saving_dir + 'permutation_test_summary_'+saving_file_name+'.csv')
elif dataset == 'testset':
    null_info_pd.to_csv(saving_dir + 'null_info_pd_'+saving_file_name+'_rep.csv')
    summary_df.to_csv(saving_dir + 'permutation_test_summary_'+saving_file_name+'_rep.csv')
