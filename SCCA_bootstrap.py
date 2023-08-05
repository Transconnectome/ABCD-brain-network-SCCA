# import packages
import sys
import os

from cca_zoo.models import SCCA_PMD

import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA

import pingouin as pg
import seaborn as sns

import argparse

import matplotlib.pyplot as plt

############################################
########## define custom function ##########
############################################

def PMD_bootstrap(cca_x, cca_y, penalty, n_comp, n_boot, n_start, n_step, save_output_dir_path, dataset):
    cca_x_variable = cca_x.columns
    cca_y_variable = cca_y.columns

    i = n_start
    while i <= n_boot:
        try:

            cca_x_boot = cca_x.sample(len(cca_x), replace=True)
            cca_y_boot = cca_y.loc[cca_x_boot.index]
            density_boot = pd.DataFrame(cca_y_boot['density'])
            
            cca_boot = SCCA_PMD(c=penalty, latent_dims=n_comp, max_iter=200).fit([cca_x_boot, cca_y_boot])
            cca_x_weight = pd.DataFrame(cca_boot.weights[0], index=cca_x_variable)
            cca_y_weight = pd.DataFrame(cca_boot.weights[1], index=cca_y_variable)
            cca_x_loading = pd.DataFrame(cca_boot.get_loadings([cca_x_boot, cca_y_boot], normalize=True)[0], index=cca_x_variable)
            cca_y_loading = pd.DataFrame(cca_boot.get_loadings([cca_x_boot, cca_y_boot], normalize=True)[1], index=cca_y_variable)
            cca_u = pd.DataFrame(cca_boot.transform([cca_x_boot, cca_y_boot])[0], index=cca_x_boot.index, columns=[f'u_{i}' for i in range(n_comp)])
            cca_v = pd.DataFrame(cca_boot.transform([cca_x_boot, cca_y_boot])[1], index=cca_x_boot.index, columns=[f'v_{i}' for i in range(n_comp)])

            # data concat for partial loading with covar=density
            data_x = pd.concat([cca_u, cca_v, cca_x_boot, density_boot], axis=1)
            data_y = pd.concat([cca_u, cca_v, cca_y_boot], axis=1)

            cca_x_cross_loading = pd.DataFrame(np.zeros((len(cca_x_boot.iloc[0]), n_comp)), index=cca_x_variable)
            cca_y_cross_loading = pd.DataFrame(np.zeros((len(cca_y_boot.iloc[0]), n_comp)), index=cca_y_variable)

            for comp_num in range(n_comp):
                for variable_num, variable_name in enumerate(cca_x_variable):
                    cca_x_cross_loading.iloc[variable_num, comp_num] = np.corrcoef(cca_x_boot.iloc[:, variable_num], cca_v.iloc[:, comp_num])[0, 1]

                for variable_num, variable_name in enumerate(cca_y_variable):
                    cca_y_cross_loading.iloc[variable_num ,comp_num] = np.corrcoef(cca_y_boot.iloc[:, variable_num], cca_u.iloc[:, comp_num])[0, 1]
                    if variable_name == 'density':
                        continue

            if dataset == 'trainset':
                cca_x_weight.to_csv(save_output_dir_path+f'/cca_x_weight_{i}.csv')
                cca_y_weight.to_csv(save_output_dir_path+f'/cca_y_weight_{i}.csv')
                cca_x_loading.to_csv(save_output_dir_path+f'/cca_x_loading_{i}.csv')
                cca_y_loading.to_csv(save_output_dir_path+f'/cca_y_loading_{i}.csv')
                cca_x_cross_loading.to_csv(save_output_dir_path+f'/cca_x_cross_loading_{i}.csv')
                cca_y_cross_loading.to_csv(save_output_dir_path+f'/cca_y_cross_loading_{i}.csv')

            elif dataset == 'testset':
                cca_x_weight.to_csv(save_output_dir_path+f'/cca_x_weight_rep_{i}.csv')
                cca_y_weight.to_csv(save_output_dir_path+f'/cca_y_weight_rep_{i}.csv')
                cca_x_loading.to_csv(save_output_dir_path+f'/cca_x_loading_rep_{i}.csv')
                cca_y_loading.to_csv(save_output_dir_path+f'/cca_y_loading_rep_{i}.csv')
                cca_x_cross_loading.to_csv(save_output_dir_path+f'/cca_x_cross_loading_rep_{i}.csv')
                cca_y_cross_loading.to_csv(save_output_dir_path+f'/cca_y_cross_loading_rep_{i}.csv')
                
            i = i + 1
            
        except:
            print(f'i = {i}, SVD did not converge')


####################################################################################

####################################################################################
#############################      main body        ################################
####################################################################################

parser = argparse.ArgumentParser()

parser.add_argument('--domain')
parser.add_argument('--dataset')
parser.add_argument('--exp_tag')
parser.add_argument('--n_boot', required=True, help="type n_perm")
parser.add_argument('--n_start', required=True, help="type bootstrap start serial number")

args = parser.parse_args()

domain = args.domain
dataset = args.dataset
exp_tag = args.exp_tag
n_boot = int(args.n_boot)
n_start = int(args.n_start)
param_tune_scheme = 'first_comp_cov'

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

saving_dir = f'/storage/connectome/seojw/data/PMD_boot_result/{domain}_BNM/{param_tune_scheme}/{exp_tag}/bootstrap_data'

os.makedirs(saving_dir, exist_ok=True)

# run permutation test and save the result
PMD_bootstrap(cca_x, cca_y, [c1, c2], 5, n_boot, n_start, 1, saving_dir, dataset)




