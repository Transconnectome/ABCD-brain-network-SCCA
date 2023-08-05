import pandas as pd
import numpy as np
import os
import time
from cca_zoo.models import SCCA_PMD
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
import pingouin as pg
###############################################################

# argumented variable setting
parser = argparse.ArgumentParser()

parser.add_argument('--domain')
parser.add_argument('--dataset')
parser.add_argument('--exp_tag')
parser.add_argument('--n_boot')

args = parser.parse_args()

param_tune_scheme = 'first_comp_cov'
domain = args.domain
dataset = args.dataset
exp_tag = args.exp_tag
n_boot = int(args.n_boot)

ncomp = 5


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
density = pd.DataFrame(cca_y['density'])

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


# calculate SCCA with original dataset for reordering bootstrap result
scca_org = (SCCA_PMD(c=[c1, c2], latent_dims=ncomp).fit([cca_x, cca_y]))

a_org = scca_org.weights[0]
b_org = scca_org.weights[1]

org_x_loading = scca_org.get_loadings([cca_x, cca_y], normalize=True)[0]
org_y_loading = scca_org.get_loadings([cca_x, cca_y], normalize=True)[1]

org_cca_u = scca_org.transform([cca_x, cca_y])[0]
org_cca_v = scca_org.transform([cca_x, cca_y])[1]


org_x_cross_loading = pd.DataFrame(np.zeros((len(cca_x.iloc[0]), ncomp)), index=cca_x.columns)
org_y_cross_loading = pd.DataFrame(np.zeros((len(cca_y.iloc[0]), ncomp)), index=cca_y.columns)

for comp_num in range(ncomp):
    for variable_num in range(len(cca_x.iloc[0])):
        org_x_cross_loading.iloc[variable_num, comp_num] = np.corrcoef(cca_x.iloc[:, variable_num], org_cca_v[:, comp_num])[0, 1]
    for variable_num in range(len(cca_y.iloc[0])):
        org_y_cross_loading.iloc[variable_num ,comp_num] = np.corrcoef(cca_y.iloc[:, variable_num], org_cca_u[:, comp_num])[0, 1]

section_num = 50

for block in ['x', 'y']:
    for metric in ['weight', 'loading', 'cross_loading']:
        for comp_num in range(1, ncomp + 1):
            globals()[f'cca_{block}_{metric}_comp{comp_num}'] = pd.DataFrame()
            for section in range(section_num):
                globals()[f'cca_{block}_{metric}_comp{comp_num}_{section}'] = pd.DataFrame()

cca_bootstrap_result_dir = f'/storage/connectome/seojw/data/PMD_boot_result/{domain}_BNM/{param_tune_scheme}/{exp_tag}/bootstrap_data/'
saving_dir = f'/storage/connectome/seojw/data/PMD_boot_result/{domain}_BNM/{param_tune_scheme}/{exp_tag}/summary_result/'
os.makedirs(saving_dir, exist_ok=True)


# bootstrap result summarize
for section in range(section_num):
    init_num = section * int(n_boot/section_num) + 1
    fin_num = (section + 1) * int(n_boot/section_num) + 1
    for file_num in range(init_num, fin_num):
        cca_x_weight = pd.read_csv(cca_bootstrap_result_dir + f'cca_x_weight_{file_num}.csv', index_col=0)
        cca_y_weight = pd.read_csv(cca_bootstrap_result_dir + f'cca_y_weight_{file_num}.csv', index_col=0)
        cca_x_loading = pd.read_csv(cca_bootstrap_result_dir + f'cca_x_loading_{file_num}.csv', index_col=0)
        cca_y_loading = pd.read_csv(cca_bootstrap_result_dir + f'cca_y_loading_{file_num}.csv', index_col=0)
        cca_x_cross_loading = pd.read_csv(cca_bootstrap_result_dir + f'cca_x_cross_loading_{file_num}.csv', index_col=0)
        cca_y_cross_loading = pd.read_csv(cca_bootstrap_result_dir + f'cca_y_cross_loading_{file_num}.csv', index_col=0)
        
        # reordering
        u_sim = a_org.T @ cca_x_weight
        v_sim = b_org.T @ cca_y_weight
        match = (np.abs(u_sim) + np.abs(v_sim)).T.idxmax()

        sign_list = np.zeros(ncomp)
        for i in range(ncomp):
            sign_list[i] = np.sign((u_sim + v_sim).iloc[i, int(match[i])])
        reordered_x_weight = (cca_x_weight.loc[:, match] * sign_list)
        reordered_y_weight = (cca_y_weight.loc[:, match] * sign_list)
        reordered_x_loading = (cca_x_loading.loc[:, match] * sign_list)
        reordered_y_loading = (cca_y_loading.loc[:, match] * sign_list)
        reordered_x_cross_loading = (cca_x_cross_loading.loc[:, match] * sign_list)
        reordered_y_cross_loading = (cca_y_cross_loading.loc[:, match] * sign_list)
        
        for block in ['x', 'y']:
            for metric in ['weight', 'loading', 'cross_loading']:
                for comp_num in range(1, ncomp+1):
                    globals()[f'cca_{block}_{metric}_comp{comp_num}_{section}'] = pd.concat([globals()[f'cca_{block}_{metric}_comp{comp_num}_{section}'], globals()[f'reordered_{block}_{metric}'].iloc[:, comp_num-1]], axis=1)

    for block in ['x', 'y']:
        for metric in ['weight', 'loading', 'cross_loading']:
            for comp_num in range(1, ncomp+1):
                globals()[f'cca_{block}_{metric}_comp{comp_num}'] = pd.concat([globals()[f'cca_{block}_{metric}_comp{comp_num}'], globals()[f'cca_{block}_{metric}_comp{comp_num}_{section}']], axis=1)


for block in ['x', 'y']:
    for metric in ['weight', 'loading', 'cross_loading']:
        for comp_num in range(1, ncomp + 1):

            estimate_list = {'x_weight':a_org, 'y_weight':b_org, 'x_loading':org_x_loading, 'y_loading':org_y_loading,
                             'x_cross_loading':org_x_cross_loading.values, 'y_cross_loading':org_y_cross_loading.values, 
                             'x_density_partial_loading':org_x_density_partial_loading.values, 'y_density_partial_loading':org_y_density_partial_loading.values,
                             'x_density_partial_cross_loading':org_x_density_partial_cross_loading.values, 'y_density_partial_cross_loading':org_y_density_partial_cross_loading.values}
            estimate = estimate_list[f'{block}_{metric}'][:, comp_num - 1]
            low_68 = globals()[f'cca_{block}_{metric}_comp{comp_num}'].quantile(0.5 - 0.34, axis=1)
            upper_68 = globals()[f'cca_{block}_{metric}_comp{comp_num}'].quantile(0.5 + 0.34, axis=1)
            low_95 = globals()[f'cca_{block}_{metric}_comp{comp_num}'].quantile(0.5 - 0.95/2, axis=1)
            upper_95 = globals()[f'cca_{block}_{metric}_comp{comp_num}'].quantile(0.5 + 0.95/2, axis=1)
            low_99 = globals()[f'cca_{block}_{metric}_comp{comp_num}'].quantile(0.5 - 0.99/2, axis=1)
            upper_99 = globals()[f'cca_{block}_{metric}_comp{comp_num}'].quantile(0.5 + 0.99/2, axis=1)
            low_997 = globals()[f'cca_{block}_{metric}_comp{comp_num}'].quantile(0.5 - 0.997/2, axis=1)
            upper_997 = globals()[f'cca_{block}_{metric}_comp{comp_num}'].quantile(0.5 + 0.997/2, axis=1)
            consistency = np.abs(np.sign(globals()[f'cca_{block}_weight_comp{comp_num}']).sum(axis=1)) / n_boot
            globals()[f'cca_{block}_{metric}_comp{comp_num}_summary'] = pd.DataFrame({f'{metric}_comp{comp_num}_estimate':estimate,
                                                                                      f'{metric}_comp{comp_num}_68%_low':low_68,
                                                                                      f'{metric}_comp{comp_num}_68%_upper':upper_68,
                                                                                      f'{metric}_comp{comp_num}_95%_low':low_95,
                                                                                      f'{metric}_comp{comp_num}_95%_upper':upper_95,
                                                                                      f'{metric}_comp{comp_num}_99%_low':low_99,
                                                                                      f'{metric}_comp{comp_num}_99%_upper':upper_99,
                                                                                      f'{metric}_comp{comp_num}_99.7%_low':low_997,
                                                                                      f'{metric}_comp{comp_num}_99.7%_upper':upper_997,
                                                                                      f'{metric}_comp{comp_num}_consistency':consistency},
                         index=globals()[f'cca_{block}_{metric}_comp{comp_num}'].index)
            globals()[f'cca_{block}_{metric}_comp{comp_num}_summary'].to_csv(saving_dir + f'bootstrap_result_summary_{block}_{metric}_comp{comp_num}.csv')
