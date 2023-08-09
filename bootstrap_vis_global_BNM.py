import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from scipy.stats import binom

parser = argparse.ArgumentParser()

parser.add_argument('--domain')
parser.add_argument('--exp_tag')
args = parser.parse_args()

domain = args.domain
exp_tag = args.exp_tag

block = 'y'

data_dir = f'/storage/connectome/seojw/data/PMD_boot_result/{domain}_BNM/first_comp_cov/{exp_tag}/summary_result'


# global measure extraction
y_loading_comp1_res = pd.read_csv(f'{data_dir}/bootstrap_result_summary_y_loading_comp1.csv', index_col=0)

ROI_list = pd.read_csv('/storage/connectome/seojw/data/ROI_list.csv',index_col=0).iloc[:, 0].values

global_measures = []
for i in range(y_loading_comp1_res.shape[0]):
    if y_loading_comp1_res.index[i].split('_')[-1] not in ROI_list:
        global_measures.append(y_loading_comp1_res.index[i])
if len(global_measures) > 300:
    ROI_list_temp = []
    for ROI in ROI_list:
        ROI_list_temp.append(ROI.replace('-', '.'))
    global_measures = []
    for i in range(y_loading_comp1_res.shape[0]):
        if y_loading_comp1_res.index[i].split('_')[-1] not in ROI_list_temp:
            global_measures.append(y_loading_comp1_res.index[i])

# boostrap result plotting
for metric in ['weight', 'loading', 'cross_loading']:
    for comp_num in range(1, 6):
        output_dir = f'{data_dir}/BNM_result/comp{comp_num}/ci_95/{metric}'
        os.makedirs(output_dir, exist_ok=True)

        data =  pd.read_csv(f'{data_dir}/bootstrap_result_summary_{block}_{metric}_comp{comp_num}.csv', index_col=0)
        
        if (domain == 'GPS') & (comp_num == 2):
            data.iloc[:, :-1] = -1 * data.iloc[:, :-1]

        var_num = len(data)
        data = data.loc[global_measures]

        lowbound_95 = data.iloc[:, 3]
        upperbound_95 = data.iloc[:, 4]
        error_bar_pos = (lowbound_95 + upperbound_95) / 2
        error_bar_size = (lowbound_95 - upperbound_95) / 2
        
        occurence_rate = data.iloc[:, -1]
        p = occurence_rate.sum() / var_num
        occurence_crit = binom.isf(0.001/var_num, 5000, p) / 5000

        significant_variable = ((lowbound_95 * upperbound_95) > 0) & (occurence_rate > occurence_crit)   # significant & consistent
        significant_variable_2 = ((lowbound_95 * upperbound_95) > 0)                       # only significant, not consistently selected

        # plot full global BNM
        fig = plt.figure(figsize=(16, 8))
        plt.scatter(np.arange(1, len(data)+1), data.iloc[:,0])
        plt.xticks(np.arange(1, len(data)+1), data.index, rotation=90)
        #plt.xticks(np.arange(1, len(data)+1)[significant_variable], data.index[significant_variable], rotation=90, c='red')
        plt.errorbar(x=np.arange(1, len(data)+1), y=error_bar_pos, yerr=error_bar_size, fmt='none', c='gray')
        plt.errorbar(x=np.arange(1, len(data)+1)[significant_variable_2], y=error_bar_pos[significant_variable_2],
                     yerr=error_bar_size[significant_variable_2], fmt='none', c='goldenrod')
        plt.errorbar(x=np.arange(1, len(data)+1)[significant_variable], y=error_bar_pos[significant_variable],
                     yerr=error_bar_size[significant_variable], fmt='none', c='red')

        plt.axhline(y=0, ls=':')
        plt.ylabel(f'{metric}')
        plt.title(f'{domain}_BNM_{exp_tag}_{metric}_comp{comp_num}')
        plt.tight_layout()
        for save_format in ['png', 'pdf', 'eps']:
            plt.savefig(f'{output_dir}/global_BNM_{metric}_comp{comp_num}_full.{save_format}', format=save_format)
        plt.close()


        # plot significant only
        fig = plt.figure(figsize=(8, 16))
        data_sig = data.loc[significant_variable]
        ranked_index = data_sig.iloc[:, 0].sort_values().index
        data_sig_ranked = data_sig.loc[ranked_index]
        error_bar_pos = (lowbound_95[ranked_index] + upperbound_95[ranked_index]) / 2
        error_bar_size = (upperbound_95[ranked_index] - lowbound_95[ranked_index]) / 2

        plt.barh(np.arange(len(data_sig)), data_sig_ranked.iloc[:, 0])
        plt.errorbar(x=error_bar_pos, y=np.arange(len(data_sig)), xerr=error_bar_size, fmt='none', c='black')

        plt.yticks(np.arange(len(data_sig)), data_sig_ranked.index)
        plt.xlabel(metric, fontsize=15)
        plt.title(f'{domain}_BNM_{exp_tag}_{metric}_comp{comp_num}')
        plt.tight_layout()
        plt.legend()

        for save_format in ['png', 'pdf', 'eps']:
            plt.savefig(f'{output_dir}/global_BNM_{metric}_comp{comp_num}_sigonly.{save_format}', format=save_format)
        plt.close()
