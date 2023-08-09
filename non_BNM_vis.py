import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
from scipy.stats import binom

parser = argparse.ArgumentParser()

parser.add_argument('--domain')
parser.add_argument('--exp_tag')

args = parser.parse_args()

domain = args.domain
exp_tag = args.exp_tag

block = 'x'

input_data_dir = f'/storage/connectome/seojw/data/PMD_boot_result/{domain}_BNM/first_comp_cov/{exp_tag}/summary_result'
output_dir = f'{input_data_dir}/non_BNM_plot'
os.makedirs(output_dir, exist_ok=True)


if domain == 'GPS':
    psychi_dis = {'DEPeur4':'DEP', 'MDDeur6':'MDD', 'ALCDEP_EURauto':'ALCDEP',
                        'ASDauto':'ASD', 'BIPauto':'BIP', 'CROSSauto':'CROSS', 'EDauto':'ED',
                        'OCDauto':'OCD', 'SCZ_EURauto':'SCZ', 'PTSDeur4':'PTSD', 'ADHDeur6':'ADHD'}
    mental_traits = {'Happieur4':'SWB', 'GHappieur2':'GHappi', 
                            'GHappiMeaneur1':'GHappi_Mean', 'GHappiHealth6':'GHappi_Health',
                            'NEUROTICISMauto':'Neuroticism', 'WORRYauto':'Worry', 
                            'ANXIETYauto':'Anxiety'}
    behavioral_traits = {'ASPauto':'ASP', 'CANNABISauto':'Cannabis',
                                'RISK4PCauto':'Risky_Behav', 'RISKTOLauto':'RiskTol', 
                                'SMOKERauto':'Smoke', 'DRINKauto':'Drinking'}
    cognitive_ability = {'CPeur2':'CP', 'EAeur1':'EA', 'IQeur2':'IQ'}
    physical_health = {'BMIeur4':'BMI', 'INSOMNIAeur6':'Insomnia', 'SNORINGeur1':'Snoring'}

    categories = [psychi_dis, mental_traits, behavioral_traits, cognitive_ability,
                physical_health]
    category_name_list = ['psychiatric disorder', 'mental traits', 'behavioral traits', 'cognitive ability', 'physical health']

else:
    if domain == 'pheno':
        ori_var = pd.read_csv('/storage/connectome/seojw/data/ENV_Pheno_separate.csv', header=None)[0].dropna()
        var_name = pd.read_csv('/storage/connectome/seojw/data/ENV_Pheno_separate.csv', header=None)[1].dropna()
        cat_info = pd.read_csv('/storage/connectome/seojw/data/ENV_Pheno_separate.csv', header=None)[2].dropna()


    if domain == 'ENV':
        ori_var = pd.read_csv('/storage/connectome/seojw/data/ENV_Pheno_separate.csv', header=None)[3].dropna()
        var_name = pd.read_csv('/storage/connectome/seojw/data/ENV_Pheno_separate.csv', header=None)[4].dropna()
        cat_info = pd.read_csv('/storage/connectome/seojw/data/ENV_Pheno_separate.csv', header=None)[5].dropna()
                

    df = pd.concat([ori_var, var_name, cat_info], axis=1)
    df.columns = ['ori_var', 'var_name', 'category']

    for cat in cat_info.unique():
        globals()[f'{cat}'] = {}
        cond = df['category'] == cat
        for i in range(sum(cond)):
            globals()[f'{cat}'][df.loc[cond].iloc[i, 0]] = df.loc[cond].iloc[i, 1]

    if domain == 'pheno':
        categories = [psychopathology, psychological_traits, cognition, medical_history,
                    physical_activity, substance_use, early_development, life_style]
        category_name_list = ['psychopathology', 'psychological traits', 'cognition', 'medical history', 'physical activity',
                            'substance use', 'early development', 'life style']
    if domain == 'ENV':
        categories = [parent_psychopathology, perinatal_events,
                            neighborhood_environment, parent_trait, family_environment,
                            school_environment, family_history]
        category_name_list =['parent_psychopathology', 'perinatal_events',
                            'neighborhood_environment', 'parent_trait', 'family_environment',
                            'school_environment', 'family_history']
                        
for metric in ['weight', 'loading', 'cross_loading']:
    for comp_num in range(1,6):

        data = pd.read_csv(f'{input_data_dir}/bootstrap_result_summary_{block}_{metric}_comp{comp_num}.csv', index_col=0)
        
        if (domain == 'GPS') & (comp_num == 2):
            data.iloc[:, :-1] = data.iloc[:, :-1] * -1

        var_num = len(data)
        occurence_rate = data.iloc[:, -1]
        p = occurence_rate.sum() / var_num
        occurence_crit = binom.isf(0.001/var_num, 5000, p) / 5000
        
        print(occurence_crit)
        # full plot
        fig = plt.figure(figsize=(16, 8))

        xticklabels = []
        i = 1
        for idx, category in enumerate(categories):
            cat_filter_data = pd.DataFrame(data.loc[category.keys()])
            cat_filter_data.index = category.values()

            lowbound_95 = cat_filter_data.iloc[:, 3]
            upperbound_95 = cat_filter_data.iloc[:, 4]
            error_bar_pos = (lowbound_95 + upperbound_95) / 2
            error_bar_size = (lowbound_95 - upperbound_95) / 2
            occurence_rate = cat_filter_data.iloc[:, -1]
            significant_variable = ((lowbound_95 * upperbound_95) > 0) & (occurence_rate > occurence_crit)   # significant & consistent
            significant_variable_2 = ((lowbound_95 * upperbound_95) > 0)                       # only significant, not consistently selected

            plt.scatter(x=np.arange(i, len(cat_filter_data) + i), y=cat_filter_data.iloc[:,0], s=150, label=category_name_list[idx])
            plt.errorbar(x=np.arange(i, len(cat_filter_data) + i), y=error_bar_pos, yerr=error_bar_size, fmt='none', c='gray')
            plt.errorbar(x=np.arange(i, len(cat_filter_data) + i)[significant_variable_2], y=error_bar_pos[significant_variable_2], 
                            yerr=error_bar_size[significant_variable_2], fmt='none', c='goldenrod')
            plt.errorbar(x=np.arange(i, len(cat_filter_data) + i)[significant_variable], y=error_bar_pos[significant_variable], 
                            yerr=error_bar_size[significant_variable], fmt='none', c='red')

            xticklabels += list(cat_filter_data.index)

            i += len(cat_filter_data)

        if domain == 'GPS':
            fontsize=15
        else:
            fontsize=10

        plt.axhline(y=0, ls=':')
        plt.xticks(np.arange(1, i), xticklabels, rotation=90, fontsize=fontsize)
        plt.ylabel(metric, fontsize=15)
        plt.title(f'BNM-{domain} mode{comp_num} {metric} pattern {exp_tag}', fontsize=16)
        plt.legend()
        plt.tight_layout()
    
        for save_format in ['png', 'pdf', 'eps']:
            plt.savefig(f'{output_dir}/{metric}_comp{comp_num}_full.{save_format}', format=save_format)
        plt.close()

        # significant variable only
        if domain in ['pheno', 'ENV']:
            data.index = var_name 
        
        for idx, cat in enumerate(categories):
            if domain == 'GPS':
                data.loc[cat.keys(), 'category'] = idx
                data = data.rename(index=cat)
            else:
                data.loc[cat.values(), 'category'] = idx


        occurence_cond = data.iloc[:, -2] > occurence_crit
        sig_ci95_cond = data.iloc[:, 3] * data.iloc[:, 4] > 0
        cond = occurence_cond * sig_ci95_cond

        significant_data = data.loc[cond, :].sort_values(by=f'{metric}_comp{comp_num}_estimate')
        sign_var = significant_data.index
        significant_data.loc[:, 'plt_x'] = np.arange(1, len(sign_var) + 1)
        
        error_pos = (significant_data.iloc[:, 3] + significant_data.iloc[:, 4]) / 2
        error_size = (significant_data.iloc[:, 4] - significant_data.iloc[:, 3]) / 2
        
        fig = plt.figure(figsize=(8, 6))
        for cat_idx in np.sort(significant_data.loc[:, 'category'].unique()):
            cat_cond = significant_data['category'] == cat_idx
            plt_x = significant_data.loc[cat_cond, 'plt_x']
            plt_y = significant_data.loc[cat_cond, f'{metric}_comp{comp_num}_estimate']
            category_name = category_name_list[int(cat_idx)]
            plt.barh(plt_x, plt_y, label=category_name)
        plt.errorbar(x=error_pos, y=np.arange(1, len(sign_var)+1), xerr=error_size, fmt='none', c='black', label=f'{metric} CI 95%')
        plt.legend()
        plt.yticks(np.arange(1, len(sign_var) + 1), sign_var)
        plt.xlabel(f'{metric}')
        plt.title(f'comp{comp_num} {metric} {exp_tag}')
        plt.tight_layout()
        for save_format in ['png', 'pdf', 'eps']:
            plt.savefig(f'{output_dir}/{metric}_comp{comp_num}_sigonly.{save_format}', format=save_format)
        plt.close()
