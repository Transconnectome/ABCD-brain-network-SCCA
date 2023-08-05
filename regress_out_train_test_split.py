import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from cca_zoo.models import SCCA_PMD
import time

# import package
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import argparse

def residualize_covariate(train_dataset, test_dataset, cov_train, cov_test):
    col_name = train_dataset.columns
    train_dataset.columns = [f'Var_{i}' for i in range(len(col_name))]
    test_dataset.columns = [f'Var_{i}' for i in range(len(col_name))]

    binary_list = []
    for i in range(len(train_dataset.iloc[0])):
        if len(train_dataset.iloc[:, i].unique()) == 2:
            binary_list.append(train_dataset.iloc[:, i].name)
    linear_reg_list = (train_dataset.columns).drop(binary_list)

    train_dataset_res = train_dataset.copy()
    test_dataset_res = test_dataset.copy()

    train_total_data = pd.merge(train_dataset, cov_train, how='inner', left_index=True, right_index=True)
    test_total_data = pd.merge(test_dataset, cov_test, how='inner', left_index=True, right_index=True)

    # linear regression for not binary variables
    for col in linear_reg_list:
        reg_formula = col + ' ~ age + C(female) + high_educ + income + C(married) + C(abcd_site) + BMI + C(race_ethnicity)'
        res = smf.ols(formula = reg_formula, data=train_total_data).fit()

        pred_train = res.predict(train_total_data.loc[:, cov_train.columns.insert(0, col)])
        res_train = train_total_data.loc[:, col] - pred_train

        pred_test = res.predict(test_total_data.loc[:, cov_test.columns.insert(0, col)])
        res_test = test_total_data.loc[:, col] - pred_test

        train_dataset_res.loc[:, col] = res_train
        test_dataset_res.loc[:, col] = res_test

    # logistic regression for binary variables and check with visualization
    for col in binary_list:
        try:
            reg_formula = col + ' ~ age + C(female) + high_educ + income + C(married) + C(abcd_site) + BMI + C(race_ethnicity)'
            res = smf.logit(formula = reg_formula, data=train_total_data).fit(maxiter=1000)

            pred_train = res.predict(train_total_data.loc[:, cov_train.columns.insert(0, col)])
            res_train = train_total_data.loc[:, col] - pred_train

            pred_test = res.predict(test_total_data.loc[:, cov_test.columns.insert(0, col)])
            res_test = test_total_data.loc[:, col] - pred_test

            train_dataset_res.loc[:, col] = res_train
            test_dataset_res.loc[:, col] = res_test

        except:
            print('except occured')


    scaler = StandardScaler().fit(train_dataset_res)
    train_dataset_res = pd.DataFrame(scaler.transform(train_dataset_res), index=train_dataset_res.index, columns=col_name)
    test_dataset_res = pd.DataFrame(scaler.transform(test_dataset_res), index=test_dataset_res.index, columns=col_name)

    return train_dataset_res, test_dataset_res


parser = argparse.ArgumentParser()
parser.add_argument('--domain', required=False)
parser.add_argument('--file_tag')
args = parser.parse_args()

domain = args.domain
file_tag = args.file_tag

saving_dir = f'/storage/connectome/seojw/data/SCCA_dataset/{domain}_BNM'

BNM = pd.read_csv(f'/storage/connectome/seojw/data/{file_tag}_rm_zeroval.csv', index_col=0)
BNM_sub = BNM.index

# covariate data import & preprocessing
covariates = pd.read_csv('/storage/connectome/seojw/data/SCCA_covariate_v2023_08_05.csv', index_col=0)
covariates = covariates.dropna()

scaler = StandardScaler()
z_scaled_variable_list = ['age', 'high.educ', 'income', 'BMI']
covariates.loc[:, z_scaled_variable_list] = scaler.fit_transform(covariates.loc[:, z_scaled_variable_list])
covariates = covariates.rename(columns = {'race.ethnicity' : 'race_ethnicity', 'high.educ':'high_educ'})

if domain == 'ENV':
    # train test split & regressing out for ENV-BNM CCA


    env = pd.read_csv('/storage/connectome/seojw/data/ABCD_CCA_ENV_v22_09_01_rm_zeroval.csv', index_col=0).dropna()
    env_sub = env.index

    sublist = env_sub & BNM_sub & covariates.index
    env = env.loc[sublist]
    BNM = BNM.loc[sublist]
    cov = covariates.loc[sublist]

    # random stratified split based on famhx.
    fam_history = np.sign(env.loc[:, 'famhx_ss_fath_prob_alc_p':]).sum(axis=1)

    train_index, test_index = train_test_split(np.arange(len(env)), stratify=fam_history, shuffle=True, test_size=0.2)
    env_train, env_test = env.iloc[train_index, :], env.iloc[test_index, :]
    BNM_train, BNM_test = BNM.iloc[train_index,:], BNM.iloc[test_index, :]
    cov_train, cov_test = cov.iloc[train_index, :], cov.iloc[test_index, :]

    env_train_res, env_test_res = residualize_covariate(env_train, env_test, cov_train, cov_test)
    BNM_train_res, BNM_test_res = residualize_covariate(BNM_train, BNM_test, cov_train, cov_test)

    env_train_res.to_csv(f'{saving_dir}/regressed_{domain}_BNM_CCA_trainset_{domain}_{file_tag[4:]}.csv')
    env_test_res.to_csv(f'{saving_dir}/regressed_{domain}_BNM_CCA_testset_{domain}_{file_tag[4:]}.csv')
    BNM_train_res.to_csv(f'{saving_dir}/regressed_{domain}_BNM_CCA_trainset_{file_tag}.csv')
    BNM_test_res.to_csv(f'{saving_dir}/regressed_{domain}_BNM_CCA_testset_{file_tag}.csv')



# train test split & regressing out for pheno-BNM CCA
if domain == 'pheno':


    pheno = pd.read_csv('/storage/connectome/seojw/data/ABCD_CCA_phenotype_v22_09_01_rm_zeroval.csv', index_col=0).dropna()
    pheno_sub = pheno.index

    sublist = pheno_sub & BNM_sub & covariates.index
    pheno = pheno.loc[sublist]
    BNM = BNM.loc[sublist]
    cov = covariates.loc[sublist]

    # random stratified split based on ksad
    ksad_list = pheno.loc[:, 'ADHD_Life':'SUI_Life_Parent'].sum(axis=1)

    train_index, test_index = train_test_split(np.arange(len(pheno)), stratify=ksad_list, shuffle=True, test_size=0.2)
    pheno_train, pheno_test = pheno.iloc[train_index, :], pheno.iloc[test_index, :]
    BNM_train, BNM_test = BNM.iloc[train_index,:], BNM.iloc[test_index, :]
    cov_train, cov_test = cov.iloc[train_index, :], cov.iloc[test_index, :]

    pheno_train_res, pheno_test_res = residualize_covariate(pheno_train, pheno_test, cov_train, cov_test)
    BNM_train_res, BNM_test_res = residualize_covariate(BNM_train, BNM_test, cov_train, cov_test)

    pheno_train_res.to_csv(f'{saving_dir}/regressed_{domain}_BNM_CCA_trainset_{domain}_{file_tag[4:]}.csv')
    pheno_test_res.to_csv(f'{saving_dir}/regressed_{domain}_BNM_CCA_testset_{domain}_{file_tag[4:]}.csv')
    BNM_train_res.to_csv(f'{saving_dir}/regressed_{domain}_BNM_CCA_trainset_{file_tag}.csv')
    BNM_test_res.to_csv(f'{saving_dir}/regressed_{domain}_BNM_CCA_testset_{file_tag}.csv')


if domain == 'GPS':

    GPS_train = pd.read_csv('/storage/connectome/seojw/data/ABCD_GPS_EUR_only_train.csv', index_col=0)
    GPS_test = pd.read_csv('/storage/connectome/seojw/data/ABCD_GPS_EUR_only_test.csv', index_col=0)

    train_index = GPS_train.index & BNM_sub & covariates.index
    test_index = GPS_test.index & BNM_sub & covariates.index

    GPS_train, GPS_test = GPS_train.loc[train_index, :], GPS_test.loc[test_index, :]
    BNM_train, BNM_test = BNM.loc[train_index,:], BNM.loc[test_index, :]
    cov_train, cov_test = covariates.loc[train_index, :], covariates.loc[test_index, :]

    BNM_train_res, BNM_test_res = residualize_covariate(BNM_train, BNM_test, cov_train, cov_test)

    GPS_train.to_csv(f'{saving_dir}/regressed_{domain}_BNM_CCA_trainset_{domain}_{file_tag[4:]}.csv')
    GPS_test.to_csv(f'{saving_dir}/regressed_{domain}_BNM_CCA_testset_{domain}_{file_tag[4:]}.csv')
    BNM_train_res.to_csv(f'{saving_dir}/regressed_{domain}_BNM_CCA_trainset_{file_tag}.csv')
    BNM_test_res.to_csv(f'{saving_dir}/regressed_{domain}_BNM_CCA_testset_{file_tag}.csv')