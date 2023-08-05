# import packages
import sys
import os

from cca_zoo.model_selection import GridSearchCV
from cca_zoo.models import SCCA_PMD

import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA

import seaborn as sns

import argparse


#####################################
####### define custom function ######
#####################################

def scorer(estimator,X):
    dim_corrs=estimator.score(X)
    return dim_corrs.mean()


def scorer_cov(estimator, X):
    U = estimator.transform(X)[0][:, 0]
    V = estimator.transform(X)[1][:, 0]
    dim_cov = np.cov(U, V)[0, 1]
    return dim_cov.mean()


def cv_result_saver(CCA_cv_results, c1, c2, savefile_dir):
    value = pd.DataFrame(CCA_cv_results.cv_results_['mean_test_score'].reshape(len(c1), len(c2)),
                         index=['c1_' + str(round(c, 2)) for c in c1], columns=['c2_' + str(round(c, 2)) for c in c2])
    value.to_csv(savefile_dir)

#####################################
#########   code body   #############
#####################################

parser = argparse.ArgumentParser()

parser.add_argument('--domain')
parser.add_argument('--dataset')
parser.add_argument('--exp_tag')

args = parser.parse_args()

domain = args.domain
dataset = args.dataset
exp_tag = args.exp_tag
param_tune_scheme = 'first_comp_cov'


file_dir = f'/storage/connectome/seojw/data/SCCA_dataset/{domain}_BNM/'

cca_x_file_name = file_dir + f'regressed_{domain}_BNM_CCA_{dataset}_{domain}_{exp_tag}.csv'
cca_y_file_name = file_dir + f'regressed_{domain}_BNM_CCA_{dataset}_BNM_{exp_tag}.csv'

cca_x = pd.read_csv(cca_x_file_name, index_col=0)
cca_y = pd.read_csv(cca_y_file_name, index_col=0)

# make sure the input data are zscaled before run SCCA
scaler = StandardScaler()
cca_x = pd.DataFrame(scaler.fit_transform(cca_x), index=cca_x.index, columns=cca_x.columns)
cca_y = pd.DataFrame(scaler.fit_transform(cca_y), index=cca_y.index, columns=cca_y.columns)

# set parameter space for grid search
c1 = list(np.arange(0.1, 1.05, 0.05))
c2 = list(np.arange(0.1, 1.05, 0.05))
c1[-1] = 1
c2[-1] = 1
param_grid = {'c': [c1,c2]} 


# set output file directory and output file name
# run hyper parameter tuning

init = time.time() 
if param_tune_scheme == 'first_comp_cov':
    save_dir = f'/storage/connectome/seojw/data/SCCA_param_tune_result/{domain}_BNM/first_comp_cov/{exp_tag}/'
    os.makedirs(save_dir, exist_ok=True)
    output_file_dir = save_dir + f'param_result_{domain}_{exp_tag}.csv'
    sparse_CCA = GridSearchCV(SCCA_PMD(latent_dims=1), param_grid=param_grid, cv=5,
                         verbose=True, scoring=scorer_cov).fit([cca_x, cca_y])
    cv_result_saver(sparse_CCA, c1, c2, output_file_dir)

print(time.time() - init)
