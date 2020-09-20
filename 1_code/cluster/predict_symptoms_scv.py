import argparse

# Essentials
import os, sys, glob
import pandas as pd
import numpy as np
import copy
import json

# Stats
import scipy as sp
from scipy import stats

# Sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA

# --------------------------------------------------------------------------------------------------------------------
# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-x", help="IVs", dest="X_file", default=None)
parser.add_argument("-y", help="DVs", dest="y_file", default=None)
parser.add_argument("-metric", help="brain feature (e.g., ac)", dest="metric", default=None)
parser.add_argument("-pheno", help="psychopathology dimension", dest="pheno", default=None)
parser.add_argument("-seed", help="seed for shuffle_data", dest="seed", default=1)
parser.add_argument("-alg", help="estimator", dest="alg", default=None)
parser.add_argument("-score", help="score set order", dest="score", default=None)
parser.add_argument("-o", help="output directory", dest="outroot", default=None)

args = parser.parse_args()
print(args)
X_file = args.X_file
y_file = args.y_file
metric = args.metric
pheno = args.pheno
# seed = int(args.seed)
# seed = int(os.environ['SGE_TASK_ID'])-1
alg = args.alg
score = args.score
outroot = args.outroot
# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# prediction functions
def corr_true_pred(y_true, y_pred):
    if type(y_true) == np.ndarray:
        y_true = y_true.flatten()
    if type(y_pred) == np.ndarray:
        y_pred = y_pred.flatten()
        
    r,p = sp.stats.pearsonr(y_true, y_pred)
    return r


def root_mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2, axis=0)
    rmse = np.sqrt(mse)
    return rmse


def get_reg():
    regs = {'rr': Ridge(),
            'lr': Lasso(),
            'krr_lin': KernelRidge(kernel='linear'),
            'krr_rbf': KernelRidge(kernel='rbf'),
            'svr_lin': SVR(kernel='linear'),
            'svr_rbf': SVR(kernel='rbf')
            }

    return regs


def get_stratified_cv(X, y, n_splits = 10):

    # sort data on outcome variable in ascending order
    idx = y.sort_values(ascending = True).index
    if X.ndim == 2: X_sort = X.loc[idx,:]
    elif X.ndim == 1: X_sort = X.loc[idx]
    y_sort = y.loc[idx]
    
    # create custom stratified kfold on outcome variable
    my_cv = []
    for k in range(n_splits):
        my_bool = np.zeros(y.shape[0]).astype(bool)
        my_bool[np.arange(k,y.shape[0],n_splits)] = True

        train_idx = np.where(my_bool == False)[0]
        test_idx = np.where(my_bool == True)[0]
        my_cv.append( (train_idx, test_idx) )

    return X_sort, y_sort, my_cv


def my_cross_val_score(X, y, my_cv, reg, my_scorer):
    
    accuracy = np.zeros(len(my_cv),)

    for k in np.arange(len(my_cv)):
        tr = my_cv[k][0]
        te = my_cv[k][1]

        # Split into train test
        X_train = X.iloc[tr,:]; X_test = X.iloc[te,:]
        y_train = y.iloc[tr]; y_test = y.iloc[te]

        # standardize predictors
        sc = StandardScaler(); sc.fit(X_train); X_train = sc.transform(X_train); X_test = sc.transform(X_test)
        X_train = pd.DataFrame(data = X_train, index = X.iloc[tr,:].index, columns = X.iloc[tr,:].columns)
        X_test = pd.DataFrame(data = X_test, index = X.iloc[te,:].index, columns = X.iloc[te,:].columns)

        reg.fit(X_train, y_train)
        accuracy[k] = my_scorer(reg, X_test, y_test)
        
    return accuracy


def run_reg_scv(X, y, reg, n_splits = 10, scoring = 'r2', run_perm = False):
    
    X_sort, y_sort, my_cv = get_stratified_cv(X = X, y = y, n_splits = n_splits)

    accuracy = my_cross_val_score(X = X_sort, y = y_sort, my_cv = my_cv, reg = reg, my_scorer = scoring)

    if run_perm:
        X_sort.reset_index(drop = True, inplace = True)

        n_perm = 5000
        permuted_acc = np.zeros((n_perm,))

        for i in np.arange(n_perm):
            np.random.seed(i)
            idx = np.arange(y_sort.shape[0])
            np.random.shuffle(idx)

            y_perm = y_sort.iloc[idx].copy()
            y_perm.reset_index(drop = True, inplace = True)
            
            permuted_acc[i] = my_cross_val_score(X = X_sort, y = y_perm, my_cv = my_cv, reg = reg, my_scorer = scoring).mean()

    if run_perm:
        return accuracy, permuted_acc
    else:
        return accuracy


# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# inputs
X = pd.read_csv(X_file)
X.set_index(['bblid', 'scanid'], inplace = True)
X = X.filter(regex = metric)

y = pd.read_csv(y_file)
y.set_index(['bblid', 'scanid'], inplace = True)
y = y.loc[:,pheno]

# outdir
outdir = os.path.join(outroot, alg + '_' + score + '_' + metric + '_' + pheno)
if not os.path.exists(outdir): os.makedirs(outdir);
# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# set scorer
if score == 'r2':
    my_scorer = make_scorer(r2_score, greater_is_better = True)
elif score == 'corr':
    my_scorer = make_scorer(corr_true_pred, greater_is_better = True)
elif score == 'mse':
    my_scorer = make_scorer(mean_squared_error, greater_is_better = False)
elif score == 'rmse':
    my_scorer = make_scorer(root_mean_squared_error, greater_is_better = False)
elif score == 'mae':
    my_scorer = make_scorer(mean_absolute_error, greater_is_better = False)

# prediction
regs = get_reg()

accuracy, permuted_acc = run_reg_scv(X = X, y = y, reg = regs[alg], scoring = my_scorer, run_perm = True)
# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# outputs
np.savetxt(os.path.join(outdir,'accuracy.txt'), accuracy)
np.savetxt(os.path.join(outdir,'accuracy_mean.txt'), np.array([accuracy.mean()]))
np.savetxt(os.path.join(outdir,'accuracy_std.txt'), np.array([accuracy.std()]))
np.savetxt(os.path.join(outdir,'permuted_acc.txt'), permuted_acc)

# --------------------------------------------------------------------------------------------------------------------

print('Finished!')
