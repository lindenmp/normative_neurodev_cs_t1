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
parser.add_argument("-c", help="DVs", dest="c_file", default=None)
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
c_file = args.c_file
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


def get_stratified_cv(X, y, c = None, n_splits = 10):

    # sort data on outcome variable in ascending order
    idx = y.sort_values(ascending = True).index
    if X.ndim == 2: X_sort = X.loc[idx,:]
    elif X.ndim == 1: X_sort = X.loc[idx]
    y_sort = y.loc[idx]
    if c is not None:
        if c.ndim == 2: c_sort = c.loc[idx,:]
        elif c.ndim == 1: c_sort = c.loc[idx]
    
    # create custom stratified kfold on outcome variable
    my_cv = []
    for k in range(n_splits):
        my_bool = np.zeros(y.shape[0]).astype(bool)
        my_bool[np.arange(k,y.shape[0],n_splits)] = True

        train_idx = np.where(my_bool == False)[0]
        test_idx = np.where(my_bool == True)[0]
        my_cv.append( (train_idx, test_idx) )

    if c is not None:
        return X_sort, y_sort, my_cv, c_sort
    else:
        return X_sort, y_sort, my_cv


def cross_val_score_nuis(X, y, c, my_cv, reg, my_scorer, c_y = None):
    
    accuracy = np.zeros(len(my_cv),)

    for k in np.arange(len(my_cv)):
        tr = my_cv[k][0]
        te = my_cv[k][1]

        # Split into train test
        X_train = X.iloc[tr,:]; X_test = X.iloc[te,:]
        y_train = y.iloc[tr]; y_test = y.iloc[te]
        c_train = c.iloc[tr,:]; c_test = c.iloc[te,:]
        if c_y is not None: c_y_train = c_y.iloc[tr,:]; c_y_test = c_y.iloc[te,:]

        # standardize predictors
        sc = StandardScaler(); sc.fit(X_train); X_train = sc.transform(X_train); X_test = sc.transform(X_test)
        X_train = pd.DataFrame(data = X_train, index = X.iloc[tr,:].index, columns = X.iloc[tr,:].columns)
        X_test = pd.DataFrame(data = X_test, index = X.iloc[te,:].index, columns = X.iloc[te,:].columns)

        # standardize covariates
        sc = StandardScaler(); sc.fit(c_train); c_train = sc.transform(c_train); c_test = sc.transform(c_test)
        c_train = pd.DataFrame(data = c_train, index = c.iloc[tr,:].index, columns = c.iloc[tr,:].columns)
        c_test = pd.DataFrame(data = c_test, index = c.iloc[te,:].index, columns = c.iloc[te,:].columns)

        if c_y is not None:
            sc = StandardScaler(); sc.fit(c_y_train); c_y_train = sc.transform(c_y_train); c_y_test = sc.transform(c_y_test)
            c_y_train = pd.DataFrame(data = c_y_train, index = c.iloc[tr,:].index, columns = c.iloc[tr,:].columns)
            c_y_test = pd.DataFrame(data = c_y_test, index = c.iloc[te,:].index, columns = c.iloc[te,:].columns)

        # regress nuisance (X)
        # nuis_reg = LinearRegression(); nuis_reg.fit(c_train, X_train)
        nuis_reg = KernelRidge(kernel='rbf'); nuis_reg.fit(c_train, X_train)
        X_pred = nuis_reg.predict(c_train); X_train = X_train - X_pred
        X_pred = nuis_reg.predict(c_test); X_test = X_test - X_pred

        # # regress nuisance (y)
        # if c_y is None:  
        #     # nuis_reg = LinearRegression(); nuis_reg.fit(c_train, y_train)
        #     nuis_reg = KernelRidge(kernel='rbf'); nuis_reg.fit(c_train, y_train)
        #     y_pred = nuis_reg.predict(c_train); y_train = y_train - y_pred
        #     y_pred = nuis_reg.predict(c_test); y_test = y_test - y_pred
        # elif c_y is not None:
        #     # nuis_reg = LinearRegression(); nuis_reg.fit(c_y_train, y_train)
        #     nuis_reg = KernelRidge(kernel='rbf'); nuis_reg.fit(c_y_train, y_train)
        #     y_pred = nuis_reg.predict(c_y_train); y_train = y_train - y_pred
        #     y_pred = nuis_reg.predict(c_y_test); y_test = y_test - y_pred

        reg.fit(X_train, y_train)
        accuracy[k] = my_scorer(reg, X_test, y_test)
        
    return accuracy



def run_reg_scv(X, y, c, reg, n_splits = 10, scoring = 'r2', run_perm = False):
    
    X_sort, y_sort, my_cv, c_sort = get_stratified_cv(X = X, y = y, c = c, n_splits = n_splits)

    accuracy_nuis = cross_val_score_nuis(X = X_sort, y = y_sort, c = c_sort, my_cv = my_cv, reg = reg, my_scorer = scoring)

    if run_perm:
        X_sort.reset_index(drop = True, inplace = True)
        c_sort.reset_index(drop = True, inplace = True)

        n_perm = 5000
        permuted_acc_nuis = np.zeros((n_perm,))

        for i in np.arange(n_perm):
            np.random.seed(i)
            idx = np.arange(y_sort.shape[0])
            np.random.shuffle(idx)

            y_perm = y_sort.iloc[idx].copy()
            y_perm.reset_index(drop = True, inplace = True)
            c_y = c_sort.iloc[idx,:]
            c_y.reset_index(drop = True, inplace = True)
            
            permuted_acc_nuis[i] = cross_val_score_nuis(X = X_sort, y = y_perm, c = c_sort, my_cv = my_cv, reg = reg, my_scorer = scoring, c_y = c_y).mean()

    if run_perm:
        return accuracy_nuis, permuted_acc_nuis
    else:
        return accuracy_nuis


# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# inputs
X = pd.read_csv(X_file)
X.set_index(['bblid', 'scanid'], inplace = True)
X = X.filter(regex = metric)

y = pd.read_csv(y_file)
y.set_index(['bblid', 'scanid'], inplace = True)
y = y.loc[:,pheno]

c = pd.read_csv(c_file)
c.set_index(['bblid', 'scanid'], inplace = True)

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

accuracy_nuis, permuted_acc_nuis = run_reg_scv(X = X, y = y, c = c, reg = regs[alg], scoring = my_scorer, run_perm = True)
# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# outputs
np.savetxt(os.path.join(outdir,'accuracy_nuis.txt'), accuracy_nuis)
np.savetxt(os.path.join(outdir,'accuracy_mean_nuis.txt'), np.array([accuracy_nuis.mean()]))
np.savetxt(os.path.join(outdir,'accuracy_std_nuis.txt'), np.array([accuracy_nuis.std()]))
np.savetxt(os.path.join(outdir,'permuted_acc_nuis.txt'), permuted_acc_nuis)

# --------------------------------------------------------------------------------------------------------------------

print('Finished!')
