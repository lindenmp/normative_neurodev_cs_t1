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
import copy

# --------------------------------------------------------------------------------------------------------------------
# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-x", help="IVs", dest="X_file", default=None)
parser.add_argument("-y", help="DVs", dest="y_file", default=None)
parser.add_argument("-c", help="DVs", dest="c_file", default=None)
parser.add_argument("-metric", help="brain feature (e.g., ac)", dest="metric", default=None)
parser.add_argument("-pheno", help="psychopathology dimension", dest="pheno", default=None)
parser.add_argument("-alg", help="estimator", dest="alg", default=None)
parser.add_argument("-score", help="score set order", dest="score", default=None)
parser.add_argument("-runperm", help="whether to run permutation test or not", dest="runperm", default=0, type=int)
parser.add_argument("-o", help="output directory", dest="outroot", default=None)

args = parser.parse_args()
print(args)
X_file = args.X_file
y_file = args.y_file
c_file = args.c_file
metric = args.metric
pheno = args.pheno
alg = args.alg
score = args.score
runperm = args.runperm
outroot = args.outroot
# --------------------------------------------------------------------------------------------------------------------
if runperm == 1:
    print('Running with permutation test')

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


def shuffle_data(X, y, c, seed = 0):
    np.random.seed(seed)
    idx = np.arange(y.shape[0])
    np.random.shuffle(idx)

    X_shuf = X.iloc[idx,:]
    y_shuf = y.iloc[idx]
    c_shuf = c.iloc[idx,:]
    
    return X_shuf, y_shuf, c_shuf


def get_reg():
    regs = {'rr': Ridge(),
            'lr': Lasso(),
            'krr_lin': KernelRidge(kernel='linear'),
            'krr_rbf': KernelRidge(kernel='rbf'),
            'svr_lin': SVR(kernel='linear'),
            'svr_rbf': SVR(kernel='rbf')
            }

    return regs


def get_cv(y, n_splits = 10):

    my_cv = []

    kf = KFold(n_splits = n_splits, shuffle = False)

    for train_idx, test_idx in kf.split(y):
        my_cv.append( (train_idx, test_idx) )

    return my_cv


def my_cross_val_score(X, y, c, my_cv, reg, my_scorer, n_components = 9):

    accuracy = np.zeros((len(my_cv),n_components))

    for k in np.arange(len(my_cv)):
        tr = my_cv[k][0]
        te = my_cv[k][1]

        # Split into train test
        X_train = X.iloc[tr,:]; X_test = X.iloc[te,:]
        y_train = y.iloc[tr]; y_test = y.iloc[te]
        c_train = c.iloc[tr,:]; c_test = c.iloc[te,:]

        # standardize predictors
        sc = StandardScaler(); sc.fit(X_train); X_train = sc.transform(X_train); X_test = sc.transform(X_test)
        X_train = pd.DataFrame(data = X_train, index = X.iloc[tr,:].index, columns = X.iloc[tr,:].columns)
        X_test = pd.DataFrame(data = X_test, index = X.iloc[te,:].index, columns = X.iloc[te,:].columns)

        # standardize covariates
        sc = StandardScaler(); sc.fit(c_train); c_train = sc.transform(c_train); c_test = sc.transform(c_test)
        c_train = pd.DataFrame(data = c_train, index = c.iloc[tr,:].index, columns = c.iloc[tr,:].columns)
        c_test = pd.DataFrame(data = c_test, index = c.iloc[te,:].index, columns = c.iloc[te,:].columns)

        # # regress nuisance (X) 
        # # nuis_reg = LinearRegression(); nuis_reg.fit(c_train, X_train)
        # nuis_reg = copy.deepcopy(reg); nuis_reg.fit(c_train, X_train)
        # X_pred = nuis_reg.predict(c_train); X_train = X_train - X_pred
        # X_pred = nuis_reg.predict(c_test); X_test = X_test - X_pred

        pca = PCA(n_components = n_components, svd_solver = 'full')
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

        # regress nuisance (y)
        nuis_reg = copy.deepcopy(reg); nuis_reg.fit(c_train, y_train)
        y_pred = nuis_reg.predict(c_train); y_train = y_train - y_pred
        y_pred = nuis_reg.predict(c_test); y_test = y_test - y_pred

        for i in np.arange(0,n_components):
            reg.fit(X_train[:,i].reshape(-1, 1), y_train)
            accuracy[k,i] = my_scorer(reg, X_test[:,i].reshape(-1, 1), y_test)
        
    return accuracy


def run_reg(X, y, c, reg, my_scorer, n_splits = 10, seed = 0, n_components = 9):

    X_shuf, y_shuf, c_shuf = shuffle_data(X = X, y = y, c = c, seed = seed)

    my_cv = get_cv(y_shuf, n_splits = n_splits)

    accuracy = my_cross_val_score(X = X_shuf, y = y_shuf, c = c_shuf, my_cv = my_cv, reg = reg, my_scorer = my_scorer, n_components = n_components)

    return accuracy


def run_perm(X, y, c, reg, my_scorer, n_splits = 10, n_components = 9):

    X.reset_index(drop = True, inplace = True)

    my_cv = get_cv(y, n_splits = n_splits)

    n_perm = 100000
    permuted_acc = np.zeros((n_perm,n_components))

    for i in np.arange(n_perm):
        np.random.seed(i)
        idx = np.arange(y.shape[0])
        np.random.shuffle(idx)

        y_perm = y.iloc[idx].copy()
        y_perm.reset_index(drop = True, inplace = True)
        c_perm = c.iloc[idx,:].copy()
        c_perm.reset_index(drop = True, inplace = True)
            
        temp_acc = my_cross_val_score(X = X, y = y_perm, c = c_perm, my_cv = my_cv, reg = reg, my_scorer = my_scorer, n_components = n_components)
        permuted_acc[i,:] = temp_acc.mean(axis=0)

    return permuted_acc


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

num_random_splits = 100

if X.shape[1] == 400:
    n_components = 9
elif X.shape[1] == 463:
    n_components = 8

accuracy_mean = np.zeros((num_random_splits,n_components))
accuracy_std = np.zeros((num_random_splits,n_components))

for i in np.arange(0,num_random_splits):
    accuracy = run_reg(X = X, y = y, c = c, reg = regs[alg], my_scorer = my_scorer, seed = i, n_components = n_components)
    accuracy_mean[i,:] = accuracy.mean(axis=0)
    accuracy_std[i,:] = accuracy.std(axis=0)

if runperm == 1:
    print('Running permutation test...')
    permuted_acc = run_perm(X = X, y = y, c = c, reg = regs[alg], my_scorer = my_scorer, n_components = n_components)
    print('...done')

# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# outputs
np.savetxt(os.path.join(outdir,'accuracy_mean.txt'), accuracy_mean)
np.savetxt(os.path.join(outdir,'accuracy_std.txt'), accuracy_std)
if runperm == 1:
    np.savetxt(os.path.join(outdir,'permuted_acc.txt'), permuted_acc)

# --------------------------------------------------------------------------------------------------------------------

print('Finished!')
