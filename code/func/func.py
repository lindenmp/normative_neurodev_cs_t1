# Functions for project: NormativeNeuroDev_CrossSec
# This project used normative modelling to examine network control theory metrics
# Linden Parkes, 2019
# lindenmp@seas.upenn.edu

from IPython.display import clear_output

import numpy as np
import scipy as sp
import pandas as pd

from numpy.matlib import repmat 
from scipy.linalg import svd, schur
from scipy import stats

from statsmodels.stats import multitest

import matplotlib.pyplot as plt

def get_cmap(which_type = 'qual1', num_classes = 8):
    # Returns a nice set of colors to make a nice colormap using the color schemes
    # from http://colorbrewer2.org/
    #
    # The online tool, colorbrewer2, is copyright Cynthia Brewer, Mark Harrower and
    # The Pennsylvania State University.

    if which_type == 'linden':
        cmap_base = np.array([[255,105,97],[97,168,255],[178,223,138],[117,112,179],[255,179,71]])
    elif which_type == 'pair':
        cmap_base = np.array([[124,230,199],[255,169,132]])
    elif which_type == 'qual1':
        cmap_base = np.array([[166,206,227],[31,120,180],[178,223,138],[51,160,44],[251,154,153],[227,26,28],
                            [253,191,111],[255,127,0],[202,178,214],[106,61,154],[255,255,153],[177,89,40]])
    elif which_type == 'qual2':
        cmap_base = np.array([[141,211,199],[255,255,179],[190,186,218],[251,128,114],[128,177,211],[253,180,98],
                            [179,222,105],[252,205,229],[217,217,217],[188,128,189],[204,235,197],[255,237,111]])
    elif which_type == 'seq_red':
        cmap_base = np.array([[255,245,240],[254,224,210],[252,187,161],[252,146,114],[251,106,74],
                            [239,59,44],[203,24,29],[165,15,21],[103,0,13]])
    elif which_type == 'seq_blu':
        cmap_base = np.array([[247,251,255],[222,235,247],[198,219,239],[158,202,225],[107,174,214],
                            [66,146,198],[33,113,181],[8,81,156],[8,48,107]])
    elif which_type == 'redblu_pair':
        cmap_base = np.array([[222,45,38],[49,130,189]])
    elif which_type == 'yeo17':
        cmap_base = np.array([[97,38,107], # VisCent
                            [194,33,39], # VisPeri
                            [79,130,165], # SomMotA
                            [44,181,140], # SomMotB
                            [75,148,72], # DorsAttnA
                            [23,116,62], # DorsAttnB
                            [149,77,158], # SalVentAttnA
                            [222,130,177], # SalVentAttnB
                            [75,87,61], # LimbicA
                            [149,166,110], # LimbicB
                            [210,135,47], # ContA
                            [132,48,73], # ContB
                            [92,107,131], # ContC
                            [218,221,50], # DefaultA
                            [175,49,69], # DefaultB
                            [41,38,99], # DefaultC
                            [53,75,158] # TempPar
                            ])
    elif which_type == 'yeo17_downsampled':
        cmap_base = np.array([[97,38,107], # VisCent
                            [79,130,165], # SomMotA
                            [75,148,72], # DorsAttnA
                            [149,77,158], # SalVentAttnA
                            [75,87,61], # LimbicA
                            [210,135,47], # ContA
                            [218,221,50], # DefaultA
                            [53,75,158] # TempPar
                            ])

    if cmap_base.shape[0] > num_classes: cmap = cmap_base[0:num_classes]
    else: cmap = cmap_base

    cmap = cmap / 255

    return cmap


def update_progress(progress, my_str = ''):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = my_str + " Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)


def node_strength(A):
    s = np.sum(A, axis = 0)

    return s


def ave_control(A, c = 1):
    # FUNCTION:
    #         Returns values of AVERAGE CONTROLLABILITY for each node in a
    #         network, given the adjacency matrix for that network. Average
    #         controllability measures the ease by which input at that node can
    #         steer the system into many easily-reachable states.
    #
    # INPUT:
    #         A is the structural (NOT FUNCTIONAL) network adjacency matrix, 
    #         such that the simple linear model of dynamics outlined in the 
    #         reference is an accurate estimate of brain state fluctuations. 
    #         Assumes all values in the matrix are positive, and that the 
    #         matrix is symmetric.
    #
    # OUTPUT:
    #         Vector of average controllability values for each node
    #
    # Bassett Lab, University of Pennsylvania, 2016.
    # Reference: Gu, Pasqualetti, Cieslak, Telesford, Yu, Kahn, Medaglia,
    #            Vettel, Miller, Grafton & Bassett, Nature Communications
    #            6:8414, 2015.

    u, s, vt = svd(A) # singluar value decomposition
    A = A/(c + s[0]) # Matrix normalization 
    T, U = schur(A,'real') # Schur stability
    midMat = np.multiply(U,U).transpose()
    v = np.matrix(np.diag(T)).transpose()
    N = A.shape[0]
    P = np.diag(1 - np.matmul(v,v.transpose()))
    P = repmat(P.reshape([N,1]), 1, N)
    values = sum(np.divide(midMat,P))
    
    return values


def modal_control(A, c = 1):
    # FUNCTION:
    #         Returns values of MODAL CONTROLLABILITY for each node in a
    #         network, given the adjacency matrix for that network. Modal
    #         controllability indicates the ability of that node to steer the
    #         system into difficult-to-reach states, given input at that node.
    #
    # INPUT:
    #         A is the structural (NOT FUNCTIONAL) network adjacency matrix, 
    #     such that the simple linear model of dynamics outlined in the 
    #     reference is an accurate estimate of brain state fluctuations. 
    #     Assumes all values in the matrix are positive, and that the 
    #     matrix is symmetric.
    #
    # OUTPUT:
    #         Vector of modal controllability values for each node
    #
    # Bassett Lab, University of Pennsylvania, 2016. 
    # Reference: Gu, Pasqualetti, Cieslak, Telesford, Yu, Kahn, Medaglia,
    #            Vettel, Miller, Grafton & Bassett, Nature Communications
    #            6:8414, 2015.
    
    u, s, vt = svd(A) # singluar value decomposition
    A = A/(c + s[0]) # Matrix normalization
    T, U = schur(A,'real') # Schur stability
    eigVals = np.diag(T)
    N = A.shape[0]
    phi = np.zeros(N,dtype = float)
    for i in range(N):
        Al = U[i,] * U[i,]
        Ar = (1.0 - np.power(eigVals,2)).transpose()
        phi[i] = np.matmul(Al, Ar)
    
    return phi


def mark_outliers(x, thresh = 3, c = 1.4826):
    my_med = np.median(x)
    mad = np.median(abs(x - my_med))/c
    cut_off = mad * thresh
    upper = my_med + cut_off
    lower = my_med - cut_off
    outliers = np.logical_or(x > upper, x < lower)
    
    return outliers


def winsorize_outliers_signed(x, thresh = 3, c = 1.4826):
    my_med = np.median(x)
    mad = np.median(abs(x - my_med))/c
    cut_off = mad * thresh
    upper = my_med + cut_off
    lower = my_med - cut_off
    pos_outliers = x > upper
    neg_outliers = x < lower

    if pos_outliers.any() and ~neg_outliers.any():
        x_out = sp.stats.mstats.winsorize(x, limits = (0,0.05))
    elif ~pos_outliers.any() and neg_outliers.any():
        x_out = sp.stats.mstats.winsorize(x, limits = (0.05,0))
    elif pos_outliers.any() and neg_outliers.any():
        x_out = sp.stats.mstats.winsorize(x, limits = 0.05)
    else:
        x_out = x
        
    return x_out


def get_synth_cov(df, cov = 'ageAtScan1_Years', stp = 1):
    # Synthetic cov data
    X_range = [np.min(df[cov]), np.max(df[cov])]
    X = np.arange(X_range[0],X_range[1],stp)
    X = X.reshape(-1,1)

    return X


def run_corr(df_X, df_y, typ = 'spearmanr'):
    df_corr = pd.DataFrame(index = df_y.columns, columns = ['coef', 'p'])
    for i, row in df_corr.iterrows():
        if typ == 'spearmanr':
            df_corr.loc[i] = sp.stats.spearmanr(df_X, df_y[i])
        elif typ == 'pearsonr':
            df_corr.loc[i] = sp.stats.pearsonr(df_X, df_y[i])

    return df_corr


def get_fdr_p(p_vals):
    out = multitest.multipletests(p_vals, alpha = 0.05, method = 'fdr_bh')
    p_fdr = out[1] 

    return p_fdr


def get_fdr_p_df(p_vals):
    p_fdr = pd.DataFrame(index = p_vals.index,
                        columns = p_vals.columns,
                        data = np.reshape(get_fdr_p(p_vals.values.flatten()), p_vals.shape))

    return p_fdr


def get_null_p(coef, null):

    num_perms = null.shape[1]
    num_parcels = len(coef)
    p_perm = np.zeros((num_parcels,))

    for i in range(num_parcels):
        r_obs = abs(coef[i])
        r_perm = abs(null[i,:])
        p_perm[i] = np.sum(r_perm >= r_obs) / num_perms

    return p_perm


def get_sys_summary(coef, p_vals, idx, method = 'mean', alpha = 0.05, signed = True):
    u_idx = np.unique(idx)
    if signed == True:
        sys_summary = np.zeros((len(u_idx),2))
    else:
        sys_summary = np.zeros((len(u_idx),1))
        
    for i in u_idx:
        # filter regions by system idx
        coef_tmp = coef[idx == i]
        p_tmp = p_vals[idx == i]
        
        # threshold out non-sig coef
        coef_tmp = coef_tmp[p_tmp < alpha]

        # proportion of signed significant coefs within system i
        if method == 'mean':
            if signed == True:
                if any(coef_tmp[coef_tmp > 0]): sys_summary[i-1,0] = np.mean(abs(coef_tmp[coef_tmp > 0]))
                if any(coef_tmp[coef_tmp < 0]): sys_summary[i-1,1] = np.mean(abs(coef_tmp[coef_tmp < 0]))
            else:
                try:
                    sys_summary[i-1,0] = np.mean(coef_tmp[coef_tmp != 0])
                except:
                    sys_summary[i-1,0] = 0
                
        elif method == 'median':
            if signed == True:
                if any(coef_tmp[coef_tmp > 0]): sys_summary[i-1,0] = np.median(abs(coef_tmp[coef_tmp > 0]))
                if any(coef_tmp[coef_tmp < 0]): sys_summary[i-1,1] = np.median(abs(coef_tmp[coef_tmp < 0]))
            else:
                try:
                    sys_summary[i-1,0] = np.median(coef_tmp[coef_tmp != 0])
                except:
                    sys_summary[i-1,0] = 0
                    
        elif method == 'max':
            if signed == True:
                if any(coef_tmp[coef_tmp > 0]): sys_summary[i-1,0] = np.max(abs(coef_tmp[coef_tmp > 0]))
                if any(coef_tmp[coef_tmp < 0]): sys_summary[i-1,1] = np.max(abs(coef_tmp[coef_tmp < 0]))
            else:
                try:
                    sys_summary[i-1,0] = np.max(coef_tmp[coef_tmp != 0])
                except:
                    sys_summary[i-1,0] = 0

        if np.any(np.isnan(sys_summary)):
            sys_summary[np.isnan(sys_summary)] = 0

    return sys_summary


def prop_bar_plot(sys_prop, sys_summary, labels = '', which_colors = 'yeo17', axlim = 'auto', title_str = '', fig_size = [4,4]):
    f, ax = plt.subplots()
    f.set_figwidth(fig_size[0])
    f.set_figheight(fig_size[1])

    y_pos = np.arange(1,sys_prop.shape[0]+1)

    if which_colors == 'solid':
        cmap = get_cmap(which_type = 'redblu_pair', num_classes = 2)
        ax.barh(y_pos, sys_prop[:,0], color = cmap[0], edgecolor = 'k', align='center')
        if sys_prop.shape[1] == 2:
            ax.barh(y_pos, -sys_prop[:,1], color = cmap[1], edgecolor = 'k', align='center')
        ax.axvline(linewidth = 1, color = 'k')
    elif which_colors == 'opac_scaler':
        cmap = get_cmap(which_type = 'redblu_pair', num_classes = 2)
        for i in range(sys_prop.shape[0]):
            ax.barh(y_pos[i], sys_prop[i,0], facecolor = np.append(cmap[0], sys_summary[i,0]), edgecolor = 'k', align='center')
            if sys_prop.shape[1] == 2:
                ax.barh(y_pos[i], -sys_prop[i,1], facecolor = np.append(cmap[1], sys_summary[i,1]), edgecolor = 'k', align='center')
        ax.axvline(linewidth = 1, color = 'k')
    else:
        cmap = get_cmap(which_type = which_colors, num_classes = sys_prop.shape[0])
        ax.barh(y_pos, sys_prop[:,0], color = cmap, edgecolor = 'k', align='center')
        if sys_prop.shape[1] == 2:
            ax.barh(y_pos, -sys_prop[:,1], color = cmap, edgecolor = 'k', align='center')
        ax.axvline(linewidth = 1, color = 'k')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)        
    ax.invert_yaxis() # labels read top-to-bottom

    if axlim == 'auto':
        anchors = np.array([0.2, 0.4, 0.6, 0.8, 1])
        the_max = np.round(np.max(sys_prop),2)
        ax_anchor = anchors[find_nearest_above(anchors, the_max)]
        ax.set_xlim([-ax_anchor-ax_anchor*.05, ax_anchor+ax_anchor*.05])
    else:
        if axlim == 0.2:
            ax.set_xticks(np.arange(-axlim, axlim+0.1, 0.1))
        if axlim == 0.1:
            ax.set_xticks(np.arange(-axlim, axlim+0.05, 0.05))
        else:
            ax.set_xlim([-axlim, axlim])

    ax.xaxis.grid(True, which='major')

    ax.xaxis.tick_top()
    if sys_prop.shape[1] == 2:
        ax.set_xticklabels([str(abs(np.round(x,2))) for x in ax.get_xticks()])
    ax.set_title(title_str)

    # Hide the right and top spines
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.show()

    return f, ax


def perc_dev(Z, thr = 2.6, sign = 'abs'):
    if sign == 'abs':
        bol = np.abs(Z) > thr;
    elif sign == 'pos':
        bol = Z > thr;
    elif sign == 'neg':
        bol = Z < -thr;
    
    # count the number that have supra-threshold z-stats and store as percentage
    Z_perc = np.sum(bol, axis = 1) / Z.shape[1] * 100
    
    return Z_perc

