# Functions for project: NormativeNeuroDev_CrossSec
# This project used normative modelling to examine network control theory metrics
# Linden Parkes, 2019
# lindenmp@seas.upenn.edu

# Essentials
import os, sys, glob
import pandas as pd
import numpy as np
import nibabel as nib

# Stats
import scipy as sp
from scipy import stats
import statsmodels.api as sm
import pingouin as pg

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

from IPython.display import clear_output
from scipy.stats import t
from numpy.matlib import repmat 
from scipy.linalg import svd, schur
from statsmodels.stats import multitest


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


def summarise_network(df_z, network_idx, roi_loc, metrics = ('ct',), method = 'mean'):
    """ Get system averages of input dataframe """

    df_out = pd.DataFrame()
    for metric in metrics:
        if metric == 'ct':
            if method == 'median': df_tmp = df_z.filter(regex = metric).groupby(network_idx[roi_loc == 1], axis = 1).median()
            if method == 'mean': df_tmp = df_z.filter(regex = metric).groupby(network_idx[roi_loc == 1], axis = 1).mean()
            if method == 'max': df_tmp = df_z.filter(regex = metric).groupby(network_idx[roi_loc == 1], axis = 1).max()
            
            my_list = [metric + '_' + str(i) for i in np.unique(network_idx[roi_loc == 1]).astype(int)]
            df_tmp.columns = my_list
        else:
            if method == 'median': df_tmp = df_z.filter(regex = metric).groupby(network_idx, axis = 1).median()
            if method == 'mean': df_tmp = df_z.filter(regex = metric).groupby(network_idx, axis = 1).mean()
            if method == 'max': df_tmp = df_z.filter(regex = metric).groupby(network_idx, axis = 1).max()
            
            my_list = [metric + '_' + str(i) for i in np.unique(network_idx).astype(int)]
            df_tmp.columns = my_list

        df_out = pd.concat((df_out, df_tmp), axis = 1)

    return df_out


def run_corr(my_series, my_dataframe, method = 'pearsonr'):
    """ Simple correlation between pandas series and columns in a dataframe """
    df_corr = pd.DataFrame(index = my_dataframe.columns, columns = ['coef', 'p'])
    if method == 'spearmanr':
        for i, row in df_corr.iterrows():
            df_corr.loc[i] = sp.stats.spearmanr(my_series, my_dataframe[i])
    elif method == 'pearsonr':
        for i, row in df_corr.iterrows():
            df_corr.loc[i] = sp.stats.pearsonr(my_series, my_dataframe[i])

    return df_corr


def get_fdr_p(p_vals, alpha = 0.05):
    out = multitest.multipletests(p_vals, alpha = alpha, method = 'fdr_bh')
    p_fdr = out[1] 

    return p_fdr


def get_fdr_p_df(p_vals, alpha = 0.05):
    p_fdr = pd.DataFrame(index = p_vals.index,
                        columns = p_vals.columns,
                        data = np.reshape(get_fdr_p(p_vals.values.flatten(), alpha = alpha), p_vals.shape))

    return p_fdr


def compute_null(df, df_z, num_perms = 1000, method = 'pearson'):
    np.random.seed(0)
    null = np.zeros((num_perms,df_z.shape[1]))

    for i in range(num_perms):
        if i%10 == 0: update_progress(i/num_perms, df.name)
        null[i,:] = df_z.reset_index(drop = True).corrwith(df.sample(frac = 1).reset_index(drop = True), method = method)
    update_progress(1, df.name)   
    
    return null


def get_null_p(coef, null):

    num_perms = null.shape[0]
    num_vars = len(coef)
    p_perm = np.zeros((num_vars,))

    for i in range(num_vars):
        r_obs = abs(coef[i])
        r_perm = abs(null[:,i])
        p_perm[i] = np.sum(r_perm >= r_obs) / num_perms

    return p_perm


def run_pheno_correlations(df_phenos, df_z, method = 'pearson', assign_p = 'permutation', nulldir = os.getcwd()):
    df_out = pd.DataFrame(columns = ['pheno','variable','coef', 'p'])
    phenos = df_phenos.columns
    
    for pheno in phenos:
        df_tmp = pd.DataFrame(index = df_z.columns, columns = ['coef', 'p'])
        if assign_p == 'permutation':
            # Get true correlation
            df_tmp.loc[:,'coef'] = df_z.corrwith(df_phenos.loc[:,pheno], method = method)
            # Get null
            if os.path.exists(os.path.join(nulldir,'null_' + pheno + '_' + method + '.npy')): # if null exists, load it
                null = np.load(os.path.join(nulldir,'null_' + pheno + '_' + method + '.npy')) 
            else: # otherwise, compute and save it out
                null = compute_null(df_phenos.loc[:,pheno], df_z, num_perms = 1000, method = method)
                np.save(os.path.join(nulldir,'null_' + pheno + '_' + method), null)
            # Compute p-values using null
            df_tmp.loc[:,'p'] = get_null_p(df_tmp.loc[:,'coef'].values, null)
        elif assign_p == 'parametric':
            if method == 'pearson':
                for col in df_z.columns:
                    df_tmp.loc[col,'coef'] = sp.stats.pearsonr(df_phenos.loc[:,pheno], df_z.loc[:,col])[0]
                    df_tmp.loc[col,'p'] = sp.stats.pearsonr(df_phenos.loc[:,pheno], df_z.loc[:,col])[1]
            if method == 'spearman':
                for col in df_z.columns:
                    df_tmp.loc[col,'coef'] = sp.stats.spearmanr(df_phenos.loc[:,pheno], df_z.loc[:,col])[0]
                    df_tmp.loc[col,'p'] = sp.stats.spearmanr(df_phenos.loc[:,pheno], df_z.loc[:,col])[1]    
        # append
        df_tmp.reset_index(inplace = True); df_tmp.rename(index=str, columns={'index': 'variable'}, inplace = True); df_tmp['pheno'] = pheno
        df_out = df_out.append(df_tmp, sort = False)
    df_out.set_index(['pheno','variable'], inplace = True)
    
    return df_out


def run_pheno_partialcorrs(df_phenos, df_z, method = 'pearson'):
    df_input = pd.concat((df_phenos, df_z), axis = 1)
    if method == 'pearson': df_out = pd.DataFrame(columns = ['pheno','variable','coef', 'p', 'BF10'])
    else: df_out = pd.DataFrame(columns = ['pheno','variable','coef', 'p'])
    phenos = list(df_phenos.columns)
    
    for pheno in phenos:
        print(pheno)
        if method == 'pearson': df_tmp = pd.DataFrame(index = df_z.columns, columns = ['coef', 'p', 'BF10'])
        else: df_tmp = pd.DataFrame(index = df_z.columns, columns = ['coef', 'p'])
        
        phenos_cov = phenos.copy(); phenos_cov.remove(pheno)
        results = pg.pairwise_corr(data = df_input, columns = [[pheno], list(df_z.columns)], covar = phenos_cov, method = method)
        results.set_index('Y', inplace = True)
        df_tmp.loc[:,'coef'] = results['r']; df_tmp.loc[:,'p'] = results['p-unc']
        if method == 'pearson': df_tmp.loc[:,'BF10'] = results['BF10'].astype(float)
        
        # append
        df_tmp.reset_index(inplace = True); df_tmp.rename(index=str, columns={'index': 'variable'}, inplace = True); df_tmp['pheno'] = pheno
        df_out = df_out.append(df_tmp, sort = False)
    df_out.set_index(['pheno','variable'], inplace = True)
    
    return df_out


# Create grouping variable
def create_dummy_vars(df, groups):
    dummy_vars = np.zeros((df.shape[0],1)).astype(bool)
    for i, group in enumerate(groups):
        x = df.loc[:,group].values == 4
        print(group+':', x.sum())
        x = x.reshape(-1,1)
        x = x.astype(bool)
        dummy_vars = np.append(dummy_vars, x, axis = 1)
    dummy_vars = dummy_vars[:,1:]
    
    # filter comorbid
    comorbid_diag = np.sum(dummy_vars, axis = 1) > 1
    print('Comorbid N:', comorbid_diag.sum())
    dummy_vars[comorbid_diag,:] = 0

    for i, group in enumerate(groups):
        print(group+':', dummy_vars[:,i].sum())
    
    return dummy_vars


def run_ttest(df_x, df_y = '', tail = 'two'):
    df_out = pd.DataFrame(index = df_x.columns)
    if type(df_y) == str:
        df_out.loc[:,'mean'] = df_x.mean(axis = 0)
        test = sp.stats.ttest_1samp(df_x, popmean = 0)
    else:
        df_out.loc[:,'mean_diff'] = df_x.mean(axis = 0) - df_y.mean(axis = 0)
        test = sp.stats.ttest_ind(df_x, df_y)
        
    df_out.loc[:,'tstat'] = test[0]
    df_out.loc[:,'p'] = test[1]
    
    if tail == 'one': df_out.loc[:,'p'] = df_out.loc[:,'p']/2
        
    df_out.loc[:,'p-corr'] = get_fdr_p(df_out.loc[:,'p'])
    
    return df_out


def get_cohend(df_x, df_y):
    df_out = pd.DataFrame(index = df_x.columns)
    df_out.loc[:,'mean_diff'] = df_x.mean(axis = 0) - df_y.mean(axis = 0)
    df_out.loc[:,'d'] = df_out.loc[:,'mean_diff'] / pd.concat((df_x,df_y), axis = 0).std()
    
    return df_out


def run_perm_test(df_x, df_y, num_perms = 1000):
    np.random.seed(0)
    num_vars = df_x.shape[1]
    df_out = pd.DataFrame(index = df_x.columns)
        
    # concatenate inputs and create labels
    df_in = pd.concat((df_x,df_y), axis = 0)
    labels = np.concatenate((np.ones(df_x.shape[0]), np.zeros(df_y.shape[0])))
    
    # get true mean difference (df_x - df_y)
    df_out.loc[:,'mean_diff'] = df_in.iloc[labels == 1,:].mean(axis = 0) - df_in.iloc[labels == 0,:].mean(axis = 0)
    
    # generate null
    null = np.zeros((num_perms,num_vars))
    for i in range(num_perms):
        np.random.shuffle(labels)
        null[i,:] = df_in.iloc[labels == 1,:].mean(axis = 0).values - df_in.iloc[labels == 0,:].mean(axis = 0).values
    
    # calculate two-sided p-value
    for i, row in enumerate(df_out.index):
        df_out.loc[row,'p'] = np.sum(null[:,i] >= df_out.loc[row,'mean_diff']) / num_perms
    
    # correct for multiple comparisons using FDR
    df_out.loc[:,'p-corr'] = get_fdr_p(df_out.loc[:,'p'])
        
    return df_out


def get_sys_prop(coef, p_vals, idx, alpha = 0.05):
    u_idx = np.unique(idx)
    sys_prop = np.zeros((len(u_idx),2))

    for i in u_idx:
        # filter regions by system idx
        coef_tmp = coef[idx == i]
        p_tmp = p_vals[idx == i]
        
        # threshold out non-sig coef
        coef_tmp = coef_tmp[p_tmp < alpha]

        # proportion of signed significant coefs within system i
        sys_prop[i-1,0] = coef_tmp[coef_tmp > 0].shape[0] / np.sum(idx == i)
        sys_prop[i-1,1] = coef_tmp[coef_tmp < 0].shape[0] / np.sum(idx == i)

    return sys_prop


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
        elif axlim == 0.1:
            ax.set_xticks(np.arange(-axlim, axlim+0.05, 0.05))
        elif axlim == 1:
            ax.set_xticks(np.arange(-axlim, axlim+0.5, 0.5))
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


def dependent_corr(xy, xz, yz, n, twotailed=True):
    """
    Calculates the statistic significance between two dependent correlation coefficients
    @param xy: correlation coefficient between x and y
    @param xz: correlation coefficient between x and z
    @param yz: correlation coefficient between y and z
    @param n: number of elements in x, y and z
    @param twotailed: whether to calculate a one or two tailed test, only works for 'steiger' method
    @param conf_level: confidence level, only works for 'zou' method
    @param method: defines the method uses, 'steiger' or 'zou'
    @return: t and p-val
    
    Author: Philipp Singer (www.philippsinger.info)
    copied on 20/1/2020 from https://github.com/psinger/CorrelationStats/blob/master/corrstats.py
    """
    d = xy - xz
    determin = 1 - xy * xy - xz * xz - yz * yz + 2 * xy * xz * yz
    av = (xy + xz)/2
    cube = (1 - yz) * (1 - yz) * (1 - yz)

    t2 = d * np.sqrt((n - 1) * (1 + yz)/(((2 * (n - 1)/(n - 3)) * determin + av * av * cube)))
    p = 1 - t.cdf(abs(t2), n - 3)

    if twotailed:
        p *= 2

    return t2, p

