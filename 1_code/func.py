# Linden Parkes, 2020
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

# Extra
from statsmodels.stats import multitest


def set_proj_env(dataset = 'PNC', train_test_str = 'squeakycleanExclude', exclude_str = 't1Exclude', parc_str = 'schaefer',
    parc_scale = 400, parc_variant = 'orig', primary_covariate = 'ageAtScan1_Years', extra_str = ''):

    # Project root directory
    projdir = '/Users/lindenmp/Google-Drive-Penn/work/research_projects/normative_neurodev_cs_t1'; os.environ['PROJDIR'] = projdir

    # Data directory
    datadir = os.path.join(projdir, '0_data'); os.environ['DATADIR'] = datadir

    # Imaging derivatives
    derivsdir = os.path.join('/Volumes/work_ssd/research_data/PNC/'); os.environ['DERIVSDIR'] = derivsdir

    # Pipeline directory
    pipelinedir = os.path.join(projdir, '2_pipeline'); os.environ['PIPELINEDIR'] = pipelinedir

    # Output directory
    outputdir = os.path.join(projdir, '3_output'); os.environ['OUTPUTDIR'] = outputdir

    # Cortical thickness directory
    ctdir = os.path.join(derivsdir, 'processedData/antsCorticalThickness'); os.environ['CTDIR'] = ctdir

    # Parcellation specifications
    # Names of parcels
    if parc_str == 'schaefer': parcel_names = np.genfromtxt(os.path.join(projdir, 'figs_support/labels/schaefer' + str(parc_scale) + 'NodeNames.txt'), dtype='str')
    # vector describing whether rois belong to cortex (1) or subcortex (0)
    if parc_str == 'schaefer': parcel_loc = np.loadtxt(os.path.join(projdir, 'figs_support/labels/schaefer' + str(parc_scale) + 'NodeNames_loc.txt'), dtype='int')

    drop_parcels = []
    num_parcels = parcel_names.shape[0]

    if parc_str == 'schaefer':
        ct_name_tmp = 'bblid/*xscanid/ct_schaefer' + str(parc_scale) + '_17.txt'; os.environ['CT_NAME_TMP'] = ct_name_tmp
        voldir = os.path.join(derivsdir, 'processedData/gm_vol_masks_native'); os.environ['VOLDIR'] = voldir
        vol_name_tmp = 'bblid/*xscanid/Schaefer2018_' + str(parc_scale) + '_17Networks_native_gm.nii.gz'; os.environ['VOL_NAME_TMP'] = vol_name_tmp

    # Yeo labels
    if parc_str == 'lausanne':
        yeo_idx = np.loadtxt(os.path.join(projdir, 'figs_support/labels/yeo7netlabelsLaus' + str(parc_scale) + '.txt')).astype(int)
        yeo_labels = ('Visual', 'Somatomator', 'Dorsal Attention', 'Ventral Attention', 'Limbic', 'Frontoparietal Control', 'Default Mode', 'Subcortical', 'Brainstem')
    elif parc_str == 'schaefer':
        yeo_idx = np.loadtxt(os.path.join(projdir, 'figs_support/labels/yeo17netlabelsSchaefer' + str(parc_scale) + '.txt')).astype(int)
        yeo_labels = ('Vis. A', 'Vis. B', 'Som. Mot. A', 'Som. Mot. B', 'Dors. Attn. A', 'Dors. Attn. B', 'Sal. Vent. Attn. A', 'Sal. Vent. Attn. B',
                    'Limbic A', 'Limbic B', 'Cont. A', 'Cont. B', 'Cont. C', 'Default A', 'Default B', 'Default C', 'Temp. Par.')
        # yeo_labels = ('V (A)', 'V (B)', 'SM (A)', 'SM (B)', 'DA (A)', 'DA (B)', 'SVA (A)', 'SVA (B)',
        #             'L (A)', 'L (B)', 'C (A)', 'C (B)', 'C (C)', 'D (A)', 'D (B)', 'D (C)', 'TP')

    return parcel_names, parcel_loc, drop_parcels, num_parcels, yeo_idx, yeo_labels


def my_get_cmap(which_type = 'qual1', num_classes = 8):
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


def rank_to_normal(rank, c, n):
    # Standard quantile function
    x = (rank - c) / (n - 2*c + 1)
    return sp.stats.norm.ppf(x)


def rank_int(series, c=3.0/8):
    # Check input
    assert(isinstance(series, pd.Series))
    assert(isinstance(c, float))

    # Set seed
    np.random.seed(123)

    # Drop NaNs
    series = series.loc[~pd.isnull(series)]

    # Get rank, ties are averaged
    rank = sp.stats.rankdata(series, method="average")

    # Convert numpy array back to series
    rank = pd.Series(rank, index=series.index)

    # Convert rank to normal distribution
    transformed = rank.apply(rank_to_normal, c=c, n=len(rank))
    
    return transformed


def get_synth_cov(df, cov = 'ageAtScan1_Years', stp = 1):
    # Synthetic cov data
    X_range = [np.min(df[cov]), np.max(df[cov])]
    X = np.arange(X_range[0],X_range[1],stp)
    X = X.reshape(-1,1)

    return X


def get_fdr_p(p_vals, alpha = 0.05):
    out = multitest.multipletests(p_vals, alpha = alpha, method = 'fdr_bh')
    p_fdr = out[1] 

    return p_fdr


def get_fdr_p_df(p_vals, alpha = 0.05, rows = False):
    
    if rows:
        p_fdr = pd.DataFrame(index = p_vals.index, columns = p_vals.columns)
        for row, data in p_vals.iterrows():
            p_fdr.loc[row,:] = get_fdr_p(data.values)
    else:
        p_fdr = pd.DataFrame(index = p_vals.index,
                            columns = p_vals.columns,
                            data = np.reshape(get_fdr_p(p_vals.values.flatten(), alpha = alpha), p_vals.shape))

    return p_fdr


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
        elif assign_p == 'none':
            df_tmp.loc[:,'coef'] = df_z.corrwith(df_phenos.loc[:,pheno], method = method)

        # append
        df_tmp.reset_index(inplace = True); df_tmp.rename(index=str, columns={'index': 'variable'}, inplace = True); df_tmp['pheno'] = pheno
        df_out = df_out.append(df_tmp, sort = False)
    df_out.set_index(['pheno','variable'], inplace = True)
    
    return df_out


def prop_bar_plot(sys_prop, sys_summary, labels = '', which_colors = 'yeo17', axlim = 'auto', title_str = '', fig_size = [4,4]):
    f, ax = plt.subplots()
    f.set_figwidth(fig_size[0])
    f.set_figheight(fig_size[1])

    y_pos = np.arange(1,sys_prop.shape[0]+1)

    if which_colors == 'solid':
        cmap = my_get_cmap(which_type = 'redblu_pair', num_classes = 2)
        ax.barh(y_pos, sys_prop[:,0], color = cmap[0], edgecolor = 'k', align='center')
        if sys_prop.shape[1] == 2:
            ax.barh(y_pos, -sys_prop[:,1], color = cmap[1], edgecolor = 'k', align='center')
        ax.axvline(linewidth = 1, color = 'k')
    elif which_colors == 'opac_scaler':
        cmap = my_get_cmap(which_type = 'redblu_pair', num_classes = 2)
        for i in range(sys_prop.shape[0]):
            ax.barh(y_pos[i], sys_prop[i,0], facecolor = np.append(cmap[0], sys_summary[i,0]), edgecolor = 'k', align='center')
            if sys_prop.shape[1] == 2:
                ax.barh(y_pos[i], -sys_prop[i,1], facecolor = np.append(cmap[1], sys_summary[i,1]), edgecolor = 'k', align='center')
        ax.axvline(linewidth = 1, color = 'k')
    else:
        cmap = my_get_cmap(which_type = which_colors, num_classes = sys_prop.shape[0])
        ax.barh(y_pos, sys_prop[:,0], color = cmap, linewidth = 0, align='center')
        if sys_prop.shape[1] == 2:
            ax.barh(y_pos, -sys_prop[:,1], color = cmap, linewidth = 0, align='center')
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
            ax.set_xticks(np.arange(axlim[0], axlim[1]+0.1, 0.1))
        elif axlim == 0.1:
            ax.set_xticks(np.arange(axlim[0], axlim[1]+0.05, 0.05))
        elif axlim == 1:
            ax.set_xticks(np.arange(axlim[0], axlim[1]+0.5, 0.5))
        else:
            ax.set_xlim([axlim[0], axlim[1]])

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


# Create grouping variable
def create_dummy_vars(df, groups, filter_comorbid = True):
    dummy_vars = np.zeros((df.shape[0],1)).astype(bool)
    for i, group in enumerate(groups):
        x = df.loc[:,group].values == 4
        print(group+':', x.sum())
        x = x.reshape(-1,1)
        x = x.astype(bool)
        dummy_vars = np.append(dummy_vars, x, axis = 1)
    dummy_vars = dummy_vars[:,1:]
    
    # filter comorbid
    if filter_comorbid:
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