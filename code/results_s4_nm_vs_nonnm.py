#!/usr/bin/env python
# coding: utf-8

# # Results, section 4: Comparing effects sizes to conventional analyses

# In[1]:


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
from mpl_toolkits.axes_grid1 import make_axes_locatable


# In[2]:


sys.path.append('/Users/lindenmp/Dropbox/Work/ResProjects/NormativeNeuroDev_CrossSec_T1/code/func/')
from proj_environment import set_proj_env
sys.path.append('/Users/lindenmp/Dropbox/Work/git/pyfunc/')
from func import run_corr, get_fdr_p, run_pheno_correlations, dependent_corr, get_sys_summary, get_fdr_p_df, get_sys_prop, run_ttest, get_cohend, create_dummy_vars


# In[3]:


train_test_str = 'squeakycleanExclude'
exclude_str = 't1Exclude' # 't1Exclude' 'fsFinalExclude'
parc_str = 'schaefer' # 'schaefer' 'lausanne'
parc_scale = 400 # 200 400 | 60 125
primary_covariate = 'ageAtScan1_Years'
extra_str = ''
parcel_names, parcel_loc, drop_parcels, num_parcels, yeo_idx, yeo_labels = set_proj_env(train_test_str = train_test_str, exclude_str = exclude_str,
                                                                            parc_str = parc_str, parc_scale = parc_scale, extra_str = extra_str)


# In[4]:


os.environ['NORMATIVEDIR']


# In[5]:


outdir = os.path.join(os.environ['NORMATIVEDIR'],'analysis_outputs')
if not os.path.exists(outdir): os.makedirs(outdir)


# In[6]:


phenos = ['Overall_Psychopathology','Psychosis_Positive','Psychosis_NegativeDisorg','AnxiousMisery','Externalizing','Fear']
phenos_label_short = ['Ov. Psych.', 'Psy. (pos.)', 'Psy. (neg.)', 'Anx.-mis.', 'Ext.', 'Fear']
phenos_label = ['Overall Psychopathology','Psychosis (Positive)','Psychosis (Negative)','Anxious-Misery','Externalizing','Fear']
metrics = ['ct', 'vol']
metrics_label_short = ['Thickness', 'Volume']
metrics_label = ['Thickness', 'Volume']

method = 'spearman'
assign_p = 'permutation'


# ## Plots

# In[7]:


if not os.path.exists(os.environ['FIGDIR']): os.makedirs(os.environ['FIGDIR'])
os.chdir(os.environ['FIGDIR'])
sns.set(style='white', context = 'paper', font_scale = 1)


# ## Load data

# In[8]:


df = pd.read_csv(os.path.join(outdir,'df.csv')); df.set_index(['bblid', 'scanid'], inplace = True)
df_z = pd.read_csv(os.path.join(outdir,'df_z.csv')); df_z.set_index(['bblid', 'scanid'], inplace = True)
df_pheno_z = pd.read_csv(os.path.join(outdir,'df_pheno_z.csv')); df_pheno_z.set_index(['pheno', 'variable'], inplace = True)
region_filter = pd.read_csv(os.path.join(outdir,'region_filter.csv'), index_col=0); region_filter = region_filter.iloc[:,0].astype(bool)


# In[9]:


# Train
df_node_train = pd.read_csv(os.path.join(os.environ['NORMATIVEDIR'], 'resp_train.csv'))
df_node_train.set_index(['bblid', 'scanid'], inplace = True)

# Test
df_node_test = pd.read_csv(os.path.join(os.environ['NORMATIVEDIR'], 'resp_test.csv'))
df_node_test.set_index(['bblid', 'scanid'], inplace = True)

# concat
df_node = pd.concat((df_node_train, df_node_test), axis = 0); print(df_node.shape)


# ### Regress age/sex out of node features

# In[10]:


df_nuis = df.loc[:,[primary_covariate,'sex_adj']]
df_nuis = sm.add_constant(df_nuis)

# df_node
cols = df_node.columns
mdl = sm.OLS(df_node.loc[:,cols], df_nuis).fit()
y_pred = mdl.predict(df_nuis)
y_pred.columns = cols
df_node.loc[:,cols] = df_node.loc[:,cols] - y_pred


# ## Get pheno-metric relationships

# In[11]:


if assign_p == 'permutation':
    nulldir = os.path.join(os.environ['NORMATIVEDIR'], 'nulls')
    if not os.path.exists(nulldir): os.makedirs(nulldir)
    df_pheno = run_pheno_correlations(df.loc[:,phenos], df_node, method = method, assign_p = assign_p, nulldir = nulldir)
elif assign_p == 'parametric':
    df_pheno = run_pheno_correlations(df.loc[:,phenos], df_node, method = method, assign_p = assign_p)


# In[12]:


# correct multiple comparisons. We do this across brain regions and phenotypes (e.g., 400*6 = 2400 tests)
df_p_corr = pd.DataFrame(index = df_pheno.index, columns = ['p-corr']) # output dataframe

for metric in metrics:
    p_corr = get_fdr_p(df_pheno.loc[:,'p'].filter(regex = metric)) # correct p-values for metric
    p_corr_tmp = pd.DataFrame(index = df_pheno.loc[:,'p'].filter(regex = metric).index, columns = ['p-corr'], data = p_corr) # set to dataframe with correct indices
    df_pheno.loc[p_corr_tmp.index, 'p-corr'] = p_corr_tmp # store using index matching


# In[13]:


alpha = 0.05
print(alpha)


# In[14]:


x = df_pheno['p-corr'].values < alpha
df_pheno['sig'] = x

x = x.reshape(1,-1)
y = np.matlib.repmat(region_filter, 1, len(phenos))

my_bool = np.concatenate((x, y), axis = 0); region_filt = np.all(my_bool, axis = 0); df_pheno['sig_smse'] = region_filt

print(str(np.sum(df_pheno['sig'] == True)) + ' significant effects (fdr)')
print(str(np.sum(df_pheno['sig_smse'] == True)) + ' significant effects (fdr)')


# In[15]:


metric = 'vol'
num_regions = pd.DataFrame(index = [metric], columns = phenos)
counts_greater = pd.DataFrame(index = [metric], columns = phenos)
counts_smaller = pd.DataFrame(index = [metric], columns = phenos)

for j, pheno in enumerate(phenos):
    df_tmp = df_pheno.loc[pheno,['coef','sig_smse']].filter(regex = metric, axis = 0).copy()
    df_tmp_z = df_pheno_z.loc[pheno,['coef','sig_smse']].filter(regex = metric, axis = 0).copy()
    mask_idx = np.logical_and(df_tmp['sig_smse'],df_tmp_z['sig_smse'])
    num_regions.loc[metric,pheno] = mask_idx.sum()

    steiger = pd.DataFrame(index = df_tmp_z.index, columns = ['t2','p'])
    count_great = 0
    count_small = 0
    for col, _ in mask_idx[mask_idx].iteritems():
        xy = np.abs(df_tmp_z.loc[col,'coef']) # correlation between phenotype and deviation
        xz = np.abs(df_tmp.loc[col,'coef']) # correlation between phenotype and brain feature
        yz = np.abs(sp.stats.spearmanr(df_node[col],df_z[col])[0]) # correlation deviation and brain feature
        t2, p = dependent_corr(xy, xz, yz, df_z.shape[0], twotailed=True) # test for difference between correlations
        steiger.loc[col,'t2'] = t2
        steiger.loc[col,'p'] = p
    steiger['p_fdr'] = get_fdr_p(steiger['p'])

    # store
    counts_greater.loc[metric,pheno] = steiger[np.logical_and(steiger['t2'] > 0,steiger['p_fdr']<.05)].shape[0]
    counts_smaller.loc[metric,pheno] = steiger[np.logical_and(steiger['t2'] < 0,steiger['p_fdr']<.05)].shape[0]


# In[16]:


num_regions


# In[17]:


counts_greater / num_regions * 100


# In[18]:


counts_smaller / num_regions * 100

