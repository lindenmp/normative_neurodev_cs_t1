#!/usr/bin/env python
# coding: utf-8

# # Results, section 3: group analysis of deviations

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
from func import my_get_cmap, run_corr, get_fdr_p, run_pheno_correlations, dependent_corr, get_sys_summary, get_fdr_p_df, get_sys_prop, run_ttest, get_cohend, create_dummy_vars, perc_dev


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


# # Group analysis

# ## Create grouping variable

# In[9]:


groups = ('goassessSmryDep', 'goassessSmryAdd')
group_idx = create_dummy_vars(df.loc[df.loc[:,train_test_str] == 1,:], groups)


# In[10]:


# pad with zeros to match dimensions with training set
zero_pad = np.zeros((np.sum(df[train_test_str] == 0),group_idx.shape[1])).astype(bool)
group_idx = np.append(zero_pad, group_idx, axis = 0); print(group_idx.shape)


# ### Demographics

# In[11]:


print(groups[0],
      'mean age:', df.loc[group_idx[:,0],primary_covariate].mean().round(2),
      '(', df.loc[group_idx[:,0],primary_covariate].std().round(2), ')')


# In[12]:


df.loc[group_idx[:,0],'sex_adj'].value_counts() / np.sum(group_idx[:,0]) * 100


# In[13]:


print(groups[1],
      'mean age:', df.loc[group_idx[:,1],primary_covariate].mean().round(2),
      '(', df.loc[group_idx[:,1],primary_covariate].std().round(2), ')')


# In[14]:


df.loc[group_idx[:,1],'sex_adj'].value_counts() / np.sum(group_idx[:,1]) * 100


# In[15]:


sp.stats.ttest_ind(df.loc[group_idx[:,0],primary_covariate], df.loc[group_idx[:,1],primary_covariate])


# In[16]:


# get pheno correlations without subjects in the diagnostic groups
# note, these correlations are generated prior to regressing out age/sex below to keep them in line with what was done in the primary analyses (results_s2_phenos)
# note, in the primary analyses i found that controlling for age/sex in the z scores made no difference to the results which is why it's not included
df_pheno_z_woutgroups = run_pheno_correlations(df.loc[group_idx.sum(axis = 1) == 0,phenos], df_z.loc[group_idx.sum(axis = 1) == 0,:], method = method, assign_p = 'parametric')


# In[17]:


df_pheno_z_woutgroups.head()


# ### Regress age/sex out of node features

# In[18]:


# Given differences in demographics between groups, we regress age/sex from z scores before calculating cohen's D
df_nuis = df.loc[:,[primary_covariate,'sex_adj']]
df_nuis = sm.add_constant(df_nuis)

# df_z
cols = df_z.columns
mdl = sm.OLS(df_z.loc[:,cols], df_nuis).fit()
y_pred = mdl.predict(df_nuis)
y_pred.columns = cols
df_z.loc[:,cols] = df_z.loc[:,cols] - y_pred


# ## Disorder groups vs. HCs

# In[19]:


for i, group in enumerate(groups):
    i_idx = group_idx[:,i]
    for metric in metrics:
        df_y = df_z.loc[df[train_test_str] == 0,region_filter].filter(regex = metric, axis = 1)
        df_x = df_z.loc[i_idx,region_filter].filter(regex = metric, axis = 1)
        df_out = run_ttest(df_x, df_y, tail = 'one')
        print(group+'_'+metric + ', significant t-tests:', (np.sum(df_out.loc[:,'p-corr'] < .05) / df_out.shape[0] * 100))


# In[20]:


metric = metrics[1]; print(metric)


# ### Spatial correlations

# In[21]:


df_y = df_z.loc[df[train_test_str] == 0,:].filter(regex = metric, axis = 1)

X = get_cohend(df_z.loc[group_idx[:,0],:].filter(regex = metric, axis = 1), df_y).loc[:,'d'].rename('Depression')
y = get_cohend(df_z.loc[group_idx[:,1],:].filter(regex = metric, axis = 1), df_y).loc[:,'d'].rename('ADHD')
# df_input = pd.concat((df_pheno_z_woutgroups.loc[pheno,'coef'].filter(regex = metric).astype(float), X, y), axis = 1)
df_input = pd.concat((X, y), axis = 1)

df_input.head()


# In[22]:


sns.set(style='white', context = 'paper', font_scale = 1)
f = sns.jointplot(x = df_input.loc[:,'Depression'], y = df_input.loc[:,'ADHD'], kind="reg")
f.fig.set_figwidth(2.5)
f.fig.set_figheight(2.5)
f.set_axis_labels('Depression (Cohen\'s D)', 'ADHD (Cohen\'s D)')
f.plot_joint(plt.scatter, c = "k", s = 5, linewidth = 2, marker = ".", alpha = 0.3)
f.annotate(sp.stats.pearsonr)
f.ax_joint.collections[0].set_alpha(0)
f.ax_joint.set_ylim([-0.5, 0.3])
f.ax_joint.set_xlim([-0.5, 0.3])
f.savefig('cohensd_spatial_corr_depression->adhd.svg', dpi = 300, bbox_inches = 'tight', pad_inches = 0)


# In[23]:


# f = sns.jointplot(x = df_input.loc[:,'coef'], y = df_input.loc[:,'Depression'], kind="reg")
# f.fig.set_figwidth(2.5)
# f.fig.set_figheight(2.5)
# f.set_axis_labels(phenos_label[p] + ' ($\\rho$)', 'Depression (Cohen\'s D)')
# f.plot_joint(plt.scatter, c = "k", s = 5, linewidth = 2, marker = ".", alpha = 0.3)
# f.annotate(sp.stats.pearsonr)
# f.ax_joint.collections[0].set_alpha(0)
# # f.ax_joint.set_xticks([-0.15, -0.05, 0.05])
# f.ax_joint.set_ylim([-0.5, 0.3])
# f.ax_joint.set_xlim([-0.175, 0.075])
# f.savefig('cohensd_spatial_corr_'+pheno+'->depression.svg', dpi = 300, bbox_inches = 'tight', pad_inches = 0)


# In[24]:


# f = sns.jointplot(x = df_input.loc[:,'coef'], y = df_input.loc[:,'ADHD'], kind="reg")
# f.fig.set_figwidth(2.5)
# f.fig.set_figheight(2.5)
# f.set_axis_labels(phenos_label[p] + ' ($\\rho$)', 'ADHD (Cohen\'s D)')
# f.plot_joint(plt.scatter, c = "k", s = 5, linewidth = 2, marker = ".", alpha = 0.3)
# f.annotate(sp.stats.pearsonr)
# f.ax_joint.collections[0].set_alpha(0)
# # f.ax_joint.set_xticks([-0.15, -0.05, 0.05])
# f.ax_joint.set_ylim([-0.5, 0.3])
# f.ax_joint.set_xlim([-0.175, 0.075])
# f.savefig('cohensd_spatial_corr_'+pheno+'->ADHD.svg', dpi = 300, bbox_inches = 'tight', pad_inches = 0)


# ### Regress p out of node features

# In[25]:


corr_diff = np.zeros(len(phenos))


# In[26]:


for pheno in phenos:
    df_nuis = df.loc[:,[primary_covariate,'sex_adj',pheno]]
    df_nuis = sm.add_constant(df_nuis)

    # df_z
    cols = df_z.columns
    mdl = sm.OLS(df_z.loc[:,cols], df_nuis).fit()
    y_pred = mdl.predict(df_nuis)
    y_pred.columns = cols
    df_z_regr = df_z.loc[:,cols] - y_pred

    df_y = df_z_regr.loc[df[train_test_str] == 0,:].filter(regex = metric, axis = 1)

    X = get_cohend(df_z_regr.loc[group_idx[:,0],:].filter(regex = metric, axis = 1), df_y).loc[:,'d'].rename('Depression')
    y = get_cohend(df_z_regr.loc[group_idx[:,1],:].filter(regex = metric, axis = 1), df_y).loc[:,'d'].rename('ADHD')
    df_input_p = pd.concat((df_pheno_z.loc[pheno,'coef'].filter(regex = metric).astype(float), X, y), axis = 1)
    
    print(pheno, np.round(sp.stats.pearsonr(X,y),2))


# In[27]:


p = 0
pheno = phenos[p]; print(pheno)


# In[28]:


# Given differences in demographics between groups, we regress age/sex from z scores before calculating cohen's D
# df_nuis = df.loc[:,pheno]
df_nuis = df.loc[:,[primary_covariate,'sex_adj',pheno]]
df_nuis = sm.add_constant(df_nuis)

# df_z
cols = df_z.columns
mdl = sm.OLS(df_z.loc[:,cols], df_nuis).fit()
y_pred = mdl.predict(df_nuis)
y_pred.columns = cols
df_z_regr = df_z.loc[:,cols] - y_pred


# In[29]:


df_y = df_z_regr.loc[df[train_test_str] == 0,:].filter(regex = metric, axis = 1)

X = get_cohend(df_z_regr.loc[group_idx[:,0],:].filter(regex = metric, axis = 1), df_y).loc[:,'d'].rename('Depression')
y = get_cohend(df_z_regr.loc[group_idx[:,1],:].filter(regex = metric, axis = 1), df_y).loc[:,'d'].rename('ADHD')
df_input_p = pd.concat((df_pheno_z.loc[pheno,'coef'].filter(regex = metric).astype(float), X, y), axis = 1)

df_input_p.head()


# In[30]:


sns.set(style='white', context = 'paper', font_scale = 1)
f = sns.jointplot(x = df_input_p.loc[:,'Depression'], y = df_input_p.loc[:,'ADHD'], kind="reg")
f.fig.set_figwidth(2.5)
f.fig.set_figheight(2.5)
f.set_axis_labels('Depression (Cohen\'s D)', 'ADHD (Cohen\'s D)')
f.plot_joint(plt.scatter, c = "k", s = 5, linewidth = 2, marker = ".", alpha = 0.3)
f.annotate(sp.stats.pearsonr)
f.ax_joint.collections[0].set_alpha(0)
f.ax_joint.set_ylim([-0.3, 0.3])
f.ax_joint.set_xlim([-0.3, 0.3])
f.savefig('cohensd_spatial_corr_depression->adhd_p_nuis.svg', dpi = 300, bbox_inches = 'tight', pad_inches = 0)


# In[31]:


cmap = my_get_cmap('pair')


# In[32]:


f, ax = plt.subplots()
f.set_figwidth(2.5)
f.set_figheight(1)
sns.kdeplot(df_input.loc[:,'ADHD'], ax = ax, color = cmap[0], label = 'no p')
sns.kdeplot(df_input_p.loc[:,'ADHD'], ax = ax, color = cmap[1], label = 'p')
ax.legend().remove()
ax.set_ylim([0, 6])
ax.set_xlim([-0.5, 0.3])
ax.set_xlabel('ADHD (Cohen\'s D)')
ax.set_ylabel('density')
f.savefig('cohensd_adhd.svg', dpi = 300, bbox_inches = 'tight', pad_inches = 0)


# In[33]:


sp.stats.ttest_rel(df_input_p.loc[:,'ADHD'],df_input.loc[:,'ADHD'])


# In[34]:


f, ax = plt.subplots()
f.set_figwidth(2.5)
f.set_figheight(1)
sns.kdeplot(df_input.loc[:,'Depression'], ax = ax, color = cmap[0], label = 'no p')
sns.kdeplot(df_input_p.loc[:,'Depression'], ax = ax, color = cmap[1], label = 'p')
ax.legend().remove()
ax.set_ylim([0, 6])
ax.set_xlim([-0.5, 0.3])
ax.set_xlabel('Depression (Cohen\'s D)')
ax.set_ylabel('density')
f.savefig('cohensd_depression.svg', dpi = 300, bbox_inches = 'tight', pad_inches = 0)


# In[35]:


sp.stats.ttest_rel(df_input_p.loc[:,'Depression'],df_input.loc[:,'Depression'])

