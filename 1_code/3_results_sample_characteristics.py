#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Essentials
import os, sys, glob
import pandas as pd
import numpy as np
import nibabel as nib
import scipy.io as sio

# Stats
import scipy as sp
from scipy import stats
import statsmodels.api as sm
import pingouin as pg

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'


# In[2]:


sys.path.append('/Users/lindenmp/Google-Drive-Penn/work/research_projects/normative_neurodev_cs_t1/1_code/')
from func import set_proj_env, my_get_cmap, get_fdr_p


# In[3]:


train_test_str = 'train_test'
exclude_str = 't1Exclude' # 't1Exclude' 'fsFinalExclude'
parc_str = 'schaefer' # 'schaefer' 'lausanne'
parc_scale = 400 # 200 400 | 60 125 250
parcel_names, parcel_loc, drop_parcels, num_parcels, yeo_idx, yeo_labels = set_proj_env(exclude_str = exclude_str, parc_str = parc_str, parc_scale = parc_scale)


# In[4]:


# output file prefix
outfile_prefix = exclude_str+'_'+parc_str+'_'+str(parc_scale)+'_'
outfile_prefix


# ### Setup directory variables

# In[5]:


figdir = os.path.join(os.environ['OUTPUTDIR'], 'figs')
print(figdir)
if not os.path.exists(figdir): os.makedirs(figdir)


# In[6]:


labels = ['Train', 'Test']
phenos = ['Overall_Psychopathology','Psychosis_Positive','Psychosis_NegativeDisorg','AnxiousMisery','Externalizing','Fear']
phenos_label_short = ['Ov. Psych.', 'Psy. (pos.)', 'Psy. (neg.)', 'Anx.-mis.', 'Ext.', 'Fear']
phenos_label = ['Overall Psychopathology','Psychosis (Positive)','Psychosis (Negative)','Anxious-Misery','Externalizing','Fear']

# phenos = ['mood_4factorv2', 'psychosis_4factorv2','externalizing_4factorv2', 'phobias_4factorv2','overall_psychopathology_4factorv2']
# phenos_label = ['mood', 'psychosis','externalizing', 'phobias','overall_psychopathology']


# ## Setup plots

# In[7]:


if not os.path.exists(figdir): os.makedirs(figdir)
os.chdir(figdir)
sns.set(style='white', context = 'talk', font_scale = 0.8)
cmap = my_get_cmap('pair')


# ## Load data

# In[8]:


df = pd.read_csv(os.path.join(os.environ['PIPELINEDIR'], '1_compute_node_features', 'out', outfile_prefix+'df.csv'))
df.set_index(['bblid', 'scanid'], inplace = True)
print(df.shape)


# In[9]:


# train/test proportion
print('train N:', np.sum(df.loc[:,train_test_str] == 0))
print('test N:', np.sum(df.loc[:,train_test_str] == 1))


# In[10]:


df['sex'].unique()


# In[11]:


print(np.sum(df.loc[:,'sex'] == 1))
print((np.sum(df.loc[:,'sex'] == 1)/df.shape[0]) * 100)


# In[12]:


print(np.sum(df.loc[:,'sex'] == 2))
print((np.sum(df.loc[:,'sex'] == 2)/df.shape[0]) * 100)


# In[13]:


df['ageAtScan1_Years'].mean()


# In[14]:


df['ageAtScan1_Years'].std()


# In[15]:


for pheno in phenos:
    print(sp.stats.spearmanr(df.loc[:,'averageManualRating'],df.loc[:,pheno]))


# 0 = Male, 1 = Female

# In[16]:


# train/test proportion
print('train, sex = 1, N:', np.sum(df.loc[df.loc[:,train_test_str] == 0,'sex'] == 1))
print('train, sex = 2, N:',np.sum(df.loc[df.loc[:,train_test_str] == 0,'sex'] == 2))

print('test, sex = 1, N:', np.sum(df.loc[df.loc[:,train_test_str] == 1,'sex'] == 1))
print('test, sex = 2, N:',np.sum(df.loc[df.loc[:,train_test_str] == 1,'sex'] == 2))


# ### Sex

# In[17]:


stats = pd.DataFrame(index = phenos, columns = ['test_stat', 'pval'])

for i, pheno in enumerate(phenos):
    x = df.loc[df.loc[:,'sex'] == 1,pheno]
    y = df.loc[df.loc[:,'sex'] == 2,pheno]
    
    test_output = sp.stats.ttest_ind(x,y)
    stats.loc[pheno,'test_stat'] = test_output[0]
    stats.loc[pheno,'pval'] = test_output[1]
    
stats.loc[:,'pval_corr'] = get_fdr_p(stats.loc[:,'pval'])
stats.loc[:,'sig'] = stats.loc[:,'pval_corr'] < 0.05

stats


# In[18]:


f, ax = plt.subplots(1,len(phenos))
f.set_figwidth(len(phenos)*5)
f.set_figheight(5)

# sex: 1=male, 2=female
for i, pheno in enumerate(phenos):
    x = df.loc[df.loc[:,'sex'] == 1,pheno]
    sns.histplot(x, ax = ax[i], label = 'male', color = 'b')

    y = df.loc[df.loc[:,'sex'] == 2,pheno]
    sns.histplot(y, ax = ax[i], label = 'female', color = 'r')
    
    if i == 0:
        ax[i].legend()
    ax[i].set_xlabel(pheno)

    if stats.loc[pheno,'sig']:
        ax[i].set_title('t-stat:' + str(np.round(stats.loc[pheno,'test_stat'],2)) + ', p-value: ' + str(np.round(stats.loc[pheno,'pval_corr'],4)), fontweight="bold")
    else:
        ax[i].set_title('t-stat:' + str(np.round(stats.loc[pheno,'test_stat'],2)) + ', p-value: ' + str(np.round(stats.loc[pheno,'pval_corr'],4)))
    ax[i].tick_params(pad = -2)
        
f.savefig(outfile_prefix+'symptoms_distributions_sex.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0)


# ### Age

# In[19]:


stats = pd.DataFrame(index = phenos, columns = ['r', 'pval'])
xvar = 'ageAtScan1_Years'
x = df[xvar]
for i, pheno in enumerate(phenos):
    y = df[pheno]
    r,p = sp.stats.pearsonr(x,y)
    
    stats.loc[pheno,'r'] = r
    stats.loc[pheno,'pval'] = p
    
stats.loc[:,'pval_corr'] = get_fdr_p(stats.loc[:,'pval'])
stats.loc[:,'sig'] = stats.loc[:,'pval_corr'] < 0.05

print(stats)

f, ax = plt.subplots(1,len(phenos))
f.set_figwidth(len(phenos)*5)
f.set_figheight(5)

x = df[xvar]
for i, pheno in enumerate(phenos):
    y = df[pheno]
    plot_data = pd.merge(x,y, on=['bblid','scanid'])
    sns.regplot(x = xvar, y = pheno, data = plot_data, ax=ax[i])
    
    if stats.loc[pheno,'sig']:
        ax[i].set_title('r:' + str(np.round(stats.loc[pheno,'r'],2)) + ', p-value: ' + str(np.round(stats.loc[pheno,'pval_corr'],4)), fontweight="bold")
    else:
        ax[i].set_title('r:' + str(np.round(stats.loc[pheno,'r'],2)) + ', p-value: ' + str(np.round(stats.loc[pheno,'pval_corr'],4)))
    ax[i].tick_params(pad = -2)
    
f.savefig(outfile_prefix+'symptoms_correlations_age.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0)


# ### Age

# In[20]:


stats = pd.DataFrame(index = phenos, columns = ['r', 'pval'])
xvar = 'medu1'
x = df[xvar]
for i, pheno in enumerate(phenos):
    y = df[pheno]
    r,p = sp.stats.pearsonr(x,y)
    
    stats.loc[pheno,'r'] = r
    stats.loc[pheno,'pval'] = p
    
stats.loc[:,'pval_corr'] = get_fdr_p(stats.loc[:,'pval'])
stats.loc[:,'sig'] = stats.loc[:,'pval_corr'] < 0.05

print(stats)

f, ax = plt.subplots(1,len(phenos))
f.set_figwidth(len(phenos)*5)
f.set_figheight(5)

x = df[xvar]
for i, pheno in enumerate(phenos):
    y = df[pheno]
    plot_data = pd.merge(x,y, on=['bblid','scanid'])
    sns.regplot(x = xvar, y = pheno, data = plot_data, ax=ax[i])
    
    if stats.loc[pheno,'sig']:
        ax[i].set_title('r:' + str(np.round(stats.loc[pheno,'r'],2)) + ', p-value: ' + str(np.round(stats.loc[pheno,'pval_corr'],4)), fontweight="bold")
    else:
        ax[i].set_title('r:' + str(np.round(stats.loc[pheno,'r'],2)) + ', p-value: ' + str(np.round(stats.loc[pheno,'pval_corr'],4)))
    ax[i].tick_params(pad = -2)
    
f.savefig(outfile_prefix+'symptoms_correlations_medu.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0)


# ### Diagnostic table

# In[21]:


# to_screen = ['goassessSmryPsy', 'goassessSmryMood', 'goassessSmryEat', 'goassessSmryAnx', 'goassessSmryBeh']
# counts = np.sum(df.loc[:,to_screen] == 4)
# print(counts)
# print(counts/df.shape[0]*100)


# In[22]:


to_screen = ['goassessSmryPsy','goassessSmryMan', 'goassessSmryDep', 'goassessSmryBul', 'goassessSmryAno', 'goassessSmrySoc',
             'goassessSmryPan', 'goassessSmryAgr', 'goassessSmryOcd', 'goassessSmryPtd', 'goassessSmryAdd',
            'goassessSmryOdd', 'goassessSmryCon']
counts = np.sum(df.loc[:,to_screen] == 4)
print(counts)
print(counts/df.shape[0]*100)


# In[23]:


to_keep = counts[counts >= 50].index
list(to_keep)


# In[24]:


counts[counts >= 50]


# In[25]:


my_xticklabels = ['Psychosis spectrum (n=96)',
                 'Depression (n=191)',
                 'Social anxiety disorder (n=318)',
                 'Agoraphobia (n=77)',
                 'PTSD (n=168)',
                 'ADHD (n=226)',
                 'ODD (n=448)',
                 'Conduct disorder (n=114)']


# In[26]:


sns.set(style='white', context = 'paper', font_scale = 1)


# In[27]:


f, ax = plt.subplots(1,len(phenos))
f.set_figwidth(len(phenos)*2.5)
f.set_figheight(2)

for i, pheno in enumerate(phenos):
    mean_scores = np.zeros(len(to_keep))
    for j, diagnostic_score in enumerate(to_keep):
        idx = df.loc[:,diagnostic_score] == 4
        mean_scores[j] = df.loc[idx,pheno].mean()
    
    ax[i].bar(x = np.arange(0,len(mean_scores)), height = mean_scores, color = 'w', edgecolor = 'k', linewidth = 1.5)
    ax[i].set_ylim([-.2,1.2])
    ax[i].set_xticks(np.arange(0,len(mean_scores)))
    ax[i].set_xticklabels(my_xticklabels, rotation = 90)
    ax[i].tick_params(pad = -2)
    ax[i].set_title(phenos_label[i])
    if i == 1:
        ax[i].set_xlabel('Diagnostic group')
    if i == 0:
        ax[i].set_ylabel('Factor score (z)')
    
f.savefig(outfile_prefix+'symptom_dimensions_groups.svg', dpi = 300, bbox_inches = 'tight', pad_inches = 0)

