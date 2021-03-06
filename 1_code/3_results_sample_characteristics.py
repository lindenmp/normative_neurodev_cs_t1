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
phenos_label_short = ['Ov. psych.', 'Psy. (pos.)', 'Psy. (neg.)', 'Anx.-mis.', 'Ext.', 'Fear']
phenos_label = ['Overall psychopathology','Psychosis (positive)','Psychosis (negative)','Anxious-misery','Externalizing','Fear']


# ## Setup plots

# In[7]:


if not os.path.exists(figdir): os.makedirs(figdir)
os.chdir(figdir)
sns.set(style='white', context = 'paper', font_scale = 0.8)
sns.set_style({'font.family':'sans-serif', 'font.sans-serif':['Public Sans']})
cmap = my_get_cmap('pair')


# ## Load data

# In[8]:


df = pd.read_csv(os.path.join(os.environ['PIPELINEDIR'], '1_compute_node_features', 'out', outfile_prefix+'df.csv'))
df.set_index(['bblid', 'scanid'], inplace = True)
print(df.shape)


# In[9]:


df['ageAtScan1_Years'].mean()


# In[10]:


df['ageAtScan1_Years'].std()


# In[11]:


df['sex'].unique()


# In[12]:


print(np.sum(df.loc[:,'sex'] == 1))
print(np.round((np.sum(df.loc[:,'sex'] == 1)/df.shape[0]) * 100,2))


# In[13]:


print(np.sum(df.loc[:,'sex'] == 2))
print(np.round((np.sum(df.loc[:,'sex'] == 2)/df.shape[0]) * 100,2))


# In[14]:


np.sum(df.loc[:,'averageManualRating'] == 2)


# In[15]:


# train/test proportion
print('train N:', np.sum(df.loc[:,train_test_str] == 0))
print(np.round(df.loc[df.loc[:,train_test_str] == 0,'ageAtScan1_Years'].mean(),2))
print(np.round(df.loc[df.loc[:,train_test_str] == 0,'ageAtScan1_Years'].std(),2))


# In[16]:


print('test N:', np.sum(df.loc[:,train_test_str] == 1))
print(np.round(df.loc[df.loc[:,train_test_str] == 1,'ageAtScan1_Years'].mean(),2))
print(np.round(df.loc[df.loc[:,train_test_str] == 1,'ageAtScan1_Years'].std(),2))


# 0 = Male, 1 = Female

# In[17]:


# train/test proportion
print('train, sex = 1, N:', np.sum(df.loc[df.loc[:,train_test_str] == 0,'sex'] == 1))
print(np.round((np.sum(df.loc[df.loc[:,train_test_str] == 0,'sex'] == 1)/np.sum(df.loc[:,train_test_str] == 0)) * 100,2))
print('train, sex = 2, N:',np.sum(df.loc[df.loc[:,train_test_str] == 0,'sex'] == 2))
print(np.round((np.sum(df.loc[df.loc[:,train_test_str] == 0,'sex'] == 2)/np.sum(df.loc[:,train_test_str] == 0)) * 100,2))

print('test, sex = 1, N:', np.sum(df.loc[df.loc[:,train_test_str] == 1,'sex'] == 1))
print(np.round((np.sum(df.loc[df.loc[:,train_test_str] == 1,'sex'] == 1)/np.sum(df.loc[:,train_test_str] == 1)) * 100,2))
print('test, sex = 2, N:',np.sum(df.loc[df.loc[:,train_test_str] == 1,'sex'] == 2))
print(np.round((np.sum(df.loc[df.loc[:,train_test_str] == 1,'sex'] == 2)/np.sum(df.loc[:,train_test_str] == 1)) * 100,2))


# ### Sex

# In[18]:


stats = pd.DataFrame(index = phenos, columns = ['test_stat', 'pval'])

for i, pheno in enumerate(phenos):
    x = df.loc[df.loc[:,'sex'] == 1,pheno]
#     x = df.loc[np.logical_and(df[train_test_str] == 1,df['sex'] == 1),pheno]
    y = df.loc[df.loc[:,'sex'] == 2,pheno]
#     y = df.loc[np.logical_and(df[train_test_str] == 1,df['sex'] == 2),pheno]
    
    test_output = sp.stats.ttest_ind(x,y)
    stats.loc[pheno,'test_stat'] = test_output[0]
    stats.loc[pheno,'pval'] = test_output[1]
    
stats.loc[:,'pval_corr'] = get_fdr_p(stats.loc[:,'pval'])
stats.loc[:,'sig'] = stats.loc[:,'pval_corr'] < 0.05

np.round(stats.astype(float),2)


# In[19]:


f, ax = plt.subplots(1,len(phenos))
f.set_figwidth(len(phenos)*1.4)
f.set_figheight(1.25)

# sex: 1=male, 2=female
for i, pheno in enumerate(phenos):
    x = df.loc[df.loc[:,'sex'] == 1,pheno]
#     x = df.loc[np.logical_and(df[train_test_str] == 1,df['sex'] == 1),pheno]
    sns.kdeplot(x, ax = ax[i], label = 'male', color = 'b')

    y = df.loc[df.loc[:,'sex'] == 2,pheno]
#     y = df.loc[np.logical_and(df[train_test_str] == 1,df['sex'] == 2),pheno]
    sns.kdeplot(y, ax = ax[i], label = 'female', color = 'r')
    
    ax[i].set_xlabel('')
    ax[i].set_title(phenos_label[i])
#     if stats.loc[pheno,'sig']:
#         ax[i].set_title('t-stat:' + str(np.round(stats.loc[pheno,'test_stat'],2)) + ', p-value: ' + str(np.round(stats.loc[pheno,'pval_corr'],4)), fontweight="bold")
#     else:
#         ax[i].set_title('t-stat:' + str(np.round(stats.loc[pheno,'test_stat'],2)) + ', p-value: ' + str(np.round(stats.loc[pheno,'pval_corr'],4)))
    ax[i].tick_params(pad = -2)
    ax[i].set_ylim([0,0.5])
    if i == 0:
        ax[i].set_ylabel('Counts')
    else:
        ax[i].set_ylabel('')
        
    if i != 0:
        ax[i].set_yticklabels('')
#     if i == 0:
#         ax[i].legend()

    if stats.loc[pheno,'sig']:
        textstr = 't = {:.2f} \np < 0.05'.format(stats.loc[pheno,'test_stat'])
    else:
        textstr = 't = {:.2f} \np = {:.2f}'.format(stats.loc[pheno,'test_stat'], stats.loc[pheno,'pval_corr'])
    ax[i].text(0.05, 0.95, textstr, transform=ax[i].transAxes,
            verticalalignment='top')
        
f.savefig(outfile_prefix+'symptoms_distributions_sex.svg', dpi = 300, bbox_inches = 'tight', pad_inches = 0)


# ### nuisance correlations

# In[20]:


stats = pd.DataFrame(index = phenos, columns = ['r', 'pval'])
covs = ['ageAtScan1_Years', 'medu1', 'mprage_antsCT_vol_TBV', 'averageManualRating', 'T1_snr']
covs_label = ['Age (yrs)', 'Maternal education \n(yrs)', 'TBV', 'T1 QA', 'T1 SNR']

for c, cov in enumerate(covs):
    x = df[cov]
    nan_filt = x.isna()
    if nan_filt.any():
        x = x[~nan_filt]
    
    for i, pheno in enumerate(phenos):
        y = df[pheno]
        if nan_filt.any():
            y = y[~nan_filt]
            
        r,p = sp.stats.pearsonr(x,y)

        stats.loc[pheno,'r'] = r
        stats.loc[pheno,'pval'] = p

    stats.loc[:,'pval_corr'] = get_fdr_p(stats.loc[:,'pval'])
    stats.loc[:,'sig'] = stats.loc[:,'pval_corr'] < 0.05

    f, ax = plt.subplots(1,len(phenos))
    f.set_figwidth(len(phenos)*1.4)
    f.set_figheight(1.25)
    
    for i, pheno in enumerate(phenos):
        y = df[pheno]
        if nan_filt.any():
            y = y[~nan_filt]
        
        plot_data = pd.merge(x,y, on=['bblid','scanid'])
        sns.kdeplot(x = cov, y = pheno, data = plot_data, ax=ax[i], color='gray', thresh=0.05)
        sns.regplot(x = cov, y = pheno, data = plot_data, ax=ax[i], scatter=False)
#         ax[i].scatter(x = plot_data[cov], y = plot_data[pheno], color='gray', s=1, alpha=0.25)

        ax[i].set_ylabel(phenos_label[i], labelpad=-1)
        ax[i].set_xlabel(covs_label[c])
        ax[i].tick_params(pad = -2.5)
#         ax[i].set_xlim([x.min()-x.min()*.10,
#                         x.max()+x.max()*.10])
        
        if stats.loc[pheno,'sig']:
            textstr = 'r = {:.2f} \np < 0.05'.format(stats.loc[pheno,'r'])
        else:
            textstr = 'r = {:.2f} \np = {:.2f}'.format(stats.loc[pheno,'r'], stats.loc[pheno,'pval_corr'])

        ax[i].text(0.05, 0.975, textstr, transform=ax[i].transAxes,
                verticalalignment='top')

    f.subplots_adjust(wspace=0.5)
    f.savefig(outfile_prefix+'symptoms_correlations_'+cov+'.png', dpi = 150, bbox_inches = 'tight', pad_inches = 0.1)
    f.savefig(outfile_prefix+'symptoms_correlations_'+cov+'.svg', dpi = 300, bbox_inches = 'tight', pad_inches = 0)


# ### Diagnostic table

# In[21]:


df['goassessDxpmr4_bin'] = df.loc[:,'goassessDxpmr4'] == '4PS'
df['goassessDxpmr4_bin'] = df['goassessDxpmr4_bin'].astype(int)*4


# In[22]:


to_screen = ['goassessDxpmr4_bin','goassessSmryMan', 'goassessSmryDep', 'goassessSmryBul', 'goassessSmryAno', 'goassessSmrySoc',
             'goassessSmryPan', 'goassessSmryAgr', 'goassessSmryOcd', 'goassessSmryPtd', 'goassessSmryAdd',
            'goassessSmryOdd', 'goassessSmryCon']
# counts = np.sum(df.loc[:,to_screen] == 4)
# counts = np.sum(df.loc[df.loc[:,train_test_str] == 0,to_screen] == 4)
counts = np.sum(df.loc[df.loc[:,train_test_str] == 1,to_screen] == 4)
print(counts)
print(np.round(counts/df.shape[0]*100,2))


# In[23]:


to_keep = counts[counts >= 50].index
list(to_keep)


# In[24]:


counts[counts >= 50]


# In[25]:


# my_xticklabels = ['Psychosis spectrum (n=389)',
#                  'Depression (n=191)',
#                  'Social anxiety disorder (n=318)',
#                  'Agoraphobia (n=77)',
#                  'PTSD (n=168)',
#                  'ADHD (n=226)',
#                  'ODD (n=448)',
#                  'Conduct disorder (n=114)']
my_xticklabels = ['Psychosis spectrum (n=364)',
                 'Depression (n=179)',
                 'Social anxiety disorder (n=295)',
                 'Agoraphobia (n=73)',
                 'PTSD (n=156)',
                 'ADHD (n=206)',
                 'ODD (n=407)',
                 'Conduct disorder (n=102)']


# In[26]:


f, ax = plt.subplots(1,len(phenos))
f.set_figwidth(len(phenos)*1.4)
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
#     if i == 1:
#         ax[i].set_xlabel('Diagnostic group')
    if i == 0:
        ax[i].set_ylabel('Factor score (z)')
    if i != 0:
        ax[i].set_yticklabels('')
    
f.savefig(outfile_prefix+'symptom_dimensions_groups.svg', dpi = 300, bbox_inches = 'tight', pad_inches = 0)


# # Siblings

# In[27]:


def count_dups(df):
    dup_bool = df['famid'].duplicated(keep=False)

    unique_famid = np.unique(df.loc[dup_bool,'famid'])

    number_dups = np.zeros(len(unique_famid),)
    for i, famid in enumerate(unique_famid):
        number_dups[i] = np.sum(df['famid'] == famid)
        
    count_of_dups = []
    unique_dups = np.unique(number_dups)
    for i in unique_dups:
        count_of_dups.append(np.sum(number_dups == i))

    
    return unique_famid, unique_dups, count_of_dups


# In[28]:


unique_famid, unique_dups, count_of_dups = count_dups(df)

print(len(unique_famid))
print(count_of_dups)
print(unique_dups)
print(np.multiply(count_of_dups,unique_dups))


# In[29]:


print(np.multiply(count_of_dups,unique_dups))


# In[30]:


unique_famid, unique_dups, count_of_dups = count_dups(df.loc[df['train_test'] == 0,:])

print(len(unique_famid))
print(count_of_dups)
print(unique_dups)
print(np.multiply(count_of_dups,unique_dups))


# In[31]:


unique_famid, unique_dups, count_of_dups = count_dups(df.loc[df['train_test'] == 1,:])

print(len(unique_famid))
print(count_of_dups)
print(unique_dups)
print(np.multiply(count_of_dups,unique_dups))

