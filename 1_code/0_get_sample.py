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
from func import set_proj_env, my_get_cmap, rank_int


# In[3]:


train_test_str = 'squeakycleanExclude'
exclude_str = 't1Exclude' # 't1Exclude' 'fsFinalExclude'
parc_str = 'schaefer' # 'schaefer' 'lausanne'
parc_scale = 400 # 200 400 | 60 125 250
_ = set_proj_env(exclude_str = exclude_str, parc_str = parc_str, parc_scale = parc_scale)


# In[4]:


# output file prefix
outfile_prefix = exclude_str+'_'
outfile_prefix


# ### Setup directory variables

# In[5]:


print(os.environ['PIPELINEDIR'])
if not os.path.exists(os.environ['PIPELINEDIR']): os.makedirs(os.environ['PIPELINEDIR'])


# In[6]:


outputdir = os.path.join(os.environ['PIPELINEDIR'], '0_get_sample', 'out')
print(outputdir)
if not os.path.exists(outputdir): os.makedirs(outputdir)


# In[7]:


figdir = os.path.join(os.environ['OUTPUTDIR'], 'figs')
print(figdir)
if not os.path.exists(figdir): os.makedirs(figdir)


# # Load in demographic and symptom data

# In[8]:


# LTN and Health Status
health = pd.read_csv(os.path.join(os.environ['DATADIR'], 'external/pncDataFreeze20170905/n1601_dataFreeze/health/n1601_health_20170421.csv'))
# Protocol
prot = pd.read_csv(os.path.join(os.environ['DATADIR'], 'external/pncDataFreeze20170905/n1601_dataFreeze/neuroimaging/n1601_pnc_protocol_validation_params_status_20161220.csv'))
# T1 QA
t1_qa = pd.read_csv(os.path.join(os.environ['DATADIR'], 'external/pncDataFreeze20170905/n1601_dataFreeze/neuroimaging/t1struct/n1601_t1QaData_20170306.csv'))
# Demographics
demog = pd.read_csv(os.path.join(os.environ['DATADIR'], 'external/pncDataFreeze20170905/n1601_dataFreeze/demographics/n1601_demographics_go1_20161212.csv'))
# Brain volume
brain_vol = pd.read_csv(os.path.join(os.environ['DATADIR'], 'external/pncDataFreeze20170905/n1601_dataFreeze/neuroimaging/t1struct/n1601_ctVol20170412.csv'))
# dti QA
dti_qa = pd.read_csv(os.path.join(os.environ['DATADIR'], 'external/pncDataFreeze20170905/n1601_dataFreeze/neuroimaging/dti/n1601_dti_qa_20170301.csv'))

# GOASSESS Bifactor scores
goassess = pd.read_csv(os.path.join(os.environ['DATADIR'], 'external/GO1_clinical_factor_scores_psychosis_split_BIFACTOR.csv'))
# Clinical diagnostic 
clinical = pd.read_csv(os.path.join(os.environ['DATADIR'], 'external/pncDataFreeze20170905/n1601_dataFreeze/clinical/n1601_goassess_psych_summary_vars_20131014.csv'))
# Psychosis summary
clinical_psychosis = pd.read_csv(os.path.join(os.environ['DATADIR'], 'external/pncDataFreeze20170905/n1601_dataFreeze/clinical/n1601_diagnosis_dxpmr_20170509.csv'))
# Cognition
cnb = pd.read_csv(os.path.join(os.environ['DATADIR'], 'external/pncDataFreeze20170905/n1601_dataFreeze/cnb/n1601_cnb_factor_scores_tymoore_20151006.csv'))

# merge
df = health
df = pd.merge(df, prot, on=['scanid', 'bblid']) # prot
df = pd.merge(df, t1_qa, on=['scanid', 'bblid']) # t1_qa
df = pd.merge(df, demog, on=['scanid', 'bblid']) # demog
df = pd.merge(df, brain_vol, on=['scanid', 'bblid']) # brain_vol
df = pd.merge(df, dti_qa, on=['scanid', 'bblid']) # dti_qa

df = pd.merge(df, clinical, on=['scanid', 'bblid']) # clinical
df = pd.merge(df, clinical_psychosis, on=['scanid', 'bblid']) # clinical
df = pd.merge(df, goassess, on=['bblid']) # goassess
df = pd.merge(df, cnb, on=['scanid', 'bblid']) # cnb

print(df.shape[0])


# In[9]:


#t1 snr
snr = pd.read_csv(os.path.join(os.environ['DATADIR'], 'external/n1601_snr.txt'), header=None, delimiter='\t')

idx = snr.iloc[:,0]

for idx, row in snr.iterrows():
    snr.loc[idx,'bblid'] = row[0].split('/')[0]
    snr.loc[idx,'scanid'] = row[0].split('/')[1].split('x')[1]
    
snr['bblid'] = snr['bblid'].astype(int)
snr['scanid'] = snr['scanid'].astype(int)
snr.set_index(['bblid', 'scanid'], inplace = True)
snr['T1_snr'] = snr.loc[:,1]

snr.drop(labels=[0,1], axis=1, inplace=True)

df = pd.merge(df, snr, on=['scanid', 'bblid'])


# In[10]:


# famid
famid = pd.read_csv(os.path.join(os.environ['DATADIR'], 'external/PNC_BBLID_FAMID.csv'))
famid.drop(labels=['cnbDatasetid','study'], axis=1, inplace=True)
famid = famid.groupby('bblid').nth(0)

df = pd.merge(df, famid, on=['bblid'])


# In[11]:


df.set_index(['bblid', 'scanid'], inplace = True)
df.head()


# # Filter subjects

# In[12]:


# 1) Primary sample filter
df = df[df['healthExcludev2'] == 0]
print('N after initial exclusion:', df.shape[0])

# 2) T1 exclusion
df = df[df[exclude_str] == 0]
print('N after T1 exclusion:', df.shape[0])


# In[13]:


# 3) filter subjects with NaN on key variables
screen = [train_test_str, 'ageAtScan1', 'sex', 'race2', 'handednessv2', 'medu1', 'mprage_antsCT_vol_TBV', 'averageManualRating', 'psychoactiveMedPsychv2', 'psychoactiveMedMedicalv2', 'T1_snr', 'famid',
          'Overall_Psychopathology','Psychosis_Positive','Psychosis_NegativeDisorg','AnxiousMisery','Externalizing','Fear']

drop_idx = df.loc[:,screen].isna().any(axis = 1)
df = df.loc[~drop_idx,:]
print('N after variable screen:', df.shape[0])


# In[14]:


# 4) filter duplicates on famid
dup_bool = df['famid'].duplicated(keep='first')
print(np.sum(dup_bool))

df = df.loc[~dup_bool,:]
print('N after duplicate screen:', df.shape[0])


# In[15]:


np.sum(df['famid'].duplicated(keep=False))


# In[16]:


df['averageManualRating'].unique()


# In[17]:


np.sum(df['averageManualRating'] == 2)


# In[18]:


# Convert age to years
df['ageAtScan1_Years'] = np.round(df.ageAtScan1/12, decimals=1)


# # Define train/test split

# In[19]:


if train_test_str == 'squeakycleanExclude':
    # 1) Use squeakycleanExclude
    # For PNC, the train/test split is based upon the 'squeakycleanExclude' criteria. From the wiki:
    # "This category excludes all subjects with any screening diagnosis as found by the GOASSESS clinical screening instrument.
    # Note that this is NOT a commonly used criteria except for sensitivity analysis, and in general we would prefer to not
    # have studies use this without a strong rationale; would discuss with your BBL liaison."
    # train = squeakycleanExclude == 0 --> retain those WITHOUT any lifetime psychopathology 
    # test = squeakycleanExclude == 1 --> retain those WITH lifetime psychopathology
    print('Train:', np.sum(df[train_test_str] == 0), 'Test:', np.sum(df[train_test_str] == 1))

    # randomly sample a healthy holdout cohort
    df['train_test'] = df[train_test_str]
    n=100
#     n=np.round(df.loc[df[train_test_str] == 0,:].shape[0]*.5).astype(int)
    print(n)
    hold_out = df.loc[df[train_test_str] == 0,:].sample(n=n, random_state=0, replace=False, axis=0).index
    df.loc[hold_out,'train_test'] = 1
    print('Train:', np.sum(df['train_test'] == 0), 'Test:', np.sum(df['train_test'] == 1))

train_test_str = 'train_test'


# # Characterise train/test split

# ## Train/Test split

# In[20]:


# find unique ages
age_unique = np.unique(df.ageAtScan1_Years)
print('There are', age_unique.shape[0], 'unique age points')

# Check if train and test represent the full unique age space
train_diff = np.setdiff1d(df[df[train_test_str] == 0].ageAtScan1_Years,age_unique)
test_diff = np.setdiff1d(df[df[train_test_str] == 1].ageAtScan1_Years,age_unique)

if train_diff.size == 0:
    print('All unique age points are represented in the training set')
elif train_diff.size != 0:
    print('All unique age points ARE NOT represented in the training set')
    
if test_diff.size == 0:
    print('All unique age points are represented in the testing set')
elif test_diff.size != 0:
    print('All unique age points ARE NOT represented in the testing set')


# ## Export

# In[21]:


phenos = ['Overall_Psychopathology','Psychosis_Positive','Psychosis_NegativeDisorg','AnxiousMisery','Externalizing','Fear','F1_Exec_Comp_Res_Accuracy','F3_Executive_Efficiency','Overall_Speed']
print(phenos)


# In[22]:


for pheno in phenos:
    if df.loc[:,pheno].isna().any():
        print('NaN replacement: ', pheno)
        x = np.nanmedian(df.loc[:,pheno])
        df.loc[df.loc[:,pheno].isna(),pheno] = x


# In[23]:


# Normalize
rank_r = np.zeros(len(phenos),)

for i, pheno in enumerate(phenos):
    # normalize regional metric
#     x = sp.stats.yeojohnson(df.loc[:,pheno])[0]
    x = rank_int(df.loc[:,pheno])
    # check if rank order is preserved
    rank_r[i] = sp.stats.spearmanr(df.loc[:,pheno],x)[0]
    # store normalized version
    df.loc[:,pheno] = x

print(np.sum(rank_r < 0.99))


# In[24]:


df.loc[:,phenos].var()


# In[25]:


header = ['squeakycleanExclude', 'train_test', 'ageAtScan1', 'ageAtScan1_Years','sex', 'race2', 'handednessv2', 'medu1', 'T1_snr', 'dti64MeanRelRMS', 'famid',
          'mprage_antsCT_vol_TBV', 'averageManualRating', 'psychoactiveMedPsychv2', 'psychoactiveMedMedicalv2',
          'Overall_Psychopathology','Psychosis_Positive','Psychosis_NegativeDisorg','AnxiousMisery','Externalizing','Fear',
          'F1_Exec_Comp_Res_Accuracy','F3_Executive_Efficiency','Overall_Speed',
          'goassessSmryMood', 'goassessSmryMan', 'goassessSmryDep', 'goassessSmryEat', 'goassessSmryBul',
          'goassessSmryAno', 'goassessSmryAnx', 'goassessSmryGad', 'goassessSmrySep', 'goassessSmryPhb', 'goassessSmrySoc',
          'goassessSmryPan', 'goassessSmryAgr', 'goassessSmryOcd', 'goassessSmryPtd', 'goassessSmryPsy', 'goassessSmryDel',
          'goassessSmryHal', 'goassessSmryHalAv', 'goassessSmryHalAs', 'goassessSmryHalVh', 'goassessSmryHalOh', 'goassessSmryHalTh',
          'goassessSmryBeh', 'goassessSmryAdd', 'goassessSmryOdd', 'goassessSmryCon', 'goassessSmryPrimePos1', 'goassessSmryPrimeTot',
          'goassessSmryPrimePos2', 'goassessSmryPsychOverallRtg', 'goassessDxpmr4']
df.to_csv(os.path.join(outputdir, outfile_prefix+'df.csv'), columns = header)


# # Plots

# In[26]:


if not os.path.exists(figdir): os.makedirs(figdir)
os.chdir(figdir)
sns.set(style='white', context = 'paper', font_scale = 1)
sns.set_style({'font.family':'sans-serif', 'font.sans-serif':['Public Sans']})
cmap = my_get_cmap('pair')

labels = ['Train', 'Test']
phenos = ['Overall_Psychopathology','Psychosis_Positive','Psychosis_NegativeDisorg','AnxiousMisery','Externalizing','Fear']
phenos_label_short = ['Ov. psych.', 'Psy. (pos.)', 'Psy. (neg.)', 'Anx.-mis.', 'Ext.', 'Fear']
phenos_label = ['Overall psychopathology','Psychosis (Positive)','Psychosis (Negative)','Anxious-Misery','Externalizing','Fear']


# Figure 2A

# In[27]:


f, axes = plt.subplots(1,2)
f.set_figwidth(6.5)
f.set_figheight(2.5)
colormap = sns.color_palette("pastel", 2)

sns.distplot(df.loc[df[train_test_str] == 0,'ageAtScan1_Years'], bins=20, hist=True, kde=False, rug=False, label = labels[0],
             hist_kws={"histtype": "step", "linewidth": 2, "alpha": 1}, color=list(cmap[0]), ax = axes[0]);
sns.distplot(df.loc[df[train_test_str] == 1,'ageAtScan1_Years'], bins=20, hist=True, kde=False, rug=False, label = labels[1],
             hist_kws={"histtype": "step", "linewidth": 2, "alpha": 1}, color=list(cmap[1]), ax = axes[0]);
axes[0].set_xlabel('Age (years)');
axes[0].set_ylabel('Number of participants');
axes[0].set_xticks(np.arange(np.min(np.round(age_unique,0)), np.max(np.round(age_unique,0)), 2))

# set width of bar
barWidth = 0.25

# Sex
y_train = [np.sum(df.loc[df[train_test_str] == 0,'sex'] == 1), np.sum(df.loc[df[train_test_str] == 0,'sex'] == 2)]
y_test = [np.sum(df.loc[df[train_test_str] == 1,'sex'] == 1), np.sum(df.loc[df[train_test_str] == 1,'sex'] == 2)]
r1 = np.arange(len(y_train))+barWidth/2
r2 = [x + barWidth for x in r1]
axes[1].bar(r1, y_train, width = barWidth, color = cmap[0], label = labels[0])
axes[1].bar(r2, y_test, width = barWidth, color = cmap[1], label = labels[1])
axes[1].set_xlabel('Sex')
# axes[1].set_ylabel('Number of participants')
axes[1].set_xticks([r + barWidth for r in range(len(y_train))])
axes[1].set_xticklabels(['Male', 'Female'])

f.savefig(outfile_prefix+'age_distributions.svg', dpi = 300, bbox_inches = 'tight', pad_inches = 0)


# ## Phenotype distributions over train/test

# Figure 2B

# In[28]:


df_rc = pd.melt(df, id_vars = train_test_str, value_vars = phenos)

f, ax = plt.subplots()
f.set_figwidth(2.5)
f.set_figheight(4)
ax = sns.violinplot(y='variable', x='value', hue=train_test_str, data=df_rc, palette = cmap, split=True, scale='width', inner = 'quartile', orient = 'h')
ax.get_legend().remove()
ax.set_yticklabels(phenos_label_short)
ax.set_ylabel('Psychopathology phenotypes')
ax.set_xlabel('Phenotype score')
f.savefig(outfile_prefix+'phenos_distributions.svg', dpi = 300, bbox_inches = 'tight', pad_inches = 0)

