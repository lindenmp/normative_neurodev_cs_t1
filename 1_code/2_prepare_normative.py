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


import numpy.matlib


# In[3]:


sys.path.append('/Users/lindenmp/Google-Drive-Penn/work/research_projects/normative_neurodev_cs_t1/1_code/')
from func import set_proj_env, get_synth_cov


# In[4]:


train_test_str = 'squeakycleanExclude'
exclude_str = 't1Exclude' # 't1Exclude' 'fsFinalExclude'
parc_str = 'schaefer' # 'schaefer' 'lausanne'
parc_scale = 400 # 200 400 | 60 125 250
extra_str = ''
parcel_names, parcel_loc, drop_parcels, num_parcels, yeo_idx, yeo_labels = set_proj_env(train_test_str = train_test_str, exclude_str = exclude_str,
                                                                                        parc_str = parc_str, parc_scale = parc_scale)


# In[5]:


# output file prefix
outfile_prefix = exclude_str+'_'+parc_str+'_'+str(parc_scale)+'_'
outfile_prefix


# ### Setup directory variables

# In[6]:


outputdir = os.path.join(os.environ['PIPELINEDIR'], '2_prepare_normative', 'out')
print(outputdir)
if not os.path.exists(outputdir): os.makedirs(outputdir)


# In[7]:


figdir = os.path.join(os.environ['OUTPUTDIR'], 'figs')
print(figdir)
if not os.path.exists(figdir): os.makedirs(figdir)


# ## Load data

# In[8]:


# Load data
df = pd.read_csv(os.path.join(os.environ['PIPELINEDIR'], '1_compute_node_features', 'out', outfile_prefix+'df.csv'))
df.set_index(['bblid', 'scanid'], inplace = True)

df_node = pd.read_csv(os.path.join(os.environ['PIPELINEDIR'], '1_compute_node_features', 'out', outfile_prefix+'df_node.csv'))
df_node.set_index(['bblid', 'scanid'], inplace = True)

# adjust sex to 0 and 1
# now: male = 0, female = 1
df['sex_adj'] = df.sex - 1
print(df.shape)
print(df_node.shape)


# In[9]:


print('Train:', np.sum(df[train_test_str] == 0), 'Test:', np.sum(df[train_test_str] == 1))


# ## Normalize

# In[10]:


metrics = ['ct', 'vol']
my_str = '|'.join(metrics); print(my_str)


# In[11]:


norm_data = False


# In[12]:


if np.any(df_node.filter(regex = my_str, axis = 1) < 0):
    print('WARNING: some regional values are <0, box cox will fail')

if np.any(df_node.filter(regex = my_str, axis = 1) == 0):
    print('WARNING: some regional values are == 0, box cox will fail')


# In[13]:


rank_r = np.zeros(df_node.filter(regex = my_str).shape[1])

# normalise
if norm_data:
    for i, col in enumerate(df_node.filter(regex = my_str).columns):
        # normalize regional metric
        x = sp.stats.boxcox(df_node.loc[:,col])[0]
        # check if rank order is preserved
        rank_r[i] = sp.stats.spearmanr(df_node.loc[:,col],x)[0]
        # store normalized version
        df_node.loc[:,col] = x
    print(np.sum(rank_r < .99))
else:
    print('Skipping...')


# # Prepare files for normative modelling

# In[14]:


# Note, 'ageAtScan1_Years' is assumed to be covs[0] and 'sex_adj' is assumed to be covs[1]
# if more than 2 covs are to be used, append to the end and age/sex will be duplicated accordingly in the forward model
covs = ['ageAtScan1_Years', 'sex_adj']

print(covs)
num_covs = len(covs)
print(num_covs)


# In[15]:


extra_str_2 = ''


# ## Primary model (train/test split)

# In[16]:


# Write out training
df[df[train_test_str] == 0].to_csv(os.path.join(outputdir, outfile_prefix+'train.csv'))
df[df[train_test_str] == 0].to_csv(os.path.join(outputdir, outfile_prefix+'cov_train.txt'), columns = covs, sep = ' ', index = False, header = False)

# Write out test
df[df[train_test_str] == 1].to_csv(os.path.join(outputdir, outfile_prefix+'test.csv'))
df[df[train_test_str] == 1].to_csv(os.path.join(outputdir, outfile_prefix+'cov_test.txt'), columns = covs, sep = ' ', index = False, header = False)


# In[17]:


# Write out training
resp_train = df_node[df_node[train_test_str] == 0].drop(train_test_str, axis = 1)
mask = np.all(np.isnan(resp_train), axis = 1)
if np.any(mask): print("Warning: NaNs in response train")
resp_train.to_csv(os.path.join(outputdir, outfile_prefix+'resp_train.csv'))
resp_train.to_csv(os.path.join(outputdir, outfile_prefix+'resp_train.txt'), sep = ' ', index = False, header = False)

# Write out test
resp_test = df_node[df_node[train_test_str] == 1].drop(train_test_str, axis = 1)
mask = np.all(np.isnan(resp_test), axis = 1)
if np.any(mask): print("Warning: NaNs in response train")
resp_test.to_csv(os.path.join(outputdir, outfile_prefix+'resp_test.csv'))
resp_test.to_csv(os.path.join(outputdir, outfile_prefix+'resp_test.txt'), sep = ' ', index = False, header = False)

print(str(resp_train.shape[1]) + ' features written out for normative modeling')


# ### Forward variants

# In[18]:


# Synthetic cov data
x = get_synth_cov(df, cov = 'ageAtScan1_Years', stp = 1)

if 'sex_adj' in covs:
    # Produce gender dummy variable for one repeat --> i.e., to account for two runs of ages, one per gender
    gender_synth = np.concatenate((np.ones(x.shape),np.zeros(x.shape)), axis = 0)

# concat
synth_cov = np.concatenate((np.matlib.repmat(x, 2, 1), np.matlib.repmat(gender_synth, 1, 1)), axis = 1)
print(synth_cov.shape)

# write out
np.savetxt(os.path.join(outputdir, outfile_prefix+'cov_test_forward.txt'), synth_cov, delimiter = ' ', fmt = ['%.1f', '%.d'])

