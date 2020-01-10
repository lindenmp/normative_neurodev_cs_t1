#!/usr/bin/env python
# coding: utf-8

# # Preamble

# In[1]:


import os, sys
import pandas as pd
import numpy as np


# In[2]:


sys.path.append('/Users/lindenmp/Dropbox/Work/ResProjects/NormativeNeuroDev_CrossSec_T1/code/func/')
from proj_environment import set_proj_env
from func import get_synth_cov


# In[3]:


train_test_str = 'squeakycleanExclude'
exclude_str = 't1Exclude' # 't1Exclude' 'fsFinalExclude'
parc_str = 'schaefer' # 'schaefer' 'lausanne'
parc_scale = 400 # 200 400 | 60 125
extra_str = ''
parcel_names, parcel_loc, drop_parcels, num_parcels, yeo_idx, yeo_labels = set_proj_env(train_test_str = train_test_str, exclude_str = exclude_str,
                                                                            parc_str = parc_str, parc_scale = parc_scale, extra_str = extra_str)


# In[4]:


print(os.environ['MODELDIR_BASE'])
print(os.environ['MODELDIR'])


# ## Load data

# In[5]:


# Load data
df = pd.read_csv(os.path.join(os.environ['MODELDIR_BASE'], 'df_pheno.csv'))
df.set_index(['bblid', 'scanid'], inplace = True)

df_node = pd.read_csv(os.path.join(os.environ['MODELDIR'], 'df_node_clean.csv'))
df_node.set_index(['bblid', 'scanid'], inplace = True)

# adjust sex to 0 and 1
df['sex_adj'] = df.sex - 1
print(df.shape)
print(df_node.shape)


# In[6]:


df.head()


# In[7]:


df_node.head()


# # Prepare files for normative modelling

# In[8]:


# Note, 'ageAtScan1_Years' is assumed to be covs[0] and 'sex_adj' is assumed to be covs[1]
# if more than 2 covs are to be used, append to the end and age/sex will be duplicated accordingly in the forward model
covs = ['ageAtScan1_Years', 'sex_adj']

print(covs)
num_covs = len(covs)
print(num_covs)


# In[9]:


extra_str_2 = ''


# ## Primary model (train/test split)

# In[10]:


# Create subdirectory for specific normative model -- labeled according to parcellation/resolution choices and covariates
normativedir = os.path.join(os.environ['MODELDIR'], '+'.join(covs) + extra_str_2 + '/')
print(normativedir)
if not os.path.exists(normativedir): os.mkdir(normativedir);


# In[11]:


# Write out training -- retaining only residuals from nuissance regression
df[df[train_test_str] == 0].to_csv(os.path.join(normativedir, 'train.csv'))
df[df[train_test_str] == 0].to_csv(os.path.join(normativedir, 'cov_train.txt'), columns = covs, sep = ' ', index = False, header = False)

resp_train = df_node[df_node[train_test_str] == 0].drop(train_test_str, axis = 1)
mask = np.all(np.isnan(resp_train), axis = 1)
if np.any(mask): print("Warning: NaNs in response train")
resp_train.to_csv(os.path.join(normativedir, 'resp_train.csv'))
resp_train.to_csv(os.path.join(normativedir, 'resp_train.txt'), sep = ' ', index = False, header = False)

# Write out test -- retaining only residuals from nuissance regression
df[df[train_test_str] == 1].to_csv(os.path.join(normativedir, 'test.csv'))
df[df[train_test_str] == 1].to_csv(os.path.join(normativedir, 'cov_test.txt'), columns = covs, sep = ' ', index = False, header = False)

resp_test = df_node[df_node[train_test_str] == 1].drop(train_test_str, axis = 1)
mask = np.all(np.isnan(resp_test), axis = 1)
if np.any(mask): print("Warning: NaNs in response train")
resp_test.to_csv(os.path.join(normativedir, 'resp_test.csv'))
resp_test.to_csv(os.path.join(normativedir, 'resp_test.txt'), sep = ' ', index = False, header = False)

print(str(resp_train.shape[1]) + ' features written out for normative modeling')


# ### Forward variants

# In[12]:


fwddir = os.path.join(normativedir, 'forward/')
if not os.path.exists(fwddir): os.mkdir(fwddir)

# Synthetic cov data
x = get_synth_cov(df, cov = 'ageAtScan1_Years', stp = 1)

if 'sex_adj' in covs:
    # Produce gender dummy variable for one repeat --> i.e., to account for two runs of ages, one per gender
    gender_synth = np.concatenate((np.ones(x.shape),np.zeros(x.shape)), axis = 0)

# concat
synth_cov = np.concatenate((np.matlib.repmat(x, 2, 1), np.matlib.repmat(gender_synth, 1, 1)), axis = 1)
print(synth_cov.shape)

# write out
np.savetxt(os.path.join(fwddir, 'synth_cov_test.txt'), synth_cov, delimiter = ' ', fmt = ['%.1f', '%.d'])


# ### Permutation test | train and test | no blocks

# In[13]:


# number of permutations
num_perms = 1000

# Set seed for reproducibility
np.random.seed(0)

for i in range(num_perms):
    permdir = os.path.join(normativedir, 'perm_all/perm_' + str(i))
    if not os.path.exists(permdir): os.makedirs(permdir)

    df_shuffed = df.copy()
    df_shuffed.loc[:,covs] = df_shuffed[covs].sample(frac = 1).values
    df_shuffed.loc[:,covs[1]] = df_shuffed.loc[:,covs[1]].astype(int)

    df_shuffed[df_shuffed[train_test_str] == 0].to_csv(os.path.join(permdir, 'cov_train.txt'), columns = covs, sep = ' ', index = False, header = False)
    df_shuffed[df_shuffed[train_test_str] == 1].to_csv(os.path.join(permdir, 'cov_test.txt'), columns = covs, sep = ' ', index = False, header = False)

