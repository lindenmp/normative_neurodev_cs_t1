#!/usr/bin/env python
# coding: utf-8

# # Preamble

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


# In[2]:


sys.path.append('/Users/lindenmp/Dropbox/Work/ResProjects/NormativeNeuroDev_CrossSec_T1/code/func/')
from proj_environment import set_proj_env
sys.path.append('/Users/lindenmp/Dropbox/Work/git/pyfunc/')
from func import get_synth_cov


# In[3]:


train_test_str = 'squeakycleanExclude'
exclude_str = 't1Exclude' # 't1Exclude' 'fsFinalExclude'
parc_str = 'schaefer' # 'schaefer' 'lausanne'
parc_scale = 400 # 200 400 | 60 125 250
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

# df_node = pd.read_csv(os.path.join(os.environ['MODELDIR'], 'df_node_clean.csv'))
df_node = pd.read_csv(os.path.join(os.environ['MODELDIR'], 'df_node_base.csv'))
df_node.set_index(['bblid', 'scanid'], inplace = True)

# adjust sex to 0 and 1
df['sex_adj'] = df.sex - 1
print(df.shape)
print(df_node.shape)


# In[6]:


print('Train:', np.sum(df[train_test_str] == 0), 'Test:', np.sum(df[train_test_str] == 1))


# In[7]:


df.head()


# In[8]:


df_node.head()


# # Prepare files for normative modelling

# In[9]:


# Note, 'ageAtScan1_Years' is assumed to be covs[0] and 'sex_adj' is assumed to be covs[1]
# if more than 2 covs are to be used, append to the end and age/sex will be duplicated accordingly in the forward model
covs = ['ageAtScan1_Years', 'sex_adj']

print(covs)
num_covs = len(covs)
print(num_covs)


# In[10]:


extra_str_2 = ''


# ## Primary model (train/test split)

# In[11]:


# Create subdirectory for specific normative model -- labeled according to parcellation/resolution choices and covariates
normativedir = os.path.join(os.environ['MODELDIR'], '+'.join(covs) + extra_str_2 + '/')
print(normativedir)
if not os.path.exists(normativedir): os.mkdir(normativedir);


# In[12]:


# Write out training
df[df[train_test_str] == 0].to_csv(os.path.join(normativedir, 'train.csv'))
df[df[train_test_str] == 0].to_csv(os.path.join(normativedir, 'cov_train.txt'), columns = covs, sep = ' ', index = False, header = False)

# Write out test
df[df[train_test_str] == 1].to_csv(os.path.join(normativedir, 'test.csv'))
df[df[train_test_str] == 1].to_csv(os.path.join(normativedir, 'cov_test.txt'), columns = covs, sep = ' ', index = False, header = False)


# In[13]:


# Write out training
resp_train = df_node[df_node[train_test_str] == 0].drop(train_test_str, axis = 1)
mask = np.all(np.isnan(resp_train), axis = 1)
if np.any(mask): print("Warning: NaNs in response train")
resp_train.to_csv(os.path.join(normativedir, 'resp_train.csv'))
resp_train.to_csv(os.path.join(normativedir, 'resp_train.txt'), sep = ' ', index = False, header = False)

# Write out test
resp_test = df_node[df_node[train_test_str] == 1].drop(train_test_str, axis = 1)
mask = np.all(np.isnan(resp_test), axis = 1)
if np.any(mask): print("Warning: NaNs in response train")
resp_test.to_csv(os.path.join(normativedir, 'resp_test.csv'))
resp_test.to_csv(os.path.join(normativedir, 'resp_test.txt'), sep = ' ', index = False, header = False)

print(str(resp_train.shape[1]) + ' features written out for normative modeling')


# ### Forward variants

# In[14]:


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


# ### Cross-val variant

# In[15]:


# Create subdirectory for specific normative model -- labeled according to parcellation/resolution choices and covariates
cvdir = os.path.join(normativedir, 'cv/')
if not os.path.exists(cvdir): os.mkdir(cvdir)

