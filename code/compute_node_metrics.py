#!/usr/bin/env python
# coding: utf-8

# # Preamble

# In[1]:


import os, sys, glob
import pandas as pd
import numpy as np
import scipy.io as sio


# In[2]:


sys.path.append('/Users/lindenmp/Dropbox/Work/ResProjects/NormativeNeuroDev_CrossSec_T1/code/func/')
from proj_environment import set_proj_env


# In[3]:


train_test_str = 'squeakycleanExclude'
exclude_str = 't1Exclude' # 't1Exclude' 'fsFinalExclude'
parc_str = 'schaefer' # 'schaefer' 'lausanne'
parc_scale = 400 # 200 400 | 60 125
parcel_names, parcel_loc, drop_parcels, num_parcels, yeo_idx, yeo_labels = set_proj_env(train_test_str = train_test_str, exclude_str = exclude_str,
                                                                            parc_str = parc_str, parc_scale = parc_scale)


# ### Setup output directory

# In[4]:


print(os.environ['MODELDIR'])
if not os.path.exists(os.environ['MODELDIR']): os.makedirs(os.environ['MODELDIR'])


# ## Load train/test .csv and setup node .csv

# In[5]:


os.path.join(os.environ['TRTEDIR'])


# In[6]:


# Load data
df = pd.read_csv(os.path.join(os.environ['TRTEDIR'], 'df_pheno.csv'))
df.set_index(['bblid', 'scanid'], inplace = True)
print(df.shape)


# In[7]:


# Corrupted jacobian file
if parc_str == 'schaefer' and parc_scale == 400:
    df.drop(labels = (133007, 6259), inplace=True)


# In[8]:


# output dataframe
ct_labels = ['ct_' + str(i) for i in range(num_parcels)]
jd_labels = ['jd_' + str(i) for i in range(num_parcels)]

df_node = pd.DataFrame(index = df.index, columns = ct_labels + jd_labels)
df_node.insert(0, train_test_str, df[train_test_str])

print(df_node.shape)


# ## Load in cortical thickness

# In[9]:


CT = np.zeros((df.shape[0], num_parcels))

for (i, (index, row)) in enumerate(df.iterrows()):
    file_name = os.environ['CT_NAME_TMP'].replace("scanid", str(index[1]))
    full_path = os.path.join(os.environ['CTDIR'], file_name)
    
    ct = np.loadtxt(full_path)
    CT[i,:] = ct
    
df_node.loc[:,ct_labels] = CT


# ## Load in jacobian determinants

# In[10]:


JD = np.zeros((df.shape[0], num_parcels))

for (i, (index, row)) in enumerate(df.iterrows()):
    file_name = os.environ['JD_NAME_TMP'].replace("scanid", str(index[1]))
    full_path = os.path.join(os.environ['JDDIR'], file_name)
    
    jd = np.loadtxt(full_path)
    JD[i,:] = jd
    
df_node.loc[:,jd_labels] = JD


# ## Save out

# In[11]:


os.environ['MODELDIR']


# In[12]:


# Save out
df_node.to_csv(os.path.join(os.environ['MODELDIR'], 'df_node_base.csv'))
df.to_csv(os.path.join(os.environ['MODELDIR'], 'df_pheno.csv'))

