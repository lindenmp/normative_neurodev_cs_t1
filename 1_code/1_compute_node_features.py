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
from func import set_proj_env


# In[3]:


train_test_str = 'squeakycleanExclude'
exclude_str = 't1Exclude' # 't1Exclude' 'fsFinalExclude'
parc_str = 'schaefer' # 'schaefer' 'lausanne'
parc_scale = 400 # 200 400 | 60 125 250
parcel_names, parcel_loc, drop_parcels, num_parcels, yeo_idx, yeo_labels = set_proj_env(train_test_str = train_test_str, exclude_str = exclude_str,
                                                                                        parc_str = parc_str, parc_scale = parc_scale)


# In[4]:


# output file prefix
outfile_prefix = exclude_str+'_'+parc_str+'_'+str(parc_scale)+'_'
outfile_prefix


# ### Setup directory variables

# In[5]:


outputdir = os.path.join(os.environ['PIPELINEDIR'], '1_compute_node_features', 'out')
print(outputdir)
if not os.path.exists(outputdir): os.makedirs(outputdir)


# In[6]:


figdir = os.path.join(os.environ['OUTPUTDIR'], 'figs')
print(figdir)
if not os.path.exists(figdir): os.makedirs(figdir)


# ## Load data

# In[7]:


# Load data
df = pd.read_csv(os.path.join(os.environ['PIPELINEDIR'], '0_get_sample', 'out', exclude_str+'_df.csv'))
df.set_index(['bblid', 'scanid'], inplace = True)
print(df.shape)


# In[8]:


# output dataframe
ct_labels = ['ct_' + str(i) for i in range(num_parcels)]
vol_labels = ['vol_' + str(i) for i in range(num_parcels)]

df_node = pd.DataFrame(index = df.index, columns = ct_labels + vol_labels)
df_node.insert(0, train_test_str, df[train_test_str])

print(df_node.shape)


# ## Load in cortical thickness

# In[9]:


# subject filter
subj_filt = np.zeros((df.shape[0],)).astype(bool)


# In[10]:


CT = np.zeros((df.shape[0], num_parcels))

for (i, (index, row)) in enumerate(df.iterrows()):
    file_name = os.environ['CT_NAME_TMP'].replace("bblid", str(index[0]))
    file_name = file_name.replace("scanid", str(index[1]))
    full_path = glob.glob(os.path.join(os.environ['CTDIR'], file_name))
    if i == 0: print(full_path)
        
    if len(full_path) > 0:
        ct = np.loadtxt(full_path[0])
        CT[i,:] = ct
    elif len(full_path) == 0:
        subj_filt[i] = True
    
df_node.loc[:,ct_labels] = CT


# In[11]:


np.sum(subj_filt)


# In[12]:


if any(subj_filt):
    df = df.loc[~subj_filt]
    df_node = df_node.loc[~subj_filt]


# ## Load in cortical volume

# In[13]:


# subject filter
subj_filt = np.zeros((df.shape[0],)).astype(bool)


# In[14]:


VOL = np.zeros((df.shape[0], num_parcels))

for (i, (index, row)) in enumerate(df.iterrows()):
    file_name = os.environ['VOL_NAME_TMP'].replace("bblid", str(index[0]))
    file_name = file_name.replace("scanid", str(index[1]))
    full_path = glob.glob(os.path.join(os.environ['VOLDIR'], file_name))
    if i == 0: print(full_path)    
    
    if len(full_path) > 0:
        img = nib.load(full_path[0])
        v = np.array(img.dataobj)
        v = v[v != 0]
        unique_elements, counts_elements = np.unique(v, return_counts=True)
        if len(unique_elements) == num_parcels:
            VOL[i,:] = counts_elements
        else:
            print(str(index) + '. Warning: not all parcels present')
            subj_filt[i] = True
    elif len(full_path) == 0:
        subj_filt[i] = True
    
df_node.loc[:,vol_labels] = VOL


# In[15]:


np.sum(subj_filt)


# In[16]:


if any(subj_filt):
    df = df.loc[~subj_filt]
    df_node = df_node.loc[~subj_filt]


# # Save out raw data

# In[17]:


print(df_node.isna().any().any())


# In[18]:


df_node.to_csv(os.path.join(outputdir, outfile_prefix+'df_node.csv'))
df.to_csv(os.path.join(outputdir, outfile_prefix+'df.csv'))

