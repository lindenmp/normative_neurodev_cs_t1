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


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[3]:


sys.path.append('/Users/lindenmp/Google-Drive-Penn/work/research_projects/normative_neurodev_cs_t1/code/')
from func import set_proj_env, my_get_cmap, get_fdr_p


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


figdir = os.path.join(os.environ['OUTPUTDIR'], 'figs')
print(figdir)
if not os.path.exists(figdir): os.makedirs(figdir)
    
outputdir = os.path.join(os.environ['PIPELINEDIR'], '8_prepare_mediation', 'out')
print(outputdir)
if not os.path.exists(outputdir): os.makedirs(outputdir)


# ## Setup plots

# In[7]:


if not os.path.exists(figdir): os.makedirs(figdir)
os.chdir(figdir)
sns.set(style='white', context = 'talk', font_scale = 1)
cmap = my_get_cmap('pair')

phenos = ['Overall_Psychopathology','Psychosis_Positive','Psychosis_NegativeDisorg','AnxiousMisery','Externalizing','Fear']
phenos_label_short = ['Ov. Psych.', 'Psy. (pos.)', 'Psy. (neg.)', 'Anx.-mis.', 'Ext.', 'Fear']
phenos_label = ['Overall Psychopathology','Psychosis (Positive)','Psychosis (Negative)','Anxious-Misery','Externalizing','Fear']
metrics = ['ct', 'vol']
metrics_label_short = ['Thickness', 'Volume']
metrics_label = ['Thickness', 'Volume']


# ## Load data

# In[8]:


# Train
df_train = pd.read_csv(os.path.join(os.environ['PIPELINEDIR'], '2_prepare_normative', 'out', outfile_prefix+'train.csv'))
df_train.set_index(['bblid', 'scanid'], inplace = True)
df_node_train = pd.read_csv(os.path.join(os.environ['PIPELINEDIR'], '2_prepare_normative', 'out', outfile_prefix+'resp_train.csv'))
df_node_train.set_index(['bblid', 'scanid'], inplace = True)

# Test
df_test = pd.read_csv(os.path.join(os.environ['PIPELINEDIR'], '2_prepare_normative', 'out', outfile_prefix+'test.csv'))
df_test.set_index(['bblid', 'scanid'], inplace = True)
df_node_test = pd.read_csv(os.path.join(os.environ['PIPELINEDIR'], '2_prepare_normative', 'out', outfile_prefix+'resp_test.csv'))
df_node_test.set_index(['bblid', 'scanid'], inplace = True)

# concat
df = pd.concat((df_train, df_test), axis = 0); print(df.shape)
df_node = pd.concat((df_node_train, df_node_test), axis = 0); print(df_node.shape)


# ### Sex effects on y

# male = 0, female = 1

# In[9]:


stats = pd.DataFrame(index = phenos, columns = ['test_stat', 'pval'])

for i, pheno in enumerate(phenos):
    x = df.loc[df.loc[:,'sex_adj'] == 1,pheno]
    y = df.loc[df.loc[:,'sex_adj'] == 0,pheno]
    
    test_output = sp.stats.ttest_ind(x,y)
    stats.loc[pheno,'test_stat'] = test_output[0]
    stats.loc[pheno,'pval'] = test_output[1]
    
stats.loc[:,'pval_corr'] = get_fdr_p(stats.loc[:,'pval'])
stats.loc[:,'sig'] = stats.loc[:,'pval_corr'] < 0.05

stats


# sex_adj = male = 0, female = 1
# 
# sex_adj_flip = female = 0, male = 1

# In[10]:


df['sex_adj_flip'] = df['sex_adj'] ^ 1
df[['sex_adj','sex_adj_flip']].head()


# In[11]:


phenos = list(stats.loc[stats.loc[:,'sig'],:].index)
phenos


# In[12]:


use_flip = stats.loc[phenos,'test_stat'] < 0
use_flip


# ## Load nispat outputs

# In[13]:


z_cv = np.loadtxt(os.path.join(os.environ['PIPELINEDIR'], '3_run_normative', outfile_prefix+'out_cv', 'Z.txt'), delimiter = ' ').transpose()
df_z_cv = pd.DataFrame(data = z_cv, index = df_node_train.index, columns = df_node_train.columns)

z = np.loadtxt(os.path.join(os.environ['PIPELINEDIR'], '3_run_normative', outfile_prefix+'out', 'Z.txt'), delimiter = ' ').transpose()
df_z_test = pd.DataFrame(data = z, index = df_node_test.index, columns = df_node_test.columns)

# concat
df_z = pd.concat((df_z_cv,df_z_test), axis = 0); print(df_z.shape)


# ### Are there sex effects on devations?

# In[14]:


for metric in metrics:
    x = df_z.loc[df.loc[:,'sex_adj'] == 0,:].filter(regex = metric)
    y = df_z.loc[df.loc[:,'sex_adj'] == 1,:].filter(regex = metric)
        
    ttest_output = sp.stats.ttest_ind(x,y)

    print(metric, np.sum(get_fdr_p(ttest_output[1]) < 0.05))
    print(df_z.filter(regex = metric).columns[get_fdr_p(ttest_output[1]) < 0.05])


# ## Examine dimensionality reduction

# In[15]:


f, ax = plt.subplots(1,2)
f.set_figwidth(10)
f.set_figheight(5)

N_components = []

for i, metric in enumerate(metrics):
    x = df_z.filter(regex = metric)

    # find number of PCs that explain 80% variance
    pca = PCA(n_components = x.shape[1], svd_solver = 'full')
    pca.fit(StandardScaler().fit_transform(x))
    cum_var = np.cumsum(pca.explained_variance_ratio_)
#     n_components = np.where(cum_var >= 0.8)[0][0]+1
    var_idx = pca.explained_variance_ratio_ >= .01
    n_components = np.sum(var_idx)
    N_components.append(n_components)
    
    var_exp = cum_var[n_components-1]

    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=n_components, svd_solver='full', random_state = 0)
    pca.fit(x)

    ax[i].plot(pca.explained_variance_ratio_)
    ax[i].set_xlabel('PCs')
    if i == 0:
        ax[i].set_ylabel('Variance explained')
    ax[i].set_title(metric)
    ax[i].tick_params(pad = -2)

    print(metric, n_components, np.sum(pca.explained_variance_ratio_))


# ## Save out input files for matlab

# In[16]:


np.savetxt(os.path.join(outputdir,outfile_prefix+'Y_labels.txt'), phenos, fmt="%s")
np.savetxt(os.path.join(outputdir,outfile_prefix+'use_flip.txt'), use_flip, fmt="%i")

np.savetxt(os.path.join(outputdir,outfile_prefix+'M_labels.txt'), metrics, fmt="%s")
np.savetxt(os.path.join(outputdir,outfile_prefix+'N_components.txt'), N_components, fmt="%i")

df.loc[:,['sex_adj','sex_adj_flip']].to_csv(os.path.join(outputdir,outfile_prefix+'X.csv'))
df_z.to_csv(os.path.join(outputdir,outfile_prefix+'M.csv'))
df.loc[:,phenos].to_csv(os.path.join(outputdir,outfile_prefix+'Y.csv'))


# ## Plot PCs

# In[17]:


import matplotlib.image as mpimg
from brain_plot_func import roi_to_vtx, brain_plot


# In[18]:


if parc_str == 'schaefer':
    subject_id = 'fsaverage'
elif parc_str == 'lausanne':
    subject_id = 'lausanne125'


# In[19]:


figs_to_delete = []

for metric in metrics:
    x = df_z.filter(regex = metric)
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=10, svd_solver='full', random_state = 0)
    pca.fit(x)

    for pc in np.arange(0,3):       
        roi_data = pca.components_[pc,:]
        for hemi in ('lh', 'rh'):
            fig_str = hemi + '_' + metric + '_pc_' + str(pc)
            figs_to_delete.append('ventral_'+fig_str)
            figs_to_delete.append('med_'+fig_str)
            figs_to_delete.append('lat_'+fig_str)

            if subject_id == 'lausanne125':
                parc_file = os.path.join('/Applications/freesurfer/subjects/', subject_id, 'label', hemi + '.myaparc_' + str(parc_scale) + '.annot')
            elif subject_id == 'fsaverage':
                parc_file = os.path.join('/Users/lindenmp/Google-Drive-Penn/work/research_projects/normative_neurodev_cs_t1/figs_support/Parcellations/FreeSurfer5.3/fsaverage/label/',
                                         hemi + '.Schaefer2018_' + str(parc_scale) + 'Parcels_17Networks_order.annot')

            # project subject's data to vertices
            brain_plot(roi_data, parcel_names, parc_file, fig_str, subject_id = subject_id, hemi = hemi, surf = 'inflated', showcolorbar = True)


# In[20]:


for metric in metrics:
    for pc in np.arange(0,3):
        f, axes = plt.subplots(3, 2)
        f.set_figwidth(3)
        f.set_figheight(5)
        plt.subplots_adjust(wspace=0, hspace=-0.465)

        print(metric, pc)
        # column 0:
        fig_str = 'lh_'+metric+'_pc_'+str(pc)+'.png'
        try:
            image = mpimg.imread('ventral_' + fig_str); axes[2,0].imshow(image); axes[2,0].axis('off')
        except FileNotFoundError: axes[2,0].axis('off')
        try:
            image = mpimg.imread('med_' + fig_str); axes[1,0].imshow(image); axes[1,0].axis('off')
        except FileNotFoundError: axes[1,0].axis('off')
        try:
        #     axes[0,0].set_title('Thickness (left)')
            image = mpimg.imread('lat_' + fig_str); axes[0,0].imshow(image); axes[0,0].axis('off')
        except FileNotFoundError: axes[0,0].axis('off')

        # column 1:
        fig_str = 'rh_'+metric+'_pc_'+str(pc)+'.png'
        try:
        #     axes[0,1].set_title('Thickness (right)')
            image = mpimg.imread('lat_' + fig_str); axes[0,1].imshow(image); axes[0,1].axis('off')
        except FileNotFoundError: axes[0,1].axis('off')
        try:
            image = mpimg.imread('med_' + fig_str); axes[1,1].imshow(image); axes[1,1].axis('off')
        except FileNotFoundError: axes[1,1].axis('off')
        try:
            image = mpimg.imread('ventral_' + fig_str); axes[2,1].imshow(image); axes[2,1].axis('off')
        except FileNotFoundError: axes[2,1].axis('off')

        plt.show()
        f.savefig(outfile_prefix+metric+'_pc_'+str(pc)+'.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
        # f.savefig(metric+'_'+pheno+'_pdm.svg', dpi = 1000, bbox_inches = 'tight', pad_inches = 0)


# In[21]:


for file in figs_to_delete:
    try:
        os.remove(os.path.join(figdir,file+'.png'))
    except:
        print(file, 'not found')

