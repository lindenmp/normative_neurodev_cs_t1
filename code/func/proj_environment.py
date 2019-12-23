# Functions for project: NormativeNeuroDev_CrossSec
# This project used normative modelling to examine network control theory metrics
# Linden Parkes, 2019
# lindenmp@seas.upenn.edu

import os, sys
import numpy as np

def set_proj_env(dataset = 'PNC', train_test_str = 'squeakycleanExclude', exclude_str = 't1Exclude',
    parc_str = 'schaefer', parc_scale = 400, parc_variant = 'orig', edge_weight = 'streamlineCount',
    primary_covariate = 'ageAtScan1_Years', extra_str = ''):

    # Project root directory
    projdir = '/Users/lindenmp/Dropbox/Work/ResProjects/NormativeNeuroDev_CrossSec'; os.environ['PROJDIR'] = projdir
    
    # Derivatives for dataset --> root directory for the dataset under analysis
    derivsdir = os.path.join('/Users/lindenmp/Dropbox/Work/ResData/PNC/'); os.environ['DERIVSDIR'] = derivsdir

    # Parcellation specifications
    # Names of parcels
    if parc_str == 'lausanne': parcel_names = np.genfromtxt(os.path.join(projdir, 'figs_support/labels/lausanne_' + str(parc_scale) + '.txt'), dtype='str')
    if parc_str == 'schaefer': parcel_names = np.genfromtxt(os.path.join(projdir, 'figs_support/labels/schaefer' + str(parc_scale) + 'NodeNames.txt'), dtype='str')

    # vector describing whether rois belong to cortex (1) or subcortex (0)
    if parc_str == 'lausanne': parcel_loc = np.loadtxt(os.path.join(projdir, 'figs_support/labels/lausanne_' + str(parc_scale) + '_loc.txt'), dtype='int')
    if parc_str == 'schaefer': parcel_loc = np.loadtxt(os.path.join(projdir, 'figs_support/labels/schaefer' + str(parc_scale) + 'NodeNames_loc.txt'), dtype='int')
    
    if parc_variant == 'no_bs':
        drop_parcels = np.where(parcel_loc == 2)
        num_parcels = parcel_names.shape[0] - drop_parcels[0].shape[0]
    elif parc_variant == 'cortex_only':
        drop_parcels = np.where(parcel_loc != 1)
        num_parcels = parcel_names.shape[0] - drop_parcels[0].shape[0]
    else:
        drop_parcels = []
        num_parcels = parcel_names.shape[0]

    # Cortical thickness directory
    ctdir = os.path.join(derivsdir, 'processedData/antsCorticalThickness_inMNI'); os.environ['CTDIR'] = ctdir

    if parc_str == 'lausanne':
        # cortical thickness text file name
        ct_file_name = 'ct_' + parc_str + str(parc_scale) + '.txt'; os.environ['CT_FILE_NAME'] = ct_file_name
        
        # Structural connectivity derivatives
        scdir = os.path.join(derivsdir, 'processedData/diffusion/deterministic_dec2016', edge_weight, 'LausanneScale' + str(parc_scale)); os.environ['SCDIR'] = scdir

        # template file name for a subject's structural connectivity .mat file (sourced from Ted's group)
        # this is a template because it requires some find and replacement -- see compute_node_metric.ipynb
        sc_name_tmp = 'scanid_' + edge_weight + '_LausanneScale' + str(parc_scale) + '.mat'; os.environ['SC_NAME_TMP'] = sc_name_tmp
        
        # field inside .mat file to reference to get the A matrix
        if edge_weight == 'streamlineCount': os.environ['CONN_STR'] = 'connectivity'
        elif edge_weight == 'volNormStreamline': os.environ['CONN_STR'] = 'volNorm_connectivity'
    elif parc_str == 'schaefer':
        ct_file_name = 'ct_' + parc_str + str(parc_scale) + '_17.txt'; os.environ['CT_FILE_NAME'] = ct_file_name
        
        scdir = os.path.join(derivsdir, 'processedData/diffusion/deterministic_20171118'); os.environ['SCDIR'] = scdir
        sc_name_tmp = 'bblid/*xscanid/tractography/connectivity/bblid_*xscanid_SchaeferPNC_' + str(parc_scale) + '_dti_streamlineCount_connectivity.mat'; os.environ['SC_NAME_TMP'] = sc_name_tmp

        os.environ['CONN_STR'] = 'connectivity'

    # Normative dir based on the train/test split --> specific combinations of parcellation/number of parcels/edge weight come off this directory
    # This is the first of the output directories for the project and is created by get_train_test.ipynb if it doesn't exist
    trtedir = os.path.join(projdir, 'analysis/normative', exclude_str, train_test_str); os.environ['TRTEDIR'] = trtedir

    # Subdirector in normative dir (TRTEDIR) that designates combination of parcellation/number of parcels/edge weight
    modeldir_base = os.path.join(trtedir, parc_str + '_' + str(num_parcels) + '_' + edge_weight); os.environ['MODELDIR_BASE'] = modeldir_base
    modeldir = os.path.join(trtedir, parc_str + '_' + str(num_parcels) + '_' + edge_weight + extra_str); os.environ['MODELDIR'] = modeldir

    # normative dir
    model = primary_covariate + '+sex_adj'
    normativedir = os.path.join(modeldir, model); os.environ['NORMATIVEDIR'] = normativedir

    # Figure directory
    figdir = os.path.join(modeldir, 'figs'); os.environ['FIGDIR'] = figdir

    # Yeo labels
    if parc_str == 'lausanne':
        yeo_idx = np.loadtxt(os.path.join(projdir, 'figs_support/labels/yeo7netlabelsLaus' + str(parc_scale) + '.txt')).astype(int)
        yeo_labels = ('Visual', 'Somatomator', 'Dorsal Attention', 'Ventral Attention', 'Limbic', 'Frontoparietal Control', 'Default Mode', 'Subcortical', 'Brainstem')
    elif parc_str == 'schaefer':
        yeo_idx = np.loadtxt(os.path.join(projdir, 'figs_support/labels/yeo17netlabelsSchaefer' + str(parc_scale) + '.txt')).astype(int)
        yeo_labels = ('VisCent', 'VisPeri', 'SomMotA', 'SomMotB', 'DorsAttnA', 'DorsAttnB', 'SalVentAttnA', 'SalVentAttnB',
                    'LimbicA', 'LimbicB', 'ContA', 'ContB', 'ContC', 'DefaultA', 'DefaultB', 'DefaultC', 'TempPar')

    return parcel_names, parcel_loc, drop_parcels, num_parcels, yeo_idx, yeo_labels
