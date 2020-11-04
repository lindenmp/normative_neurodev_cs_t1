#!/usr/bin/env python
# coding: utf-8

# # Submit jobs to cubic

# In[ ]:


import os
import numpy as np
import subprocess
import json

py_exec = '/cbica/home/parkesl/miniconda3/envs/neurodev_cs_predictive/bin/python'

my_str = 't1Exclude_schaefer_400'
# my_str = 't1Exclude_schaefer_200'
# my_str = 'fsFinalExclude_lausanne_250'
# my_str = 'fsFinalExclude_lausanne_125'

indir = '/cbica/home/parkesl/research_projects/normative_neurodev_cs_t1/2_pipeline/8_prediction_fixedpcs/store/'+my_str+'_'
outdir = '/cbica/home/parkesl/research_projects/normative_neurodev_cs_t1/2_pipeline/8_prediction_fixedpcs/out/'+my_str+'_'

phenos = ['Overall_Psychopathology','Psychosis_Positive','Psychosis_NegativeDisorg','AnxiousMisery','Externalizing','Fear']

print(indir)
print(phenos)

metrics = ['vol', 'ct']
algs = ['rr',]
scores = ['corr', 'rmse', 'mae']

num_algs = len(algs)
num_metrics = len(metrics)
num_phenos = len(phenos)
num_scores = len(scores)

print(num_algs * num_metrics * num_phenos * num_scores * 2)


# ## Nuisance regression

# In[ ]:


covs = ['ageAtScan1_Years', 'sex_adj']
# covs = ['ageAtScan1_Years', 'sex_adj', 'medu1']

#########
for metric in metrics:
    if metric == 'vol':
        py_script = '/cbica/home/parkesl/research_projects/normative_neurodev_cs_t1/1_code/cluster/predict_symptoms_rcv_nuis_vol.py'
    elif metric == 'ct':
        py_script = '/cbica/home/parkesl/research_projects/normative_neurodev_cs_t1/1_code/cluster/predict_symptoms_rcv_nuis_ct.py'

    for alg in algs:
        for pheno in phenos:
            for score in scores:
                modeldir = outdir+'predict_symptoms_rcv_nuis_'+'_'.join(covs)
                subprocess_str = '{0} {1} -x {2}X.csv -y {2}y.csv -c {2}c_{3}.csv -alg {4} -metric {5} -pheno {6} -score {7} -runpca 3 -runperm 1 -o {8}'.format(py_exec, py_script, indir, '_'.join(covs), alg, metric, pheno, score, modeldir)
                
                name = 'raw_' + metric[0] + '_' + alg + '_' + pheno[0] + '_' + score[0]
                qsub_call = 'qsub -N {0} -l h_vmem=1G,s_vmem=1G -pe threaded 2 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ '.format(name)

                os.system(qsub_call + subprocess_str)
                
                #########
                modeldir = outdir+'predict_symptoms_rcv_nuis_'+'_'.join(covs)+'_z'
                subprocess_str = '{0} {1} -x {2}X_z.csv -y {2}y.csv -c {2}c_{3}.csv -alg {4} -metric {5} -pheno {6} -score {7} -runpca 3 -runperm 1 -o {8}'.format(py_exec, py_script, indir, '_'.join(covs), alg, metric, pheno, score, modeldir)

                name = 'z_' + metric[0] + '_' + alg + '_' + pheno[0] + '_' + score[0]
                qsub_call = 'qsub -N {0} -l h_vmem=1G,s_vmem=1G -pe threaded 2 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ '.format(name)

                os.system(qsub_call + subprocess_str)

