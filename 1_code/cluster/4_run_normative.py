# NOTE: this is less of a script and more of a scrapbook.
# it simply stores chunks of code for 'copy/paste' purposes to run normative models on the cluster

#import nispat
import os, sys
from nispat.normative import estimate
from nispat.normative_parallel import execute_nm, collect_nm, delete_nm

# ------------------------------------------------------------------------------
# parallel (batch)
# ------------------------------------------------------------------------------
# settings and paths
python_path = '/cbica/home/parkesl/miniconda3/envs/nispat/bin/python'
normative_path = '/cbica/home/parkesl/miniconda3/envs/nispat/nispat/nispat/normative.py'
batch_size = 4
memory = '2G'
duration = '2:00:00'
cluster_spec = 'cubica'

# ------------------------------------------------------------------------------
# directories
# ------------------------------------------------------------------------------
file_prefix = 't1Exclude_schaefer_400_'

inputdir = '/cbica/home/parkesl/research_projects/normative_neurodev_cs_t1/2_pipeline/2_prepare_normative/out/'
print(inputdir)

outputdir = '/cbica/home/parkesl/research_projects/normative_neurodev_cs_t1/2_pipeline/4_run_normative/'
print(outputdir)
if not os.path.exists(outputdir): os.makedirs(outputdir)

# ------------------------------------------------------------------------------
# Primary model
job_name = 'main_'
workdir = os.path.join(outputdir,file_prefix+'out/')
if not os.path.exists(workdir): os.makedirs(workdir)
os.chdir(workdir)

# input files and paths
cov_train = os.path.join(inputdir, file_prefix+'cov_train.txt')
resp_train = os.path.join(inputdir, file_prefix+'resp_train.txt')
cov_test = os.path.join(inputdir, file_prefix+'cov_test.txt')
resp_test = os.path.join(inputdir, file_prefix+'resp_test.txt')

# run normative
execute_nm(workdir,
	python_path=python_path,
	normative_path=normative_path,
	job_name=job_name,
	covfile_path=cov_train,
	respfile_path=resp_train,
	batch_size=batch_size,
	memory=memory,
	duration=duration,
	cluster_spec=cluster_spec,
	cv_folds=None,
	testcovfile_path=cov_test,
	testrespfile_path=resp_test,
	alg = 'gpr')
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Forward model
job_name = 'fwd_'
workdir = os.path.join(outputdir,file_prefix+'out_forward/')
if not os.path.exists(workdir): os.makedirs(workdir)
os.chdir(workdir)

# input files and paths
cov_train = os.path.join(inputdir, file_prefix+'cov_train.txt');
resp_train = os.path.join(inputdir, file_prefix+'resp_train.txt');
cov_test = os.path.join(inputdir, file_prefix+'cov_test_forward.txt');

# run normative
execute_nm(workdir,
	python_path=python_path,
	normative_path=normative_path,
	job_name=job_name,
	covfile_path=cov_train,
	respfile_path=resp_train,
	batch_size=batch_size,
	memory=memory,
	duration=duration,
	cluster_spec=cluster_spec,
	cv_folds=None,
	testcovfile_path=cov_test,
	testrespfile_path=None,
	alg = 'gpr')
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Cross val model
job_name = 'cv_'
workdir = os.path.join(outputdir,file_prefix+'out_cv/')
if not os.path.exists(workdir): os.makedirs(workdir)
os.chdir(workdir)

# input files and paths
cov_train = os.path.join(inputdir, file_prefix+'cov_train.txt')
resp_train = os.path.join(inputdir, file_prefix+'resp_train.txt')

# run normative
execute_nm(workdir,
	python_path=python_path,
	normative_path=normative_path,
	job_name=job_name,
	covfile_path=cov_train,
	respfile_path=resp_train,
	batch_size=batch_size,
	memory=memory,
	duration=duration,
	cluster_spec=cluster_spec,
	cv_folds=10,
	testcovfile_path=None,
	testrespfile_path=None,
	alg = 'gpr')
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# NOTE: only run *after* cluster jobs are done
workdir = os.path.join(outputdir,file_prefix+'out/')
collect_nm(workdir, collect=True)
delete_nm(workdir)

workdir = os.path.join(outputdir,file_prefix+'out_forward/')
collect_nm(workdir, collect=True)
delete_nm(workdir)

workdir = os.path.join(outputdir,file_prefix+'out_cv/')
collect_nm(workdir, collect=True)
delete_nm(workdir)

# ------------------------------------------------------------------------------
