# Normative neurodevelopment
This repository includes code used to analyze the relationship between dimensional psychopathology phenotypes and deviations from normative neurodevelopment in the Philadelphia Neurodevelopmental Cohort.

# Environment build

    conda create -n NormativeNeuroDev_CrossSec_T1 python=3.7
    conda activate NormativeNeuroDev_CrossSec_T1
    # Essentials
    pip install jupyterlab ipython pandas numpy scipy seaborn matplotlib
    pip install jupyter_contrib_nbextensions && jupyter contrib nbextension install
	# Statistics
	pip install statsmodels sklearn pingouin
	# Extras
    pip install nibabel torch glob3
    cd /Users/lindenmp/Dropbox/Work/ResProjects/NormativeNeuroDev_CrossSec_T1
    conda env export > environment.yml
	pip freeze > requirements.txt

# Code

In the **code** subdirectory you will find the following Jupyter notebooks and .py scripts:
1. Pre-normative modeling scripts:
- `get_train_test.ipynb`
	- Performs initial ingest of PNC demographic data, participant exclusion based on various quality control.
	- Produces Figures 2A and 2B.
	- Designates train/test split.
- `compute_node_metrics.ipynb`
	- Reads in neuroimaging data.
	- Sets up feature table of regional brain features.
- `clean_node_metrics.ipynb`
	- Performs nuisance regression on feature table.
- `prepare_normative.ipynb`
	- Prepares input files for normative modeling.

2. Run normative modeling:
- `run_normative_local.py`
	- Runs primary normative models on local machine.
- `cluster/run_normative_perm.sh`
	- Submits each of the permuted normative models to the cluster as a separate job

3. Results:
- `results_s1.ipynb`
	- Produces Figure 2C
- `results_s2.ipynb`
	- Produces Figures 3 and 4
- `results_s3.ipynb`
	- Produces Figure 5
