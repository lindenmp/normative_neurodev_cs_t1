# Normative neurodevelopment
This repository includes code used to analyze the relationship between dimensional psychopathology phenotypes and deviations from normative neurodevelopment in the Philadelphia Neurodevelopmental Cohort.

# Environment build

    conda create -n normative_neurodev_cs_t1 python=3.7
    conda activate normative_neurodev_cs_t1

    # Essentials
    pip install jupyterlab ipython pandas numpy seaborn matplotlib nibabel glob3 nilearn ipywidgets
    pip install jupyter_contrib_nbextensions && jupyter contrib nbextension install

	# Statistics
	pip install scipy statsmodels sklearn pingouin pygam brainspace bctpy

	# Extra
	pip install mat73

	# Pysurfer for plotting
	pip install vtk==8.1.2
	pip install mayavi
	pip install PyQt5
	jupyter nbextension install --py mayavi --user
	jupyter nbextension enable --py mayavi --user
	jupyter nbextension enable --py widgetsnbextension
	pip install pysurfer

    cd /Users/lindenmp/Google-Drive-Penn/work/research_projects/normative_neurodev_cs_t1
    conda env export > environment.yml
	pip freeze > requirements.txt

# Code

In the **code** subdirectory you will find the following Jupyter notebooks and .py scripts:
1. Pre-normative modeling scripts:
- `0_get_sample.ipynb`
	- Performs initial ingest of PNC demographic data, participant exclusion based on various quality control.
	- Produces Figures 2A and 2B.
	- Designates train/test split.
- `1_compute_node_metrics.ipynb`
	- Reads in neuroimaging data.
	- Sets up feature table of regional brain features.
- `2_prepare_normative.ipynb`
	- Prepares input files for normative modeling.
- `3_results_sample_characteristics.ipynb`
	- Characterizing sample demographics
	- Produces Figures S1, S2, S3, and S9

2. Run normative modeling:
- `cluster/4_run_normative.py`
	- Submits normative models to the cbica cluster

3. Results:
- `5_results_forward.ipynb`
	- Plots predictions from the normative model as annualized percent change.
	- Produces Figures S4
- `6_results_correlations.ipynb`
	- Computes regional correlations between psychopathology dimensions and deviations from the normative model
	- Produces Figures 2, 3, and S8
- `7_results_case_control.ipynb`
	- Computes regional Cohen's d comparing deviations from depression and ADHD groups against healthy controls
	- Computes spatial correlation of Cohen's d values between depression and ADHD groups
	- Repeats above analyses controlling for overall psychopathology
	- Produces Figure 4

4. Prediction:
- `8_prepare_prediction.ipynb`
	- Prepares input files for prediction modeling.
	- Produces Figures S5, S6, and S7
- `cluster/9_job_submitter.ipynb`
	- Submits prediction models to the cbica cluster

5. Results (cont.):
- `10_results_model_performance.ipynb`
	- Produces Figures 1 and S10

<!-- - `results_s4_nm_vs_nonnm.ipynb`
	- Computes regional correlations between psychopathology dimensions and brain features, bypassing the normative model
	- Loads in results from `results_s2_phenos.ipynb` and performs regional Steiger's tests to compare analyses
	- Produces Table S1 -->

<!-- # Dependencies

This code depends upon https://github.com/lindenmp/pyfunc -->
