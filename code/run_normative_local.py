#import nispat
import os, sys
sys.path.append('/Users/lindenmp/Dropbox/Work/git/nispat/nispat') # Linden's Macbook Pro
from normative import estimate

# primary directory for normative model
exclude_str = 't1Exclude'
# exclude_str = 'fsFinalExclude'

combo_label = 'schaefer_400_streamlineCount'
# combo_label = 'schaefer_400_streamlineCount_nuis-netdens'
# combo_label = 'schaefer_400_streamlineCount_nuis-str'
# combo_label = 'schaefer_200_streamlineCount'

# combo_label = 'lausanne_234_streamlineCount'
# combo_label = 'lausanne_129_streamlineCount'

# Cross-sec
normativedir = os.path.join('/Users/lindenmp/Dropbox/Work/ResProjects/NormativeNeuroDev_CrossSec/analysis/normative',
	exclude_str, 'squeakycleanExclude', combo_label, 'ageAtScan1_Years+sex_adj/') # Linden's Macbook Pro

print(normativedir)

# ################################################################################################
# Primary model
wdir = normativedir; os.chdir(wdir)
# input files and paths
cov_train = os.path.join(normativedir, 'cov_train.txt');
resp_train = os.path.join(normativedir, 'resp_train.txt');
cov_test = os.path.join(normativedir, 'cov_test.txt');
resp_test = os.path.join(normativedir, 'resp_test.txt');
# run normative
estimate(resp_train, cov_train, testcov=cov_test, testresp=resp_test, cvfolds=None, alg = "gpr")

################################################################################################
# Forward model
wdir = os.path.join(normativedir, 'forward'); os.chdir(wdir);
# input files and paths
cov_train = os.path.join(normativedir, 'cov_train.txt');
resp_train = os.path.join(normativedir, 'resp_train.txt');
cov_test = os.path.join(wdir, 'synth_cov_test.txt');
# run normative
estimate(resp_train, cov_train, testcov=cov_test, testresp=None, cvfolds=None, alg = "gpr")
