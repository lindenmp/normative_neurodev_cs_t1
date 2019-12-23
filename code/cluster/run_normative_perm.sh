#!/bin/bash

#SBATCH --job-name=nm_perm
#SBATCH --account=dq13
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-999

echo "Running on `hostname` at `date`"
perm=perm_${SLURM_ARRAY_TASK_ID}
echo -e " ----- ${perm} ----- "

export OMP_NUM_THREADS=1

module load python/3.6.2
source /home/lindenmp/virtual_env/NeuroDev_NetworkControl/bin/activate

combo_label=schaefer_400_streamlineCount
# combo_label=schaefer_400_streamlineCount_nuis-netdens
# combo_label=schaefer_400_streamlineCount_nuis-str
# combo_label=schaefer_200_streamlineCount

# combo_label=lausanne_234_streamlineCount
# combo_label=lausanne_129_streamlineCount

normativedir=/scratch/kg98/Linden/ResProjects/NormativeNeuroDev_CrossSec/analysis/normative/t1Exclude/squeakycleanExclude/${combo_label}/ageAtScan1_Years+sex_adj/
# normativedir=/scratch/kg98/Linden/ResProjects/NormativeNeuroDev_CrossSec/analysis/normative/fsFinalExclude/squeakycleanExclude/${combo_label}/ageAtScan1_Years+sex_adj/

wdir=${normativedir}perm_all/${perm}/
cd ${wdir}
/home/lindenmp/virtual_env/NeuroDev_NetworkControl/bin/python \
	/scratch/kg98/Linden/ResProjects/NormativeNeuroDev_CrossSec/code/nispat/nispat/normative.py \
	-c ${wdir}cov_train.txt \
	-t ${wdir}cov_test.txt \
	-r ${normativedir}resp_test.txt \
	-a gpr \
	${normativedir}resp_train.txt | ts
