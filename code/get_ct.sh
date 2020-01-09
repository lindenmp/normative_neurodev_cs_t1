# input data directories
in_dir='/Users/lindenmp/Dropbox/Work/ResData/PNC/processedData/voxelwiseMaps_antsCt/'

cd ${in_dir}
for scan in *
do
    echo ${scan}
    name=$(echo "$scan" | cut -f 1 -d '.')

    parc_file='/Users/lindenmp/Dropbox/Work/ResProjects/NormativeNeuroDev_CrossSec_T1/templates/schaefer2018PNC2mm/Schaefer2018_400_17Networks_PNC_2mm.nii.gz'
    fslmeants -i ${scan} --label=${parc_file} -o ${name}_schaefer400_17.txt
done
