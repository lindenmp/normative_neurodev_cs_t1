# This script takes the Schaefer parcellation in PNC template and multiplies it by each subject's GM hard segmentation mask
# This produces a Schaefer parc for each subjects where only voxels inside GM are retained.
# This new mask is used to extract region CT/JD measures for each subjects

# input data directories
in_dir='/Users/lindenmp/Dropbox/Work/ResData/PNC/processedData/gm_masks_template/'

cd ${in_dir}
for gm_mask in *.nii.gz
do
    echo ${gm_mask}
    out_label=$(echo "$gm_mask" | cut -f 1 -d '.')

    parc_file='/Users/lindenmp/Dropbox/Work/ResProjects/NormativeNeuroDev_CrossSec_T1/templates/schaefer2018PNC2mm/Schaefer2018_400_17Networks_PNC_2mm.nii.gz'
	fslmaths ${parc_file} -mul ${gm_mask} ${out_label}_Schaefer400_17.nii.gz
done
