# input data directories
in_dir='/Users/lindenmp/Dropbox/Work/ResData/PNC/processedData/voxelwiseMaps_antsCt/'

cd ${in_dir}
for scan in *.nii.gz
do
    echo ${scan}
    out_label=$(echo "$scan" | cut -f 1 -d '.')
    scanid=$(echo "$out_label" | cut -f 1 -d '_')

    parc_file='/Users/lindenmp/Dropbox/Work/ResData/PNC/processedData/gm_masks_template/'${scanid}'_atropos3class_seg_GmMask_Template_Schaefer400_17.nii.gz'
    fslmeants -i ${scan} --label=${parc_file} -o ${out_label}_schaefer400_17_gm.txt
done
