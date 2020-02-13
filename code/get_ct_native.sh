ctdir=/Users/lindenmp/Dropbox/Work/ResData/PNC/processedData/antsCorticalThickness/
parcdir=/Users/lindenmp/Dropbox/Work/ResData/PNC/processedData/gm_vol_masks_native/
parc_file=Schaefer2018_400_17Networks_native_gm.nii.gz
cd ${ctdir}

for i in */; do
	cd ${ctdir}${i}; for j in */; do
		echo ${i}${j}
    	fslmeants -i ${ctdir}${i}${j}CorticalThickness.nii.gz --label=${parcdir}${i}${j}${parc_file} -o ${ctdir}${i}${j}ct_schaefer400_17.txt
    done
done
