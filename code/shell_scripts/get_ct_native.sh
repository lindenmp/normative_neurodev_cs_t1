ctdir=/Users/lindenmp/Dropbox/Work/ResData/PNC/processedData/antsCorticalThickness/
# parcdir=/Users/lindenmp/Dropbox/Work/ResData/PNC/processedData/gm_vol_masks_native/
# parc_file=Schaefer2018_400_17Networks_native_gm.nii.gz
# parc_file=Schaefer2018_200_17Networks_native_gm.nii.gz

parcdir=/Users/lindenmp/Dropbox/Work/ResData/PNC/processedData/freesurfer53/
cd ${ctdir}

for i in *; do
	cd ${ctdir}${i};
	for j in *; do
		echo ${i}/${j}
    	# fslmeants -i ${ctdir}${i}/${j}/CorticalThickness.nii.gz --label=${parcdir}${i}/${j}/${parc_file} -o ${ctdir}${i}/${j}/ct_schaefer400_17.txt
    	# fslmeants -i ${ctdir}${i}/${j}/CorticalThickness.nii.gz --label=${parcdir}${i}/${j}/${parc_file} -o ${ctdir}${i}/${j}/ct_schaefer200_17.txt
    	
		parc_file=${parcdir}${i}/${j}/label/${i}_${j}_Lausanne_ROIv_scale60_T1.nii.gz
		if test -f "${parc_file}"; then fslmeants -i ${ctdir}${i}/${j}/CorticalThickness.nii.gz --label=${parc_file} -o ${ctdir}${i}/${j}/ct_lausanne60.txt
		else echo "No parc file"; fi

		parc_file=${parcdir}${i}/${j}/label/${i}_${j}_Lausanne_ROIv_scale125_T1.nii.gz
		if test -f "${parc_file}"; then fslmeants -i ${ctdir}${i}/${j}/CorticalThickness.nii.gz --label=${parc_file} -o ${ctdir}${i}/${j}/ct_lausanne125.txt
		else echo "No parc file"; fi

		parc_file=${parcdir}${i}/${j}/label/${i}_${j}_Lausanne_ROIv_scale250_T1.nii.gz
		if test -f "${parc_file}"; then fslmeants -i ${ctdir}${i}/${j}/CorticalThickness.nii.gz --label=${parc_file} -o ${ctdir}${i}/${j}/ct_lausanne250.txt
		else echo "No parc file"; fi
    done
done
