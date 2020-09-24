server_dir=/data/joy/BBL/studies/pnc/processedData/structural/antsCorticalThickness/
cd ${server_dir}
out_dir=/data/jag/bassett-lab/lindenmp/NeuroDev_NetworkControl/data/PNC/derivatives/gm_vol_masks_native/

for i in */; do
	cd ${server_dir}${i}; for j in */; do
		echo ${i}${j}
		mkdir -p ${out_dir}${i}${j}

		# Get cortical gray matter mask in native space
		fslmaths ${server_dir}${i}${j}BrainSegmentation.nii.gz -thr 2 -uthr 2 -bin ${out_dir}${i}${j}BrainSegmentation_cortical_gm.nii.gz

		# Move Schaefer from PNC to native space
		/data/joy/BBL/applications/ANTSlatest/build/bin/antsApplyTransforms -d 3 -e 0 -i \
			/data/jag/bassett-lab/lindenmp/NeuroDev_NetworkControl/data/PNC/derivatives/Schaefer2018_400_17Networks_PNC_2mm.nii.gz \
			-r ${out_dir}${i}${j}BrainSegmentation_cortical_gm.nii.gz \
			-o ${out_dir}${i}${j}Schaefer2018_400_17Networks_native.nii.gz \
			-n GenericLabel \
			-t ${server_dir}${i}${j}TemplateToSubject0Warp.nii.gz -t ${server_dir}${i}${j}TemplateToSubject1GenericAffine.mat

		# Mask parcellation by cortical gm
		fslmaths ${out_dir}${i}${j}BrainSegmentation_cortical_gm.nii.gz \
			-mul ${out_dir}${i}${j}Schaefer2018_400_17Networks_native.nii.gz \
			${out_dir}${i}${j}Schaefer2018_400_17Networks_native_gm.nii.gz

		# Move Schaefer from PNC 200 to native space
		/data/joy/BBL/applications/ANTSlatest/build/bin/antsApplyTransforms -d 3 -e 0 -i \
			/data/jag/bassett-lab/lindenmp/NeuroDev_NetworkControl/data/PNC/derivatives/Schaefer2018_200_17Networks_PNC_2mm.nii.gz \
			-r ${out_dir}${i}${j}BrainSegmentation_cortical_gm.nii.gz \
			-o ${out_dir}${i}${j}Schaefer2018_200_17Networks_native.nii.gz \
			-n GenericLabel \
			-t ${server_dir}${i}${j}TemplateToSubject0Warp.nii.gz -t ${server_dir}${i}${j}TemplateToSubject1GenericAffine.mat

		# Mask parcellation by cortical gm
		fslmaths ${out_dir}${i}${j}BrainSegmentation_cortical_gm.nii.gz \
			-mul ${out_dir}${i}${j}Schaefer2018_200_17Networks_native.nii.gz \
			${out_dir}${i}${j}Schaefer2018_200_17Networks_native_gm.nii.gz
	done
done
