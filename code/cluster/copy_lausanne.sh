server_dir=/data/joy/BBL/studies/pnc/processedData/structural/freesurfer53/
out_dir=/data/jag/bassett-lab/lindenmp/NeuroDev_NetworkControl/data/PNC/derivatives/freesurfer53/

cd ${server_dir}
for i in *; do cd ${server_dir}${i}; for j in *x*; do echo ${i}/${j}; mkdir -p ${out_dir}${i}/${j}/label/; cp ${server_dir}${i}/${j}/label/${i}_${j}_Lausanne_ROIv_scale125_T1.nii.gz ${out_dir}${i}/${j}/label/; done; done

cd ${server_dir}
for i in *; do cd ${server_dir}${i}; for j in *x*; do echo ${i}/${j}; cp ${server_dir}${i}/${j}/label/${i}_${j}_Lausanne_ROIv_scale60_T1.nii.gz ${out_dir}${i}/${j}/label/; done; done

cd ${server_dir}
for i in *; do cd ${server_dir}${i}; for j in *x*; do echo ${i}/${j}; cp ${server_dir}${i}/${j}/label/${i}_${j}_Lausanne_ROIv_scale250_T1.nii.gz ${out_dir}${i}/${j}/label/; done; done
