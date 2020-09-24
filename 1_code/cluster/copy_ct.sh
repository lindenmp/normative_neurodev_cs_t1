server_dir=/data/joy/BBL/studies/pnc/processedData/structural/antsCorticalThickness/
cd ${server_dir}
out_dir=/data/jag/bassett-lab/lindenmp/NeuroDev_NetworkControl/data/PNC/derivatives/antsCorticalThickness/

for i in */; do cd ${server_dir}${i}; for j in */; do echo ${i}${j}; mkdir -p ${out_dir}${i}${j}; cp ${server_dir}${i}${j}CorticalThickness.nii.gz ${out_dir}${i}${j}; done; done
