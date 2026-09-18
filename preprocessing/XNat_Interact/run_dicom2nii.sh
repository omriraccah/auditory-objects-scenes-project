#!/usr/bin/env bash
#SBATCH --output=logs/%j.out
#SBATCH -p psych_day
#SBATCH -t 2:00:00
#SBATCH --mem 4GB
#SBATCH -n 1

module load dcm2niix/1.0.20230411-GCCcore-12.2.0

user=$1
sess_ID=$2
echo "running ${sess_ID}"

# set up paths
export top_dir="/gpfs/milgram/scratch60/turk-browne/$1"
export dcm_dir=${top_dir}/sandbox/auditory-object-scenes-data/$2/SCANS
export nii_dir=${top_dir}/sandbox/auditory-object-scenes-data/$2_nii
mkdir -p $nii_dir; cd $dcm_dir

# looping through files in the dicom directories and run 
for k in *
do
    if [ -d "${k}" ]; then
        dcm2niix -o "$nii_dir" -f "${sess_ID}_%t_%f" "$dcm_dir/$k"
    fi
done
