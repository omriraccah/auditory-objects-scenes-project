#!/usr/bin/env bash
#SBATCH --output=logs/%j-renaming.out
#SBATCH -p psych_day
#SBATCH -t 2:00:00
#SBATCH --mem 5GB
#SBATCH -n 1

events_dir="/gpfs/milgram/scratch60/turk-browne/$2/sandbox_av/events"
nii_dir="/gpfs/milgram/scratch60/turk-browne/$2/sandbox_av/${1}_nii"
if [[ ! -d "$events_dir" ]]; then
    echo "Error: events directory does not exist: $events_dir" >&2
    exit 1
fi

shopt -s nullglob
mat_files=("$events_dir"/*"$1".mat)
shopt -u nullglob
if (( ${#mat_files[@]} > 0 )); then
    cp -rf "${mat_files[@]}" "$nii_dir/" || exit 1
fi
echo "Copied ${#mat_files[@]} .mat files to $nii_dir/"

module load miniconda
conda activate "/gpfs/milgram/project/turk-browne/$2/conda_envs/myenv"

python "/gpfs/milgram/project/turk-browne/$2/auditory-objects-scenes-project/preprocessing/XNat_Interact/rename.py" -sub "$1" -u "$2" -e "$3" -v "$4"
