#!/usr/bin/env bash
#SBATCH --output=logs/%j-build-rois.out
#SBATCH -p psych_day
#SBATCH -t 2:00:00
#SBATCH --mem 5GB
#SBATCH -n 1

module load miniconda
conda activate "/gpfs/milgram/project/turk-browne/$2/conda_envs/myenv"

python "/gpfs/milgram/project/turk-browne/$2/auditory-objects-scenes-project/preprocessing/ROIs/build_rois.py" -sub "$1" -u "$2" -e "$3" -v "$4"