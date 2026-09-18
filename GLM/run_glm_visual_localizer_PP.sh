#!/usr/bin/env bash
#SBATCH --output logs/%j-GLM-visual-localizer_%A_%a.out
#SBATCH --job-name GLM_VISUAL_LOCALIZER-PP
#SBATCH --array=1-1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=30:00
#SBATCH --mail-type ALL
#SBATCH --partition=psych_day

module load miniconda
conda activate "/gpfs/milgram/project/turk-browne/$1/conda_envs/myenv"

SUBJECTS=("pp05")

SUBJECT_ID=${SUBJECTS[$SLURM_ARRAY_TASK_ID - 1]}

echo "Running visual localizer GLM for subject $SUBJECT_ID"

python "/gpfs/milgram/project/turk-browne/$1/auditory-objects-scenes-project/GLM/glm_visual_localizer.py" -s "$SUBJECT_ID"
