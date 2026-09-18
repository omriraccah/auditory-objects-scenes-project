#!/usr/bin/env bash
#SBATCH --output=/gpfs/milgram/project/turk-browne/aa2842/auditory-objects-scenes-project/GLMsingle/logs/%j-GLMsingle_%A_%a.out
#SBATCH --job-name GLMSingle
#SBATCH --array=1-2
#SBATCH -n 16
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type ALL
#SBATCH --partition=psych_week

module load miniconda
conda activate "/gpfs/milgram/project/turk-browne/aa2842/conda_envs/myenv"

MODALITIES=("visual" "auditory")

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    MODALITY=${MODALITIES[$SLURM_ARRAY_TASK_ID - 1]}
else
    MODALITY=${1:-auditory}
fi

echo "Running $MODALITY GLMsingle for subject pp06"

python "/gpfs/milgram/project/turk-browne/aa2842/auditory-objects-scenes-project/GLMsingle/glm_single.py" \
    -s pp06 -m "$MODALITY" "${@:2}"
