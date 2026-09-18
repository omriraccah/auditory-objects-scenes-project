#!/bin/bash

# SLURM Options: Log file, 16 CPUs, 6GB per CPU, 100 hours time limit

#SBATCH --output logs/%j-fmriprep.out
#SBATCH --job-name preproc_objects_scenes
#SBATCH -n 16 -t 100:00:00 
#SBATCH --mem-per-cpu 10G
#SBATCH --mail-type ALL
#SBATCH --mail-user=aryan.agarwal@yale.edu
#SBATCH --partition=psych_week

# Set the subject ID in BIDS (ignoring sub- prefix)
SUB=$1
USE_FIELDMAPS=$3
USE_FS_NO_RECONALL=$4

# Load the version of fMRIprep to be used
FMRIPREP_SIF="/gpfs/milgram/project/turk-browne/aa2842/containers/fmriprep-25.2.5.sif"

# Set paths for root directory to the data, freesurfer licence, 
# output directory for derivatives, and work directory for intermediate files.

ROOT="/gpfs/milgram/scratch60/turk-browne/$2/sandbox/auditory-object-scenes-data/objects_scenes_bids/"

export FS_LICENSE=/gpfs/milgram/project/turk-browne/aa2842/license.txt

cd $ROOT
OUT="/gpfs/milgram/scratch60/turk-browne/$2/sandbox/auditory-object-scenes-data/objects_scenes_bids/derivatives/"
WORK="/gpfs/milgram/scratch60/turk-browne/$2/sandbox/auditory-object-scenes-data/bids_workdir/sub-${SUB}_${SLURM_JOB_ID}/"

mkdir -p "$OUT"
mkdir -p "$WORK"

echo $SUB $ROOT $OUT $WORK

# This is the command that runs fmriprep. Specify the directories, 'participant' level processing, and templates for spatial normalization (e.g. MNI152).

FMRIPREP_CMD=(
    apptainer run --cleanenv
    -B "${ROOT}:/data:ro"
    -B "${OUT}:/out"
    -B "${WORK}:/work"
    -B "${FS_LICENSE}:/opt/freesurfer/license.txt:ro"
    "$FMRIPREP_SIF"
    /data
    /out
    participant
    --participant-label "$SUB"
    --nthreads 16
    --random-seed 12345
    --skull-strip-fixed-seed
    -w /work
    --output-spaces T1w MNI152Lin anat MNI152NLin2009cAsym
    --fs-license-file /opt/freesurfer/license.txt
)

if [ "$USE_FIELDMAPS" = 0 ]; then
    FMRIPREP_CMD+=(--ignore fieldmaps)
fi

if [ "$USE_FS_NO_RECONALL" = 1 ]; then
    FMRIPREP_CMD+=(--fs-no-reconall)
fi

echo "Running fMRIPrep command:"
printf '%q ' "${FMRIPREP_CMD[@]}"
echo

"${FMRIPREP_CMD[@]}"

PREPROC="/gpfs/milgram/scratch60/turk-browne/$2/sandbox/auditory-object-scenes-data/preprocessed"
if [ ! -d $PREPROC ]; then
        mkdir $PREPROC
fi

cp -r /gpfs/milgram/scratch60/turk-browne/$2/sandbox/auditory-object-scenes-data/objects_scenes_bids/derivatives/sub-$1* /gpfs/milgram/scratch60/turk-browne/$2/sandbox/auditory-object-scenes-data/preprocessed/

cp /gpfs/milgram/scratch60/turk-browne/$2/sandbox/auditory-object-scenes-data/objects_scenes_bids/sub-$1/func/*.tsv /gpfs/milgram/scratch60/turk-browne/$2/sandbox/auditory-object-scenes-data/preprocessed/sub-$1/func/
