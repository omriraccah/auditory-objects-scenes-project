#!/bin/bash
#SBATCH --job-name=segmentation-multimem
#SBATCH --ntasks=1 --nodes=1
#SBATCH --output=logs/ASHS_segmentation-%j.out
#SBATCH -p psych_day
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=25000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aryan.agarwal@yale.edu

sub_num=$1
sub="$sub_num"

# shellcheck disable=SC1090
LOG_DIR="/gpfs/milgram/project/turk-browne/$2/auditory-objects-scenes-project/preprocessing/ASHS/logs"
DATA_DIR="/gpfs/milgram/scratch60/turk-browne/$2/sandbox_av/objects_scenes_bids/sub-$1"
SUBJ_DIR="/gpfs/milgram/scratch60/turk-browne/$2/sandbox_av/preprocessed/sub-$1"
ANAT_DIR="$DATA_DIR/anat"
ROI_DIR="$SUBJ_DIR/rois/ASHS_maguire"

#delete existing roi directory
#rm -r "$SUBJ_DIR/rois/"

if [ ! -d $ROI_DIR ]; then
        mkdir -p $ROI_DIR
fi

echo "segmenting T2 scan"

# run ASHS segmentation
# `-I` participant Id (This ID gets propagated throughout the pipeline)
# `-a` location of the ASHS trained model 
# `-g` the location of the T1 
# `-f` the location of the T2
# `-w` the output directory

export ASHS_ROOT=/gpfs/milgram/scratch60/turk-browne/$2/sandbox_av/ashs-fastashs_2_beta

T1_PATH="$ANAT_DIR/sub-$1_T1w.nii" 
T2_PATH="$ANAT_DIR/sub-$1_acq-hipp_T2w.nii" 

ASHS_TRAINED_MODEL="/gpfs/milgram/scratch60/turk-browne/$2/sandbox/ashs_maguire_atlas_3T"
now=`date +%Y-%m-%d_%H:%M:%S`

bash $ASHS_ROOT/bin/ashs_main.sh -I $sub -a $ASHS_TRAINED_MODEL -g $T1_PATH -f $T2_PATH -w $ROI_DIR >>${LOG_DIR}/step5A_roi_segment_ASHS_$sub_$now.txt 2>&1

HPC_DIR="$ROI_DIR/final"

FINAL_DIR="$HPC_DIR/func_masks"

if [ ! -d $FINAL_DIR ]; then
        mkdir -p $FINAL_DIR
fi