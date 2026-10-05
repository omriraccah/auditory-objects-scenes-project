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
DATA_DIR="/gpfs/milgram/scratch60/turk-browne/$2/sandbox/pp02_nii"
#T2_DIR="/gpfs/milgram/scratch60/turk-browne/$2/sandbox/bids/"
SUBJ_DIR="$DATA_DIR"
ANAT_DIR="$SUBJ_DIR/anat"
ROI_DIR="$SUBJ_DIR/rois/ASHS_princeton_atlas_052_isoo"

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

#export ASHS_ROOT=/gpfs/milgram/pi/turk-browne/shared_resources/packages/ashs-fastashs_beta #(normally used for Princeton)
export ASHS_ROOT=/gpfs/milgram/scratch60/turk-browne/$2/sandbox/ashs-fastashs-2017 

#ASHS_TRAINED_MODEL="/gpfs/milgram/project/turk-browne/projects/stat_episodic/ASHS/ashs_atlas_princeton"
#ASHS_TRAINED_MODEL=/gpfs/milgram/pi/turk-browne/projects/differint/ASHS/princeton_atlas # BIG ATLAS (ABOVE IS SMALL)
T1_PATH="$DATA_DIR/pp02_NewT1_0.8.nii" 
T2_PATH="$DATA_DIR/acq-hipp_T2w_52_iso.nii" 

ASHS_TRAINED_MODEL="/gpfs/milgram/scratch60/turk-browne/$2/sandbox/ashs_atlas_3T_maguire"
#ASHS_TRAINED_MODEL="/gpfs/milgram/scratch60/turk-browne/$2/sandbox/ashs_atlas_upennpmc_t1ext_20240617/final"
#T1_PATH="$ANAT_DIR/sub-${sub}_desc-preproc_T1w.nii.gz"
# T2_PATH="$ANAT_DIR/sub-${sub}_desc-preproc_T2w.nii.gz" THIS IS BEFORE FIX TO HIGHER RES T2w image

now=`date +%Y-%m-%d_%H:%M:%S`

bash $ASHS_ROOT/run_ashs.sh -I $sub -a $ASHS_TRAINED_MODEL -g $T1_PATH -f $T2_PATH -w $ROI_DIR >>${LOG_DIR}/step5A_roi_segment_ASHS_$sub_$now.txt 2>&1

HPC_DIR="$ROI_DIR/final"

FINAL_DIR="$HPC_DIR/func_masks"

if [ ! -d $FINAL_DIR ]; then
        mkdir -p $FINAL_DIR
fi