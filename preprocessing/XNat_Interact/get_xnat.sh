#!/usr/bin/env bash
#SBATCH --output=logs/%j-GETDATA.out
#SBATCH -p psych_day
#SBATCH -t 1:00:00
#SBATCH --mem 2GB
#SBATCH -n 1

source "/gpfs/milgram/project/turk-browne/aa2842/auditory-objects-scenes-project/preprocessing/XNat_Interact/globals.sh"

module load XNATClientTools

sess_ID=$2
password=$PWD
OUTDIR="/gpfs/milgram/scratch60/turk-browne/$1/sandbox/auditory-object-scenes-data/"
xnat_user=$USER_ID

cd $OUTDIR
echo $xnat_user
echo $sess_ID
echo $password
ArcGet -host https://xnat-milgram.hpc.yale.edu/ -u $xnat_user -p $password -s $sess_ID
unzip ${sess_ID}.zip
#cp -r "/gpfs/milgram/scratch60/turk-browne/$1/sandbox/$2" "/gpfs/milgram/project/turk-browne/$1/auditory-objects-scenes-project/raw_data"