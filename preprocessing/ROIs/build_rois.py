from nilearn import datasets, plotting, image
from ants import image_read, apply_transforms
import os 
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":

    ############ Parse CL args ###############
    parser = argparse.ArgumentParser()
    parser.add_argument("-sub", "--subject_id", type=str, help="multimodal_sub_01", default="01")
    parser.add_argument('-u', '--user', type=str, default="aa2842")
    parser.add_argument("-e", '--execute', type=int, default=1)
    parser.add_argument("-v", "--verbose", type=int, default=1)

    args = parser.parse_args()
    sub = args.subject_id
    execute = args.execute
    user = args.user
    verbose = args.verbose
    
    # Step 1: Download the Nilearn atlas in the same location each time. 

    # Save the Nilearn atlas here
    atlas_dir = f"/gpfs/milgram/scratch60/turk-browne/{user}/sandbox/atlases/harvard-oxford/"
    os.makedirs(atlas_dir, exist_ok=True)

    # Save the specific MNI ROI masks here
    roi_dir = f"/gpfs/milgram/project/turk-browne/{user}/auditory-objects-scenes-project/preprocessing/ROIs/results/harvard-oxford/"
    os.makedirs(roi_dir, exist_ok=True)

    # Save the T1 transformed ROI masks here
    sub_roi_dir = f"/gpfs/milgram/scratch60/turk-browne/{user}/sandbox/auditory-object-scenes-data/preprocessed/sub-pp{sub}/rois/harvard-oxford/"
    os.makedirs(sub_roi_dir, exist_ok=True)
    
    # Fetching the Harvard-Oxford cortical atlas with 25% threshold
    dataset_ho = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm', data_dir=atlas_dir)
    atlas_ho_filename = dataset_ho.filename
    atlas_ho_labels = dataset_ho.labels
    atlas_img = image.load_img(atlas_ho_filename)

    # Step 2: Extract ROIs and save in the MNI space folder. 
    # OR COMMENTED THIS OUT TO PRODUCE ALL ROIS FROM THE ATLAS (Jan 10 2025)
    #rois = ['Lateral Occipital Cortex, inferior division', 'Superior Temporal Gyrus, anterior division', 'Intracalcarine Cortex', "Heschl's Gyrus (includes H1 and H2)", 'Occipital Pole', 'Parahippocampal Gyrus, anterior division', 'Parahippocampal Gyrus, posterior division', 'Temporal Pole']
    #roi_names = ['LOC', 'STG_A', 'IC', 'HG', 'OP', 'PPA_A', 'PPA_P', 'TP']
    
    rois = atlas_ho_labels
    roi_names = atlas_ho_labels
    roi_indices = [atlas_ho_labels.index(x) for x in rois]
    
    # Create a mask for each ROI and save it
    mni_roi_paths = []
    for roi_index, roi_name in zip(roi_indices, roi_names):
        roi_mask = image.math_img(f"img == {roi_index}", img=atlas_img)
        roi_path = f"{roi_dir}{roi_name}.nii.gz"
        roi_mask.to_filename(roi_path)
        mni_roi_paths.append(roi_path)
    
    # Step 3: Convert masks into the T1w space and store appropriately
    
    # Now we loop through all ROIs and use ANTS to transform them to T1 space and save them. 
    final_roi_paths = []
    for roi_path in mni_roi_paths:
        # Set the paths
        roi_name = os.path.basename(roi_path)
        transform_path = f"/gpfs/milgram/scratch60/turk-browne/{user}/sandbox/auditory-object-scenes-data/preprocessed/sub-pp{sub}/anat/sub-pp{sub}_from-MNI152Lin_to-T1w_mode-image_xfm.h5"
        t1_path = f"/gpfs/milgram/scratch60/turk-browne/{user}/sandbox/auditory-object-scenes-data/preprocessed/sub-pp{sub}/anat/sub-pp{sub}_desc-preproc_T1w.nii.gz"
        
        
        transformed_roi = apply_transforms(fixed=image_read(t1_path), moving=image_read(roi_path), transformlist=transform_path, interpolator='nearestNeighbor')
        
        # Save the file
        transformed_path = f"{sub_roi_dir}{roi_name}"
        transformed_roi.to_filename(transformed_path)
        
        final_roi_paths.append(transformed_path)

    print("Analysis completed; all ROIs saved successfully", flush=True)
    