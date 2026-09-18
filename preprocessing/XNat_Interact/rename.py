import os
import json
import shutil
import pandas as pd
import numpy as np
import argparse
import sys
from scipy.io import loadmat

# Helper function to find the files we need to rename
def find_files(directory):

    # Choose all non '.' files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.json') or f.endswith('.nii') or f.endswith('.tsv') or f.endswith('.mat')]
    files = [f for f in files if 'practice' not in f and 'order' not in f and 'seq' not in f and 'visual_loc_block' not in f]

    # Sort so that we rename nii files first (allows JSON reference with same name)
    files = sorted(files, key=lambda x: (x.endswith('.json'), x))

    if verbose:
        for file in files:
            print(file)

    return files


def matlab_value_to_string(value):
    """Convert a scalar MATLAB cell/string value to a plain Python string."""
    while isinstance(value, np.ndarray):
        if value.size != 1:
            if value.dtype.kind in {"U", "S"}:
                return "".join(value.ravel().astype(str))
            raise ValueError(f"Expected one MATLAB string value, got shape {value.shape}")
        value = value.reshape(-1)[0]

    return str(value)


def classify_visual_stimulus(stimulus_path):
    """Map a visual stimulus filename to its block-level GLM condition."""
    stimulus_path = matlab_value_to_string(stimulus_path).strip().replace("\\", "/")
    path_lower = stimulus_path.lower()
    filename_upper = stimulus_path.rsplit("/", 1)[-1].upper()

    if "/objects/" in path_lower:
        return "visual_object"
    if "/scenes/" in path_lower and filename_upper.startswith("S_"):
        return "visual_scene"
    if "/scenes/" in path_lower and filename_upper.startswith("C_"):
        return "visual_scrambled_scene"

    raise ValueError(f"Could not classify visual stimulus: {stimulus_path}")


def visual_block_conditions(stimulus_sequence, number_of_blocks):
    """Derive one condition per block from the ordered stimulus sequence."""
    stimulus_sequence = list(stimulus_sequence)

    if number_of_blocks <= 0:
        raise ValueError("Visual events file does not contain any blocks")
    if len(stimulus_sequence) % number_of_blocks != 0:
        raise ValueError(
            f"Cannot divide {len(stimulus_sequence)} visual stimuli into "
            f"{number_of_blocks} blocks"
        )

    stimuli_per_block = len(stimulus_sequence) // number_of_blocks
    conditions = []

    for block_index in range(number_of_blocks):
        start = block_index * stimuli_per_block
        block_stimuli = stimulus_sequence[start:start + stimuli_per_block]
        block_conditions = {classify_visual_stimulus(value) for value in block_stimuli}

        if len(block_conditions) != 1:
            raise ValueError(
                f"Visual block {block_index + 1} contains multiple conditions: "
                f"{sorted(block_conditions)}"
            )

        conditions.append(block_conditions.pop())

    return np.asarray(conditions, dtype=object)


# Helper function to transform .mat files to TSV files
def transform_mat_to_tsv(path_to_events_file, directory, execute, verbose):
    print(path_to_events_file)
    events_name = os.path.basename(path_to_events_file).lower()
    is_visual = 'visual' in events_name or 'vislocalizer' in events_name
    is_auditory = 'auditory' in events_name or 'audlocalizer' in events_name

    # read data from nested .mat structure
    if is_visual:
        events = loadmat(file_name=path_to_events_file)
        onset = events['data']['blockOnset'][0][0]
        duration = events['data']['blockDurations'][0][0]
        stimulus_sequence = events['data']['stimulus_sequence'][0][0]
    elif is_auditory:
        events = loadmat(file_name=path_to_events_file)
        onset = events['data']['Onsets'][0][0]
        duration = events['data']['Durations'][0][0]
        stimulus_name = events['data']['trialType'][0][0]
    else:
        raise ValueError("The given file is neither visual nor auditory - not a valid events .mat")
        
    # convert values to arrays
    onset = np.concatenate(onset).ravel()
    duration = np.concatenate(duration).ravel()
    if is_visual:
        stimulus_sequence = np.concatenate(stimulus_sequence).ravel()
        stimulus_name = visual_block_conditions(stimulus_sequence, len(onset))
    else:
        stimulus_name = np.concatenate(stimulus_name).ravel()

    if not (len(onset) == len(duration) == len(stimulus_name)):
        raise ValueError(
            "Events onset, duration, and trial_type arrays do not have matching lengths: "
            f"{len(onset)}, {len(duration)}, {len(stimulus_name)}"
        )

    # create panda with columns onset duration and stimulus name
    events_panda = pd.DataFrame({'onset': onset, 'duration': duration, 'trial_type': stimulus_name})

    # Save the file as a tsv with the same name
    print(f"Ready to save {path_to_events_file} as a tsv")
    if execute:
        temp_path = os.path.join(directory, os.path.basename(path_to_events_file).replace(".mat", ".tsv"))
        events_panda.to_csv(temp_path, sep='\t', index=False)
        print(f"saved to {temp_path}")
    if verbose: print(events_panda)

    return 0

# Helper function to locate the JSON file corresponding to a TSV events file
def find_json(directory, files, task, run):
    if run == "0":
        print("This is the practice run")
        x = "practice"
    elif task == "auditory":
        x = f"audlocalizer_run-{run}"
    elif task == "visual":
        x = f"vislocalizer"
    else:
        raise ValueError("Wrong task ID given")

    for file in files:
        if file.endswith('.json'):
            with open(os.path.join(directory, file), 'r') as json_file:
                json_data = json.load(json_file)
                if x in json_data['SeriesDescription']:
                    return file
    raise ValueError(f"Wrong task ID – {task} and run – {run} combination")
    return 0

# This function performs the main renaming of the files when given arguments
# called files and directory, which specify the files to be renamed
def rename(directory, files, execute, verbose):
    names = []
    # loop over every file to be renamed
    for filename in files:
        file_path = os.path.join(directory, filename)

        # This conditional sets the path to reference the .json where the "code" is extracted form
        # The code specifies the type of scan and for nii files we reference the json for extraction

        if filename.endswith('.json'):
            json_path = file_path
        elif filename.endswith('.nii'):
            json_path = os.path.join(directory, f"{filename.split('.')[0]}.json")
        elif filename.endswith('.tsv'):
            run_id = filename.split("_")[3]
            run_id = run_id if run_id == "0" else run_id.zfill(2)
            task_id = filename.split("_")[1]
            json_path = os.path.join(directory, find_json(directory, files, task_id, run_id))
        elif filename.endswith('.mat'):
            run_id = filename.split("_")[3]
            run_id = run_id if run_id == "0" else run_id.zfill(2)
            task_id = filename.split("_")[1]
            json_path = os.path.join(directory, find_json(directory, files, task_id, run_id))

        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)
            code = json_data['SeriesDescription']
            # This conditional adds the acquisition time to audio test file names so if we repeat then we don't over write files
            if 'audiotest' in code:
                acq_time = f"_{json_data['AcquisitionTime']}"
            else:
                acq_time = ""

        # These conditionals allow for specific formatting (according to BIDS) for the various scans
        if 'T1' in code or 'T2' in code:
            # This is an anatomical run
            sub_id = filename.split('_202')[0]
            new_path = f"sub-{sub_id}_{code}.{filename.split('.')[-1]}"
        elif 'scout' in code:
            # This is a scout run
            sub_id = filename.split('_202')[0]
            new_path = f"sub-{sub_id}_{code}_scout.{filename.split('.')[-1]}"
        elif 'epi' in code:
            # This is a fieldmap
            sub_id = filename.split('_202')[0]
            new_path = f"sub-{sub_id}_{code}.{filename.split('.')[-1]}"
        else:
            # This is a functional run or the .csv file for the run or the .mat file for the run (including audio test)
            if filename.endswith('.nii') or filename.endswith('.json'): # the nii data or jsons
                sub_id = filename.split('_202')[0]
                new_path = f"sub-{sub_id}_task-{code}_bold{acq_time}.{filename.split('.')[-1]}"
            elif filename.endswith('.tsv') or filename.endswith('.mat'): # events files
                sub_id = json_path.split('/')[-1].split('_202')[0]
                new_path = f"sub-{sub_id}_task-{code}_events{acq_time}.{filename.split('.')[-1]}"

        # Conditional to rename/print.
        if execute:
            os.rename(file_path, os.path.join(directory, new_path))
            if verbose: print(f"{filename} renamed to {new_path}")
        else:
            if verbose: print(f"{filename} renamed to {new_path}")
        names.append(new_path)
    print("Names have been assigned")

    return names

# This function moves all the files to the BIDS directory
# Any files not in bids go to /bids/sourcedata/sub-[]/other/
def move(subject_id, niidir, datadir, execute, verbose):
    subject_id = "sub-" + subject_id
    subject_dir = datadir + "/" + subject_id
    if execute:
        if os.path.exists(datadir):
            os.chdir(datadir)
        else:
            os.makedirs(datadir)
            os.chdir(datadir)

    maps = {'.mat': 'other', 'scout': 'other', 'practice': 'other', 'order': 'other', 'seq': 'other', 'audiotest': 'other', 'T1': 'anat',
            'T2': 'anat', 'run': 'func', 'bold': 'func', 'epi': 'fmap', 'vislocalizer_events.tsv': 'func'}

    for file in os.listdir(niidir):
        for val in maps.keys():
            if val in file:
                if val == 'run' and '.mat' in file: # Moves .mat events to other instead of func
                    dest = 'other'
                else:
                    dest = maps[val]
                if execute:
                    if not os.path.exists(f"{subject_dir}/{dest}"): os.makedirs(f"{subject_dir}/{dest}")
                    shutil.move(os.path.join(niidir, file), f"{subject_dir}/{dest}")
                if verbose: print(f"{file} sent to {dest}")
                break
    if execute:
        os.chdir(subject_dir)
        shutil.move(f"{subject_dir}/other", f"{datadir}/sourcedata/{subject_id}/other")
    else:
        if verbose: print(f"Would have moved {subject_dir}/other to {datadir}/sourcedata/{subject_id}/other")
    return 1


def format_json(subject_id, bids_dir, verbose):
    """
    This function formats JSON files for field map files, functional files, and anatomical files
    Functional: Adds 'TaskName', deletes 'AcquisitionDuration'
    Anatomical: Deletes 'RepetitionTime'
    Field maps: Adds 'IntendedFor', deletes 'RepetitionTime'
    """

    path_func = bids_dir + "/sub-" + subject_id + "/func"
    path_anat = bids_dir + "/sub-" + subject_id + "/anat"
    fmap_1 = bids_dir + "/sub-" + subject_id + "/fmap/" + "sub-" + subject_id + "_dir-AP_epi.json"
    fmap_2 = bids_dir + "/sub-" + subject_id + "/fmap/" + "sub-" + subject_id + "_dir-PA_epi.json"
    fieldmaps_exist = os.path.exists(fmap_1) and os.path.exists(fmap_2)

    if not fieldmaps_exist and verbose:
        print("Fieldmap JSONs not found; skipping fieldmap formatting")

    for file in os.listdir(path_func):
        if file.endswith(".nii") and fieldmaps_exist:
            # Edit the first fieldmap JSON. Add IntendedFor and delete repetition time
            with open(fmap_1, 'r') as json_file_1:
                json_data_1 = json.load(json_file_1)

            if 'IntendedFor' not in json_data_1: json_data_1['IntendedFor'] = []

            intended = f"func/{file}"
            if intended not in json_data_1['IntendedFor']:
                json_data_1['IntendedFor'].append(intended)

            if 'RepetitionTime' in json_data_1: del json_data_1['RepetitionTime']

            with open(fmap_1, 'w') as json_file_1:
                json.dump(json_data_1, json_file_1)

            # Edit the second fieldmap JSON. Add IntendedFor and delete repetition time
            with open(fmap_2, 'r') as json_file_2:
                json_data_2 = json.load(json_file_2)

            if 'IntendedFor' not in json_data_2: json_data_2['IntendedFor'] = []

            if intended not in json_data_2['IntendedFor']:
                json_data_2['IntendedFor'].append(intended)

            if 'RepetitionTime' in json_data_2: del json_data_2['RepetitionTime']

            with open(fmap_2, 'w') as json_file_2:
                json.dump(json_data_2, json_file_2)

            if verbose: print(f"Fieldmap jsons updated for {file}")
        elif file.endswith(".json"):

            # Check if this json is a functional one
            if 'bold' in file:
                # Edit the function JSON files. Add in TaskName and remove AcquisitionDuration
                with open(os.path.join(path_func, file), 'r') as func_json:
                    func_data = json.load(func_json)

                # Calculate the right task name
                task_name = func_data['SeriesDescription'].split('_')[0]

                # Add in task field
                func_data['TaskName'] = task_name

                # Delete acq duration
                if 'AcquisitionDuration' in func_data: del func_data['AcquisitionDuration']

                with open(os.path.join(path_func, file), 'w') as func_json:
                    json.dump(func_data, func_json)

                if verbose: print(f"Functional jsons updated for {file}")

    for file in os.listdir(path_anat):
        # Edit the anatomical data JSON files. Remove repetition time.
        if file.endswith(".json"):
            with open(os.path.join(path_anat, file), 'r') as anat_json:
                anat_data = json.load(anat_json)

            if 'RepetitionTime' in anat_data: del anat_data['RepetitionTime']

            with open(os.path.join(path_anat, file), 'w') as anat_json:
                json.dump(anat_data, anat_json)

            if verbose: print(f"Anatomical JSON edited for {file}")

    print("Formatting of JSONs is done")

    return 0


def format_events(subject_id, bids_dir, descriptions, levels_dict, execute, verbose):
    """
    This function creates the required .json files for each .tsv events file which describe the column headers.

    Inputs:
    bids_dir: Path to the BIDS dataset
    descriptions: The description field as a dictionary for the .json file
    levels_dict: The levels field as a dictionary for the .json file

    Output:
    If execute = True, it will update the JSON event files
    Otherwise, it will print out path to JSON files and the data to be sent to them
    """

    path = bids_dir + "/sub-" + subject_id + "/func"

    for file in os.listdir(path):
        if file.endswith(".tsv"):
            # Loading events file
            temp = pd.read_csv(os.path.join(path, file), sep='\t')

            # Creating the required JSON files
            data_dict = {}
            for column in temp.columns:
                data_dict[column] = {"Description": f"{descriptions[column]}"}
                if column in levels_dict.keys(): data_dict[column]["Levels"] = levels_dict[column]

            json_filename = os.path.splitext(os.path.join(path, file))[0] + ".json"

            if execute:
                # Write a new .json file
                with open(json_filename, 'w') as f:
                    json.dump(data_dict, f)

                if verbose: print(f"Events file updated and JSON created for {file}")
            else:
                if verbose: print(json_filename)
                if verbose: print(data_dict)

    return 0


if __name__ == "__main__":

    ############ Parse CL args ###############
    parser = argparse.ArgumentParser()
    parser.add_argument("-sub", "--subject_id", type=str, help="multimodal_sub_01", default="002")
    parser.add_argument('-u', '--user', type=str, default="aa2842")
    parser.add_argument("-e", '--execute', type=int, default=1)
    parser.add_argument("-v", "--verbose", type=int, default=1)

    args = parser.parse_args()
    subject_id = args.subject_id
    execute = args.execute
    user = args.user
    verbose = args.verbose

    directory = f'/gpfs/milgram/scratch60/turk-browne/{user}/sandbox/auditory-object-scenes-data/{subject_id}_nii/'
    bids_dir = f'/gpfs/milgram/scratch60/turk-browne/{user}/sandbox/auditory-object-scenes-data/objects_scenes_bids/'

    mat_files = [f for f in os.listdir(directory) if f.endswith('.mat')]
    for mat_file in mat_files:
        transform_mat_to_tsv(os.path.join(directory, mat_file), directory, execute, verbose)

    files = find_files(directory)
    new_names = rename(directory, files, execute, verbose)
    move(subject_id, directory, bids_dir, execute, verbose)

    if execute: format_json(subject_id, bids_dir, verbose)

    levels_dict = {'stimulus_name': {'category 1': 'some explanation', 'category 2': 'some explanation'},
                   'event': {'category 1': 'some explanation', 'category 2': 'some explanation'},
                   'trial_type': {'category 1': 'some explanation', 'category 2': 'some explanation'}
                   }

    descriptions = {'onset': 'something', 'duration': 'something', 'tr': 'something', 'stimulus_name': 'something',
                    'event': 'something', 'trial_type': 'something'}

    if execute: format_events(subject_id, bids_dir, descriptions, levels_dict, execute, verbose)

    print("Completed renaming and moving to BIDS directory")
