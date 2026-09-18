import numpy as np
import nilearn as ni
import nilearn.image
import nibabel as nib
import pandas as pd
import os
import glob
import argparse
import ast
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.plotting import plot_design_matrix
from nilearn.glm import threshold_stats_img


DEFAULT_BIDS_DIR = "/gpfs/milgram/scratch60/turk-browne/aa2842/sandbox/auditory-object-scenes-data/objects_scenes_bids"
DEFAULT_PREPROC_DIR = "/gpfs/milgram/scratch60/turk-browne/aa2842/sandbox/auditory-object-scenes-data/preprocessed"
DEFAULT_SAVE_DIR = "/gpfs/milgram/project/turk-browne/aa2842/auditory-objects-scenes-project/GLM/results"
DEFAULT_PP03_EVENTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sub-pp03_task-vislocalizer_events_three_conditions.tsv"
)
VISUAL_CONDITIONS = (
    "visual_object",
    "visual_scene",
    "visual_scrambled_scene"
)


# Helper to standardize subject strings for this project
def format_subject(sub):
    if sub is None:
        raise ValueError("Please pass a subject with -s/--subject, e.g. pp03")

    sub = str(sub).strip()
    sub = sub.replace("sub-", "")

    if not sub.startswith("pp"):
        sub = f"pp{sub}"

    return sub


# Helper to pull BIDS entities out of filenames
def get_bids_entities(path):
    filename = os.path.basename(path)
    entities = {}

    for part in filename.split("_"):
        if "-" in part:
            key, val = part.split("-", 1)
            entities[key] = val.split(".")[0]

    return entities


# Function to find the single visual localizer run and its matching files
def find_visual_run(
    bids_dir,
    preproc_dir,
    sub,
    task="vislocalizer",
    space="T1w",
    events_file=None
):
    """
    :param bids_dir: String path to raw BIDS data
    :param preproc_dir: String path to fMRIPrep derivatives
    :param sub: String subject ID, e.g. "pp03"
    :param task: String task label in the BIDS/fMRIPrep filenames
    :param space: String fMRIPrep output space to model
    :param events_file: Optional events TSV override
    :return: Dictionary with BOLD/events/mask/confounds paths and BIDS entities
    """

    sub = format_subject(sub)
    sub_bids = f"sub-{sub}"

    func_dir = os.path.join(preproc_dir, sub_bids, "func")
    bids_func_dir = os.path.join(bids_dir, sub_bids, "func")

    if not os.path.exists(func_dir):
        raise FileNotFoundError(f"Could not find fMRIPrep func directory: {func_dir}")
    if not os.path.exists(bids_func_dir):
        raise FileNotFoundError(f"Could not find BIDS func directory: {bids_func_dir}")

    bold_pattern = os.path.join(func_dir, f"{sub_bids}_task-{task}*_space-{space}_desc-preproc_bold.nii.gz")
    bold_paths = sorted(glob.glob(bold_pattern))

    if len(bold_paths) == 0:
        raise FileNotFoundError(f"Could not find visual localizer BOLD with pattern: {bold_pattern}")
    if len(bold_paths) > 1:
        raise ValueError(f"Found more than one visual localizer BOLD file: {bold_paths}")

    bold_path = bold_paths[0]
    bold_entities = get_bids_entities(bold_path)
    run_label = bold_entities.get("run", None)

    if events_file is not None:
        events_path = os.path.abspath(os.path.expanduser(events_file))
        if not os.path.isfile(events_path):
            raise FileNotFoundError(f"Could not find events override file: {events_path}")

        events_entities = get_bids_entities(events_path)
        if events_entities.get("sub") not in (None, sub):
            raise ValueError(
                f"Events override is for sub-{events_entities['sub']}, not {sub_bids}: "
                f"{events_path}"
            )
        if events_entities.get("task") not in (None, task):
            raise ValueError(
                f"Events override is for task-{events_entities['task']}, not task-{task}: "
                f"{events_path}"
            )
    else:
        events_pattern = os.path.join(bids_func_dir, f"{sub_bids}_task-{task}*_events.tsv")
        events_paths = sorted(glob.glob(events_pattern))

        if run_label is not None:
            events_paths = [p for p in events_paths if f"_run-{run_label}_" in os.path.basename(p)]
        else:
            events_no_run = [p for p in events_paths if "_run-" not in os.path.basename(p)]
            events_paths = events_no_run

        if len(events_paths) == 0:
            raise FileNotFoundError(f"Could not find matching events file with pattern: {events_pattern}")
        if len(events_paths) > 1:
            raise ValueError(f"Found more than one matching visual localizer events file: {events_paths}")

        events_path = events_paths[0]
    bold_prefix = os.path.basename(bold_path).split(f"_space-{space}_desc-preproc_bold.nii.gz")[0]
    mask_path = os.path.join(func_dir, f"{bold_prefix}_space-{space}_desc-brain_mask.nii.gz")
    confounds_path = os.path.join(func_dir, f"{bold_prefix}_desc-confounds_timeseries.tsv")
    bold_json_path = bold_path.replace(".nii.gz", ".json")

    for path in [mask_path, confounds_path, bold_json_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find matching file: {path}")

    print(f"fMRIPrep directory: {preproc_dir}", flush=True)
    print(f"BIDS directory: {bids_dir}", flush=True)
    print(f"Subject: {sub_bids}", flush=True)
    print(f"Task: {bold_entities.get('task')}", flush=True)
    print(f"Run label: {run_label}", flush=True)
    print(f"BOLD file: {bold_path}", flush=True)
    print(f"Events file: {events_path}", flush=True)
    print(f"Brain mask file: {mask_path}", flush=True)
    print(f"Confounds file: {confounds_path}", flush=True)

    return {
        "subject": sub,
        "subject_bids": sub_bids,
        "task": bold_entities.get("task"),
        "run": run_label,
        "space": space,
        "bold_path": bold_path,
        "bold_json_path": bold_json_path,
        "events_path": events_path,
        "mask_path": mask_path,
        "confounds_path": confounds_path
    }


# Helper to map the actual visual localizer labels into GLM condition names
def clean_visual_label(label):
    original = label

    if isinstance(label, str):
        label = label.strip()
        try:
            parsed_label = ast.literal_eval(label)
            if isinstance(parsed_label, list) and len(parsed_label) == 1:
                label = parsed_label[0]
            elif isinstance(parsed_label, str):
                label = parsed_label
        except (ValueError, SyntaxError):
            pass

    label = str(label).strip().strip("'").strip('"').lower()

    label_map = {
        "object": "visual_object",
        "objects": "visual_object",
        "visual_object": "visual_object",
        "visual objects": "visual_object",
        "scene": "visual_scene",
        "scenes": "visual_scene",
        "visual_scene": "visual_scene",
        "visual scenes": "visual_scene",
        "scrambled_scene": "visual_scrambled_scene",
        "scrambled scene": "visual_scrambled_scene",
        "scrambled scenes": "visual_scrambled_scene",
        "phase_scrambled_scene": "visual_scrambled_scene",
        "phase scrambled scene": "visual_scrambled_scene",
        "visual_scrambled_scene": "visual_scrambled_scene",
        "visual scrambled scene": "visual_scrambled_scene"
    }

    return label_map.get(label, original)


# Function to load and clean the events file
def load_and_clean_events(events_path):
    """
    :param events_path: String path to visual localizer BIDS events.tsv
    :return: Cleaned events dataframe with one of the three visual condition labels
    """

    events = pd.read_csv(events_path, sep="\t")
    required_columns = ["onset", "duration", "trial_type"]

    for col in required_columns:
        if col not in events.columns:
            raise ValueError(f"Events file is missing required column: {col}")

    original_labels = sorted(events["trial_type"].dropna().astype(str).unique().tolist())
    events = events.copy()
    events["trial_type"] = events["trial_type"].apply(clean_visual_label)
    events = events[events["trial_type"].isin(VISUAL_CONDITIONS)].copy()

    if events.empty:
        raise ValueError(
            f"No recognized visual events found after mapping labels: {original_labels}"
        )

    missing_conditions = sorted(set(VISUAL_CONDITIONS) - set(events["trial_type"]))
    if missing_conditions:
        raise ValueError(f"Events file is missing visual conditions: {missing_conditions}")

    events["onset"] = pd.to_numeric(events["onset"])
    events["duration"] = pd.to_numeric(events["duration"])

    if events["duration"].isna().any() or events["onset"].isna().any():
        raise ValueError("Events file has NaN onset or duration values")
    if (events["duration"] <= 0).any():
        raise ValueError("Events file has non-positive durations")

    print(f"Events columns: {events.columns.tolist()}", flush=True)
    print(f"Original event labels: {original_labels}", flush=True)
    print(f"Cleaned event counts: {events['trial_type'].value_counts().to_dict()}", flush=True)
    print(f"Using actual event durations: {sorted(events['duration'].unique().tolist())}", flush=True)

    return events


# Function to load the BOLD data and build frame times from the header
def load_bold(bold_path):
    """
    :param bold_path: String path to preprocessed BOLD
    :return: BOLD image, frame times, TR, and number of scans
    """

    fmri_img = ni.image.load_img(bold_path)
    header_img = nib.load(bold_path)

    if len(fmri_img.shape) != 4:
        raise ValueError(f"BOLD image is not 4D: {bold_path}")

    t_r = float(header_img.header.get_zooms()[3])
    n_scans = int(fmri_img.shape[3])
    frame_times = np.arange(n_scans) * t_r

    print(f"BOLD shape: {fmri_img.shape}", flush=True)
    print(f"TR from image header: {t_r}", flush=True)
    print(f"Number of volumes from image header: {n_scans}", flush=True)

    return fmri_img, frame_times, t_r, n_scans


# Function to load fMRIPrep confounds for the visual localizer
def load_visual_confounds(bold_path, n_scans):
    """
    :param bold_path: String path to preprocessed BOLD
    :param n_scans: Int number of BOLD volumes
    :return: Confounds dataframe and optional sample mask
    """

    confounds, sample_mask = load_confounds(
        bold_path,
        strategy=["high_pass", "motion", "scrub"],
        motion="basic",
        scrub=0,
        fd_threshold=0.5,
        std_dvars_threshold=1.5
    )

    if confounds.shape[0] != n_scans:
        raise ValueError(f"Confounds rows ({confounds.shape[0]}) do not match BOLD volumes ({n_scans})")

    print(f"Confounds shape: {confounds.shape}", flush=True)
    print(f"Confounds used: {confounds.columns.tolist()}", flush=True)
    print(f"Sample mask from load_confounds: {sample_mask}", flush=True)

    return confounds, sample_mask


# Function to make the design matrix
def make_design_matrix(frame_times, events, confounds):
    """
    :param frame_times: Numpy array of scan times
    :param events: Cleaned events dataframe
    :param confounds: fMRIPrep confounds dataframe
    :return: First-level design matrix
    """

    designmat = make_first_level_design_matrix(
        frame_times,
        events,
        drift_model=None,  # loading it from fmriprep instead
        add_regs=confounds,
        hrf_model="glover + derivative + dispersion"
    )

    print(f"Design matrix shape: {designmat.shape}", flush=True)
    print(f"Design matrix columns: {designmat.columns.tolist()}", flush=True)

    return designmat


# Function to save the design matrix for checking
def save_design_matrix(design_matrix, save_dir, sub_bids, task, space):
    design_dir = os.path.join(save_dir, "first_level_glm", sub_bids, "design_matrix")
    os.makedirs(design_dir, exist_ok=True)

    csv_path = os.path.join(design_dir, f"{sub_bids}_task-{task}_space-{space}_design_matrix.csv")
    png_path = os.path.join(design_dir, f"{sub_bids}_task-{task}_space-{space}_design_matrix.png")

    design_matrix.to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(14, 6))
    plot_design_matrix(design_matrix, ax=ax)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print(f"Saved design matrix CSV: {csv_path}", flush=True)
    print(f"Saved design matrix plot: {png_path}", flush=True)

    return csv_path, png_path


# Function to estimate glm for one subject
def fit_subject_glm(bold_img, design_matrix, mask_path, sample_mask=None):
    """
    :param bold_img: Nilearn image for the visual localizer
    :param design_matrix: First-level design matrix
    :param mask_path: String path to brain mask
    :param sample_mask: Optional sample mask from fMRIPrep confounds
    :return: Fitted Nilearn FirstLevelModel
    """

    fmri_glm = FirstLevelModel(
        slice_time_ref=0.5,  # IMPORTANT, when using slicetiming in fmriprep
        noise_model="ar1",
        smoothing_fwhm=None,
        mask_img=mask_path
    )

    if sample_mask is None:
        sample_masks = None
    else:
        sample_masks = [sample_mask]

    fmri_glm = fmri_glm.fit(
        [bold_img],
        design_matrices=[design_matrix],
        sample_masks=sample_masks
    )

    return fmri_glm


# Function to make the visual localizer contrasts
def make_contrasts(design_matrix):
    """
    :param design_matrix: First-level design matrix
    :return: Dictionary of condition and contrast vectors
    """

    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_contrasts = {column: contrast_matrix[i] for i, column in enumerate(design_matrix.columns)}

    for condition in VISUAL_CONDITIONS:
        if condition not in basic_contrasts:
            raise ValueError(f"Condition is missing from the design matrix: {condition}")

    contrasts = {
        "visual_object": basic_contrasts["visual_object"],
        "visual_scene": basic_contrasts["visual_scene"],
        "visual_scrambled_scene": basic_contrasts["visual_scrambled_scene"],
        "visual_object-visual_scene": (
            basic_contrasts["visual_object"] - basic_contrasts["visual_scene"]
        ),
        "visual_scene-visual_object": (
            basic_contrasts["visual_scene"] - basic_contrasts["visual_object"]
        ),
        "visual_scrambled_scene-visual_scene": (
            basic_contrasts["visual_scrambled_scene"] - basic_contrasts["visual_scene"]
        ),
        "visual_scene-visual_scrambled_scene": (
            basic_contrasts["visual_scene"] - basic_contrasts["visual_scrambled_scene"]
        ),
        "visual_scrambled_scene-visual_object": (
            basic_contrasts["visual_scrambled_scene"] - basic_contrasts["visual_object"]
        ),
        "visual_object-visual_scrambled_scene": (
            basic_contrasts["visual_object"] - basic_contrasts["visual_scrambled_scene"]
        )
    }

    print(f"Contrasts to compute: {list(contrasts.keys())}", flush=True)

    return contrasts


# Function to save condition maps and contrast maps
def save_contrast_maps(fmri_glm, contrasts, save_dir, sub_bids, task, space, fdr_alpha=0.05):
    """
    Save unthresholded effect-size maps, unthresholded z-maps,
    and FDR-thresholded z-maps.
    """

    saved_paths = []

    for contrast_id, contrast_val in contrasts.items():
        if contrast_id in VISUAL_CONDITIONS:
            map_type = "condition"
        else:
            map_type = "contrast"

        map_dir = os.path.join(save_dir, "first_level_glm", sub_bids, map_type, contrast_id)
        os.makedirs(map_dir, exist_ok=True)

        # 1. Save effect-size map
        effect_map = fmri_glm.compute_contrast(contrast_val, output_type="effect_size")
        effect_path = os.path.join(
            map_dir,
            f"{sub_bids}_task-{task}_{map_type}-{contrast_id}_space-{space}_stat-effect_size.nii.gz"
        )
        effect_map.to_filename(effect_path)
        saved_paths.append(effect_path)
        print(f"Saved effect_size map for {contrast_id}: {effect_path}", flush=True)

        # 2. Save unthresholded z-map
        z_map = fmri_glm.compute_contrast(contrast_val, output_type="z_score")
        z_path = os.path.join(
            map_dir,
            f"{sub_bids}_task-{task}_{map_type}-{contrast_id}_space-{space}_stat-z_score.nii.gz"
        )
        z_map.to_filename(z_path)
        saved_paths.append(z_path)
        print(f"Saved z_score map for {contrast_id}: {z_path}", flush=True)

        # 3. Save FDR-thresholded z-map
        fdr_map, fdr_threshold = threshold_stats_img(
            z_map,
            alpha=fdr_alpha,
            height_control="fdr"
        )

        fdr_path = os.path.join(
            map_dir,
            f"{sub_bids}_task-{task}_{map_type}-{contrast_id}_space-{space}_stat-z_score_desc-fdr_alpha-0p05.nii.gz"
        )
        fdr_map.to_filename(fdr_path)
        saved_paths.append(fdr_path)

        threshold_path = os.path.join(
            map_dir,
            f"{sub_bids}_task-{task}_{map_type}-{contrast_id}_space-{space}_stat-z_score_desc-fdr_alpha-0p05_threshold.txt"
        )
        with open(threshold_path, "w") as f:
            f.write(str(fdr_threshold))

        saved_paths.append(threshold_path)

        print(
            f"Saved FDR-thresholded z-map for {contrast_id}: {fdr_path}",
            flush=True
        )
        print(
            f"FDR alpha={fdr_alpha}, z-threshold={fdr_threshold}",
            flush=True
        )

    return saved_paths

# Function to run the visual localizer GLM analysis for one subject
def run_glm_analysis(
    bids_dir,
    preproc_dir,
    sub,
    save_dir,
    task="vislocalizer",
    space="T1w",
    events_file=None,
    dry_run=False,
    threshold=0.05
):

    normalized_sub = format_subject(sub)
    if events_file is None and normalized_sub == "pp03":
        events_file = DEFAULT_PP03_EVENTS_FILE
        print(f"Using pp03 three-condition events override: {events_file}", flush=True)

    # Find the visual localizer run and all matching files
    visual_run = find_visual_run(
        bids_dir,
        preproc_dir,
        normalized_sub,
        task=task,
        space=space,
        events_file=events_file
    )

    # Load and clean visual object / scene / scrambled-scene events
    events = load_and_clean_events(visual_run["events_path"])

    # Load BOLD and frame timing
    bold_img, frame_times, t_r, n_scans = load_bold(visual_run["bold_path"])

    # Load fMRIPrep confounds
    confounds, sample_mask = load_visual_confounds(visual_run["bold_path"], n_scans)

    # Make first-level design matrix
    design_matrix = make_design_matrix(frame_times, events, confounds)

    # Make visual localizer contrasts
    contrasts = make_contrasts(design_matrix)

    if dry_run:
        print("Dry run requested; stopping before FirstLevelModel.fit().", flush=True)
        print(f"Validated TR={t_r}, n_scans={n_scans}, events={events.shape[0]}, confounds={confounds.shape[1]}", flush=True)
        return 1

    # Save the design matrix for checking
    save_design_matrix(design_matrix, save_dir, visual_run["subject_bids"], visual_run["task"], visual_run["space"])

    # Estimate the beta coefficients
    fmri_glm = fit_subject_glm(bold_img, design_matrix, visual_run["mask_path"], sample_mask=sample_mask)

    # Save effect size and z-score maps
    save_contrast_maps(fmri_glm, contrasts, save_dir, visual_run["subject_bids"], visual_run["task"], visual_run["space"], threshold)

    print(f"Successfully completed visual localizer GLM for subject: {visual_run['subject_bids']}", flush=True)

    return 1


if __name__ == "__main__":
    ############ Parse CL args ###############
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", type=str)
    parser.add_argument("--bids_dir", type=str, default=DEFAULT_BIDS_DIR)
    parser.add_argument("--preproc_dir", type=str, default=DEFAULT_PREPROC_DIR)
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--task", type=str, default="vislocalizer")
    parser.add_argument("--space", type=str, default="T1w")
    parser.add_argument(
        "--events-file",
        type=str,
        default=None,
        help="Optional events TSV override (pp03 defaults to the GLM-local three-condition file)"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.05)

    args = parser.parse_args()
    given_sub = args.subject

    run_glm_analysis(
        args.bids_dir,
        args.preproc_dir,
        given_sub,
        args.save_dir,
        task=args.task,
        space=args.space,
        events_file=args.events_file,
        dry_run=args.dry_run,
        threshold=args.threshold
    )
