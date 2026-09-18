"""Run GLMsingle for the visual and auditory localizers."""

import argparse
import ast
import glob
import json
import os
import time
from pprint import pprint

import nibabel as nib
import numpy as np
import pandas as pd
from glmsingle.glmsingle import GLM_single
from nilearn.maskers import NiftiMasker
from nilearn.masking import intersect_masks


DEFAULT_PREPROC_DIR = (
    "/gpfs/milgram/scratch60/turk-browne/aa2842/sandbox/"
    "auditory-object-scenes-data/preprocessed"
)
DEFAULT_SAVE_DIR = (
    "/gpfs/milgram/project/turk-browne/aa2842/"
    "auditory-objects-scenes-project/GLMsingle/results"
)

VISUAL_CONDITIONS = (
    "visual_object",
    "visual_scene",
    "visual_scrambled_scene",
)
AUDITORY_XVAL_SCHEME = [[0, 1], [2, 3], [4, 5], [6, 7]]


def format_subject(subject):
    """Return the subject label without the ``sub-`` prefix."""

    subject = str(subject).strip().replace("sub-", "")
    if not subject.startswith("pp"):
        subject = "pp{}".format(subject)
    return subject


def find_runs(preproc_dir, subject, modality, space="T1w"):
    """Find the localizer BOLD runs and their matching files."""

    subject_bids = "sub-{}".format(subject)
    task = "vislocalizer" if modality == "visual" else "audlocalizer"
    func_dir = os.path.join(preproc_dir, subject_bids, "func")
    run_part = "" if modality == "visual" else "_run-*"
    pattern = os.path.join(
        func_dir,
        "{}_task-{}{}_space-{}_desc-preproc_bold.nii.gz".format(
            subject_bids, task, run_part, space
        ),
    )
    bold_paths = sorted(glob.glob(pattern))
    expected_runs = 1 if modality == "visual" else 8
    if len(bold_paths) != expected_runs:
        raise ValueError(
            "Expected {} {} run(s), found {}".format(
                expected_runs, modality, len(bold_paths)
            )
        )

    bold_suffix = "_space-{}_desc-preproc_bold.nii.gz".format(space)
    run_infos = []
    for bold_path in bold_paths:
        prefix = os.path.basename(bold_path)[: -len(bold_suffix)]
        run = next(
            (part.replace("run-", "") for part in prefix.split("_") if part.startswith("run-")),
            "01",
        )
        run_info = {
            "run": run,
            "bold_path": bold_path,
            "bold_json_path": bold_path.replace(".nii.gz", ".json"),
            "events_path": os.path.join(func_dir, prefix + "_events.tsv"),
            "mask_path": os.path.join(
                func_dir,
                prefix + "_space-{}_desc-brain_mask.nii.gz".format(space),
            ),
        }
        for path in (
            run_info["bold_json_path"],
            run_info["events_path"],
            run_info["mask_path"],
        ):
            if not os.path.isfile(path):
                raise FileNotFoundError("Could not find matching file: {}".format(path))
        run_infos.append(run_info)

    print("Subject: {}".format(subject_bids), flush=True)
    print("Task: {}".format(task), flush=True)
    print("Runs: {}".format([run["run"] for run in run_infos]), flush=True)
    return task, run_infos


def clean_event_label(label):
    """Unwrap MATLAB-style one-item strings used in the auditory events."""

    if isinstance(label, str):
        label = label.strip()
        try:
            parsed = ast.literal_eval(label)
            if isinstance(parsed, (list, tuple)) and len(parsed) == 1:
                label = parsed[0]
        except (ValueError, SyntaxError):
            pass
    return str(label).strip().strip("'").strip('"').replace("\\", "/")


def auditory_category(label):
    """Return the condition-level category for one sound path."""

    label_lower = label.lower()
    filename = os.path.basename(label_lower)
    if "/objects/" in label_lower and filename.startswith("o_"):
        return "auditory_object"
    if "/scenes/" in label_lower and filename.startswith("s_"):
        return "auditory_scene"
    if "/scenes/" in label_lower and filename.startswith("c_"):
        return "auditory_scrambled_scene"
    raise ValueError("Could not classify auditory event: {}".format(label))


def load_events(events_path, modality):
    """Load block events for visual or sound events for auditory."""

    events = pd.read_csv(events_path, sep="\t")
    events = events.loc[:, ["onset", "duration", "trial_type"]].copy()
    events["onset"] = pd.to_numeric(events["onset"])
    events["duration"] = pd.to_numeric(events["duration"])

    if modality == "visual":
        events["condition"] = events["trial_type"].apply(clean_event_label)
        events = events[events["condition"].isin(VISUAL_CONDITIONS)].copy()
        events["category"] = events["condition"]
    else:
        events["stimulus"] = events["trial_type"].apply(clean_event_label)
        events = events[
            ~events["stimulus"].str.lower().str.startswith("iti_")
        ].copy()
        events["condition"] = events["stimulus"].apply(os.path.basename)
        events["category"] = events["stimulus"].apply(auditory_category)

    return events.sort_values("onset").reset_index(drop=True)


def load_timing(run_info):
    """Read the BOLD header and make one frame time per acquired volume."""

    bold_img = nib.load(run_info["bold_path"])
    n_scans = bold_img.shape[3]
    t_r = float(bold_img.header.get_zooms()[3])
    with open(run_info["bold_json_path"], "r") as stream:
        metadata = json.load(stream)
    start_time = float(metadata.get("StartTime", 0))
    frame_times = start_time + np.arange(n_scans) * t_r
    return bold_img.shape, n_scans, t_r, frame_times


def create_design_matrices(run_infos, modality):
    """Create native-TR binary onset matrices for GLMsingle."""

    events_by_run = [
        load_events(run_info["events_path"], modality) for run_info in run_infos
    ]
    if modality == "visual":
        conditions = list(VISUAL_CONDITIONS)
    else:
        conditions = sorted(
            set(pd.concat(events_by_run, ignore_index=True)["condition"])
        )
    condition_index = {condition: index for index, condition in enumerate(conditions)}

    design = []
    manifest = []
    t_rs = []
    beta_index = 0
    for run_info, events in zip(run_infos, events_by_run):
        shape, n_scans, t_r, frame_times = load_timing(run_info)
        design_matrix = np.zeros((n_scans, len(conditions)), dtype=bool)

        for trial, event in events.iterrows():
            design_row = int(np.argmin(np.abs(frame_times - event["onset"])))
            if design_matrix[design_row].any():
                raise ValueError(
                    "Two events map to volume {} in run-{}".format(
                        design_row, run_info["run"]
                    )
                )
            design_matrix[design_row, condition_index[event["condition"]]] = True
            manifest.append(
                {
                    "beta_index": beta_index,
                    "run": run_info["run"],
                    "trial": trial + 1,
                    "condition": event["condition"],
                    "category": event["category"],
                    "onset": event["onset"],
                    "duration": event["duration"],
                    "design_row": design_row,
                }
            )
            beta_index += 1

        design.append(design_matrix)
        t_rs.append(t_r)
        print(
            "Run-{}: BOLD {}, design {}, {} events".format(
                run_info["run"], shape, design_matrix.shape, int(design_matrix.sum())
            ),
            flush=True,
        )

    if not np.allclose(t_rs, t_rs[0]):
        raise ValueError("BOLD runs do not have the same TR")

    if modality == "auditory":
        all_conditions = set(range(len(conditions)))
        for fold in AUDITORY_XVAL_SCHEME:
            present = set(
                np.flatnonzero(np.any(np.vstack([design[index] for index in fold]), axis=0))
            )
            if present != all_conditions:
                raise ValueError("Auditory cross-validation fold {} is incomplete".format(fold))

    manifest = pd.DataFrame(manifest)
    print("Conditions: {}".format(len(conditions)), flush=True)
    print("Single-trial betas: {}".format(len(manifest)), flush=True)
    return design, manifest, t_rs[0]


def make_mask(run_infos):
    """Use the visual run mask or the intersection of auditory run masks."""

    mask_paths = [run_info["mask_path"] for run_info in run_infos]
    if len(mask_paths) == 1:
        mask_img = nib.load(mask_paths[0])
    else:
        mask_img = intersect_masks(mask_paths, threshold=1.0, connected=False)
    voxel_count = int(np.count_nonzero(mask_img.get_fdata()))
    print("Mask voxels: {}".format(voxel_count), flush=True)
    return mask_img


def load_bold_data(run_infos, mask_img):
    """Load the BOLD runs as masked voxel-by-time arrays."""

    masker = NiftiMasker(mask_img=mask_img)
    masker.fit()
    data = []
    for run_info in run_infos:
        run_data = masker.transform(run_info["bold_path"]).T.astype(np.float32)
        data.append(run_data)
        print(
            "Loaded run-{} data: {}".format(run_info["run"], run_data.shape),
            flush=True,
        )
    return data, masker


def make_glmsingle(modality):
    """Set the GLMsingle options used for each localizer."""

    if modality == "visual":
        options = {
            "wantlibrary": 1,
            "wantglmdenoise": 0,
            "wantfracridge": 0,
            "wantfileoutputs": [1, 1, 0, 0],
            "wantmemoryoutputs": [0, 1, 0, 0],
        }
        return GLM_single(options), 7.5, "typeb", "TypeB"

    options = {
        "wantlibrary": 1,
        "wantglmdenoise": 1,
        "wantfracridge": 1,
        # GLMsingle 1.2 expects NumPy arrays for held-out run groups.
        "xvalscheme": np.asarray(AUDITORY_XVAL_SCHEME, dtype=int),
        "wantfileoutputs": [1, 1, 1, 1],
        "wantmemoryoutputs": [0, 0, 0, 1],
    }
    return GLM_single(options), 4.0, "typed", "TypeD"


def run_glm_single(args):
    """Prepare the inputs and optionally fit GLMsingle."""

    subject = format_subject(args.subject)
    task, run_infos = find_runs(args.preproc_dir, subject, args.modality, args.space)
    design, manifest, t_r = create_design_matrices(run_infos, args.modality)
    mask_img = make_mask(run_infos)
    glmsingle_obj, stimdur, result_key, result_label = make_glmsingle(args.modality)

    print("TR: {} seconds".format(t_r), flush=True)
    print("Stimulus duration: {} seconds".format(stimdur), flush=True)
    print("GLMsingle parameters:", flush=True)
    pprint(glmsingle_obj.params)

    if args.dry_run:
        print(
            "Dry run complete: inputs and designs are valid; GLMsingle.fit was not called.",
            flush=True,
        )
        return

    subject_bids = "sub-{}".format(subject)
    save_dir = os.path.abspath(
        os.path.join(args.save_dir, subject_bids, "task-{}".format(task))
    )
    output_dir = os.path.join(save_dir, "glmsingle")
    os.makedirs(save_dir, exist_ok=True)

    manifest_path = os.path.join(
        save_dir, "{}_task-{}_desc-betaManifest.tsv".format(subject_bids, task)
    )
    mask_path = os.path.join(
        save_dir,
        "{}_task-{}_space-{}_desc-GLMsingle_mask.nii.gz".format(
            subject_bids, task, args.space
        ),
    )
    manifest.to_csv(manifest_path, sep="\t", index=False)
    mask_img.to_filename(mask_path)

    data, masker = load_bold_data(run_infos, mask_img)
    print("Running GLMsingle...", flush=True)
    start_time = time.time()
    os.chdir(save_dir)
    results = glmsingle_obj.fit(
        design,
        data,
        stimdur,
        t_r,
        outputdir=output_dir,
    )
    print(
        "Elapsed time: {}".format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))),
        flush=True,
    )

    betas = np.squeeze(results[result_key]["betasmd"])
    beta_img = masker.inverse_transform(betas.T)
    beta_path = os.path.join(
        save_dir,
        "{}_task-{}_space-{}_desc-GLMsingle{}_betas.nii.gz".format(
            subject_bids, task, args.space, result_label
        ),
    )
    beta_img.to_filename(beta_path)
    print("Saved beta image: {}".format(beta_path), flush=True)
    print("Saved beta manifest: {}".format(manifest_path), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", default="pp06")
    parser.add_argument("-m", "--modality", choices=["visual", "auditory"], required=True)
    parser.add_argument("--preproc-dir", default=DEFAULT_PREPROC_DIR)
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--space", default="T1w")
    parser.add_argument("--dry-run", action="store_true")
    run_glm_single(parser.parse_args())
