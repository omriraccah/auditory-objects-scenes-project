"""Condition-level, multi-run GLM for the auditory localizer.

The auditory task is event-related.  A separate design matrix and AR(1) noise
model are estimated for every run; condition contrasts are then combined over
runs with Nilearn's fixed-effects machinery.
"""

import argparse
import ast
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.masking import intersect_masks
from nilearn.plotting import plot_design_matrix


DEFAULT_PREPROC_DIR = (
    "/gpfs/milgram/scratch60/turk-browne/aa2842/sandbox/"
    "auditory-object-scenes-data/preprocessed"
)
DEFAULT_SAVE_DIR = (
    "/gpfs/milgram/project/turk-browne/aa2842/"
    "auditory-objects-scenes-project/GLM/results"
)

AUDITORY_CONDITIONS = (
    "auditory_object",
    "auditory_scene",
    "auditory_scrambled_scene",
)


def format_subject(subject):
    """Return this project's subject label without the ``sub-`` prefix."""

    if subject is None:
        raise ValueError("Please pass a subject, e.g. pp06")

    subject = str(subject).strip().replace("sub-", "")
    if not subject.startswith("pp"):
        subject = "pp{}".format(subject)
    return subject


def get_bids_entities(path):
    """Extract the simple BIDS entities needed by this script."""

    entities = {}
    for part in os.path.basename(path).split("_"):
        if "-" in part:
            key, value = part.split("-", 1)
            entities[key] = value.split(".")[0]
    return entities


def normalize_run_label(run):
    run = str(run).strip().replace("run-", "")
    return run.zfill(2) if run.isdigit() else run


def run_sort_key(run_info):
    run = run_info["run"]
    return (0, int(run)) if run.isdigit() else (1, run)


def find_auditory_runs(
    preproc_dir,
    subject,
    task="audlocalizer",
    space="T1w",
    requested_runs=None,
):
    """Find every auditory BOLD run and its matching events/confounds/mask."""

    subject = format_subject(subject)
    subject_bids = "sub-{}".format(subject)
    func_dir = os.path.join(preproc_dir, subject_bids, "func")
    if not os.path.isdir(func_dir):
        raise FileNotFoundError("Could not find fMRIPrep func directory: {}".format(func_dir))

    pattern = os.path.join(
        func_dir,
        "{}_task-{}_run-*_space-{}_desc-preproc_bold.nii.gz".format(
            subject_bids, task, space
        ),
    )
    bold_paths = sorted(glob.glob(pattern))
    if not bold_paths:
        raise FileNotFoundError("Could not find auditory BOLD files with pattern: {}".format(pattern))

    requested = None
    if requested_runs:
        requested = {normalize_run_label(run) for run in requested_runs}

    run_infos = []
    seen_runs = set()
    bold_suffix = "_space-{}_desc-preproc_bold.nii.gz".format(space)

    for bold_path in bold_paths:
        entities = get_bids_entities(bold_path)
        run = entities.get("run")
        if run is None:
            raise ValueError("Auditory BOLD file has no run entity: {}".format(bold_path))
        run = normalize_run_label(run)
        if requested is not None and run not in requested:
            continue
        if run in seen_runs:
            raise ValueError("Found duplicate task-{} run-{} BOLD files".format(task, run))
        seen_runs.add(run)

        filename = os.path.basename(bold_path)
        if not filename.endswith(bold_suffix):
            raise ValueError("Unexpected BOLD filename: {}".format(filename))
        run_prefix = filename[: -len(bold_suffix)]

        run_info = {
            "subject": subject,
            "subject_bids": subject_bids,
            "task": task,
            "run": run,
            "space": space,
            "bold_path": bold_path,
            "bold_json_path": bold_path.replace(".nii.gz", ".json"),
            "events_path": os.path.join(func_dir, run_prefix + "_events.tsv"),
            "confounds_path": os.path.join(
                func_dir, run_prefix + "_desc-confounds_timeseries.tsv"
            ),
            "mask_path": os.path.join(
                func_dir,
                run_prefix + "_space-{}_desc-brain_mask.nii.gz".format(space),
            ),
        }

        for kind in ("bold_json_path", "events_path", "confounds_path", "mask_path"):
            if not os.path.isfile(run_info[kind]):
                raise FileNotFoundError(
                    "Missing matching {} for run-{}: {}".format(kind, run, run_info[kind])
                )
        run_infos.append(run_info)

    if requested is not None:
        missing = sorted(requested - seen_runs)
        if missing:
            raise ValueError("Requested auditory runs were not found: {}".format(missing))

    run_infos.sort(key=run_sort_key)
    if not run_infos:
        raise ValueError("No auditory runs remain after applying --runs")

    print("fMRIPrep directory: {}".format(preproc_dir), flush=True)
    print("Subject: {}".format(subject_bids), flush=True)
    print("Task: {}".format(task), flush=True)
    print("Space: {}".format(space), flush=True)
    print(
        "Runs ({}): {}".format(len(run_infos), [info["run"] for info in run_infos]),
        flush=True,
    )
    return run_infos


def matlab_string_to_scalar(value):
    """Unwrap strings such as ``['/Auditory/Objects/O_Anvil.wav']``."""

    if isinstance(value, str):
        value = value.strip()
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)) and len(parsed) == 1:
                value = parsed[0]
            elif isinstance(parsed, str):
                value = parsed
        except (ValueError, SyntaxError):
            pass
    return str(value).strip().strip("'").strip('"')


def classify_auditory_event(label):
    """Map a logged stimulus path to a condition, or return None for an ITI.

    The experiment uses the same scene filename convention already encoded for
    the visual localizer in ``preprocessing/XNat_Interact/rename.py``:
    ``S_`` is an intact scene and ``C_`` is its scrambled/control version.
    """

    label = matlab_string_to_scalar(label).replace("\\", "/")
    label_lower = label.lower()
    filename_lower = label.rsplit("/", 1)[-1].lower()

    explicit_labels = {
        "auditory_object": "auditory_object",
        "auditory object": "auditory_object",
        "object": "auditory_object",
        "objects": "auditory_object",
        "auditory_scene": "auditory_scene",
        "auditory scene": "auditory_scene",
        "scene": "auditory_scene",
        "scenes": "auditory_scene",
        "auditory_scrambled_scene": "auditory_scrambled_scene",
        "auditory scrambled scene": "auditory_scrambled_scene",
        "scrambled_scene": "auditory_scrambled_scene",
        "scrambled scene": "auditory_scrambled_scene",
    }
    if label_lower in explicit_labels:
        return explicit_labels[label_lower]

    if filename_lower.startswith("iti_"):
        return None
    if "/objects/" in label_lower and filename_lower.startswith("o_"):
        return "auditory_object"
    if "/scenes/" in label_lower and filename_lower.startswith("s_"):
        return "auditory_scene"
    if "/scenes/" in label_lower and filename_lower.startswith("c_"):
        return "auditory_scrambled_scene"

    raise ValueError("Could not classify auditory event label: {!r}".format(label))


def load_and_clean_events(events_path, run):
    """Load one events file and retain its condition-level sound events."""

    events = pd.read_csv(events_path, sep="\t")
    required = ("onset", "duration", "trial_type")
    missing_columns = [column for column in required if column not in events.columns]
    if missing_columns:
        raise ValueError(
            "Run-{} events file is missing columns: {}".format(run, missing_columns)
        )

    events = events.loc[:, list(required)].copy()
    events["condition"] = events["trial_type"].apply(classify_auditory_event)
    ignored_count = int(events["condition"].isna().sum())
    events = events[events["condition"].notna()].copy()
    events["trial_type"] = events.pop("condition")
    events["onset"] = pd.to_numeric(events["onset"], errors="raise")
    events["duration"] = pd.to_numeric(events["duration"], errors="raise")

    if events.empty:
        raise ValueError("Run-{} has no recognized auditory events".format(run))
    if events[["onset", "duration"]].isna().any().any():
        raise ValueError("Run-{} events contain NaN onset/duration values".format(run))
    if (events["onset"] < 0).any():
        raise ValueError("Run-{} events contain negative onsets".format(run))
    if (events["duration"] <= 0).any():
        raise ValueError("Run-{} events contain non-positive durations".format(run))

    counts = events["trial_type"].value_counts().to_dict()
    missing_conditions = [condition for condition in AUDITORY_CONDITIONS if condition not in counts]
    if missing_conditions:
        raise ValueError(
            "Run-{} is missing auditory conditions: {}".format(run, missing_conditions)
        )

    events = events.sort_values("onset").reset_index(drop=True)
    print(
        "Run-{} events: counts={}, ignored_ITIs={}, duration_range=({:.3f}, {:.3f}) s".format(
            run,
            counts,
            ignored_count,
            events["duration"].min(),
            events["duration"].max(),
        ),
        flush=True,
    )
    return events


def load_bold_and_timing(run_info, slice_time_ref_override=None):
    """Load a BOLD image and derive acquisition-aligned frame times."""

    # nibabel.load is lazy.  nilearn.image.load_img can touch the full gzipped
    # array during validation, which is needlessly expensive for a dry run.
    bold_img = nib.load(run_info["bold_path"])
    if len(bold_img.shape) != 4:
        raise ValueError("BOLD image is not 4D: {}".format(run_info["bold_path"]))

    t_r = float(bold_img.header.get_zooms()[3])
    n_scans = int(bold_img.shape[3])
    with open(run_info["bold_json_path"], "r") as stream:
        metadata = json.load(stream)

    json_t_r = metadata.get("RepetitionTime")
    if json_t_r is not None and not np.isclose(float(json_t_r), t_r, atol=1e-5):
        raise ValueError(
            "Run-{} TR differs between NIfTI ({}) and JSON ({})".format(
                run_info["run"], t_r, json_t_r
            )
        )

    if slice_time_ref_override is not None:
        slice_time_ref = float(slice_time_ref_override)
        timing_source = "CLI override"
    elif metadata.get("StartTime") is not None:
        slice_time_ref = float(metadata["StartTime"]) / t_r
        timing_source = "fMRIPrep StartTime"
    elif metadata.get("SliceTimingCorrected", False):
        slice_time_ref = 0.5
        timing_source = "fMRIPrep default"
    else:
        slice_time_ref = 0.0
        timing_source = "volume onset"

    if not 0.0 <= slice_time_ref < 1.0:
        raise ValueError(
            "Run-{} slice-time reference must be in [0, 1), got {}".format(
                run_info["run"], slice_time_ref
            )
        )
    frame_times = (np.arange(n_scans) + slice_time_ref) * t_r

    print(
        "Run-{} BOLD: shape={}, TR={}, scans={}, slice_time_ref={:.4f} ({})".format(
            run_info["run"], bold_img.shape, t_r, n_scans, slice_time_ref, timing_source
        ),
        flush=True,
    )
    return bold_img, frame_times, t_r, n_scans, slice_time_ref


def load_auditory_confounds(bold_path, n_scans, run):
    """Use the same fMRIPrep confound strategy as the visual GLM."""

    confounds, sample_mask = load_confounds(
        bold_path,
        strategy=["high_pass", "motion", "scrub"],
        motion="basic",
        scrub=0,
        fd_threshold=0.5,
        std_dvars_threshold=1.5,
    )
    if confounds.shape[0] != n_scans:
        raise ValueError(
            "Run-{} confound rows ({}) do not match BOLD volumes ({})".format(
                run, confounds.shape[0], n_scans
            )
        )

    if sample_mask is not None:
        sample_mask = np.asarray(sample_mask)
        kept = int(sample_mask.sum()) if sample_mask.dtype == bool else int(sample_mask.size)
    else:
        kept = n_scans
    print(
        "Run-{} confounds: {} regressors; kept {}/{} volumes after scrubbing".format(
            run, confounds.shape[1], kept, n_scans
        ),
        flush=True,
    )
    return confounds, sample_mask


def make_run_design_matrix(frame_times, events, confounds, sample_mask, run):
    """Build and validate one event-related design matrix."""

    design_matrix = make_first_level_design_matrix(
        frame_times,
        events=events,
        hrf_model="glover + derivative + dispersion",
        drift_model=None,  # Cosine high-pass regressors come from fMRIPrep.
        add_regs=confounds,
    )
    missing = [condition for condition in AUDITORY_CONDITIONS if condition not in design_matrix]
    if missing:
        raise ValueError("Run-{} design is missing conditions: {}".format(run, missing))

    if sample_mask is None:
        fitted_design = design_matrix.to_numpy()
    else:
        fitted_design = design_matrix.iloc[sample_mask].to_numpy()
    rank = int(np.linalg.matrix_rank(fitted_design))
    n_columns = int(fitted_design.shape[1])
    if fitted_design.shape[0] <= n_columns:
        raise ValueError(
            "Run-{} has too few retained volumes ({}) for {} regressors".format(
                run, fitted_design.shape[0], n_columns
            )
        )
    if rank < n_columns:
        raise ValueError(
            "Run-{} design is rank deficient after scrubbing: rank {} of {}".format(
                run, rank, n_columns
            )
        )

    print(
        "Run-{} design: shape={}, fitted_rank={}".format(run, design_matrix.shape, rank),
        flush=True,
    )
    return design_matrix, rank


def prepare_run(run_info, slice_time_ref_override=None):
    """Load and validate all inputs for one run."""

    events = load_and_clean_events(run_info["events_path"], run_info["run"])
    bold_img, frame_times, t_r, n_scans, slice_time_ref = load_bold_and_timing(
        run_info, slice_time_ref_override=slice_time_ref_override
    )

    last_event_end = float((events["onset"] + events["duration"]).max())
    acquisition_end = float(frame_times[-1] + t_r)
    if last_event_end > acquisition_end + 1e-6:
        raise ValueError(
            "Run-{} event ends at {:.3f}s, after acquisition end {:.3f}s".format(
                run_info["run"], last_event_end, acquisition_end
            )
        )

    confounds, sample_mask = load_auditory_confounds(
        run_info["bold_path"], n_scans, run_info["run"]
    )
    design_matrix, design_rank = make_run_design_matrix(
        frame_times, events, confounds, sample_mask, run_info["run"]
    )

    prepared = dict(run_info)
    prepared.update(
        {
            "events": events,
            "bold_img": bold_img,
            "t_r": t_r,
            "n_scans": n_scans,
            "slice_time_ref": slice_time_ref,
            "sample_mask": sample_mask,
            "design_matrix": design_matrix,
            "design_rank": design_rank,
        }
    )
    return prepared


def make_common_mask(prepared_runs):
    """Intersect run masks so every fitted voxel is present in every run."""

    mask_paths = [run["mask_path"] for run in prepared_runs]
    common_mask = intersect_masks(mask_paths, threshold=1.0, connected=False)
    voxel_count = int(np.count_nonzero(common_mask.get_fdata()))
    if voxel_count == 0:
        raise ValueError("The intersection of the run brain masks is empty")
    print("Common all-run mask: {} voxels".format(voxel_count), flush=True)
    return common_mask, voxel_count


def make_contrasts(design_matrices):
    """Create per-run vectors for condition and pairwise fixed-effects maps."""

    contrast_specs = {
        "auditory_object": ("auditory_object", None),
        "auditory_scene": ("auditory_scene", None),
        "auditory_scrambled_scene": ("auditory_scrambled_scene", None),
        "auditory_object-auditory_scene": ("auditory_object", "auditory_scene"),
        "auditory_scene-auditory_object": ("auditory_scene", "auditory_object"),
        "auditory_scrambled_scene-auditory_scene": (
            "auditory_scrambled_scene",
            "auditory_scene",
        ),
        "auditory_scene-auditory_scrambled_scene": (
            "auditory_scene",
            "auditory_scrambled_scene",
        ),
        "auditory_scrambled_scene-auditory_object": (
            "auditory_scrambled_scene",
            "auditory_object",
        ),
        "auditory_object-auditory_scrambled_scene": (
            "auditory_object",
            "auditory_scrambled_scene",
        ),
    }

    contrasts = {}
    for contrast_name, (positive, negative) in contrast_specs.items():
        run_vectors = []
        for design_matrix in design_matrices:
            vector = np.zeros(design_matrix.shape[1], dtype=float)
            vector[design_matrix.columns.get_loc(positive)] = 1.0
            if negative is not None:
                vector[design_matrix.columns.get_loc(negative)] = -1.0
            run_vectors.append(vector)
        contrasts[contrast_name] = run_vectors

    print("Contrasts: {}".format(list(contrasts)), flush=True)
    return contrasts


def output_root(save_dir, subject_bids):
    return os.path.join(save_dir, "first_level_glm", subject_bids)


def save_design_matrices(prepared_runs, save_dir):
    design_dir = os.path.join(
        output_root(save_dir, prepared_runs[0]["subject_bids"]), "design_matrix"
    )
    os.makedirs(design_dir, exist_ok=True)

    saved = []
    for run in prepared_runs:
        stem = "{}_task-{}_run-{}_space-{}_design_matrix".format(
            run["subject_bids"], run["task"], run["run"], run["space"]
        )
        csv_path = os.path.join(design_dir, stem + ".csv")
        png_path = os.path.join(design_dir, stem + ".png")
        run["design_matrix"].to_csv(csv_path)

        fig, axis = plt.subplots(figsize=(14, 6))
        plot_design_matrix(run["design_matrix"], ax=axis)
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        saved.extend((csv_path, png_path))

    print("Saved {} run design matrices to {}".format(len(prepared_runs), design_dir), flush=True)
    return saved


def save_common_mask(common_mask, prepared_runs, save_dir):
    first = prepared_runs[0]
    mask_dir = os.path.join(output_root(save_dir, first["subject_bids"]), "mask")
    os.makedirs(mask_dir, exist_ok=True)
    path = os.path.join(
        mask_dir,
        "{}_task-{}_space-{}_desc-allRuns_brain_mask.nii.gz".format(
            first["subject_bids"], first["task"], first["space"]
        ),
    )
    common_mask.to_filename(path)
    print("Saved common mask: {}".format(path), flush=True)
    return path


def fit_subject_glm(prepared_runs, common_mask):
    """Fit run-specific AR(1) models in a single FirstLevelModel object."""

    sample_masks = [run["sample_mask"] for run in prepared_runs]
    if all(sample_mask is None for sample_mask in sample_masks):
        sample_masks = None
    elif any(sample_mask is None for sample_mask in sample_masks):
        sample_masks = [
            np.arange(run["n_scans"]) if sample_mask is None else sample_mask
            for run, sample_mask in zip(prepared_runs, sample_masks)
        ]

    model = FirstLevelModel(
        noise_model="ar1",
        smoothing_fwhm=None,
        mask_img=common_mask,
    )
    model.fit(
        [run["bold_img"] for run in prepared_runs],
        design_matrices=[run["design_matrix"] for run in prepared_runs],
        sample_masks=sample_masks,
    )
    return model


def alpha_label(alpha):
    return ("{:g}".format(alpha)).replace(".", "p")


def save_contrast_maps(model, contrasts, prepared_runs, save_dir, fdr_alpha=0.05):
    """Save all-run fixed-effects effect maps, z maps, and FDR z maps."""

    first = prepared_runs[0]
    saved = []
    alpha_text = alpha_label(fdr_alpha)

    for contrast_name, run_vectors in contrasts.items():
        map_type = "condition" if contrast_name in AUDITORY_CONDITIONS else "contrast"
        map_dir = os.path.join(
            output_root(save_dir, first["subject_bids"]), map_type, contrast_name
        )
        os.makedirs(map_dir, exist_ok=True)
        stem = "{}_task-{}_{}_space-{}_desc-fixedEffects".format(
            first["subject_bids"], first["task"],
            "{}-{}".format(map_type, contrast_name), first["space"]
        )

        maps = model.compute_contrast(run_vectors, output_type="all")
        effect_path = os.path.join(map_dir, stem + "_stat-effect_size.nii.gz")
        z_path = os.path.join(map_dir, stem + "_stat-z_score.nii.gz")
        maps["effect_size"].to_filename(effect_path)
        maps["z_score"].to_filename(z_path)
        saved.extend((effect_path, z_path))

        fdr_map, fdr_threshold = threshold_stats_img(
            maps["z_score"], alpha=fdr_alpha, height_control="fdr", two_sided=True
        )
        fdr_stem = stem + "FdrAlpha{}".format(alpha_text)
        fdr_path = os.path.join(map_dir, fdr_stem + "_stat-z_score.nii.gz")
        threshold_path = os.path.join(map_dir, fdr_stem + "_threshold.txt")
        fdr_map.to_filename(fdr_path)
        with open(threshold_path, "w") as stream:
            stream.write("{}\n".format(fdr_threshold))
        saved.extend((fdr_path, threshold_path))

        print(
            "Saved fixed-effects maps for {} (FDR z threshold={})".format(
                contrast_name, fdr_threshold
            ),
            flush=True,
        )
    return saved


def save_model_summary(prepared_runs, save_dir, common_mask_path, common_mask_voxels):
    first = prepared_runs[0]
    summary = {
        "subject": first["subject_bids"],
        "task": first["task"],
        "space": first["space"],
        "model": "separate run designs and AR(1) fits; subject fixed effects across runs",
        "conditions": list(AUDITORY_CONDITIONS),
        "hrf_model": "glover + derivative + dispersion",
        "confounds": {
            "strategy": ["high_pass", "motion", "scrub"],
            "motion": "basic",
            "fd_threshold": 0.5,
            "std_dvars_threshold": 1.5,
        },
        "common_mask": common_mask_path,
        "common_mask_voxels": common_mask_voxels,
        "runs": [],
    }
    for run in prepared_runs:
        sample_mask = run["sample_mask"]
        if sample_mask is None:
            retained = run["n_scans"]
        else:
            retained = int(sample_mask.sum()) if sample_mask.dtype == bool else int(sample_mask.size)
        summary["runs"].append(
            {
                "run": run["run"],
                "bold": run["bold_path"],
                "events": run["events_path"],
                "mask": run["mask_path"],
                "confounds": run["confounds_path"],
                "tr": run["t_r"],
                "n_scans": run["n_scans"],
                "retained_scans": retained,
                "slice_time_ref": run["slice_time_ref"],
                "event_counts": {
                    key: int(value)
                    for key, value in run["events"]["trial_type"].value_counts().items()
                },
                "design_shape": list(run["design_matrix"].shape),
                "design_rank_after_scrubbing": run["design_rank"],
            }
        )

    path = os.path.join(
        output_root(save_dir, first["subject_bids"]),
        "{}_task-{}_space-{}_model_summary.json".format(
            first["subject_bids"], first["task"], first["space"]
        ),
    )
    with open(path, "w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print("Saved model summary: {}".format(path), flush=True)
    return path


def run_glm_analysis(
    preproc_dir,
    subject,
    save_dir,
    task="audlocalizer",
    space="T1w",
    runs=None,
    slice_time_ref=None,
    dry_run=False,
    threshold=0.05,
):
    if not 0.0 < threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    run_infos = find_auditory_runs(
        preproc_dir,
        subject,
        task=task,
        space=space,
        requested_runs=runs,
    )
    prepared_runs = [
        prepare_run(run_info, slice_time_ref_override=slice_time_ref)
        for run_info in run_infos
    ]

    trs = {run["t_r"] for run in prepared_runs}
    if len(trs) != 1:
        raise ValueError("Runs have inconsistent TRs: {}".format(sorted(trs)))

    common_mask, common_mask_voxels = make_common_mask(prepared_runs)
    contrasts = make_contrasts([run["design_matrix"] for run in prepared_runs])

    if dry_run:
        print(
            "Dry run successful: validated {} runs, {} sound events, and {} contrasts; "
            "stopping before writes and model fitting.".format(
                len(prepared_runs),
                sum(len(run["events"]) for run in prepared_runs),
                len(contrasts),
            ),
            flush=True,
        )
        return 0

    save_design_matrices(prepared_runs, save_dir)
    common_mask_path = save_common_mask(common_mask, prepared_runs, save_dir)
    model = fit_subject_glm(prepared_runs, common_mask)
    save_contrast_maps(
        model,
        contrasts,
        prepared_runs,
        save_dir,
        fdr_alpha=threshold,
    )
    save_model_summary(
        prepared_runs, save_dir, common_mask_path, common_mask_voxels
    )
    print(
        "Successfully completed auditory fixed-effects GLM for {} using runs {}".format(
            prepared_runs[0]["subject_bids"], [run["run"] for run in prepared_runs]
        ),
        flush=True,
    )
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the multi-run, event-related auditory localizer GLM"
    )
    parser.add_argument("-s", "--subject")
    parser.add_argument("--preproc-dir", default=DEFAULT_PREPROC_DIR)
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--task", default="audlocalizer")
    parser.add_argument("--space", default="T1w")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Optional run subset, e.g. --runs 01 02 (default: all available runs)",
    )
    parser.add_argument(
        "--slice-time-ref",
        type=float,
        default=None,
        help=(
            "Optional reference as a fraction of TR. By default, use StartTime "
            "from each fMRIPrep BOLD JSON."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="FDR alpha for thresholded z maps (default: 0.05)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        run_glm_analysis(
            args.preproc_dir,
            args.subject,
            args.save_dir,
            task=args.task,
            space=args.space,
            runs=args.runs,
            slice_time_ref=args.slice_time_ref,
            dry_run=args.dry_run,
            threshold=args.threshold,
        )
    )
