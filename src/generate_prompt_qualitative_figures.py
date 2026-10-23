
import ast
import hashlib
import shutil
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis
from monai.transforms import Resize
from monai.networks.nets import UNet
from medpy.metric.binary import hd95
from dataset import build_dataset


# ============================================================
# Configuration
# ============================================================

ROOT = Path("/content/mnms-domain-generalization")
DATASET_ROOT = "/content/OpenDataset"

REFERENCE_SCRIPT = ROOT / "src/evaluate_prompt_comparison.py"

RESULTS_CSV = (
    ROOT / "prompt_comparison/patient_level_results.csv"
)

LATENT_DIR = ROOT / "latent_domain_discovery"

OUTPUT_DIR = (
    ROOT / "prompt_comparison/qualitative_analysis/figures"
)

DRIVE_OUTPUT = Path(
    "/content/drive/MyDrive/mnms_prompt_qualitative_figures"
)

MODEL_PATHS = {
    "source_only": Path(
        "/content/drive/MyDrive/"
        "mnms_checkpoints_strict_ab_cd_seed42/best_model.pth"
    ),
    "generic_prompt": Path(
        "/content/drive/MyDrive/"
        "mnms_prompt_baselines_seed42/"
        "generic_prompt/best_model.pth"
    ),
    "latent_domain_prompt": Path(
        "/content/drive/MyDrive/"
        "mnms_prompt_baselines_seed42/"
        "latent_domain_prompt/best_model.pth"
    ),
}

CASES = [
    ("N2O7U5", "ES", "D", "boundary_success"),
    ("C8J7L5", "ES", "C", "dice_success"),
    ("C5L0R0", "ES", "D", "failure"),
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SPATIAL_SIZE = (256, 256)
INFERENCE_CHUNK_SIZE = 32
PROMPT_DIM = 4
NUM_PROMPTS = 2

# Consistent class colours: LV, myocardium, RV.
CLASS_COLOURS = {
    1: "#e63946",
    2: "#f4a261",
    3: "#29b6b6",
}

CLASS_NAMES = {
    1: "LV",
    2: "MYO",
    3: "RV",
}


# ============================================================
# File verification
# ============================================================

def sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def verified_copy(source, destination):
    source = Path(source)
    destination = Path(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)

    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty file: {source}")

    if destination.exists():
        if sha256(source) != sha256(destination):
            raise RuntimeError(
                f"Existing output differs: {destination}. "
                "Do not overwrite it automatically."
            )
        return

    shutil.copy2(source, destination)

    if sha256(source) != sha256(destination):
        raise RuntimeError(
            f"Backup verification failed: {destination}"
        )


def save_figure(fig, filename):
    destination = OUTPUT_DIR / filename

    if destination.exists():
        raise FileExistsError(
            f"Figure already exists: {destination}"
        )

    fig.savefig(
        destination,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    verified_copy(
        destination,
        DRIVE_OUTPUT / filename,
    )

    print("Saved and verified:", filename)


# ============================================================
# Load EXACT function/class definitions from the evaluator
#
# This does NOT import or run its top-level evaluation loop.
# Only the specifically named functions/classes are compiled.
# ============================================================

def load_evaluator_definitions(namespace):

    source = REFERENCE_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)

    required = [
        "resize_pred_to_native",
        "create_source_model",
        "PromptedUNet",
        "normalize_slice",
        "get_robust_voxels",
        "compute_entropy",
        "extract_style_features",
        "assign_latent_cluster",
        "prepare_volume_tensor",
        "run_source_inference",
        "run_prompt_inference",
        "compute_volume_metrics",
    ]

    selected = []

    for name in required:
        matches = [
            node
            for node in tree.body
            if isinstance(
                node,
                (ast.FunctionDef, ast.ClassDef)
            )
            and node.name == name
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one evaluator definition: {name}"
            )

        selected.append(matches[0])

    module = ast.fix_missing_locations(
        ast.Module(
            body=selected,
            type_ignores=[],
        )
    )

    compiled = compile(
        module,
        str(REFERENCE_SCRIPT),
        "exec",
    )

    exec(compiled, namespace)

    print(
        "Loaded exact preprocessing and inference definitions "
        "without running full evaluation."
    )


# ============================================================
# Slice-wise metrics for objective visual selection
# ============================================================

def slice_foreground_dice(prediction, truth):

    pred_fg = prediction > 0
    gt_fg = truth > 0

    denominator = pred_fg.sum() + gt_fg.sum()

    if denominator == 0:
        return 1.0

    intersection = np.logical_and(
        pred_fg,
        gt_fg,
    ).sum()

    return float(
        (2.0 * intersection) / denominator
    )


def select_slice(gt, source_pred, latent_pred, role):

    n_slices = gt.shape[2]

    # Use the previously documented peripheral failure slice
    # for the specific N2O7U5 boundary-error example.
    if role == "boundary_success":
        return 0

    candidates = []

    for z in range(n_slices):
        truth = gt[:, :, z]

        # For overlap success/failure, select a slice
        # containing substantial reference anatomy.
        if np.count_nonzero(truth) < 100:
            continue

        source_dice = slice_foreground_dice(
            source_pred[:, :, z],
            truth,
        )

        latent_dice = slice_foreground_dice(
            latent_pred[:, :, z],
            truth,
        )

        difference = latent_dice - source_dice

        candidates.append((difference, z))

    if not candidates:
        raise RuntimeError(
            "No suitable anatomical slice found."
        )

    if role == "dice_success":
        return max(candidates)[1]

    if role == "failure":
        return min(candidates)[1]

    raise ValueError(f"Unknown case role: {role}")


# ============================================================
# Draw matched segmentation contours
# ============================================================

def draw_contours(
    ax,
    mask,
    linestyle="-",
    linewidth=1.7,
    alpha=1.0,
):

    for cls in (1, 2, 3):
        binary = mask == cls

        if not binary.any():
            continue

        if binary.all():
            continue

        ax.contour(
            binary.astype(float).T,
            levels=[0.5],
            colors=[CLASS_COLOURS[cls]],
            linewidths=linewidth,
            linestyles=linestyle,
            alpha=alpha,
        )


def make_comparison_figure(
    image,
    gt,
    predictions,
    patient,
    phase,
    vendor,
    role,
    slice_idx,
    metrics,
):

    image_slice = image[:, :, slice_idx]
    gt_slice = gt[:, :, slice_idx]

    images = [
        None,
        gt_slice,
        predictions["source_only"][:, :, slice_idx],
        predictions["generic_prompt"][:, :, slice_idx],
        predictions["latent_domain_prompt"][:, :, slice_idx],
    ]

    titles = [
        "MRI",
        "Ground truth",
        "Source-only",
        "Generic prompt",
        "Latent-domain prompt",
    ]

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(15, 3.9),
        constrained_layout=True,
    )

    low, high = np.percentile(
        image_slice[np.isfinite(image_slice)],
        [1, 99],
    )

    for ax, title, mask in zip(axes, titles, images):

        ax.imshow(
            image_slice.T,
            cmap="gray",
            origin="lower",
            vmin=low,
            vmax=high if high > low else low + 1,
        )

        if title == "Ground truth":
            draw_contours(
                ax,
                gt_slice,
                linestyle="-",
                linewidth=1.8,
            )

        elif mask is not None:
            # Ground truth: solid, finer contours.
            draw_contours(
                ax,
                gt_slice,
                linestyle="-",
                linewidth=1.0,
                alpha=0.8,
            )

            # Prediction: dashed, thicker contours.
            draw_contours(
                ax,
                mask,
                linestyle="--",
                linewidth=2.0,
            )

        ax.set_title(title, fontsize=11)
        ax.axis("off")

    source = metrics["source_only"]
    latent = metrics["latent_domain_prompt"]

    fig.suptitle(
        f"{patient} / {phase} / Vendor {vendor} / "
        f"slice {slice_idx} / {role.replace('_', ' ')}\n"
        f"Volume mean Dice: {source['Mean_Dice']:.3f}"
        f" -> {latent['Mean_Dice']:.3f} | "
        f"HD95: {source['Mean_HD95']:.2f}"
        f" -> {latent['Mean_HD95']:.2f} mm",
        fontsize=11,
    )

    fig.text(
        0.5,
        0.01,
        "Solid contours: ground truth | "
        "Dashed contours: model prediction | "
        "LV: red, myocardium: orange, RV: cyan",
        ha="center",
        fontsize=9,
    )

    return fig


# ============================================================
# Main
# ============================================================

def main():

    required = [
        REFERENCE_SCRIPT,
        RESULTS_CSV,
        LATENT_DIR / "clustering_metadata.json",
        LATENT_DIR / "style_scaler.joblib",
        LATENT_DIR / "style_pca.joblib",
        LATENT_DIR / "style_kmeans.joblib",
        *MODEL_PATHS.values(),
    ]

    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Missing or empty required file: {path}"
            )

    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Mount Google Drive first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DRIVE_OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------------------------
    # Frozen cluster metadata and pipeline
    # --------------------------------------------------------

    import json

    with (
        LATENT_DIR / "clustering_metadata.json"
    ).open() as f:
        metadata = json.load(f)

    if metadata["selected_k"] != 2:
        raise ValueError("Expected frozen K=2 clustering.")

    if metadata["vendor_centre_used_for_clustering"] is not False:
        raise ValueError("Clustering used domain metadata.")

    feature_columns = metadata["feature_columns"]
    n_pca_components = int(metadata["pca_components_95"])

    # Load only your own trusted clustering objects.
    scaler = joblib.load(
        LATENT_DIR / "style_scaler.joblib"
    )

    pca = joblib.load(
        LATENT_DIR / "style_pca.joblib"
    )

    kmeans = joblib.load(
        LATENT_DIR / "style_kmeans.joblib"
    )

    resize_image_to_model = Resize(
        spatial_size=SPATIAL_SIZE
    )

    # --------------------------------------------------------
    # Exact evaluator functions with required globals
    # --------------------------------------------------------

    namespace = {
        "__builtins__": __builtins__,
        "np": np,
        "torch": torch,
        "nn": nn,
        "UNet": UNet,
        "Resize": Resize,
        "skew": skew,
        "kurtosis": kurtosis,
        "hd95": hd95,
        "DEVICE": DEVICE,
        "SPATIAL_SIZE": SPATIAL_SIZE,
        "INFERENCE_CHUNK_SIZE": INFERENCE_CHUNK_SIZE,
        "PROMPT_DIM": PROMPT_DIM,
        "NUM_PROMPTS": NUM_PROMPTS,
        "resize_image_to_model": resize_image_to_model,
        "feature_columns": feature_columns,
        "n_pca_components": n_pca_components,
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "STRUCTURE_NAMES": [
            "LV",
            "Myocardium",
            "RV",
        ],
    }

    load_evaluator_definitions(namespace)

    # --------------------------------------------------------
    # Frozen models
    # --------------------------------------------------------

    source_model = namespace["create_source_model"]()
    generic_model = namespace["PromptedUNet"]().to(DEVICE)
    latent_model = namespace["PromptedUNet"]().to(DEVICE)

    for model, name in [
        (source_model, "source_only"),
        (generic_model, "generic_prompt"),
        (latent_model, "latent_domain_prompt"),
    ]:

        state = torch.load(
            MODEL_PATHS[name],
            map_location=DEVICE,
            weights_only=True,
        )

        model.load_state_dict(state, strict=True)
        model.eval()

        print("Loaded frozen model:", name)

    namespace["source_model"] = source_model

    # --------------------------------------------------------
    # Locate only requested C/D patient-phase records
    # --------------------------------------------------------

    all_samples = build_dataset(
        split="val",
        root_dir=DATASET_ROOT,
    )

    sample_map = {}

    for sample in all_samples:
        key = (
            str(sample["patient"]),
            str(sample["phase"]),
        )

        if key in sample_map:
            raise ValueError(f"Duplicate sample: {key}")

        sample_map[key] = sample

    previous_results = pd.read_csv(RESULTS_CSV)

    run_source = namespace["run_source_inference"]
    run_prompt = namespace["run_prompt_inference"]
    prepare_volume = namespace["prepare_volume_tensor"]
    assign_cluster = namespace["assign_latent_cluster"]
    calculate_metrics = namespace["compute_volume_metrics"]

    verification_rows = []

    for patient, phase, vendor, role in CASES:

        print(f"\nProcessing {patient} / {phase} / {vendor}")

        key = (patient, phase)

        if key not in sample_map:
            raise FileNotFoundError(
                f"Patient-phase not found: {key}"
            )

        sample = sample_map[key]

        if sample["vendor"] != vendor:
            raise ValueError(
                f"Unexpected vendor for {key}"
            )

        image_obj = nib.load(sample["image_path"])
        mask_obj = nib.load(sample["mask_path"])

        image_vol = image_obj.get_fdata(
            dtype=np.float32
        )

        mask_vol = mask_obj.get_fdata(
            dtype=np.float32
        )

        frame_idx = int(sample["frame_idx"])

        spacing = tuple(
            float(x)
            for x in image_obj.header.get_zooms()[:3]
        )

        image_phase = image_vol[:, :, :, frame_idx]

        gt = mask_vol[
            :, :, :, frame_idx
        ].astype(np.int64)

        latent_cluster = assign_cluster(image_phase)

        batch_tensor, native_hw, num_slices = prepare_volume(
            image_vol,
            frame_idx,
        )

        predictions = {
            "source_only": run_source(
                batch_tensor,
                native_hw,
                num_slices,
            ),

            "generic_prompt": run_prompt(
                generic_model,
                batch_tensor,
                native_hw,
                num_slices,
                prompt_id=0,
            ),

            "latent_domain_prompt": run_prompt(
                latent_model,
                batch_tensor,
                native_hw,
                num_slices,
                prompt_id=latent_cluster,
            ),
        }

        metrics = {}

        for model_name, prediction in predictions.items():

            if prediction.shape != gt.shape:
                raise ValueError(
                    f"Prediction shape mismatch: {model_name}"
                )

            result = calculate_metrics(
                prediction,
                gt,
                spacing,
            )

            metrics[model_name] = result

            recorded = previous_results[
                (previous_results["patient"] == patient)
                & (previous_results["phase"] == phase)
                & (previous_results["model"] == model_name)
            ]

            if len(recorded) != 1:
                raise RuntimeError(
                    f"Expected one saved metric row: "
                    f"{patient} {phase} {model_name}"
                )

            for metric_name in [
                "Mean_Dice",
                "Mean_HD95",
                "Dice_LV",
                "HD95_LV",
                "Dice_Myocardium",
                "HD95_Myocardium",
                "Dice_RV",
                "HD95_RV",
            ]:

                computed = result[metric_name]
                saved = float(recorded.iloc[0][metric_name])

                tolerance = (
                    0.1 if "HD95" in metric_name else 0.001
                )

                if np.isfinite(computed) and np.isfinite(saved):
                    if abs(computed - saved) > tolerance:
                        raise RuntimeError(
                            f"Metric mismatch: {patient} {phase} "
                            f"{model_name} {metric_name}: "
                            f"computed={computed}, saved={saved}"
                        )

            verification_rows.append({
                "patient": patient,
                "phase": phase,
                "vendor": vendor,
                "model": model_name,
                "latent_cluster": latent_cluster,
                "Mean_Dice": result["Mean_Dice"],
                "Mean_HD95": result["Mean_HD95"],
                "HD95_LV": result["HD95_LV"],
            })

        print("All three model metrics verified.")

        selected_slice = select_slice(
            gt,
            predictions["source_only"],
            predictions["latent_domain_prompt"],
            role,
        )

        slices_to_render = [selected_slice]

        # Include the central slice from the documented
        # N2O7U5 case as an additional comparison.
        if role == "boundary_success":
            if gt.shape[2] > 7 and 7 not in slices_to_render:
                slices_to_render.append(7)

        for z in slices_to_render:

            fig = make_comparison_figure(
                image_phase,
                gt,
                predictions,
                patient,
                phase,
                vendor,
                role,
                z,
                metrics,
            )

            filename = (
                f"{role}_{patient}_{phase}_slice{z}.png"
            )

            save_figure(fig, filename)

        print(
            "Volume mean Dice:",
            {
                name: round(value["Mean_Dice"], 4)
                for name, value in metrics.items()
            },
        )

        print(
            "Volume mean HD95:",
            {
                name: round(value["Mean_HD95"], 2)
                for name, value in metrics.items()
            },
        )

    # --------------------------------------------------------
    # Save metric verification CSV
    # --------------------------------------------------------

    verification = pd.DataFrame(verification_rows)

    csv_path = OUTPUT_DIR / "figure_metric_verification.csv"

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        if not existing.equals(verification):
            raise RuntimeError(
                "Existing verification CSV differs."
            )
    else:
        verification.to_csv(csv_path, index=False)

    verified_copy(
        csv_path,
        DRIVE_OUTPUT / csv_path.name,
    )

    print("\nQUALITATIVE FIGURES COMPLETE")
    print("Local figures:", OUTPUT_DIR)
    print("Verified Drive backup:", DRIVE_OUTPUT)
    print("No model training or checkpoint modification occurred.")


if __name__ == "__main__":
    main()
