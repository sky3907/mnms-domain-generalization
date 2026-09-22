
import os
import time

import numpy as np
import pandas as pd
import torch
import nibabel as nib

from monai.transforms import Resize
from monai.networks.nets import UNet

from tqdm import tqdm
from medpy.metric.binary import hd95

from dataset import build_dataset


# =======================================================
# Configuration
# =======================================================

ROOT_DIR = "/content/OpenDataset"

MODEL_PATH = (
    "/content/drive/MyDrive/"
    "mnms_checkpoints_strict_ab_cd_seed42/"
    "best_model.pth"
)

RESULTS_DIR = "strict_ab_cd"

SPATIAL_SIZE = (256, 256)

INFERENCE_CHUNK_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

print("Using device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# =======================================================
# Safety check
# =======================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Strict checkpoint not found: {MODEL_PATH}"
    )

print("Checkpoint:", MODEL_PATH)


# =======================================================
# Resize operations
#
# Must match training preprocessing.
# =======================================================

resize_image_to_model = Resize(
    spatial_size=SPATIAL_SIZE
)


def resize_pred_to_native(
    pred_tensor,
    native_hw,
):

    resizer = Resize(
        spatial_size=native_hw,
        mode="nearest",
    )

    return resizer(
        pred_tensor
    )


# =======================================================
# Model
#
# Identical architecture to training.
# =======================================================

print("\nCreating U-Net...")

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=4,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(DEVICE)


state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    state_dict
)

model.eval()

print("Strict A/B model loaded.")


# =======================================================
# Official validation data
#
# IMPORTANT:
# These cases are used ONLY for final evaluation.
# They were not used for checkpoint selection.
# =======================================================

print("\nLoading official validation set...")

val_samples = build_dataset(
    split="val",
    root_dir=ROOT_DIR,
)

print(
    "Validation patient-phase volumes:",
    len(val_samples)
)


# =======================================================
# Verify validation vendor distribution
# =======================================================

vendor_counts = {}

for sample in val_samples:

    vendor = sample["vendor"]

    vendor_counts[vendor] = (
        vendor_counts.get(vendor, 0) + 1
    )


print("\nValidation vendor distribution:")

for vendor in sorted(vendor_counts):

    print(
        f"Vendor {vendor}: "
        f"{vendor_counts[vendor]} patient-phase volumes"
    )


# =======================================================
# Preprocessing
# =======================================================

def normalize_slice(img_slice):

    return (
        img_slice - img_slice.min()
    ) / (
        img_slice.max()
        - img_slice.min()
        + 1e-8
    )


# =======================================================
# Inference
# =======================================================

def run_inference_on_volume(
    image_vol,
    frame_idx,
):

    H, W, num_slices, _ = image_vol.shape

    native_hw = (
        H,
        W,
    )

    resized_slices = []

    for slice_idx in range(num_slices):

        img_slice = image_vol[
            :,
            :,
            slice_idx,
            frame_idx
        ].astype(
            np.float32
        )

        img_slice = normalize_slice(
            img_slice
        )

        img_tensor = torch.tensor(
            img_slice,
            dtype=torch.float32,
        ).unsqueeze(0)

        img_tensor = resize_image_to_model(
            img_tensor
        )

        resized_slices.append(
            img_tensor
        )


    batch_tensor = torch.stack(
        resized_slices,
        dim=0,
    )

    pred_native_slices = np.zeros(
        (
            H,
            W,
            num_slices,
        ),
        dtype=np.int64,
    )


    with torch.inference_mode():

        for start in range(
            0,
            num_slices,
            INFERENCE_CHUNK_SIZE,
        ):

            end = min(
                start + INFERENCE_CHUNK_SIZE,
                num_slices,
            )

            chunk = batch_tensor[
                start:end
            ].to(
                DEVICE
            )

            outputs = model(
                chunk
            )

            preds = torch.argmax(
                outputs,
                dim=1,
            )

            preds = (
                preds
                .cpu()
                .float()
            )


            for i in range(
                preds.shape[0]
            ):

                pred_slice = (
                    preds[i]
                    .unsqueeze(0)
                )

                pred_native = resize_pred_to_native(
                    pred_slice,
                    native_hw,
                )

                pred_native = (
                    pred_native
                    .squeeze(0)
                    .round()
                    .long()
                    .numpy()
                )

                pred_native_slices[
                    :,
                    :,
                    start + i
                ] = pred_native


    assert (
        pred_native_slices.shape
        ==
        (
            H,
            W,
            num_slices,
        )
    )

    return pred_native_slices


# =======================================================
# Metrics
# =======================================================

STRUCTURE_NAMES = [
    "LV",
    "Myocardium",
    "RV",
]


def compute_volume_metrics(
    pred_vol,
    gt_vol,
    spacing_3d,
):

    assert (
        pred_vol.shape
        ==
        gt_vol.shape
    )

    result = {}

    for cls, name in zip(
        [1, 2, 3],
        STRUCTURE_NAMES,
    ):

        pred_binary = (
            pred_vol == cls
        )

        gt_binary = (
            gt_vol == cls
        )

        pred_sum = (
            pred_binary.sum()
        )

        gt_sum = (
            gt_binary.sum()
        )


        if gt_sum == 0:

            result[
                f"Dice_{name}"
            ] = np.nan

            result[
                f"HD95_{name}"
            ] = np.nan

            continue


        intersection = np.logical_and(
            pred_binary,
            gt_binary,
        ).sum()


        dice = (
            2.0 * intersection
            + 1e-8
        ) / (
            pred_sum
            + gt_sum
            + 1e-8
        )


        if pred_sum == 0:

            hd95_value = np.nan

        else:

            try:

                hd95_value = hd95(
                    pred_binary,
                    gt_binary,
                    voxelspacing=spacing_3d,
                )

            except RuntimeError:

                hd95_value = np.nan


        result[
            f"Dice_{name}"
        ] = float(dice)

        result[
            f"HD95_{name}"
        ] = float(hd95_value)


    dice_values = [
        result[f"Dice_{name}"]
        for name in STRUCTURE_NAMES
    ]

    hd95_values = [
        result[f"HD95_{name}"]
        for name in STRUCTURE_NAMES
    ]


    result["Mean_Dice"] = float(
        np.nanmean(
            dice_values
        )
    )

    result["Mean_HD95"] = float(
        np.nanmean(
            hd95_values
        )
    )


    return result


# =======================================================
# Run official validation evaluation
# =======================================================

print(
    "\n======================================"
)

print(
    "STRICT A/B -> C/D VOLUME EVALUATION"
)

print(
    "======================================"
)


start_time = time.time()

rows = []
skipped = []


for sample in tqdm(
    val_samples
):

    patient = sample[
        "patient"
    ]

    phase = sample[
        "phase"
    ]

    vendor = sample[
        "vendor"
    ]

    vendor_name = sample[
        "vendor_name"
    ]

    centre = sample[
        "centre"
    ]

    frame_idx = sample[
        "frame_idx"
    ]


    try:

        image_obj = nib.load(
            sample["image_path"]
        )

        image_vol = image_obj.get_fdata(
            dtype=np.float32
        )

        mask_obj = nib.load(
            sample["mask_path"]
        )

        mask_vol = mask_obj.get_fdata(
            dtype=np.float32
        )

        zooms = (
            image_obj
            .header
            .get_zooms()
        )

        spacing_3d = (
            float(zooms[0]),
            float(zooms[1]),
            float(zooms[2]),
        )


        pred_vol = run_inference_on_volume(
            image_vol,
            frame_idx,
        )

        gt_vol = mask_vol[
            :,
            :,
            :,
            frame_idx
        ].astype(
            np.int64
        )


        assert (
            pred_vol.shape
            ==
            gt_vol.shape
        )


        metrics = compute_volume_metrics(
            pred_vol,
            gt_vol,
            spacing_3d,
        )


        domain = (
            "in-domain"
            if vendor in {"A", "B"}
            else "unseen"
        )


        row = {

            "patient": patient,

            "phase": phase,

            "vendor": vendor,

            "vendor_name":
                vendor_name,

            "centre": centre,

            "domain": domain,
        }


        row.update(
            metrics
        )

        rows.append(
            row
        )


    except Exception as exc:

        print(
            f"\n[WARNING] "
            f"{patient} ({phase}) "
            f"Vendor {vendor}: {exc}"
        )

        skipped.append({

            "patient":
                patient,

            "phase":
                phase,

            "vendor":
                vendor,

            "reason":
                str(exc),
        })


elapsed = (
    time.time()
    - start_time
)


# =======================================================
# Save patient-level results
# =======================================================

results_df = pd.DataFrame(
    rows
)

patient_csv = os.path.join(
    RESULTS_DIR,
    "patient_level_results.csv",
)

results_df.to_csv(
    patient_csv,
    index=False,
)


print(
    "\nPatient-level results saved:",
    patient_csv
)


# =======================================================
# Vendor summary
# =======================================================

metric_columns = [

    "Dice_LV",
    "Dice_Myocardium",
    "Dice_RV",

    "HD95_LV",
    "HD95_Myocardium",
    "HD95_RV",

    "Mean_Dice",
    "Mean_HD95",
]


vendor_summary = (
    results_df
    .groupby("vendor")[
        metric_columns
    ]
    .mean()
    .round(4)
)


vendor_summary.insert(
    0,
    "N_patient_phases",
    results_df
    .groupby("vendor")
    .size()
)


vendor_csv = os.path.join(
    RESULTS_DIR,
    "vendor_summary.csv",
)

vendor_summary.to_csv(
    vendor_csv
)


print(
    "\n======================================"
)

print(
    "VENDOR-WISE RESULTS"
)

print(
    "======================================"
)

print(
    vendor_summary
)


# =======================================================
# In-domain vs unseen summary
#
# A/B = source/reference
# C/D = unseen vendors
# =======================================================

domain_summary = (
    results_df
    .groupby("domain")[
        metric_columns
    ]
    .mean()
    .round(4)
)


domain_summary.insert(
    0,
    "N_patient_phases",
    results_df
    .groupby("domain")
    .size()
)


domain_csv = os.path.join(
    RESULTS_DIR,
    "domain_summary.csv",
)

domain_summary.to_csv(
    domain_csv
)


print(
    "\n======================================"
)

print(
    "SOURCE vs UNSEEN DOMAIN RESULTS"
)

print(
    "======================================"
)

print(
    domain_summary
)


# =======================================================
# Phase summary
# =======================================================

phase_summary = (
    results_df
    .groupby("phase")[
        metric_columns
    ]
    .mean()
    .round(4)
)


phase_csv = os.path.join(
    RESULTS_DIR,
    "phase_summary.csv",
)

phase_summary.to_csv(
    phase_csv
)


# =======================================================
# Skip report
# =======================================================

if skipped:

    skipped_df = pd.DataFrame(
        skipped
    )

    skipped_path = os.path.join(
        RESULTS_DIR,
        "skipped_samples.csv",
    )

    skipped_df.to_csv(
        skipped_path,
        index=False,
    )

    print(
        "\nSkipped samples:",
        len(skipped)
    )

else:

    print(
        "\nSkipped samples: 0"
    )


# =======================================================
# Finish
# =======================================================

print(
    f"\nEvaluation time: "
    f"{elapsed:.2f} seconds"
)

print(
    "\nSaved:"
)

print(
    patient_csv
)

print(
    vendor_csv
)

print(
    domain_csv
)

print(
    phase_csv
)
