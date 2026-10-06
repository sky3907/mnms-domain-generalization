
import os
import time
import json
import joblib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import nibabel as nib

from scipy.stats import skew, kurtosis

from monai.transforms import Resize
from monai.networks.nets import UNet

from tqdm import tqdm
from medpy.metric.binary import hd95

from dataset import build_dataset


# ============================================================
# Configuration
# ============================================================

ROOT_DIR = "/content/OpenDataset"

SOURCE_MODEL_PATH = (
    "/content/drive/MyDrive/"
    "mnms_checkpoints_strict_ab_cd_seed42/"
    "best_model.pth"
)

GENERIC_MODEL_PATH = (
    "/content/drive/MyDrive/"
    "mnms_prompt_baselines_seed42/"
    "generic_prompt/"
    "best_model.pth"
)

LATENT_MODEL_PATH = (
    "/content/drive/MyDrive/"
    "mnms_prompt_baselines_seed42/"
    "latent_domain_prompt/"
    "best_model.pth"
)

LATENT_DIR = "latent_domain_discovery"

RESULTS_DIR = "prompt_comparison"

SPATIAL_SIZE = (256, 256)

INFERENCE_CHUNK_SIZE = 32

PROMPT_DIM = 4

NUM_PROMPTS = 2


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)


print(
    "Using device:",
    DEVICE,
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )


# ============================================================
# Safety checks
# ============================================================

required_files = [

    SOURCE_MODEL_PATH,
    GENERIC_MODEL_PATH,
    LATENT_MODEL_PATH,

    os.path.join(
        LATENT_DIR,
        "style_scaler.joblib",
    ),

    os.path.join(
        LATENT_DIR,
        "style_pca.joblib",
    ),

    os.path.join(
        LATENT_DIR,
        "style_kmeans.joblib",
    ),

    os.path.join(
        LATENT_DIR,
        "clustering_metadata.json",
    ),
]


for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


print(
    "\nSource checkpoint:",
    SOURCE_MODEL_PATH,
)

print(
    "Generic checkpoint:",
    GENERIC_MODEL_PATH,
)

print(
    "Latent checkpoint:",
    LATENT_MODEL_PATH,
)


# ============================================================
# Resize operations
#
# Same geometry as strict evaluation.
# ============================================================

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


# ============================================================
# Source-only model
#
# EXACT architecture from frozen strict baseline.
# ============================================================

def create_source_model():

    model = UNet(

        spatial_dims=2,

        in_channels=1,

        out_channels=4,

        channels=(
            16,
            32,
            64,
            128,
            256,
        ),

        strides=(
            2,
            2,
            2,
            2,
        ),

        num_res_units=2,

    ).to(DEVICE)

    return model


# ============================================================
# Prompted U-Net
#
# EXACT architecture used during prompt training.
# ============================================================

class PromptedUNet(
    nn.Module
):

    def __init__(
        self,
        prompt_dim=PROMPT_DIM,
        num_prompts=NUM_PROMPTS,
    ):

        super().__init__()

        self.prompt_dim = (
            prompt_dim
        )

        self.prompt_embedding = (
            nn.Embedding(
                num_prompts,
                prompt_dim,
            )
        )

        self.unet = UNet(

            spatial_dims=2,

            in_channels=(
                1
                + prompt_dim
            ),

            out_channels=4,

            channels=(
                16,
                32,
                64,
                128,
                256,
            ),

            strides=(
                2,
                2,
                2,
                2,
            ),

            num_res_units=2,
        )


    def forward(
        self,
        images,
        prompt_ids,
    ):

        prompt = (
            self.prompt_embedding(
                prompt_ids
            )
        )

        prompt = prompt[
            :,
            :,
            None,
            None,
        ]

        prompt = prompt.expand(

            -1,
            -1,

            images.shape[-2],
            images.shape[-1],
        )

        prompted_images = torch.cat(

            [
                images,
                prompt,
            ],

            dim=1,
        )

        return self.unet(
            prompted_images
        )


# ============================================================
# Load all three FROZEN models
# ============================================================

print(
    "\nLoading frozen models..."
)


source_model = (
    create_source_model()
)

source_state = torch.load(
    SOURCE_MODEL_PATH,
    map_location=DEVICE,
)

source_model.load_state_dict(
    source_state
)

source_model.eval()


generic_model = PromptedUNet().to(
    DEVICE
)

generic_state = torch.load(
    GENERIC_MODEL_PATH,
    map_location=DEVICE,
)

generic_model.load_state_dict(
    generic_state
)

generic_model.eval()


latent_model = PromptedUNet().to(
    DEVICE
)

latent_state = torch.load(
    LATENT_MODEL_PATH,
    map_location=DEVICE,
)

latent_model.load_state_dict(
    latent_state
)

latent_model.eval()


print(
    "Source-only model loaded."
)

print(
    "Generic-prompt model loaded."
)

print(
    "Latent-domain-prompt model loaded."
)


# ============================================================
# Load FROZEN latent-domain assignment pipeline
#
# IMPORTANT:
# These objects were fit ONLY on the A/B source-training set.
#
# During this script we ONLY call:
#     transform()
#     predict()
#
# We never fit/refit anything on validation A/B or C/D.
# ============================================================

print(
    "\nLoading frozen latent-domain pipeline..."
)


with open(
    os.path.join(
        LATENT_DIR,
        "clustering_metadata.json",
    ),
    "r",
) as f:

    latent_metadata = json.load(f)


assert (
    latent_metadata[
        "selected_k"
    ]
    == 2
)

assert (
    latent_metadata[
        "vendor_centre_used_for_clustering"
    ]
    is False
)


feature_columns = (
    latent_metadata[
        "feature_columns"
    ]
)

n_pca_components = int(
    latent_metadata[
        "pca_components_95"
    ]
)


scaler = joblib.load(

    os.path.join(
        LATENT_DIR,
        "style_scaler.joblib",
    )
)


pca = joblib.load(

    os.path.join(
        LATENT_DIR,
        "style_pca.joblib",
    )
)


kmeans = joblib.load(

    os.path.join(
        LATENT_DIR,
        "style_kmeans.joblib",
    )
)


print(
    "Frozen latent K:",
    latent_metadata[
        "selected_k"
    ],
)

print(
    "Frozen PCA components:",
    n_pca_components,
)


# ============================================================
# Official validation set
#
# This is the FINAL held-out evaluation.
#
# A/B:
#     reference/in-domain
#
# C/D:
#     unseen-vendor
#
# No parameter fitting or model selection occurs below.
# ============================================================

print(
    "\nLoading official validation set..."
)


val_samples = build_dataset(
    split="val",
    root_dir=ROOT_DIR,
)


print(
    "Validation patient-phase volumes:",
    len(val_samples),
)


# ============================================================
# Verify expected vendor distribution
# ============================================================

vendor_counts = {}


for sample in val_samples:

    vendor = sample[
        "vendor"
    ]

    vendor_counts[
        vendor
    ] = (
        vendor_counts.get(
            vendor,
            0,
        )
        + 1
    )


print(
    "\nValidation vendor distribution:"
)


for vendor in sorted(
    vendor_counts
):

    print(
        f"Vendor {vendor}: "
        f"{vendor_counts[vendor]} "
        "patient-phase volumes"
    )


# Expected official validation composition.
assert vendor_counts.get(
    "A",
    0,
) == 8

assert vendor_counts.get(
    "B",
    0,
) == 20

assert vendor_counts.get(
    "C",
    0,
) == 20

assert vendor_counts.get(
    "D",
    0,
) == 20


# ============================================================
# Segmentation preprocessing
#
# EXACT same per-slice min-max normalization as strict
# evaluation.
# ============================================================

def normalize_slice(
    img_slice,
):

    return (

        img_slice
        - img_slice.min()

    ) / (

        img_slice.max()
        - img_slice.min()
        + 1e-8
    )


# ============================================================
# Latent-domain preprocessing
#
# EXACT same preprocessing used during frozen source-only
# latent-domain discovery.
# ============================================================

def get_robust_voxels(
    volume,
):

    volume = np.asarray(
        volume,
        dtype=np.float32,
    )

    finite = volume[
        np.isfinite(volume)
    ]

    if finite.size == 0:

        raise ValueError(
            "Volume contains no finite voxels."
        )


    nonzero = finite[
        finite != 0
    ]


    if nonzero.size > 100:

        voxels = nonzero

    else:

        voxels = finite


    low, high = np.percentile(
        voxels,
        [1, 99],
    )


    clipped = np.clip(
        voxels,
        low,
        high,
    )


    median = np.median(
        clipped
    )


    q25, q75 = np.percentile(
        clipped,
        [25, 75],
    )


    iqr = (
        q75
        - q25
    )


    normalized = (

        clipped
        - median

    ) / (

        iqr
        + 1e-8
    )


    return normalized.astype(
        np.float32
    )


# ============================================================
# Entropy
# ============================================================

def compute_entropy(
    values,
):

    hist, _ = np.histogram(
        values,
        bins=64,
        density=False,
    )


    probabilities = (

        hist.astype(
            np.float64
        )

        / max(
            hist.sum(),
            1,
        )
    )


    probabilities = probabilities[
        probabilities > 0
    ]


    return float(

        -np.sum(

            probabilities

            * np.log2(
                probabilities
            )
        )
    )


# ============================================================
# EXACT frozen style features
# ============================================================

def extract_style_features(
    volume,
):

    x = get_robust_voxels(
        volume
    )


    (
        p1,
        p5,
        p25,
        _p50,
        p75,
        p95,
        p99,

    ) = np.percentile(

        x,

        [
            1,
            5,
            25,
            50,
            75,
            95,
            99,
        ],
    )


    return {

        "intensity_mean":
            float(
                np.mean(x)
            ),

        "intensity_std":
            float(
                np.std(x)
            ),

        "p01":
            float(p1),

        "p05":
            float(p5),

        "p25":
            float(p25),

        "p75":
            float(p75),

        "p95":
            float(p95),

        "p99":
            float(p99),

        "skewness":
            float(
                skew(x)
            ),

        "kurtosis":
            float(
                kurtosis(x)
            ),

        "entropy":
            compute_entropy(x),
    }


# ============================================================
# Frozen latent-cluster assignment
#
# This is inference only.
#
# NO fit()
# NO fit_transform()
# NO C/D adaptation
# ============================================================

def assign_latent_cluster(
    patient_phase_volume,
):

    features = (
        extract_style_features(
            patient_phase_volume
        )
    )


    X = np.asarray(
        [[
            features[name]
            for name
            in feature_columns
        ]],
        dtype=np.float64,
    )


    if not np.all(
        np.isfinite(X)
    ):

        raise ValueError(
            "Non-finite style features."
        )


    # FROZEN scaler
    X_scaled = (
        scaler.transform(
            X
        )
    )


    # FROZEN PCA
    X_pca = (
        pca.transform(
            X_scaled
        )
    )


    X_pca = X_pca[
        :,
        :n_pca_components,
    ]


    # FROZEN source-trained K-means
    cluster = int(

        kmeans.predict(
            X_pca
        )[0]
    )


    assert cluster in {
        0,
        1,
    }


    return cluster


# ============================================================
# Prepare a volume for segmentation inference
#
# Per-slice preprocessing is done ONCE and shared across all
# three models.
# ============================================================

def prepare_volume_tensor(
    image_vol,
    frame_idx,
):

    H, W, num_slices, _ = (
        image_vol.shape
    )


    resized_slices = []


    for slice_idx in range(
        num_slices
    ):

        img_slice = image_vol[

            :,
            :,
            slice_idx,
            frame_idx,

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


        img_tensor = (
            resize_image_to_model(
                img_tensor
            )
        )


        resized_slices.append(
            img_tensor
        )


    batch_tensor = torch.stack(

        resized_slices,

        dim=0,
    )


    return (
        batch_tensor,
        (H, W),
        num_slices,
    )


# ============================================================
# Source-only inference
# ============================================================

def run_source_inference(
    batch_tensor,
    native_hw,
    num_slices,
):

    H, W = native_hw


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

                start
                + INFERENCE_CHUNK_SIZE,

                num_slices,
            )


            chunk = batch_tensor[
                start:end
            ].to(
                DEVICE
            )


            outputs = source_model(
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


                pred_native = (
                    resize_pred_to_native(

                        pred_slice,

                        native_hw,
                    )
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
                    start + i,

                ] = pred_native


    return pred_native_slices


# ============================================================
# Prompted inference
# ============================================================

def run_prompt_inference(
    model,
    batch_tensor,
    native_hw,
    num_slices,
    prompt_id,
):

    H, W = native_hw


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

                start
                + INFERENCE_CHUNK_SIZE,

                num_slices,
            )


            chunk = batch_tensor[
                start:end
            ].to(
                DEVICE
            )


            current_batch_size = (
                chunk.shape[0]
            )


            prompt_ids = torch.full(

                (
                    current_batch_size,
                ),

                fill_value=
                    int(
                        prompt_id
                    ),

                dtype=torch.long,

                device=DEVICE,
            )


            outputs = model(

                chunk,

                prompt_ids,
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


                pred_native = (
                    resize_pred_to_native(

                        pred_slice,

                        native_hw,
                    )
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
                    start + i,

                ] = pred_native


    return pred_native_slices


# ============================================================
# Metrics
#
# EXACT same volumetric Dice / HD95 calculation as strict
# evaluation.
# ============================================================

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
        == gt_vol.shape
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


        intersection = (
            np.logical_and(

                pred_binary,
                gt_binary,

            ).sum()
        )


        dice = (

            2.0
            * intersection
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

                    voxelspacing=
                        spacing_3d,
                )

            except RuntimeError:

                hd95_value = np.nan


        result[
            f"Dice_{name}"
        ] = float(
            dice
        )


        result[
            f"HD95_{name}"
        ] = float(
            hd95_value
        )


    dice_values = [

        result[
            f"Dice_{name}"
        ]

        for name
        in STRUCTURE_NAMES
    ]


    hd95_values = [

        result[
            f"HD95_{name}"
        ]

        for name
        in STRUCTURE_NAMES
    ]


    result[
        "Mean_Dice"
    ] = float(

        np.nanmean(
            dice_values
        )
    )


    result[
        "Mean_HD95"
    ] = float(

        np.nanmean(
            hd95_values
        )
    )


    return result


# ============================================================
# FINAL held-out evaluation
# ============================================================

print(
    "\n=============================================="
)

print(
    "FINAL SOURCE vs PROMPT A/B -> C/D EVALUATION"
)

print(
    "=============================================="
)


start_time = time.time()


rows = []

latent_assignment_rows = []

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

        # ----------------------------------------------------
        # Load image and GT once
        # ----------------------------------------------------

        image_obj = nib.load(
            sample[
                "image_path"
            ]
        )


        image_vol = (
            image_obj.get_fdata(
                dtype=np.float32
            )
        )


        mask_obj = nib.load(
            sample[
                "mask_path"
            ]
        )


        mask_vol = (
            mask_obj.get_fdata(
                dtype=np.float32
            )
        )


        zooms = (

            image_obj
            .header
            .get_zooms()
        )


        spacing_3d = (

            float(
                zooms[0]
            ),

            float(
                zooms[1]
            ),

            float(
                zooms[2]
            ),
        )


        # ----------------------------------------------------
        # Patient-phase volume for frozen style assignment
        # ----------------------------------------------------

        patient_phase_volume = (

            image_vol[

                :,
                :,
                :,
                frame_idx,

            ]
        )


        latent_cluster = (
            assign_latent_cluster(
                patient_phase_volume
            )
        )


        domain = (

            "in-domain"

            if vendor in {
                "A",
                "B",
            }

            else "unseen"
        )


        latent_assignment_rows.append({

            "patient":
                patient,

            "phase":
                phase,

            "vendor":
                vendor,

            "vendor_name":
                vendor_name,

            "centre":
                centre,

            "domain":
                domain,

            "latent_cluster":
                latent_cluster,
        })


        # ----------------------------------------------------
        # Prepare segmentation input ONCE
        # ----------------------------------------------------

        (
            batch_tensor,
            native_hw,
            num_slices,

        ) = prepare_volume_tensor(

            image_vol,

            frame_idx,
        )


        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        gt_vol = mask_vol[

            :,
            :,
            :,
            frame_idx,

        ].astype(
            np.int64
        )


        assert (
            gt_vol.shape
            ==
            (
                native_hw[0],
                native_hw[1],
                num_slices,
            )
        )


        # ====================================================
        # MODEL 1: SOURCE-ONLY
        # ====================================================

        source_pred = (
            run_source_inference(

                batch_tensor,

                native_hw,

                num_slices,
            )
        )


        assert (
            source_pred.shape
            == gt_vol.shape
        )


        source_metrics = (
            compute_volume_metrics(

                source_pred,

                gt_vol,

                spacing_3d,
            )
        )


        source_row = {

            "model":
                "source_only",

            "patient":
                patient,

            "phase":
                phase,

            "vendor":
                vendor,

            "vendor_name":
                vendor_name,

            "centre":
                centre,

            "domain":
                domain,

            # Recorded for analysis only.
            # Source-only model does not use this.
            "latent_cluster":
                latent_cluster,
        }


        source_row.update(
            source_metrics
        )


        rows.append(
            source_row
        )


        # ====================================================
        # MODEL 2: GENERIC PROMPT
        #
        # Always prompt ID 0.
        # ====================================================

        generic_pred = (
            run_prompt_inference(

                generic_model,

                batch_tensor,

                native_hw,

                num_slices,

                prompt_id=0,
            )
        )


        assert (
            generic_pred.shape
            == gt_vol.shape
        )


        generic_metrics = (
            compute_volume_metrics(

                generic_pred,

                gt_vol,

                spacing_3d,
            )
        )


        generic_row = {

            "model":
                "generic_prompt",

            "patient":
                patient,

            "phase":
                phase,

            "vendor":
                vendor,

            "vendor_name":
                vendor_name,

            "centre":
                centre,

            "domain":
                domain,

            # Analysis only.
            # Generic model always uses prompt 0.
            "latent_cluster":
                latent_cluster,
        }


        generic_row.update(
            generic_metrics
        )


        rows.append(
            generic_row
        )


        # ====================================================
        # MODEL 3: LATENT-DOMAIN PROMPT
        #
        # Prompt determined by frozen source-trained
        # style pipeline.
        # ====================================================

        latent_pred = (
            run_prompt_inference(

                latent_model,

                batch_tensor,

                native_hw,

                num_slices,

                prompt_id=
                    latent_cluster,
            )
        )


        assert (
            latent_pred.shape
            == gt_vol.shape
        )


        latent_metrics = (
            compute_volume_metrics(

                latent_pred,

                gt_vol,

                spacing_3d,
            )
        )


        latent_row = {

            "model":
                "latent_domain_prompt",

            "patient":
                patient,

            "phase":
                phase,

            "vendor":
                vendor,

            "vendor_name":
                vendor_name,

            "centre":
                centre,

            "domain":
                domain,

            "latent_cluster":
                latent_cluster,
        }


        latent_row.update(
            latent_metrics
        )


        rows.append(
            latent_row
        )


    except Exception as exc:

        print(
            f"\n[WARNING] "
            f"{patient} ({phase}) "
            f"Vendor {vendor}: "
            f"{exc}"
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


# ============================================================
# Save patient-level results
#
# Expected:
# 68 patient-phase volumes x 3 models = 204 rows
# ============================================================

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
    "\nPatient-level results:",
    patient_csv,
)


if not skipped:

    assert len(
        results_df
    ) == (
        len(val_samples)
        * 3
    ), (
        "Expected exactly three model results "
        "per validation patient-phase volume."
    )


# ============================================================
# Save frozen latent assignments
# ============================================================

latent_assignment_df = (
    pd.DataFrame(
        latent_assignment_rows
    )
)


latent_assignment_csv = (
    os.path.join(

        RESULTS_DIR,

        "validation_latent_assignments.csv",
    )
)


latent_assignment_df.to_csv(

    latent_assignment_csv,

    index=False,
)


# ============================================================
# Metrics
# ============================================================

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


# ============================================================
# Model x Vendor summary
# ============================================================

vendor_summary = (

    results_df

    .groupby(
        [
            "model",
            "vendor",
        ]
    )[metric_columns]

    .mean()

    .round(4)
)


vendor_counts_summary = (

    results_df

    .groupby(
        [
            "model",
            "vendor",
        ]
    )

    .size()
)


vendor_summary.insert(

    0,

    "N_patient_phases",

    vendor_counts_summary,
)


vendor_csv = os.path.join(

    RESULTS_DIR,

    "model_vendor_summary.csv",
)


vendor_summary.to_csv(
    vendor_csv
)


print(
    "\n=============================================="
)

print(
    "MODEL x VENDOR RESULTS"
)

print(
    "=============================================="
)

print(
    vendor_summary
)


# ============================================================
# Model x Domain summary
#
# This is the MAIN comparison table.
# ============================================================

domain_summary = (

    results_df

    .groupby(
        [
            "model",
            "domain",
        ]
    )[metric_columns]

    .mean()

    .round(4)
)


domain_counts_summary = (

    results_df

    .groupby(
        [
            "model",
            "domain",
        ]
    )

    .size()
)


domain_summary.insert(

    0,

    "N_patient_phases",

    domain_counts_summary,
)


domain_csv = os.path.join(

    RESULTS_DIR,

    "model_domain_summary.csv",
)


domain_summary.to_csv(
    domain_csv
)


print(
    "\n=============================================="
)

print(
    "MODEL x DOMAIN RESULTS"
)

print(
    "=============================================="
)

print(
    domain_summary
)


# ============================================================
# Model x Phase summary
# ============================================================

phase_summary = (

    results_df

    .groupby(
        [
            "model",
            "phase",
        ]
    )[metric_columns]

    .mean()

    .round(4)
)


phase_csv = os.path.join(

    RESULTS_DIR,

    "model_phase_summary.csv",
)


phase_summary.to_csv(
    phase_csv
)


# ============================================================
# Latent-cluster assignment composition
#
# IMPORTANT:
# This is descriptive analysis only.
#
# It does not alter any model or cluster.
# ============================================================

cluster_vendor_counts = pd.crosstab(

    latent_assignment_df[
        "latent_cluster"
    ],

    latent_assignment_df[
        "vendor"
    ],
)


cluster_vendor_csv = os.path.join(

    RESULTS_DIR,

    "validation_cluster_vendor_composition.csv",
)


cluster_vendor_counts.to_csv(
    cluster_vendor_csv
)


print(
    "\n=============================================="
)

print(
    "FROZEN LATENT ASSIGNMENT x VENDOR"
)

print(
    "=============================================="
)

print(
    cluster_vendor_counts
)


# ============================================================
# Compact final comparison
#
# One row per model with:
# - A/B reference Dice / HD95
# - C/D unseen Dice / HD95
# - domain degradation
# ============================================================

comparison_rows = []


model_order = [

    "source_only",

    "generic_prompt",

    "latent_domain_prompt",
]


for model_name in model_order:

    model_df = results_df[
        results_df[
            "model"
        ] == model_name
    ]


    reference_df = model_df[
        model_df[
            "domain"
        ] == "in-domain"
    ]


    unseen_df = model_df[
        model_df[
            "domain"
        ] == "unseen"
    ]


    reference_dice = float(

        reference_df[
            "Mean_Dice"
        ].mean()
    )


    reference_hd95 = float(

        reference_df[
            "Mean_HD95"
        ].mean()
    )


    unseen_dice = float(

        unseen_df[
            "Mean_Dice"
        ].mean()
    )


    unseen_hd95 = float(

        unseen_df[
            "Mean_HD95"
        ].mean()
    )


    comparison_rows.append({

        "model":
            model_name,

        "AB_reference_mean_dice":
            reference_dice,

        "AB_reference_mean_hd95":
            reference_hd95,

        "CD_unseen_mean_dice":
            unseen_dice,

        "CD_unseen_mean_hd95":
            unseen_hd95,

        # Negative means Dice became worse on unseen.
        "dice_change_AB_to_CD":
            (
                unseen_dice
                - reference_dice
            ),

        # Positive means HD95 became worse on unseen.
        "hd95_change_AB_to_CD":
            (
                unseen_hd95
                - reference_hd95
            ),
    })


comparison_df = pd.DataFrame(
    comparison_rows
)


# ============================================================
# Improvement relative to source-only
# ============================================================

source_unseen_dice = float(

    comparison_df.loc[

        comparison_df[
            "model"
        ] == "source_only",

        "CD_unseen_mean_dice",

    ].iloc[0]
)


source_unseen_hd95 = float(

    comparison_df.loc[

        comparison_df[
            "model"
        ] == "source_only",

        "CD_unseen_mean_hd95",

    ].iloc[0]
)


comparison_df[
    "CD_dice_delta_vs_source"
] = (

    comparison_df[
        "CD_unseen_mean_dice"
    ]

    - source_unseen_dice
)


# Positive = improvement because lower HD95 is better.
comparison_df[
    "CD_hd95_improvement_vs_source"
] = (

    source_unseen_hd95

    - comparison_df[
        "CD_unseen_mean_hd95"
    ]
)


# ============================================================
# Latent vs generic improvement
# ============================================================

generic_unseen_dice = float(

    comparison_df.loc[

        comparison_df[
            "model"
        ] == "generic_prompt",

        "CD_unseen_mean_dice",

    ].iloc[0]
)


generic_unseen_hd95 = float(

    comparison_df.loc[

        comparison_df[
            "model"
        ] == "generic_prompt",

        "CD_unseen_mean_hd95",

    ].iloc[0]
)


comparison_df[
    "CD_dice_delta_vs_generic"
] = (

    comparison_df[
        "CD_unseen_mean_dice"
    ]

    - generic_unseen_dice
)


comparison_df[
    "CD_hd95_improvement_vs_generic"
] = (

    generic_unseen_hd95

    - comparison_df[
        "CD_unseen_mean_hd95"
    ]
)


comparison_df = (
    comparison_df.round(4)
)


comparison_csv = os.path.join(

    RESULTS_DIR,

    "final_model_comparison.csv",
)


comparison_df.to_csv(

    comparison_csv,

    index=False,
)


print(
    "\n=============================================="
)

print(
    "FINAL MODEL COMPARISON"
)

print(
    "=============================================="
)

print(

    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# Per-structure unseen-vendor comparison
# ============================================================

unseen_structure_summary = (

    results_df[

        results_df[
            "domain"
        ] == "unseen"

    ]

    .groupby(
        "model"
    )[
        [
            "Dice_LV",
            "Dice_Myocardium",
            "Dice_RV",
            "HD95_LV",
            "HD95_Myocardium",
            "HD95_RV",
        ]
    ]

    .mean()

    .reindex(
        model_order
    )

    .round(4)
)


unseen_structure_csv = os.path.join(

    RESULTS_DIR,

    "unseen_structure_summary.csv",
)


unseen_structure_summary.to_csv(
    unseen_structure_csv
)


print(
    "\n=============================================="
)

print(
    "C/D UNSEEN PER-STRUCTURE RESULTS"
)

print(
    "=============================================="
)

print(
    unseen_structure_summary
)


# ============================================================
# Skip report
# ============================================================

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
        len(skipped),
    )


else:

    print(
        "\nSkipped samples: 0"
    )


# ============================================================
# Save evaluation metadata
# ============================================================

evaluation_metadata = {

    "evaluation_protocol":
        (
            "Frozen A/B source training -> "
            "official validation A/B reference "
            "and C/D unseen-vendor evaluation"
        ),

    "models": [

        "source_only",
        "generic_prompt",
        "latent_domain_prompt",
    ],

    "source_checkpoint":
        SOURCE_MODEL_PATH,

    "generic_prompt_checkpoint":
        GENERIC_MODEL_PATH,

    "latent_domain_prompt_checkpoint":
        LATENT_MODEL_PATH,

    "latent_assignment":
        (
            "Frozen A/B-trained StandardScaler + "
            "PCA + KMeans; transform/predict only"
        ),

    "latent_k":
        2,

    "prompt_dim":
        PROMPT_DIM,

    "generic_prompt_id":
        0,

    "c_d_used_for_training":
        False,

    "c_d_used_for_checkpoint_selection":
        False,

    "c_d_used_for_latent_pipeline_fitting":
        False,

    "segmentation_preprocessing":
        (
            "Per-slice min-max normalization "
            "and resize to 256x256"
        ),

    "evaluation_geometry":
        (
            "Predictions resized to native "
            "in-plane resolution and reconstructed "
            "as full 3D patient-phase volumes"
        ),

    "hd95_spacing":
        (
            "Original NIfTI voxel spacing"
        ),
}


with open(

    os.path.join(
        RESULTS_DIR,
        "evaluation_metadata.json",
    ),

    "w",

) as f:

    json.dump(
        evaluation_metadata,
        f,
        indent=4,
    )


# ============================================================
# Finish
# ============================================================

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

print(
    latent_assignment_csv
)

print(
    cluster_vendor_csv
)

print(
    comparison_csv
)

print(
    unseen_structure_csv
)


print(
    "\nFINAL HELD-OUT EVALUATION COMPLETE."
)

print(
    "No fitting or checkpoint selection "
    "was performed on C/D."
)
