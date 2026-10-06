
import os
import json
import random
import joblib

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from scipy.stats import skew, kurtosis

from monai.transforms import (
    Compose,
    ResizeD,
)

from monai.networks.nets import UNet
from monai.losses import DiceCELoss

from dataset import build_dataset
from mnms_dataset import MNMSDataset


# ============================================================
# Configuration
# ============================================================

ROOT_DIR = "/content/OpenDataset"

# Frozen strict A/B split
SPLIT_PATH = (
    "strict_ab_cd/source_split.json"
)

# Frozen latent-domain discovery
LATENT_DIR = (
    "latent_domain_discovery"
)

OUTPUT_DIR = (
    "prompt_baselines"
)

CHECKPOINT_ROOT = (
    "/content/drive/MyDrive/"
    "mnms_prompt_baselines_seed42"
)

BATCH_SIZE = 8
NUM_EPOCHS = 10
LR = 1e-3

SEED = 42

NUM_WORKERS = 0

# Minimal prompt representation.
PROMPT_DIM = 4

# K=2 was frozen during latent-domain discovery.
NUM_PROMPTS = 2


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

os.makedirs(
    CHECKPOINT_ROOT,
    exist_ok=True,
)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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
# Reproducibility
# ============================================================

def reset_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


reset_seed()


# ============================================================
# Transforms
#
# EXACTLY the same preprocessing as strict baseline.
# ============================================================

transforms = Compose([

    ResizeD(
        keys=["image"],
        spatial_size=(256, 256),
    ),

    ResizeD(
        keys=["mask"],
        spatial_size=(256, 256),
        mode="nearest",
    ),

])


# ============================================================
# Load FROZEN strict A/B patient split
#
# We do NOT recreate or modify the split.
# ============================================================

print(
    "\nLoading frozen strict A/B split..."
)

with open(
    SPLIT_PATH,
    "r",
) as f:

    split_info = json.load(f)


train_patient_ids = set(
    split_info[
        "train_patients"
    ]
)

internal_val_patient_ids = set(
    split_info[
        "internal_validation_patients"
    ]
)


assert train_patient_ids.isdisjoint(
    internal_val_patient_ids
)

assert len(train_patient_ids) == 120

assert len(
    internal_val_patient_ids
) == 30


# ============================================================
# Load official A/B training set ONLY
#
# C/D is never loaded in this script.
# ============================================================

all_source_samples = build_dataset(
    split="train",
    root_dir=ROOT_DIR,
)


train_samples = [

    sample

    for sample in all_source_samples

    if sample["patient"]
    in train_patient_ids

]


internal_val_samples = [

    sample

    for sample in all_source_samples

    if sample["patient"]
    in internal_val_patient_ids

]


# ============================================================
# Safety checks
# ============================================================

assert len(train_samples) == 240

assert len(
    internal_val_samples
) == 60


assert {
    sample["vendor"]
    for sample in train_samples
}.issubset(
    {"A", "B"}
)


assert {
    sample["vendor"]
    for sample in internal_val_samples
}.issubset(
    {"A", "B"}
)


print(
    "Train patients:",
    len(train_patient_ids),
)

print(
    "Internal-val patients:",
    len(internal_val_patient_ids),
)

print(
    "Train patient-phase volumes:",
    len(train_samples),
)

print(
    "Internal-val patient-phase volumes:",
    len(internal_val_samples),
)


# ============================================================
# Load FROZEN latent-domain pipeline
#
# All of these were fit using ONLY the 120 A/B training
# patients.
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

print(
    "Style features:",
    feature_columns,
)


# ============================================================
# Load FROZEN cluster assignments for A/B training volumes
# ============================================================

assignment_df = pd.read_csv(

    os.path.join(
        LATENT_DIR,
        "source_cluster_assignments.csv",
    ),

    dtype={
        "patient": str,
        "phase": str,
    },
)


train_latent_map = {

    (
        str(row.patient),
        str(row.phase),
    ):
        int(
            row.latent_cluster
        )

    for row
    in assignment_df.itertuples(
        index=False
    )
}


expected_train_keys = {

    (
        str(sample["patient"]),
        str(sample["phase"]),
    )

    for sample
    in train_samples
}


assert (
    set(
        train_latent_map.keys()
    )
    == expected_train_keys
), (
    "Frozen training cluster assignments "
    "do not exactly match the strict "
    "source-training split."
)


print(
    "\nFrozen training latent-cluster counts:"
)

print(
    assignment_df[
        "latent_cluster"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# EXACT style preprocessing used during latent discovery
#
# This is required to assign latent prompts to the internal
# A/B validation volumes without refitting anything.
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

    iqr = q75 - q25

    normalized = (
        clipped - median
    ) / (
        iqr + 1e-8
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
# EXACT 11 style features from frozen discovery
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
# Assign latent domains WITHOUT refitting
#
# scaler.transform
# pca.transform
# kmeans.predict
#
# No fit() or fit_transform() is allowed here.
# ============================================================

def assign_frozen_latent_clusters(
    samples,
):

    rows = []

    X = []


    for sample in samples:

        image_obj = nib.load(
            sample[
                "image_path"
            ]
        )

        image = image_obj.get_fdata(
            dtype=np.float32
        )

        frame_idx = sample[
            "frame_idx"
        ]

        volume = image[
            :,
            :,
            :,
            frame_idx,
        ]

        features = (
            extract_style_features(
                volume
            )
        )

        X.append([

            features[name]

            for name
            in feature_columns

        ])


        rows.append({

            "patient":
                str(
                    sample[
                        "patient"
                    ]
                ),

            "phase":
                str(
                    sample[
                        "phase"
                    ]
                ),

            # Metadata saved only for analysis.
            "vendor":
                sample[
                    "vendor"
                ],

            "centre":
                sample[
                    "centre"
                ],
        })


    X = np.asarray(
        X,
        dtype=np.float64,
    )


    assert np.all(
        np.isfinite(X)
    )


    # IMPORTANT:
    # transform only.
    X_scaled = scaler.transform(
        X
    )

    X_pca = pca.transform(
        X_scaled
    )

    X_pca = X_pca[
        :,
        :n_pca_components,
    ]


    # IMPORTANT:
    # predict only.
    labels = (
        kmeans.predict(
            X_pca
        )
        .astype(int)
    )


    assignment_map = {}


    for row, label in zip(
        rows,
        labels,
    ):

        key = (
            row["patient"],
            row["phase"],
        )

        assignment_map[
            key
        ] = int(label)

        row[
            "latent_cluster"
        ] = int(label)


    return (
        assignment_map,
        pd.DataFrame(
            rows
        ),
    )


# ============================================================
# Assign INTERNAL A/B validation latent prompts
#
# The clustering model is frozen.
# No validation labels are involved.
# ============================================================

print(
    "\nAssigning internal A/B validation "
    "using frozen latent-domain pipeline..."
)


(
    internal_val_latent_map,
    internal_val_assignment_df,

) = assign_frozen_latent_clusters(
    internal_val_samples
)


expected_val_keys = {

    (
        str(
            sample["patient"]
        ),
        str(
            sample["phase"]
        ),
    )

    for sample
    in internal_val_samples
}


assert (
    set(
        internal_val_latent_map.keys()
    )
    == expected_val_keys
)


internal_val_assignment_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "internal_ab_validation_"
        "cluster_assignments.csv",
    ),

    index=False,
)


print(
    "\nInternal A/B validation "
    "cluster counts:"
)

print(
    internal_val_assignment_df[
        "latent_cluster"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# Build base slice datasets
#
# EXACT same MNMSDataset and transforms as source-only model.
# ============================================================

print(
    "\nBuilding base slice datasets..."
)


train_base_dataset = MNMSDataset(
    train_samples,
    transform=transforms,
)


internal_val_base_dataset = MNMSDataset(
    internal_val_samples,
    transform=transforms,
)


print(
    "Train slices:",
    len(
        train_base_dataset
    ),
)

print(
    "Internal validation slices:",
    len(
        internal_val_base_dataset
    ),
)


# ============================================================
# Prompt-aware dataset wrapper
#
# generic:
#     every slice gets prompt ID 0
#
# latent:
#     every slice gets the frozen cluster ID belonging
#     to its patient-phase volume.
# ============================================================

class PromptSliceDataset(
    Dataset
):

    def __init__(
        self,
        base_dataset,
        mode,
        latent_map,
    ):

        if mode not in {
            "generic",
            "latent",
        }:

            raise ValueError(
                f"Unknown prompt mode: {mode}"
            )

        self.base_dataset = (
            base_dataset
        )

        self.mode = mode

        self.latent_map = (
            latent_map
        )


    def __len__(
        self,
    ):

        return len(
            self.base_dataset
        )


    def __getitem__(
        self,
        idx,
    ):

        (
            image,
            mask,
            metadata,

        ) = self.base_dataset[
            idx
        ]


        if self.mode == "generic":

            # Every image receives
            # exactly the same prompt.
            prompt_id = 0


        else:

            patient = str(
                metadata[
                    "patient"
                ]
            )

            phase = str(
                metadata[
                    "phase"
                ]
            )

            key = (
                patient,
                phase,
            )


            if key not in self.latent_map:

                raise KeyError(
                    "No frozen latent prompt "
                    f"for {key}"
                )


            prompt_id = int(
                self.latent_map[
                    key
                ]
            )


        return (
            image,
            mask,
            torch.tensor(
                prompt_id,
                dtype=torch.long,
            ),
            metadata,
        )


# ============================================================
# Minimal Prompted U-Net
#
# Prompt mechanism:
#
# 1. Learn a small prompt embedding.
# 2. Expand it spatially.
# 3. Concatenate it with the MRI image.
# 4. Feed the result into the same MONAI U-Net architecture.
#
# Generic and latent models have EXACTLY the same
# architecture and parameter count.
#
# Generic:
#     only prompt ID 0 is presented.
#
# Latent:
#     prompt ID 0 or 1 according to frozen cluster.
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

        # [B, prompt_dim]
        prompt = (
            self.prompt_embedding(
                prompt_ids
            )
        )


        # [B, prompt_dim, 1, 1]
        prompt = prompt[
            :,
            :,
            None,
            None,
        ]


        # [B, prompt_dim, H, W]
        prompt = prompt.expand(

            -1,
            -1,

            images.shape[-2],
            images.shape[-1],
        )


        # [B, 1 + prompt_dim, H, W]
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
# Internal A/B validation Dice
#
# EXACT same calculation as strict baseline.
# ============================================================

def evaluate_internal_validation(
    model,
    loader,
):

    model.eval()


    classes = [
        1,
        2,
        3,
    ]


    intersection = np.zeros(
        3,
        dtype=np.float64,
    )

    prediction = np.zeros(
        3,
        dtype=np.float64,
    )

    ground_truth = np.zeros(
        3,
        dtype=np.float64,
    )


    with torch.inference_mode():

        for (
            images,
            masks,
            prompt_ids,
            _,

        ) in loader:


            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True,
            )

            prompt_ids = (
                prompt_ids.to(
                    DEVICE,
                    non_blocking=True,
                )
            )


            outputs = model(
                images,
                prompt_ids,
            )


            preds = torch.argmax(
                outputs,
                dim=1,
            )


            for i, cls in enumerate(
                classes
            ):

                pred_binary = (
                    preds == cls
                )

                gt_binary = (
                    masks == cls
                )


                intersection[i] += (

                    torch.logical_and(
                        pred_binary,
                        gt_binary,
                    )

                    .sum()

                    .item()
                )


                prediction[i] += (

                    pred_binary

                    .sum()

                    .item()
                )


                ground_truth[i] += (

                    gt_binary

                    .sum()

                    .item()
                )


    dice = (

        2.0
        * intersection
        + 1e-8

    ) / (

        prediction
        + ground_truth
        + 1e-8
    )


    return {

        "Dice_LV":
            float(
                dice[0]
            ),

        "Dice_Myocardium":
            float(
                dice[1]
            ),

        "Dice_RV":
            float(
                dice[2]
            ),

        "Mean_Dice":
            float(
                np.mean(
                    dice
                )
            ),
    }


# ============================================================
# Train one prompt variant
# ============================================================

def train_variant(
    variant_name,
    prompt_mode,
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "TRAINING:",
        variant_name,
    )

    print(
        "=" * 60
    )


    # --------------------------------------------------------
    # Reset RNG before EACH model.
    #
    # This makes generic and latent prompt models start from
    # the same random state and receive the same shuffle order.
    # --------------------------------------------------------

    reset_seed(
        SEED
    )


    train_dataset = (
        PromptSliceDataset(

            train_base_dataset,

            mode=prompt_mode,

            latent_map=
                train_latent_map,
        )
    )


    val_dataset = (
        PromptSliceDataset(

            internal_val_base_dataset,

            mode=prompt_mode,

            latent_map=
                internal_val_latent_map,
        )
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = PromptedUNet().to(
        DEVICE
    )


    # --------------------------------------------------------
    # Same DiceCE loss as strict baseline
    # --------------------------------------------------------

    loss_fn = DiceCELoss(

        to_onehot_y=True,

        softmax=True,
    )


    # --------------------------------------------------------
    # Same optimizer and LR
    # --------------------------------------------------------

    optimizer = (
        torch.optim.Adam(

            model.parameters(),

            lr=LR,
        )
    )


    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=
            NUM_WORKERS,

        pin_memory=
            torch.cuda.is_available(),
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=
            NUM_WORKERS,

        pin_memory=
            torch.cuda.is_available(),
    )


    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    variant_output_dir = (
        os.path.join(
            OUTPUT_DIR,
            variant_name,
        )
    )


    variant_checkpoint_dir = (
        os.path.join(
            CHECKPOINT_ROOT,
            variant_name,
        )
    )


    os.makedirs(
        variant_output_dir,
        exist_ok=True,
    )

    os.makedirs(
        variant_checkpoint_dir,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_dice = -1.0

    best_epoch = None

    history = []


    for epoch in range(
        NUM_EPOCHS
    ):

        model.train()

        running_loss = 0.0


        for (
            images,
            masks,
            prompt_ids,
            _,

        ) in train_loader:


            images = images.to(

                DEVICE,

                non_blocking=True,
            )


            masks = masks.to(

                DEVICE,

                non_blocking=True,
            )


            prompt_ids = (
                prompt_ids.to(

                    DEVICE,

                    non_blocking=True,
                )
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            outputs = model(
                images,
                prompt_ids,
            )


            loss = loss_fn(

                outputs,

                masks.unsqueeze(1),
            )


            loss.backward()

            optimizer.step()


            running_loss += (
                loss.item()
            )


        avg_loss = (

            running_loss
            / len(
                train_loader
            )
        )


        # ----------------------------------------------------
        # Internal A/B validation ONLY
        # ----------------------------------------------------

        val_metrics = (
            evaluate_internal_validation(

                model,

                val_loader,
            )
        )


        val_dice = (
            val_metrics[
                "Mean_Dice"
            ]
        )


        print(
            f"\nEpoch "
            f"[{epoch + 1}/{NUM_EPOCHS}]"
        )

        print(
            f"Training Loss: "
            f"{avg_loss:.4f}"
        )

        print(
            "Internal A/B validation:"
        )

        print(
            "  LV Dice         : "
            f"{val_metrics['Dice_LV']:.4f}"
        )

        print(
            "  Myocardium Dice : "
            f"{val_metrics['Dice_Myocardium']:.4f}"
        )

        print(
            "  RV Dice         : "
            f"{val_metrics['Dice_RV']:.4f}"
        )

        print(
            "  Mean Dice       : "
            f"{val_dice:.4f}"
        )


        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history.append({

            "epoch":
                epoch + 1,

            "training_loss":
                avg_loss,

            "internal_val_dice_lv":
                val_metrics[
                    "Dice_LV"
                ],

            "internal_val_dice_myocardium":
                val_metrics[
                    "Dice_Myocardium"
                ],

            "internal_val_dice_rv":
                val_metrics[
                    "Dice_RV"
                ],

            "internal_val_mean_dice":
                val_dice,
        })


        pd.DataFrame(
            history
        ).to_csv(

            os.path.join(
                variant_output_dir,
                "training_history.csv",
            ),

            index=False,
        )


        # ----------------------------------------------------
        # Full epoch checkpoint
        # ----------------------------------------------------

        checkpoint = {

            "epoch":
                epoch + 1,

            "variant":
                variant_name,

            "prompt_mode":
                prompt_mode,

            "prompt_dim":
                PROMPT_DIM,

            "num_prompts":
                NUM_PROMPTS,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "internal_val_mean_dice":
                val_dice,

            "seed":
                SEED,
        }


        torch.save(

            checkpoint,

            os.path.join(

                variant_checkpoint_dir,

                f"checkpoint_epoch_"
                f"{epoch + 1}.pth",
            ),
        )


        # ----------------------------------------------------
        # Model selection ONLY using internal A/B
        # ----------------------------------------------------

        if val_dice > best_dice:

            best_dice = (
                val_dice
            )

            best_epoch = (
                epoch + 1
            )


            torch.save(

                model.state_dict(),

                os.path.join(
                    variant_checkpoint_dir,
                    "best_model.pth",
                ),
            )


            print(
                f"New best "
                f"{variant_name} model: "
                f"{best_dice:.4f}"
            )


    # --------------------------------------------------------
    # Variant complete
    # --------------------------------------------------------

    print(
        "\nTraining complete:",
        variant_name,
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best internal A/B Dice:",
        f"{best_dice:.4f}",
    )


    return {

        "variant":
            variant_name,

        "prompt_mode":
            prompt_mode,

        "best_epoch":
            best_epoch,

        "best_internal_ab_mean_dice":
            best_dice,

        "best_model_path":
            os.path.join(
                variant_checkpoint_dir,
                "best_model.pth",
            ),
    }


# ============================================================
# Experiment 1:
# Generic prompt
#
# Every image receives prompt ID 0.
# ============================================================

generic_result = train_variant(

    variant_name=
        "generic_prompt",

    prompt_mode=
        "generic",
)


# ============================================================
# Experiment 2:
# Latent-domain prompt
#
# Prompt ID = frozen latent cluster 0 or 1.
# ============================================================

latent_result = train_variant(

    variant_name=
        "latent_domain_prompt",

    prompt_mode=
        "latent",
)


# ============================================================
# Save training comparison
# ============================================================

summary_df = pd.DataFrame([

    generic_result,

    latent_result,

])


summary_path = os.path.join(

    OUTPUT_DIR,

    "prompt_training_summary.csv",
)


summary_df.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# Experiment metadata
# ============================================================

experiment_metadata = {

    "protocol":
        (
            "Frozen A/B source training "
            "with internal A/B model selection"
        ),

    "source_only_reference_checkpoint":
        (
            "/content/drive/MyDrive/"
            "mnms_checkpoints_strict_ab_cd_seed42/"
            "best_model.pth"
        ),

    "prompt_mechanism":
        (
            "Learned prompt embedding expanded "
            "spatially and concatenated with the "
            "input MRI as constant prompt channels"
        ),

    "prompt_dim":
        PROMPT_DIM,

    "num_prompt_embeddings":
        NUM_PROMPTS,

    "generic_prompt":
        (
            "All samples use prompt ID 0"
        ),

    "latent_domain_prompt":
        (
            "Samples use frozen latent-domain "
            "cluster ID 0 or 1"
        ),

    "latent_assignment_pipeline":
        (
            "Frozen StandardScaler + PCA + KMeans "
            "fit only on A/B training data"
        ),

    "c_d_used_for_training_or_model_selection":
        False,

    "batch_size":
        BATCH_SIZE,

    "epochs":
        NUM_EPOCHS,

    "learning_rate":
        LR,

    "loss":
        "DiceCELoss",

    "optimizer":
        "Adam",

    "seed":
        SEED,
}


with open(

    os.path.join(
        OUTPUT_DIR,
        "prompt_experiment_metadata.json",
    ),

    "w",

) as f:

    json.dump(
        experiment_metadata,
        f,
        indent=4,
    )


# ============================================================
# Final summary
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "PROMPT BASELINE TRAINING COMPLETE"
)

print(
    "=" * 60
)


print(
    summary_df.to_string(
        index=False
    )
)


print(
    "\nSummary saved to:",
    summary_path,
)


print(
    "\nC/D has NOT been loaded "
    "or used by this script."
)
