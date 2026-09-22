
import os
import json
import random

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from monai.transforms import (
    Compose,
    ResizeD,
)

from monai.networks.nets import UNet
from monai.losses import DiceCELoss

from dataset import build_dataset
from mnms_dataset import MNMSDataset


# =======================================================
# Configuration
# =======================================================

ROOT_DIR = "/content/OpenDataset"

OUTPUT_DIR = "strict_ab_cd"

CHECKPOINT_DIR = (
    "/content/drive/MyDrive/"
    "mnms_checkpoints_strict_ab_cd_seed42"
)

BATCH_SIZE = 8
NUM_EPOCHS = 10
LR = 1e-3

SEED = 42
VAL_FRACTION = 0.20

NUM_WORKERS = 0

# First run:
#     DRY_RUN = False
#
# This checks the split and dataset construction without training.
#
# After the counts are correct, change this to False.
DRY_RUN = False


# =======================================================
# Reproducibility
# =======================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

print("Using device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# =======================================================
# Transforms
# Keep identical to the original baseline
# =======================================================

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


# =======================================================
# Load OFFICIAL TRAINING set
#
# Important:
# Only official training A/B cases are used here.
#
# Official validation C/D are NOT touched during
# model training or model selection.
# =======================================================

print("\nLoading official source training cases...")

all_source_samples = build_dataset(
    split="train",
    root_dir=ROOT_DIR,
)

print(
    "Patient-phase source samples:",
    len(all_source_samples)
)


# =======================================================
# Build patient -> vendor mapping
# =======================================================

patient_vendor = {}
patient_phases = {}

for sample in all_source_samples:

    patient = sample["patient"]
    vendor = sample["vendor"]
    phase = sample["phase"]

    # Mahapatra protocol:
    # official training should contain source vendors A/B.
    if vendor not in {"A", "B"}:

        raise ValueError(
            f"Unexpected source vendor {vendor} "
            f"for patient {patient}"
        )

    if patient in patient_vendor:

        assert patient_vendor[patient] == vendor, (
            f"Patient {patient} has inconsistent vendors."
        )

    else:

        patient_vendor[patient] = vendor
        patient_phases[patient] = set()

    patient_phases[patient].add(phase)


# Each patient should have both ED and ES.
for patient, phases in patient_phases.items():

    assert phases == {"ED", "ES"}, (
        f"Patient {patient} does not contain "
        f"both ED and ES: {phases}"
    )


# =======================================================
# Group UNIQUE PATIENTS by vendor
# =======================================================

vendor_patients = {
    "A": [],
    "B": [],
}

for patient, vendor in patient_vendor.items():

    vendor_patients[vendor].append(patient)


for vendor in ["A", "B"]:

    vendor_patients[vendor] = sorted(
        vendor_patients[vendor]
    )


print("\nOfficial training patient counts")

for vendor in ["A", "B"]:

    print(
        f"Vendor {vendor}:",
        len(vendor_patients[vendor]),
        "patients"
    )


# =======================================================
# PATIENT-LEVEL internal split
#
# Split separately within A and B so both source
# vendors are represented equally.
#
# ED and ES can never cross train/validation because
# the split is done using patient IDs.
# =======================================================

rng = random.Random(SEED)

train_patient_ids = set()
internal_val_patient_ids = set()

for vendor in ["A", "B"]:

    patients = vendor_patients[vendor].copy()

    rng.shuffle(patients)

    n_val = int(
        round(
            len(patients) * VAL_FRACTION
        )
    )

    val_ids = patients[:n_val]
    train_ids = patients[n_val:]

    internal_val_patient_ids.update(
        val_ids
    )

    train_patient_ids.update(
        train_ids
    )


# =======================================================
# Safety: absolutely no patient leakage
# =======================================================

assert train_patient_ids.isdisjoint(
    internal_val_patient_ids
), "Patient leakage detected."


# =======================================================
# Convert patient IDs back to ED/ES sample dictionaries
# =======================================================

train_samples = [
    sample
    for sample in all_source_samples
    if sample["patient"] in train_patient_ids
]

internal_val_samples = [
    sample
    for sample in all_source_samples
    if sample["patient"] in internal_val_patient_ids
]


# =======================================================
# More safety checks
# =======================================================

train_sample_patients = {
    sample["patient"]
    for sample in train_samples
}

val_sample_patients = {
    sample["patient"]
    for sample in internal_val_samples
}

assert train_sample_patients == train_patient_ids
assert val_sample_patients == internal_val_patient_ids

assert train_sample_patients.isdisjoint(
    val_sample_patients
)

assert {
    sample["vendor"]
    for sample in train_samples
}.issubset({"A", "B"})

assert {
    sample["vendor"]
    for sample in internal_val_samples
}.issubset({"A", "B"})


# =======================================================
# Print split
# =======================================================

print("\n======================================")
print("STRICT A/B SOURCE PATIENT SPLIT")
print("======================================")

for vendor in ["A", "B"]:

    train_count = sum(
        patient_vendor[p] == vendor
        for p in train_patient_ids
    )

    val_count = sum(
        patient_vendor[p] == vendor
        for p in internal_val_patient_ids
    )

    print(
        f"Vendor {vendor}: "
        f"{train_count} train patients, "
        f"{val_count} internal-val patients"
    )


print()

print(
    "Total train patients:",
    len(train_patient_ids)
)

print(
    "Total internal-val patients:",
    len(internal_val_patient_ids)
)

print(
    "Train ED/ES samples:",
    len(train_samples)
)

print(
    "Internal-val ED/ES samples:",
    len(internal_val_samples)
)


# =======================================================
# Save exact split
#
# This is important for reproducibility and should
# eventually be committed to GitHub.
# =======================================================

split_manifest = {

    "protocol": (
        "Official A/B source-vendor training with "
        "patient-level internal A/B validation"
    ),

    "seed": SEED,

    "internal_validation_fraction": VAL_FRACTION,

    "train_patients": sorted(
        train_patient_ids
    ),

    "internal_validation_patients": sorted(
        internal_val_patient_ids
    ),

    "train_patient_count": len(
        train_patient_ids
    ),

    "internal_validation_patient_count": len(
        internal_val_patient_ids
    ),

    "vendor_counts": {}
}


for vendor in ["A", "B"]:

    split_manifest["vendor_counts"][vendor] = {

        "train": sum(
            patient_vendor[p] == vendor
            for p in train_patient_ids
        ),

        "internal_validation": sum(
            patient_vendor[p] == vendor
            for p in internal_val_patient_ids
        ),
    }


split_path = os.path.join(
    OUTPUT_DIR,
    "source_split.json"
)

with open(
    split_path,
    "w"
) as f:

    json.dump(
        split_manifest,
        f,
        indent=4
    )


print(
    "\nSaved patient split:",
    split_path
)


# =======================================================
# Build slice datasets
#
# MNMSDataset converts each ED/ES volume into
# individual 2D slices.
# =======================================================

print("\nBuilding training dataset...")

train_dataset = MNMSDataset(
    train_samples,
    transform=transforms,
)


print("\nBuilding internal validation dataset...")

internal_val_dataset = MNMSDataset(
    internal_val_samples,
    transform=transforms,
)


print(
    "\nTrain slices:",
    len(train_dataset)
)

print(
    "Internal validation slices:",
    len(internal_val_dataset)
)


# =======================================================
# DRY RUN
# =======================================================

if DRY_RUN:

    print("\n======================================")
    print("DRY RUN COMPLETE")
    print("======================================")

    print(
        "Split and datasets were created successfully."
    )

    print(
        "No model training was performed."
    )

    print(
        "\nIf the counts above are correct, "
        "set DRY_RUN = False and run again."
    )

    raise SystemExit


# =======================================================
# DataLoaders
#
# MNMSDataset returns:
# image, mask, metadata
# =======================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

internal_val_loader = DataLoader(
    internal_val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# =======================================================
# Model
#
# Same U-Net as the original baseline.
# =======================================================

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=4,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(DEVICE)


# =======================================================
# Loss and optimizer
#
# Same setup as baseline.
# =======================================================

loss_fn = DiceCELoss(
    to_onehot_y=True,
    softmax=True,
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
)


# =======================================================
# Validation Dice
#
# Compute LV / Myocardium / RV independently.
#
# This score is used ONLY on the internal A/B
# validation patients for checkpoint selection.
# =======================================================

def evaluate_internal_validation(
    model,
    loader,
):

    model.eval()

    classes = [1, 2, 3]

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

        for images, masks, _ in loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True,
            )

            outputs = model(images)

            preds = torch.argmax(
                outputs,
                dim=1,
            )

            for i, cls in enumerate(classes):

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
        2.0 * intersection + 1e-8
    ) / (
        prediction
        + ground_truth
        + 1e-8
    )

    mean_dice = float(
        np.mean(dice)
    )

    return {
        "Dice_LV": float(dice[0]),
        "Dice_Myocardium": float(dice[1]),
        "Dice_RV": float(dice[2]),
        "Mean_Dice": mean_dice,
    }


# =======================================================
# Training
# =======================================================

best_dice = -1.0

history = []


print("\n======================================")
print("TRAINING STRICT A/B BASELINE")
print("======================================")

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0

    for images, masks, _ in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)

        loss = loss_fn(
            outputs,
            masks.unsqueeze(1),
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()


    avg_loss = (
        running_loss /
        len(train_loader)
    )


    # -----------------------------------------------
    # Internal A/B validation ONLY
    # -----------------------------------------------

    val_metrics = evaluate_internal_validation(
        model,
        internal_val_loader,
    )

    val_dice = val_metrics[
        "Mean_Dice"
    ]


    print(
        f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]"
    )

    print(
        f"Training Loss: {avg_loss:.4f}"
    )

    print(
        "Internal A/B validation:"
    )

    print(
        f"  LV Dice         : "
        f"{val_metrics['Dice_LV']:.4f}"
    )

    print(
        f"  Myocardium Dice : "
        f"{val_metrics['Dice_Myocardium']:.4f}"
    )

    print(
        f"  RV Dice         : "
        f"{val_metrics['Dice_RV']:.4f}"
    )

    print(
        f"  Mean Dice       : "
        f"{val_dice:.4f}"
    )


    # -----------------------------------------------
    # Save training history
    # -----------------------------------------------

    history.append({

        "epoch": epoch + 1,

        "training_loss": avg_loss,

        "internal_val_dice_lv":
            val_metrics["Dice_LV"],

        "internal_val_dice_myocardium":
            val_metrics["Dice_Myocardium"],

        "internal_val_dice_rv":
            val_metrics["Dice_RV"],

        "internal_val_mean_dice":
            val_dice,
    })


    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "training_history.csv",
        ),
        index=False,
    )


    # -----------------------------------------------
    # Save full epoch checkpoint
    # -----------------------------------------------

    checkpoint = {

        "epoch": epoch + 1,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "internal_val_mean_dice":
            val_dice,

        "seed": SEED,
    }


    torch.save(
        checkpoint,
        os.path.join(
            CHECKPOINT_DIR,
            f"checkpoint_epoch_{epoch + 1}.pth",
        ),
    )


    # -----------------------------------------------
    # Select model ONLY using internal A/B
    # -----------------------------------------------

    if val_dice > best_dice:

        best_dice = val_dice

        torch.save(
            model.state_dict(),
            os.path.join(
                CHECKPOINT_DIR,
                "best_model.pth",
            ),
        )

        print(
            f"New best model: "
            f"{best_dice:.4f}"
        )


print("\n======================================")
print("TRAINING COMPLETE")
print("======================================")

print(
    f"Best internal A/B Dice: "
    f"{best_dice:.4f}"
)

print(
    "Best model:"
)

print(
    os.path.join(
        CHECKPOINT_DIR,
        "best_model.pth",
    )
)

print(
    "\nTraining history:"
)

print(
    os.path.join(
        OUTPUT_DIR,
        "training_history.csv",
    )
)
