import os
import time
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from monai.transforms import (
    Compose,
    ResizeD
)

from monai.networks.nets import UNet

from tqdm import tqdm

from dataset import build_dataset
from mnms_dataset import MNMSDataset


# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

ROOT_DIR = "/content/OpenDataset"

MODEL_PATH = (
    "/content/drive/MyDrive/mnms_checkpoints/best_model.pth"
)

RESULTS_DIR = "evaluation_results"

BATCH_SIZE = 64

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

print(f"Using device: {DEVICE}")


# -------------------------------------------------------
# Validation Dataset
# -------------------------------------------------------

transforms = Compose([

    ResizeD(
        keys=["image"],
        spatial_size=(256, 256)
    ),

    ResizeD(
        keys=["mask"],
        spatial_size=(256, 256),
        mode="nearest"
    )

])


print("Loading validation samples...")

val_samples = build_dataset(

    split="val",
    root_dir=ROOT_DIR

)

print(
    f"Validation patients: {len(val_samples)}"
)


print("Building validation dataset...")

val_dataset = MNMSDataset(

    val_samples,
    transform=transforms

)

print(
    f"Validation slices: {len(val_dataset)}"
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS

)

print("Validation loader ready.")


# -------------------------------------------------------
# Build Model
# -------------------------------------------------------

print("Creating U-Net...")

model = UNet(

    spatial_dims=2,

    in_channels=1,

    out_channels=4,

    channels=(16, 32, 64, 128, 256),

    strides=(2, 2, 2, 2),

    num_res_units=2,

).to(DEVICE)


print("Loading trained weights...")

state_dict = torch.load(

    MODEL_PATH,

    map_location=DEVICE

)

model.load_state_dict(
    state_dict
)

model.eval()

print("Model ready.")
# -------------------------------------------------------
# Manual Dice Evaluation
# -------------------------------------------------------

print("\nStarting evaluation...")

# Background
# LV
# Myocardium
# RV

intersection = np.zeros(4, dtype=np.float64)
prediction = np.zeros(4, dtype=np.float64)
ground_truth = np.zeros(4, dtype=np.float64)

start_time = time.time()

with torch.no_grad():

    for batch_idx, (images, masks, _) in enumerate(tqdm(val_loader)):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)

        preds = torch.argmax(
            outputs,
            dim=1
        )

        preds = preds.cpu().numpy()
        masks = masks.cpu().numpy()

        for cls in range(4):

            pred_mask = preds == cls

            gt_mask = masks == cls

            intersection[cls] += np.logical_and(
                pred_mask,
                gt_mask
            ).sum()

            prediction[cls] += pred_mask.sum()

            ground_truth[cls] += gt_mask.sum()


dice = (

    2 * intersection + 1e-8

) / (

    prediction + ground_truth + 1e-8

)

elapsed = time.time() - start_time
# -------------------------------------------------------
# Results
# -------------------------------------------------------

results = pd.DataFrame({

    "Structure": [

        "Background",

        "LV",

        "Myocardium",

        "RV"

    ],

    "Dice": dice

})

mean_dice = np.mean(dice[1:])

print("\n==============================")

print("Evaluation Results")

print("==============================")

print(results)

print(f"\nMean Dice (Foreground): {mean_dice:.4f}")

print(f"Evaluation Time: {elapsed:.2f} seconds")


csv_path = os.path.join(

    RESULTS_DIR,

    "dice_results.csv"

)

results.to_csv(

    csv_path,

    index=False

)

print(f"\nResults saved to: {csv_path}")