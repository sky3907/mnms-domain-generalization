
import os
import numpy as np
import pandas as pd
import torch
import nibabel as nib
import matplotlib.pyplot as plt

from monai.transforms import Resize
from monai.networks.nets import UNet

from dataset import build_dataset


# ============================================================
# Configuration
# ============================================================

ROOT_DIR = "/content/OpenDataset"

MODEL_PATH = (
    "/content/drive/MyDrive/"
    "mnms_checkpoints_strict_ab_cd_seed42/"
    "best_model.pth"
)

SELECTION_PATH = (
    "strict_ab_cd/"
    "qualitative_case_selection.csv"
)

OUTPUT_DIR = (
    "strict_ab_cd/"
    "qualitative_results"
)

SPATIAL_SIZE = (256, 256)
INFERENCE_CHUNK_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("Device:", DEVICE)


# ============================================================
# Load model
# ============================================================

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
    map_location=DEVICE
)

model.load_state_dict(state_dict)
model.eval()

print("Strict checkpoint loaded.")


# ============================================================
# Resize
# ============================================================

resize_image = Resize(
    spatial_size=SPATIAL_SIZE
)


def normalize_slice(img):
    return (
        img - img.min()
    ) / (
        img.max()
        - img.min()
        + 1e-8
    )


# ============================================================
# Full-volume inference
# ============================================================

def predict_volume(image_vol, frame_idx):

    H, W, Z, _ = image_vol.shape

    processed = []

    for z in range(Z):

        image_slice = image_vol[
            :, :, z, frame_idx
        ].astype(np.float32)

        image_slice = normalize_slice(
            image_slice
        )

        tensor = torch.tensor(
            image_slice,
            dtype=torch.float32
        ).unsqueeze(0)

        tensor = resize_image(tensor)

        processed.append(tensor)

    batch = torch.stack(processed)

    prediction = np.zeros(
        (H, W, Z),
        dtype=np.int64
    )

    with torch.inference_mode():

        for start in range(
            0,
            Z,
            INFERENCE_CHUNK_SIZE
        ):

            end = min(
                start + INFERENCE_CHUNK_SIZE,
                Z
            )

            logits = model(
                batch[start:end].to(DEVICE)
            )

            pred = torch.argmax(
                logits,
                dim=1
            ).cpu().float()

            for i in range(
                pred.shape[0]
            ):

                native_resize = Resize(
                    spatial_size=(H, W),
                    mode="nearest"
                )

                native_pred = native_resize(
                    pred[i].unsqueeze(0)
                )

                prediction[
                    :, :, start + i
                ] = (
                    native_pred
                    .squeeze()
                    .round()
                    .long()
                    .numpy()
                )

    return prediction


# ============================================================
# Select representative anatomical slice
#
# Choose slice with largest GT foreground area.
# This avoids empty basal/apical slices for the main figure.
# ============================================================

def choose_slice(gt_volume):

    foreground = (
        gt_volume > 0
    )

    areas = foreground.sum(
        axis=(0, 1)
    )

    return int(
        np.argmax(areas)
    )


# ============================================================
# Plot
# ============================================================

def make_figure(
    image_slice,
    gt_slice,
    pred_slice,
    vendor,
    patient,
    phase,
    slice_idx,
    mean_dice,
    mean_hd95,
):

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 5)
    )

    # MRI
    axes[0].imshow(
        image_slice,
        cmap="gray"
    )

    axes[0].set_title(
        "MRI"
    )


    # GT
    axes[1].imshow(
        image_slice,
        cmap="gray"
    )

    axes[1].imshow(
        gt_slice,
        alpha=0.55,
        vmin=0,
        vmax=3
    )

    axes[1].set_title(
        "Ground Truth"
    )


    # Prediction
    axes[2].imshow(
        image_slice,
        cmap="gray"
    )

    axes[2].imshow(
        pred_slice,
        alpha=0.55,
        vmin=0,
        vmax=3
    )

    axes[2].set_title(
        "Prediction"
    )


    # Contour comparison
    axes[3].imshow(
        image_slice,
        cmap="gray"
    )

    # GT = solid
    # Prediction = dashed

    for cls in [1, 2, 3]:

        if np.any(
            gt_slice == cls
        ):

            axes[3].contour(
                gt_slice == cls,
                levels=[0.5],
                linewidths=2,
                linestyles="solid"
            )

        if np.any(
            pred_slice == cls
        ):

            axes[3].contour(
                pred_slice == cls,
                levels=[0.5],
                linewidths=2,
                linestyles="dashed"
            )

    axes[3].set_title(
        "GT (solid) vs Pred (dashed)"
    )


    for ax in axes:
        ax.axis("off")


    fig.suptitle(
        f"Vendor {vendor} | "
        f"{patient} | {phase} | "
        f"Slice {slice_idx} | "
        f"Mean Dice={mean_dice:.3f} | "
        f"Mean HD95={mean_hd95:.2f} mm",
        fontsize=13
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"Vendor_{vendor}_"
        f"{patient}_{phase}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )


# ============================================================
# Load selected cases
# ============================================================

selection_df = pd.read_csv(
    SELECTION_PATH
)

print("\nSelected cases:")
print(
    selection_df[
        [
            "vendor",
            "patient",
            "phase",
            "Mean_Dice",
            "Mean_HD95"
        ]
    ].to_string(index=False)
)


# ============================================================
# Official validation samples
# ============================================================

validation_samples = build_dataset(
    split="val",
    root_dir=ROOT_DIR
)


# ============================================================
# Generate figures
# ============================================================

for _, selected in selection_df.iterrows():

    vendor = str(
        selected["vendor"]
    )

    patient = str(
        selected["patient"]
    )

    phase = str(
        selected["phase"]
    )

    matching = [
        sample
        for sample in validation_samples
        if (
            str(sample["patient"])
            == patient
            and
            str(sample["phase"])
            == phase
            and
            str(sample["vendor"])
            == vendor
        )
    ]

    if len(matching) != 1:

        raise RuntimeError(
            f"Expected exactly one match "
            f"for Vendor {vendor}, "
            f"{patient} {phase}; "
            f"found {len(matching)}"
        )

    sample = matching[0]

    print(
        f"\nProcessing Vendor {vendor}: "
        f"{patient} {phase}"
    )


    # ----------------------------
    # Load image and GT
    # ----------------------------

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


    frame_idx = int(
        sample["frame_idx"]
    )


    gt_vol = mask_vol[
        :, :, :, frame_idx
    ].astype(np.int64)


    # ----------------------------
    # Predict full volume
    # ----------------------------

    pred_vol = predict_volume(
        image_vol,
        frame_idx
    )


    assert (
        pred_vol.shape
        ==
        gt_vol.shape
    )


    # ----------------------------
    # Choose representative slice
    # ----------------------------

    slice_idx = choose_slice(
        gt_vol
    )

    image_slice = image_vol[
        :, :, slice_idx, frame_idx
    ]

    gt_slice = gt_vol[
        :, :, slice_idx
    ]

    pred_slice = pred_vol[
        :, :, slice_idx
    ]


    print(
        "Selected slice:",
        slice_idx
    )

    print(
        "GT labels:",
        np.unique(gt_slice)
    )

    print(
        "Prediction labels:",
        np.unique(pred_slice)
    )


    # ----------------------------
    # Plot
    # ----------------------------

    make_figure(
        image_slice=image_slice,
        gt_slice=gt_slice,
        pred_slice=pred_slice,
        vendor=vendor,
        patient=patient,
        phase=phase,
        slice_idx=slice_idx,
        mean_dice=float(
            selected["Mean_Dice"]
        ),
        mean_hd95=float(
            selected["Mean_HD95"]
        ),
    )


print(
    "\n======================================"
)

print(
    "QUALITATIVE ANALYSIS COMPLETE"
)

print(
    "======================================"
)

print(
    "Output directory:",
    OUTPUT_DIR
)
