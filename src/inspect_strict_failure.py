
import os
import math
import numpy as np
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

PATIENT = "N2O7U5"
PHASE = "ES"
VENDOR = "D"

OUTPUT_DIR = (
    "strict_ab_cd/"
    "failure_analysis/"
    "N2O7U5_ES"
)

SPATIAL_SIZE = (256, 256)

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
# Model
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

print("Checkpoint loaded.")


# ============================================================
# Helpers
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


def predict_volume(image_vol, frame_idx):

    H, W, Z, _ = image_vol.shape

    tensors = []

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

        tensor = resize_image(
            tensor
        )

        tensors.append(
            tensor
        )

    batch = torch.stack(
        tensors
    )

    prediction = np.zeros(
        (H, W, Z),
        dtype=np.int64
    )

    with torch.inference_mode():

        outputs = model(
            batch.to(DEVICE)
        )

        pred = torch.argmax(
            outputs,
            dim=1
        ).cpu().float()

    native_resize = Resize(
        spatial_size=(H, W),
        mode="nearest"
    )

    for z in range(Z):

        native_pred = native_resize(
            pred[z].unsqueeze(0)
        )

        prediction[:, :, z] = (
            native_pred
            .squeeze()
            .round()
            .long()
            .numpy()
        )

    return prediction


# ============================================================
# Find case
# ============================================================

val_samples = build_dataset(
    split="val",
    root_dir=ROOT_DIR
)

matches = [
    sample
    for sample in val_samples
    if (
        str(sample["patient"]) == PATIENT
        and
        str(sample["phase"]) == PHASE
        and
        str(sample["vendor"]) == VENDOR
    )
]

if len(matches) != 1:

    raise RuntimeError(
        f"Expected 1 case, found {len(matches)}"
    )

sample = matches[0]

print(
    f"Inspecting Vendor {VENDOR}: "
    f"{PATIENT} {PHASE}"
)


# ============================================================
# Load volume
# ============================================================

image_obj = nib.load(
    sample["image_path"]
)

mask_obj = nib.load(
    sample["mask_path"]
)

image_vol = image_obj.get_fdata(
    dtype=np.float32
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


# ============================================================
# Predict
# ============================================================

pred_vol = predict_volume(
    image_vol,
    frame_idx
)

assert pred_vol.shape == gt_vol.shape

Z = gt_vol.shape[2]

print("Number of slices:", Z)


# ============================================================
# Print per-slice foreground information
# ============================================================

print(
    "\nSlice | GT pixels | Pred pixels | "
    "GT LV | Pred LV"
)

print("-" * 55)

for z in range(Z):

    gt_fg = int(
        (gt_vol[:, :, z] > 0).sum()
    )

    pred_fg = int(
        (pred_vol[:, :, z] > 0).sum()
    )

    gt_lv = int(
        (gt_vol[:, :, z] == 1).sum()
    )

    pred_lv = int(
        (pred_vol[:, :, z] == 1).sum()
    )

    print(
        f"{z:5d} | "
        f"{gt_fg:9d} | "
        f"{pred_fg:11d} | "
        f"{gt_lv:5d} | "
        f"{pred_lv:7d}"
    )


# ============================================================
# Whole-volume montage
#
# Each slice:
# MRI + GT contour + prediction contour
# ============================================================

cols = 4
rows = math.ceil(
    Z / cols
)

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(16, 4 * rows)
)

axes = np.array(
    axes
).reshape(-1)


for z in range(Z):

    ax = axes[z]

    image_slice = image_vol[
        :, :, z, frame_idx
    ]

    gt_slice = gt_vol[
        :, :, z
    ]

    pred_slice = pred_vol[
        :, :, z
    ]

    ax.imshow(
        image_slice,
        cmap="gray"
    )


    # Ground truth = solid
    for cls in [1, 2, 3]:

        if np.any(
            gt_slice == cls
        ):

            ax.contour(
                gt_slice == cls,
                levels=[0.5],
                linewidths=2,
                linestyles="solid"
            )


    # Prediction = dashed
    for cls in [1, 2, 3]:

        if np.any(
            pred_slice == cls
        ):

            ax.contour(
                pred_slice == cls,
                levels=[0.5],
                linewidths=2,
                linestyles="dashed"
            )


    gt_fg = int(
        (gt_slice > 0).sum()
    )

    pred_fg = int(
        (pred_slice > 0).sum()
    )

    ax.set_title(
        f"Slice {z}\n"
        f"GT={gt_fg}, Pred={pred_fg}"
    )

    ax.axis("off")


for i in range(
    Z,
    len(axes)
):

    axes[i].axis(
        "off"
    )


fig.suptitle(
    "Vendor D | N2O7U5 | ES\n"
    "GT = solid contours | "
    "Prediction = dashed contours\n"
    "Mean Dice = 0.809 | "
    "Mean HD95 = 58.89 mm | "
    "LV HD95 = 146.57 mm",
    fontsize=15
)

plt.tight_layout()

montage_path = os.path.join(
    OUTPUT_DIR,
    "N2O7U5_ES_all_slices.png"
)

plt.savefig(
    montage_path,
    dpi=180,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSaved montage:",
    montage_path
)


# ============================================================
# Save individual slices
# ============================================================

for z in range(Z):

    image_slice = image_vol[
        :, :, z, frame_idx
    ]

    gt_slice = gt_vol[
        :, :, z
    ]

    pred_slice = pred_vol[
        :, :, z
    ]


    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4)
    )


    # MRI
    axes[0].imshow(
        image_slice,
        cmap="gray"
    )

    axes[0].set_title(
        f"MRI - Slice {z}"
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


    # Overlay contours
    axes[3].imshow(
        image_slice,
        cmap="gray"
    )

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
        "GT solid / Pred dashed"
    )


    for ax in axes:
        ax.axis("off")


    plt.tight_layout()

    slice_path = os.path.join(
        OUTPUT_DIR,
        f"slice_{z:02d}.png"
    )

    plt.savefig(
        slice_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


print(
    "Individual slice figures saved."
)
# ============================================================
# Compact failure summary: Slice 0 vs Slice 7
# ============================================================

failure_slices = [0, 7]

fig, axes = plt.subplots(
    2,
    4,
    figsize=(16, 8)
)

for row, z in enumerate(failure_slices):

    image_slice = image_vol[:, :, z, frame_idx]
    gt_slice = gt_vol[:, :, z]
    pred_slice = pred_vol[:, :, z]

    # MRI
    axes[row, 0].imshow(
        image_slice,
        cmap="gray"
    )
    axes[row, 0].set_title(
        f"Slice {z}: MRI"
    )

    # Ground truth
    axes[row, 1].imshow(
        image_slice,
        cmap="gray"
    )
    axes[row, 1].imshow(
        gt_slice,
        alpha=0.55,
        vmin=0,
        vmax=3
    )
    axes[row, 1].set_title(
        "Ground Truth"
    )

    # Prediction
    axes[row, 2].imshow(
        image_slice,
        cmap="gray"
    )
    axes[row, 2].imshow(
        pred_slice,
        alpha=0.55,
        vmin=0,
        vmax=3
    )
    axes[row, 2].set_title(
        "Prediction"
    )

    # Contours
    axes[row, 3].imshow(
        image_slice,
        cmap="gray"
    )

    for cls in [1, 2, 3]:

        if np.any(gt_slice == cls):
            axes[row, 3].contour(
                gt_slice == cls,
                levels=[0.5],
                linewidths=2,
                linestyles="solid"
            )

        if np.any(pred_slice == cls):
            axes[row, 3].contour(
                pred_slice == cls,
                levels=[0.5],
                linewidths=2,
                linestyles="dashed"
            )

    axes[row, 3].set_title(
        "GT solid / Pred dashed"
    )

    for col in range(4):
        axes[row, col].axis("off")


fig.suptitle(
    "Vendor D | N2O7U5 | ES\n"
    "Peripheral failure (Slice 0) vs central segmentation (Slice 7)\n"
    "Mean Dice = 0.809 | Mean HD95 = 58.89 mm | LV HD95 = 146.57 mm",
    fontsize=14
)

plt.tight_layout()

summary_path = os.path.join(
    OUTPUT_DIR,
    "N2O7U5_ES_failure_summary.png"
)

plt.savefig(
    summary_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved failure summary:",
    summary_path
)
