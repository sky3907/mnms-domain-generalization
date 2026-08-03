import os

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

import torch

from monai.transforms import Resize
from monai.networks.nets import UNet

from dataset import build_dataset


# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

ROOT_DIR = "/content/OpenDataset"

MODEL_PATH = (
    "/content/drive/MyDrive/mnms_checkpoints/best_model.pth"
)

RESULTS_DIR = "qualitative_results"

SPATIAL_SIZE = (256, 256)

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
print("Loading model...")

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

print("Model ready.")
resize_to_model = Resize(
    spatial_size=SPATIAL_SIZE
)


def resize_to_native(pred, native_hw):

    resize = Resize(
        spatial_size=native_hw,
        mode="nearest"
    )

    return resize(pred)
# -------------------------------------------------------
# Select Patient
# -------------------------------------------------------

TARGET_PATIENT = "P8V0Y7"
TARGET_PHASE = "ES"

samples = build_dataset(
    split="val",
    root_dir=ROOT_DIR
)

sample = None

for s in samples:

    if (
        s["patient"] == TARGET_PATIENT
        and s["phase"] == TARGET_PHASE
    ):

        sample = s
        break

if sample is None:

    raise ValueError(
        "Patient not found."
    )

print(
    f"Loaded {TARGET_PATIENT} ({TARGET_PHASE})"
)
# -------------------------------------------------------
# Load Volume
# -------------------------------------------------------

image_obj = nib.load(
    sample["image_path"]
)

mask_obj = nib.load(
    sample["mask_path"]
)

image = image_obj.get_fdata(
    dtype=np.float32
)

mask = mask_obj.get_fdata(
    dtype=np.float32
)

frame_idx = sample["frame_idx"]

image = image[:, :, :, frame_idx]

mask = mask[:, :, :, frame_idx]

native_hw = (
    image.shape[0],
    image.shape[1]
)

print(
    f"Volume shape: {image.shape}"
)
# -------------------------------------------------------
# Run Inference
# -------------------------------------------------------

prediction_volume = np.zeros(

    image.shape,

    dtype=np.int64

)

with torch.inference_mode():

    for slice_idx in range(image.shape[2]):

        img = image[:, :, slice_idx]

        img = (

            img - img.min()

        ) / (

            img.max() - img.min() + 1e-8

        )

        img = torch.tensor(

            img,

            dtype=torch.float32

        ).unsqueeze(0)

        img = resize_to_model(

            img

        )

        img = img.unsqueeze(0).to(DEVICE)

        output = model(

            img

        )

        pred = torch.argmax(

            output,

            dim=1

        )

        pred = pred.cpu().float()

        pred = resize_to_native(

            pred,

            native_hw

        )

        pred = (

            pred.squeeze(0)

                .long()

                .numpy()

        )

        prediction_volume[:, :, slice_idx] = pred

print("Inference complete.")
assert prediction_volume.shape == mask.shape

print("Prediction volume shape:", prediction_volume.shape)
print(np.unique(mask))
print(np.unique(prediction_volume))
# -------------------------------------------------------
# Choose Best Slice
# -------------------------------------------------------

best_slice = 0
largest_area = 0

for s in range(mask.shape[2]):

    area = np.sum(mask[:, :, s] == 2)

    if area > largest_area:

        largest_area = area
        best_slice = s

print(f"Selected slice: {best_slice}")
# -------------------------------------------------------
# Create Qualitative Figure
# -------------------------------------------------------

# -------------------------------------------------------
# Save Every Slice
# -------------------------------------------------------

patient_dir = os.path.join(
    RESULTS_DIR,
    f"{TARGET_PATIENT}_{TARGET_PHASE}"
)

os.makedirs(
    patient_dir,
    exist_ok=True
)

for s in range(image.shape[2]):

    image_slice = image[:, :, s]
    gt_slice = mask[:, :, s]
    pred_slice = prediction_volume[:, :, s]

    plt.figure(figsize=(18,5))

    # MRI
    plt.subplot(1,4,1)
    plt.imshow(image_slice, cmap="gray")
    plt.title(f"MRI (Slice {s})")
    plt.axis("off")

    # Ground Truth
    plt.subplot(1,4,2)
    plt.imshow(gt_slice, cmap="jet", vmin=0, vmax=3)
    plt.title("Ground Truth")
    plt.axis("off")

    # Prediction
    plt.subplot(1,4,3)
    plt.imshow(pred_slice, cmap="jet", vmin=0, vmax=3)
    plt.title("Prediction")
    plt.axis("off")

    # Overlay
    plt.subplot(1,4,4)

    plt.imshow(image_slice, cmap="gray")

    for cls in [1,2,3]:

        plt.contour(
            gt_slice == cls,
            levels=[0.5],
            colors=["lime"],
            linewidths=2
        )

        plt.contour(
            pred_slice == cls,
            levels=[0.5],
            colors=["red"],
            linewidths=1.5
        )

    plt.title("GT (Green) vs Prediction (Red)")
    plt.axis("off")

    plt.tight_layout()

    save_path = os.path.join(
        patient_dir,
        f"slice_{s:02d}.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print(f"\nSaved all slices to:\n{patient_dir}")