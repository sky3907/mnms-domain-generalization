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


# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

ROOT_DIR = "/content/OpenDataset"

MODEL_PATH = "/content/drive/MyDrive/mnms_checkpoints/best_model.pth"

RESULTS_DIR = "volume_evaluation_results"

SPATIAL_SIZE = (256, 256)

# Slices are fed to the model in chunks along the batch dimension
# to avoid pushing an entire volume through at once on limited GPU memory.
INFERENCE_CHUNK_SIZE = 16

# If True: prints detailed shape/label info and plots MRI/GT/prediction for
# every slice of the FIRST patient only, then stops the evaluation loop
# after that one patient (via `break`). Useful for visually sanity-checking
# geometry/labels before committing to a full ~68-patient run.
# Set to False for a real full evaluation run.
DEBUG_FIRST_PATIENT_ONLY = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")


# -------------------------------------------------------
# Resize transforms
# -------------------------------------------------------
# These mirror mnms_dataset.py / train.py / evaluate.py exactly:
#   - image key: ResizeD(spatial_size=(256, 256))  -> default mode ("area")
#   - mask key : ResizeD(spatial_size=(256, 256), mode="nearest")
#
# Here we apply the equivalent non-dict Resize transforms manually,
# since we only ever resize one tensor (image) at a time going into
# the model, and manually resize predictions back to native resolution.

resize_image_to_model = Resize(spatial_size=SPATIAL_SIZE)  # default mode == "area", same as training

# resize_pred_to_native is built per-patient once native (H, W) is known,
# using mode="nearest" (matches the mask resize mode used in training/eval,
# and is the correct choice for resizing an integer label map).


def resize_pred_to_native(pred_tensor, native_hw):
    """
    pred_tensor: torch tensor of shape (1, 256, 256), float, integer-valued class labels
    native_hw: (H, W) tuple of the original slice resolution
    Returns: torch tensor of shape (1, H, W), float, integer-valued class labels

    Uses mode="nearest", the same interpolation mode used for the mask in
    ResizeD(keys=["mask"], mode="nearest") during training. Verified below
    (see _run_resize_sanity_check) that MONAI's Resize and ResizeD produce
    bit-identical output for both "area" (image) and "nearest" (mask) modes,
    so this is safe to use as a standalone transform.
    """
    resizer = Resize(spatial_size=native_hw, mode="nearest")
    return resizer(pred_tensor)


def _run_resize_sanity_check():
    """
    One-time startup check confirming that the standalone Resize transforms
    used in this script produce identical output to the ResizeD dictionary
    transforms used in train.py / evaluate.py, for both the image ("area"
    mode) and the label map ("nearest" mode). If this check ever fails,
    preprocessing has silently diverged from the baseline and the script
    aborts rather than producing numbers that are not comparable.
    """
    from monai.transforms import Compose, ResizeD

    rng = np.random.default_rng(0)
    dummy_img = rng.random((210, 180)).astype(np.float32)
    dummy_mask = (rng.random((210, 180)) * 4).astype(np.float32)

    img_t = torch.tensor(dummy_img, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.tensor(dummy_mask, dtype=torch.float32).unsqueeze(0)

    training_transforms = Compose([
        ResizeD(keys=["image"], spatial_size=SPATIAL_SIZE),
        ResizeD(keys=["mask"], spatial_size=SPATIAL_SIZE, mode="nearest"),
    ])

    data = training_transforms({"image": img_t.clone(), "mask": mask_t.clone()})

    standalone_image = resize_image_to_model(img_t.clone())
    standalone_mask = Resize(spatial_size=SPATIAL_SIZE, mode="nearest")(mask_t.clone())

    max_diff_image = torch.abs(data["image"] - standalone_image).max().item()
    max_diff_mask = torch.abs(data["mask"] - standalone_mask).max().item()

    if max_diff_image > 1e-6 or max_diff_mask > 1e-6:
        raise RuntimeError(
            "Resize sanity check FAILED: standalone Resize transforms diverge "
            f"from training ResizeD pipeline (image diff={max_diff_image}, "
            f"mask diff={max_diff_mask}). Preprocessing is not consistent with "
            "the baseline — aborting rather than producing incomparable metrics."
        )

    print(
        f"Resize sanity check passed (image diff={max_diff_image}, "
        f"mask diff={max_diff_mask}). Preprocessing matches training pipeline."
    )


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

model.load_state_dict(state_dict)

model.eval()

print("Model ready.")


# -------------------------------------------------------
# Confirm preprocessing matches the training/evaluate.py pipeline
# -------------------------------------------------------

print("\nRunning resize sanity check against training ResizeD pipeline...")

_run_resize_sanity_check()


# -------------------------------------------------------
# Validation samples (patient/phase level, not slice level)
# -------------------------------------------------------


print("Loading validation samples...")

val_samples = build_dataset(
    split="val",
    root_dir=ROOT_DIR
)

print(f"Validation patient-phase volumes: {len(val_samples)}")


# -------------------------------------------------------
# Per-volume inference + reconstruction
# -------------------------------------------------------

STRUCTURE_NAMES = ["LV", "Myocardium", "RV"]  # classes 1, 2, 3


def normalize_slice(img_slice):
    """Exactly matches mnms_dataset.py per-slice min-max normalization."""
    return (
        img_slice - img_slice.min()
    ) / (
        img_slice.max() - img_slice.min() + 1e-8
    )


def run_inference_on_volume(image_vol, frame_idx):
    """
    image_vol: numpy array of shape (H, W, num_slices, num_frames)
    frame_idx: int, ED or ES frame index for this sample

    Returns: numpy array of shape (H, W, num_slices), integer class labels,
             at native resolution, reconstructed from per-slice 256x256 inference.
    """
    H, W, num_slices, _ = image_vol.shape

    native_hw = (H, W)

    resized_slices = []

    for slice_idx in range(num_slices):

        img_slice = image_vol[:, :, slice_idx, frame_idx].astype(np.float32)

        img_slice = normalize_slice(img_slice)

        img_tensor = torch.tensor(
            img_slice,
            dtype=torch.float32
        ).unsqueeze(0)  # (1, H, W)

        img_tensor = resize_image_to_model(img_tensor)  # (1, 256, 256)

        resized_slices.append(img_tensor)

    # Stack all slices into a single batch tensor: (num_slices, 1, 256, 256)
    batch_tensor = torch.stack(resized_slices, dim=0)

    pred_native_slices = np.zeros((H, W, num_slices), dtype=np.int64)

    with torch.inference_mode():

        for start in range(0, num_slices, INFERENCE_CHUNK_SIZE):

            end = min(start + INFERENCE_CHUNK_SIZE, num_slices)

            chunk = batch_tensor[start:end].to(DEVICE)  # (chunk, 1, 256, 256)

            outputs = model(chunk)  # (chunk, 4, 256, 256)

            preds = torch.argmax(outputs, dim=1)  # (chunk, 256, 256)

            preds = preds.cpu().float()

            for i in range(preds.shape[0]):

                pred_slice = preds[i].unsqueeze(0)  # (1, 256, 256)

                pred_slice_native = resize_pred_to_native(
                    pred_slice,
                    native_hw
                )  # (1, H, W)

                pred_slice_native = pred_slice_native.squeeze(0).round().long().numpy()

                pred_native_slices[:, :, start + i] = pred_slice_native

    assert pred_native_slices.shape == (H, W, num_slices), (
        f"Reconstructed volume shape {pred_native_slices.shape} does not match "
        f"expected native shape {(H, W, num_slices)}"
    )

    return pred_native_slices


def compute_volume_metrics(pred_vol, gt_vol, spacing_3d):
    """
    pred_vol, gt_vol: numpy arrays of shape (H, W, num_slices), integer class labels
    spacing_3d: (dx, dy, dz) tuple in mm

    Returns: dict with Dice and HD95 per structure (LV, Myocardium, RV)
    """
    assert pred_vol.shape == gt_vol.shape, (
        f"Prediction volume shape {pred_vol.shape} does not match "
        f"ground-truth volume shape {gt_vol.shape}"
    )

    result = {}

    for cls, name in zip([1, 2, 3], STRUCTURE_NAMES):

        pred_binary = pred_vol == cls
        gt_binary = gt_vol == cls

        gt_sum = gt_binary.sum()

        if gt_sum == 0:
            # Structure genuinely absent from ground truth for this patient/phase.
            # Reporting NaN (rather than forcing a Dice/HD95 value) avoids
            # distorting the mean with an artificial score for a structure
            # that was never there to segment.
            result[f"Dice_{name}"] = np.nan
            result[f"HD95_{name}"] = np.nan
            continue

        pred_sum = pred_binary.sum()

        intersection = np.logical_and(pred_binary, gt_binary).sum()

        dice = (2 * intersection + 1e-8) / (pred_sum + gt_sum + 1e-8)

        if pred_sum == 0:
            # Ground truth has the structure but the model predicted nothing:
            # Dice correctly falls out near 0 above; HD95 is undefined
            # (no predicted surface to measure distance from).
            hd95_value = np.nan
        else:
            try:
                hd95_value = hd95(
                    pred_binary,
                    gt_binary,
                    voxelspacing=spacing_3d
                )
            except RuntimeError:
                hd95_value = np.nan

        result[f"Dice_{name}"] = dice
        result[f"HD95_{name}"] = hd95_value

    return result


# -------------------------------------------------------
# Main evaluation loop
# -------------------------------------------------------

print("\nStarting patient/volume-level evaluation...")

start_time = time.time()

per_patient_rows = []

skipped_samples = []

for sample in tqdm(val_samples):

    patient = sample["patient"]
    phase = sample["phase"]
    frame_idx = sample["frame_idx"]

    try:
        img_obj = nib.load(sample["image_path"])
        image_vol = img_obj.get_fdata(dtype=np.float32)  # (H, W, num_slices, num_frames)

        mask_obj = nib.load(sample["mask_path"])
        mask_vol_full = mask_obj.get_fdata(dtype=np.float32)  # (H, W, num_slices, num_frames)

        zooms = img_obj.header.get_zooms()
        spacing_3d = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

    except Exception as exc:
        print(f"\n[WARNING] Skipping {patient} ({phase}): failed to load NIfTI ({exc})")
        skipped_samples.append({"patient": patient, "phase": phase, "reason": str(exc)})
        continue

    try:
        pred_vol = run_inference_on_volume(image_vol, frame_idx)

        gt_vol = mask_vol_full[:, :, :, frame_idx].astype(np.int64)

        assert pred_vol.shape == gt_vol.shape, (
            f"pred_vol shape {pred_vol.shape} != gt_vol shape {gt_vol.shape} "
            f"for {patient} ({phase})"
        )

        if DEBUG_FIRST_PATIENT_ONLY and len(per_patient_rows) == 0 and not skipped_samples:

            print("\n==============================")
            print(f"Patient : {patient}")
            print(f"Phase   : {phase}")
            print(f"Frame   : {frame_idx}")
            print(f"Image   : {image_vol.shape}")
            print(f"GT      : {gt_vol.shape}")
            print(f"Pred    : {pred_vol.shape}")
            print(f"Spacing : {spacing_3d}")
            print("GT labels   :", np.unique(gt_vol))
            print("Pred labels :", np.unique(pred_vol))
            print("==============================")

            import matplotlib.pyplot as plt

            for s in range(pred_vol.shape[2]):

                plt.figure(figsize=(15, 5))

                plt.subplot(1, 3, 1)
                plt.imshow(
                    image_vol[:, :, s, frame_idx],
                    cmap="gray"
                )
                plt.title(f"MRI Slice {s}")

                plt.subplot(1, 3, 2)
                plt.imshow(
                    gt_vol[:, :, s]
                )
                plt.title("Ground Truth")

                plt.subplot(1, 3, 3)
                plt.imshow(
                    pred_vol[:, :, s]
                )
                plt.title("Prediction")

                plt.savefig(f"slice_{s}.png")
                plt.close()

        metrics = compute_volume_metrics(pred_vol, gt_vol, spacing_3d)

    except Exception as exc:
        print(f"\n[WARNING] Skipping {patient} ({phase}): failed during inference/metrics ({exc})")
        skipped_samples.append({"patient": patient, "phase": phase, "reason": str(exc)})
        continue

    row = {
        "patient": patient,
        "phase": phase,
    }
    row.update(metrics)

    per_patient_rows.append(row)

    if DEBUG_FIRST_PATIENT_ONLY:
        # Stop after the first patient so you can eyeball the printed shapes/
        # labels and the slice-by-slice plots above before committing to a
        # full ~68-patient run. Set DEBUG_FIRST_PATIENT_ONLY = False at the
        # top of this file to evaluate all validation patients.
        print("\n[DEBUG] DEBUG_FIRST_PATIENT_ONLY is True — stopping after 1 patient.")
        break

if skipped_samples:
    print(f"\n{len(skipped_samples)} sample(s) skipped due to errors:")
    for s in skipped_samples:
        print(f"  - {s['patient']} ({s['phase']}): {s['reason']}")


elapsed = time.time() - start_time


# -------------------------------------------------------
# Aggregate and save results
# -------------------------------------------------------

results_df = pd.DataFrame(per_patient_rows)

csv_path = os.path.join(RESULTS_DIR, "patient_level_results.csv")

results_df.to_csv(csv_path, index=False)

print("\n==============================")
print("Patient-Level Volume Evaluation Results")
print("==============================")
print(f"({len(results_df)} rows total, showing first 5)")
print(results_df.head())

print("\n------------------------------")
print("Mean across patients (NaN-safe)")
print("------------------------------")

mean_row = {}

for name in STRUCTURE_NAMES:

    mean_dice = np.nanmean(results_df[f"Dice_{name}"].values)
    mean_hd95 = np.nanmean(results_df[f"HD95_{name}"].values)

    mean_row[f"Dice_{name}"] = mean_dice
    mean_row[f"HD95_{name}"] = mean_hd95

    print(f"{name:12s} Dice: {mean_dice:.4f}   HD95: {mean_hd95:.4f} mm")

overall_mean_dice = np.mean([mean_row[f"Dice_{n}"] for n in STRUCTURE_NAMES])
overall_mean_hd95 = np.nanmean([mean_row[f"HD95_{n}"] for n in STRUCTURE_NAMES])

print(f"\nOverall Mean Dice : {overall_mean_dice:.4f}")
print(f"Overall Mean HD95 : {overall_mean_hd95:.4f} mm")
print(f"\nEvaluation Time: {elapsed:.2f} seconds")

print(f"\nResults saved to: {csv_path}")


# -------------------------------------------------------
# Summary CSV (means only, for easy reporting)
# -------------------------------------------------------

summary_rows = []

for name in STRUCTURE_NAMES:
    summary_rows.append({
        "Structure": name,
        "Mean_Dice": mean_row[f"Dice_{name}"],
        "Mean_HD95_mm": mean_row[f"HD95_{name}"],
    })

summary_rows.append({
    "Structure": "Overall",
    "Mean_Dice": overall_mean_dice,
    "Mean_HD95_mm": overall_mean_hd95,
})

summary_df = pd.DataFrame(summary_rows)

summary_csv_path = os.path.join(RESULTS_DIR, "patient_summary.csv")

summary_df.to_csv(summary_csv_path, index=False)

print(f"Summary saved to: {summary_csv_path}")