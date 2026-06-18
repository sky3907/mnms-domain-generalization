import os
import pandas as pd
import nibabel as nib
import numpy as np

CSV_PATH = r"data/raw/OpenDataset/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
LABELED_DIR = r"data/raw/OpenDataset/Training/Labeled"

df = pd.read_csv(CSV_PATH)

processed_samples = []

for patient_id in os.listdir(LABELED_DIR):

    patient_folder = os.path.join(LABELED_DIR, patient_id)

    if not os.path.isdir(patient_folder):
        continue

    image_path = os.path.join(
        patient_folder,
        f"{patient_id}_sa.nii.gz"
    )

    mask_path = os.path.join(
        patient_folder,
        f"{patient_id}_sa_gt.nii.gz"
    )

    row = df[df["External code"] == patient_id]

    if len(row) == 0:
        continue

    row = row.iloc[0]

    ED = int(row["ED"])
    ES = int(row["ES"])

    image = nib.load(image_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()

    for phase_name, frame_idx in [("ED", ED), ("ES", ES)]:

        phase_img = image[:, :, :, frame_idx]
        phase_mask = mask[:, :, :, frame_idx]

        # z-score normalization
        phase_img = phase_img.astype(np.float32)

        mean = phase_img.mean()
        std = phase_img.std()

        if std > 0:
            phase_img = (phase_img - mean) / std

        processed_samples.append(
            {
                "patient": patient_id,
                "phase": phase_name,
                "vendor": row["Vendor"],
                "vendor_name": row["VendorName"],
                "centre": row["Centre"],
                "shape": phase_img.shape,
                "labels": np.unique(phase_mask).tolist()
            }
        )

    del image
    del mask

print(f"Patients processed: {len(processed_samples)//2}")
print(f"Total ED/ES samples: {len(processed_samples)}")

print("\nExample:")
print(processed_samples[0])