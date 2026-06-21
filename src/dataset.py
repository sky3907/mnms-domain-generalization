import os
import pandas as pd


def build_dataset(split="train", root_dir="/content/OpenDataset"):

    csv_path = os.path.join(
        root_dir,
        "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
    )

    if split == "train":
        data_dir = os.path.join(
            root_dir,
            "Training",
            "Labeled"
        )

    elif split == "val":
        data_dir = os.path.join(
            root_dir,
            "Validation"
        )

    elif split == "test":
        data_dir = os.path.join(
            root_dir,
            "Testing"
        )

    else:
        raise ValueError(
            "split must be train, val, or test"
        )

    df = pd.read_csv(csv_path)

    dataset = []

    for patient_id in os.listdir(data_dir):

        patient_folder = os.path.join(
            data_dir,
            patient_id
        )

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

        if not (
            os.path.exists(image_path)
            and os.path.exists(mask_path)
        ):
            continue

        row = df[df["External code"] == patient_id]

        if len(row) == 0:
            continue

        row = row.iloc[0]

        dataset.append({
            "patient": patient_id,
            "phase": "ED",
            "frame_idx": int(row["ED"]),
            "vendor": row["Vendor"],
            "vendor_name": row["VendorName"],
            "centre": int(row["Centre"]),
            "image_path": image_path,
            "mask_path": mask_path,
        })

        dataset.append({
            "patient": patient_id,
            "phase": "ES",
            "frame_idx": int(row["ES"]),
            "vendor": row["Vendor"],
            "vendor_name": row["VendorName"],
            "centre": int(row["Centre"]),
            "image_path": image_path,
            "mask_path": mask_path,
        })

    return dataset