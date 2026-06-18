import os
import nibabel as nib
from collections import Counter

LABELED_DIR = r"data/raw/OpenDataset/Training/Labeled"

shapes = Counter()

for patient_id in os.listdir(LABELED_DIR):

    patient_folder = os.path.join(LABELED_DIR, patient_id)

    if not os.path.isdir(patient_folder):
        continue

    image_path = os.path.join(
        patient_folder,
        f"{patient_id}_sa.nii.gz"
    )

    img = nib.load(image_path)

    shapes[img.shape] += 1

print("Unique shapes:")
for shape, count in shapes.items():
    print(shape, count)
