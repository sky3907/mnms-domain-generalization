import nibabel as nib
import numpy as np

# Paths
image_path = r"data/raw/OpenDataset/Training/Labeled/K4T7Y0/K4T7Y0_sa.nii.gz"
mask_path = r"data/raw/OpenDataset/Training/Labeled/K4T7Y0/K4T7Y0_sa_gt.nii.gz"

# Load MRI and mask
image = nib.load(image_path).get_fdata()
mask = nib.load(mask_path).get_fdata()

# Basic information
print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("All labels:", np.unique(mask))

# ED and ES frames from CSV
ED = 24
ES = 12

# Extract ED and ES volumes
ed_img = image[:, :, :, ED]
es_img = image[:, :, :, ES]

ed_mask = mask[:, :, :, ED]
es_mask = mask[:, :, :, ES]

# Verify extraction
print("\nED image shape:", ed_img.shape)
print("ES image shape:", es_img.shape)

print("\nED labels:", np.unique(ed_mask))
print("ES labels:", np.unique(es_mask))