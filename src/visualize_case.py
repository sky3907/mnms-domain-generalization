import nibabel as nib
import matplotlib.pyplot as plt

image_path = r"data/raw/OpenDataset/Training/Labeled/K4T7Y0/K4T7Y0_sa.nii.gz"
mask_path = r"data/raw/OpenDataset/Training/Labeled/K4T7Y0/K4T7Y0_sa_gt.nii.gz"

image = nib.load(image_path).get_fdata()
mask = nib.load(mask_path).get_fdata()

ED = 24

ed_img = image[:, :, :, ED]
ed_mask = mask[:, :, :, ED]

slice_idx = ed_img.shape[2] // 2

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(ed_img[:, :, slice_idx], cmap="gray")
plt.title("ED MRI")

plt.subplot(1, 2, 2)
plt.imshow(ed_mask[:, :, slice_idx])
plt.title("ED Mask")

plt.show()