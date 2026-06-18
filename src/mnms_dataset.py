import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class MNMSDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

        self.slice_samples = []

        for sample in samples:

            image = nib.load(sample["image_path"])

            _, _, num_slices, _ = image.shape

            for slice_idx in range(num_slices):

                self.slice_samples.append(
                    {
                        **sample,
                        "slice_idx": slice_idx
                    }
                )

    def __len__(self):
        return len(self.slice_samples)

    def __getitem__(self, idx):

        sample = self.slice_samples[idx]

        image = nib.load(
            sample["image_path"]
        ).get_fdata()

        mask = nib.load(
            sample["mask_path"]
        ).get_fdata()

        frame_idx = sample["frame_idx"]
        slice_idx = sample["slice_idx"]

        image = image[:, :, slice_idx, frame_idx]
        mask = mask[:, :, slice_idx, frame_idx]

        image = image.astype(np.float32)

        mean = image.mean()
        std = image.std()

        if std > 0:
            image = (image - mean) / std

        image = np.expand_dims(image, axis=0)

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask