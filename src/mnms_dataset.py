import nibabel as nib
import numpy as np
import torch

from torch.utils.data import Dataset


class MNMSDataset(Dataset):

    def __init__(self, samples, transform=None):

        self.transform = transform
        self.slice_samples = []

        for sample in samples:

            image = nib.load(
                sample["image_path"]
            ).get_fdata(dtype=np.float32)

            num_slices = image.shape[2]

            for slice_idx in range(num_slices):

                self.slice_samples.append({
                    **sample,
                    "slice_idx": slice_idx
                })

    def __len__(self):
        return len(self.slice_samples)

    def __getitem__(self, idx):

        sample = self.slice_samples[idx]

        image = nib.load(
            sample["image_path"]
        ).get_fdata(dtype=np.float32)

        mask = nib.load(
            sample["mask_path"]
        ).get_fdata(dtype=np.float32)

        frame_idx = sample["frame_idx"]
        slice_idx = sample["slice_idx"]

        image = image[:, :, slice_idx, frame_idx]
        mask = mask[:, :, slice_idx, frame_idx]

        image = (
            image - image.min()
        ) / (
            image.max() - image.min() + 1e-8
        )

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).unsqueeze(0)

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        if self.transform:

            data = {
                "image": image,
                "mask": mask
            }

            data = self.transform(data)

            image = data["image"]
            mask = data["mask"]

        mask = mask.squeeze(0).long()

        return image, mask