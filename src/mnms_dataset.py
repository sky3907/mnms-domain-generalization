import nibabel as nib
import numpy as np
import torch

from torch.utils.data import Dataset


class MNMSDataset(Dataset):

    def __init__(self, samples, transform=None):

        self.transform = transform

        # Final processed dataset
        self.slice_samples = []

        print("Preparing dataset...")

        for sample in samples:

            image = nib.load(
                sample["image_path"]
            ).get_fdata(dtype=np.float32)

            mask = nib.load(
                sample["mask_path"]
            ).get_fdata(dtype=np.float32)

            frame_idx = sample["frame_idx"]

            num_slices = image.shape[2]

            for slice_idx in range(num_slices):

                img = image[:, :, slice_idx, frame_idx]
                msk = mask[:, :, slice_idx, frame_idx]

                img = (
                    img - img.min()
                ) / (
                    img.max() - img.min() + 1e-8
                )

                img = torch.tensor(
                    img,
                    dtype=torch.float32
                ).unsqueeze(0)

                msk = torch.tensor(
                    msk,
                    dtype=torch.float32
                ).unsqueeze(0)

                if self.transform:

                    data = {
                        "image": img,
                        "mask": msk
                    }

                    data = self.transform(data)

                    img = data["image"]
                    msk = data["mask"]

                msk = msk.squeeze(0).long()

                self.slice_samples.append(
                    (
                        img,
                        msk,
                        {
                            "patient": sample["patient"],
                            "vendor": sample["vendor"],
                            "centre": sample["centre"],
                            "phase": sample["phase"],
                            "slice_idx": slice_idx,
                            "frame_idx": frame_idx,
                        }
                    )
                )

        print(
            f"Prepared {len(self.slice_samples)} slices."
        )

    def __len__(self):

        return len(self.slice_samples)

    def __getitem__(self, idx):

        return self.slice_samples[idx]