from dataset import build_dataset
from mnms_dataset import MNMSDataset

from monai.transforms import (
    Compose,
    ResizeD
)

ROOT_DIR = "/content/OpenDataset"

train_samples = build_dataset(
    split="train",
    root_dir=ROOT_DIR
)

transforms = Compose([
    ResizeD(
        keys=["image"],
        spatial_size=(256, 256)
    ),
    ResizeD(
        keys=["mask"],
        spatial_size=(256, 256),
        mode="nearest"
    )
])

dataset = MNMSDataset(
    train_samples,
    transform=transforms
)

print("Total slices:", len(dataset))

image, mask = dataset[100]

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Mask labels:", mask.unique())