import os
import glob
import torch

from torch.utils.data import DataLoader

from monai.transforms import (
    Compose,
    ResizeD
)

from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

from dataset import build_dataset
from mnms_dataset import MNMSDataset


ROOT_DIR = "/content/OpenDataset"

CHECKPOINT_DIR = "mnms_checkpoints"

BATCH_SIZE = 8
NUM_EPOCHS = 5
LR = 1e-3

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)

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

train_samples = build_dataset(
    split="train",
    root_dir=ROOT_DIR
)

val_samples = build_dataset(
    split="val",
    root_dir=ROOT_DIR
)

train_dataset = MNMSDataset(
    train_samples,
    transform=transforms
)

val_dataset = MNMSDataset(
    val_samples,
    transform=transforms
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

print("Train slices:", len(train_dataset))
print("Val slices:", len(val_dataset))

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=4,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(DEVICE)

loss_fn = DiceCELoss(
    to_onehot_y=True,
    softmax=True
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

dice_metric = DiceMetric(
    include_background=False,
    reduction="mean"
)

start_epoch = 0
best_dice = 0.0

checkpoints = sorted(
    glob.glob(
        os.path.join(
            CHECKPOINT_DIR,
            "checkpoint_epoch_*.pth"
        )
    )
)

if checkpoints:

    latest_checkpoint = checkpoints[-1]

    checkpoint = torch.load(
        latest_checkpoint,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"]
    best_dice = checkpoint["val_dice"]

    print(
        f"Resuming from epoch {start_epoch}"
    )

for epoch in range(
    start_epoch,
    NUM_EPOCHS
):

    model.train()

    running_loss = 0.0

    for images, masks in train_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_fn(
            outputs,
            masks.unsqueeze(1)
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = (
        running_loss /
        len(train_loader)
    )

    print(
        f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]"
    )

    print(
        f"Training Loss: {avg_loss:.4f}"
    )

    model.eval()

    dice_metric.reset()

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            preds = torch.argmax(
                outputs,
                dim=1,
                keepdim=True
            )

            dice_metric(
                y_pred=preds,
                y=masks.unsqueeze(1)
            )

    val_dice = (
        dice_metric.aggregate().item()
    )

    print(
        f"Validation Dice: {val_dice:.4f}"
    )

    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_dice": val_dice,
    }

    torch.save(
        checkpoint,
        os.path.join(
            CHECKPOINT_DIR,
            f"checkpoint_epoch_{epoch+1}.pth"
        )
    )

    if val_dice > best_dice:

        best_dice = val_dice

        torch.save(
            model.state_dict(),
            os.path.join(
                CHECKPOINT_DIR,
                "best_model.pth"
            )
        )

        print(
            f"New best model: {best_dice:.4f}"
        )

print(
    f"\nTraining complete."
)

print(
    f"Best Dice: {best_dice:.4f}"
)