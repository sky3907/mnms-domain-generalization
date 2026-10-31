
import argparse
import hashlib
import json
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.transforms import Compose, ResizeD
from torch.utils.data import Dataset, DataLoader

from dataset import build_dataset
from mnms_dataset import MNMSDataset


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DRIVE = Path("/content/drive/MyDrive")

SPLIT = ROOT / "strict_ab_cd/source_split.json"

CLUSTER_META = (
    ROOT / "latent_domain_discovery/clustering_metadata.json"
)

TRAIN_CLUSTERS = (
    ROOT / "latent_domain_discovery/source_cluster_assignments.csv"
)

VAL_CLUSTERS = (
    ROOT / "prompt_baselines/internal_ab_validation_cluster_assignments.csv"
)

DATA_ROOT = "/content/OpenDataset"

VARIANTS = (
    "source_only",
    "generic_prompt",
    "latent_domain_prompt",
)

MAX_EPOCHS = 30
MIN_EPOCHS = 15
PATIENCE = 8

BATCH = 8
LR = 1e-3
PROMPT_DIM = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Verified file saving
# ============================================================

def checksum(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(4 * 1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def verified_copy(source, dest):
    source = Path(source)
    dest = Path(dest)

    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty file: {source}")

    dest.parent.mkdir(parents=True, exist_ok=True)

    temporary = dest.with_name(dest.name + ".copying")

    if temporary.exists():
        raise RuntimeError(
            f"Unfinished backup exists: {temporary}"
        )

    if dest.exists():
        if checksum(dest) == checksum(source):
            return
        raise RuntimeError(
            f"Conflicting Drive backup: {dest}"
        )

    with source.open("rb") as r, temporary.open("wb") as w:
        shutil.copyfileobj(r, w)
        w.flush()
        os.fsync(w.fileno())

    if checksum(temporary) != checksum(source):
        raise RuntimeError(
            f"Backup checksum mismatch: {dest}"
        )

    temporary.replace(dest)

    if checksum(dest) != checksum(source):
        raise RuntimeError(
            f"Final backup checksum mismatch: {dest}"
        )


def save_mutable(local, remote, obj, kind):
    local = Path(local)
    remote = Path(remote)

    local.parent.mkdir(parents=True, exist_ok=True)
    remote.parent.mkdir(parents=True, exist_ok=True)

    for target in (local, remote):
        tmp = target.with_name(target.name + ".tmp")

        if kind == "torch":
            torch.save(obj, tmp)
            torch.load(
                tmp,
                map_location="cpu",
                weights_only=True,
            )

        elif kind == "csv":
            obj.to_csv(tmp, index=False)

        elif kind == "json":
            with tmp.open("w") as f:
                json.dump(obj, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

        else:
            raise ValueError(kind)

        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise RuntimeError(f"Empty output: {tmp}")

        tmp.replace(target)

    if checksum(local) != checksum(remote):
        raise RuntimeError(
            f"Local/Drive mismatch: {local} and {remote}"
        )

    if kind == "torch":
        torch.load(
            remote,
            map_location="cpu",
            weights_only=True,
        )


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Frozen cluster assignments
# ============================================================

def load_ids(path):
    df = pd.read_csv(
        path,
        dtype={"patient": str, "phase": str},
    )

    required = {"patient", "phase", "latent_cluster"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing columns in {path}: "
            f"{required - set(df.columns)}"
        )

    keys = list(zip(df.patient, df.phase))

    if (
        len(keys) != len(set(keys))
        or not df.latent_cluster.isin((0, 1)).all()
    ):
        raise ValueError(
            f"Invalid cluster assignments: {path}"
        )

    return dict(
        zip(keys, df.latent_cluster.astype(int))
    )


class RoutedSliceDataset(Dataset):
    def __init__(self, base, cluster_map, variant):
        self.base = base
        self.cluster_map = cluster_map
        self.variant = variant

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        image, mask, meta = self.base[i]

        image = torch.as_tensor(
            image, dtype=torch.float32
        )
        mask = torch.as_tensor(
            mask, dtype=torch.long
        )

        if self.variant == "latent_domain_prompt":
            key = (
                str(meta["patient"]),
                str(meta["phase"]),
            )
            pid = self.cluster_map[key]
        else:
            pid = 0

        return image, mask, torch.tensor(
            pid, dtype=torch.long
        )


# ============================================================
# Matched model architectures
# ============================================================

class MatchedUNet(nn.Module):
    def __init__(self, variant):
        super().__init__()

        self.variant = variant
        prompted = variant != "source_only"

        if prompted:
            self.prompt_embedding = nn.Embedding(
                2, PROMPT_DIM
            )

        self.unet = UNet(
            spatial_dims=2,
            in_channels=1 + (PROMPT_DIM if prompted else 0),
            out_channels=4,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

    def forward(self, x, ids):
        if self.variant != "source_only":
            prompt = self.prompt_embedding(ids)
            prompt = prompt[:, :, None, None]

            prompt = prompt.expand(
                -1, -1,
                x.shape[-2],
                x.shape[-1],
            )

            x = torch.cat([x, prompt], dim=1)

        return self.unet(x)


# ============================================================
# Matched spatial and intensity augmentation
# ============================================================

def augment_batch(images, masks, seed, epoch, step):
    b = images.shape[0]

    generator = torch.Generator(
        device="cpu"
    ).manual_seed(
        seed * 100000000
        + epoch * 10000
        + step * 37
        + 107
    )

    z = torch.rand(
        (b, 6),
        generator=generator,
    ).to(images.device)

    angle = (
        (z[:, 0] * 2 - 1)
        * (10 * math.pi / 180)
    )

    scale = 0.95 + z[:, 1] * 0.10

    tx = (z[:, 2] * 2 - 1) * 0.04
    ty = (z[:, 3] * 2 - 1) * 0.04

    theta = torch.zeros(
        (b, 2, 3),
        device=images.device,
        dtype=images.dtype,
    )

    theta[:, 0, 0] = scale * torch.cos(angle)
    theta[:, 0, 1] = -scale * torch.sin(angle)
    theta[:, 1, 0] = scale * torch.sin(angle)
    theta[:, 1, 1] = scale * torch.cos(angle)
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    grid = F.affine_grid(
        theta,
        images.shape,
        align_corners=False,
    )

    images = F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    masks = F.grid_sample(
        masks[:, None].float(),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )[:, 0].long()

    gain = (
        0.9 + z[:, 4] * 0.2
    )[:, None, None, None]

    bias = (
        (z[:, 5] * 2 - 1) * 0.08
    )[:, None, None, None]

    noise_generator = torch.Generator(
        device=images.device
    ).manual_seed(
        seed * 100000000
        + epoch * 10000
        + step * 37
        + 109
    )

    noise = torch.randn(
        images.shape,
        device=images.device,
        generator=noise_generator,
        dtype=images.dtype,
    ) * 0.015

    return (
        (images * gain + bias + noise).clamp_(0, 1),
        masks,
    )


# ============================================================
# Internal A/B validation
# ============================================================

@torch.inference_mode()
def validate(model, loader):
    model.eval()

    intersection = torch.zeros(
        3, dtype=torch.float64, device=DEVICE
    )

    prediction = torch.zeros_like(intersection)
    ground_truth = torch.zeros_like(intersection)

    for images, masks, ids in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        ids = ids.to(DEVICE)

        preds = model(
            images, ids
        ).argmax(dim=1)

        for i, cls in enumerate((1, 2, 3)):
            p = preds == cls
            gt = masks == cls

            intersection[i] += (
                torch.logical_and(p, gt).sum()
            )

            prediction[i] += p.sum()
            ground_truth[i] += gt.sum()

    dice = (
        2 * intersection + 1e-8
    ) / (
        prediction + ground_truth + 1e-8
    )

    return dice.cpu().tolist(), float(
        dice.mean().cpu()
    )


# ============================================================
# Train one model
# ============================================================

def train_variant(
    variant,
    seed,
    base_train,
    base_val,
    train_map,
    val_map,
    config,
):
    local = (
        ROOT / "augmented_training_v1"
        / f"seed{seed}" / variant
    )

    remote = (
        DRIVE / f"mnms_augmented_training_v1_seed{seed}"
        / variant
    )

    local.mkdir(parents=True, exist_ok=True)
    remote.mkdir(parents=True, exist_ok=True)

    metadata = {
        **config,
        "variant": variant,
        "seed": seed,
    }

    meta_local = local / "experiment_metadata.json"
    meta_remote = remote / "experiment_metadata.json"

    if meta_remote.exists():
        if json.loads(meta_remote.read_text()) != metadata:
            raise RuntimeError(
                f"Configuration mismatch: {meta_remote}"
            )

    save_mutable(
        meta_local, meta_remote, metadata, "json"
    )

    verified_copy(
        Path(__file__),
        remote / Path(__file__).name,
    )

    set_seed(seed)

    model = MatchedUNet(variant).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    loss_fn = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
    )

    train_dataset = RoutedSliceDataset(
        base_train, train_map, variant
    )

    val_dataset = RoutedSliceDataset(
        base_val, val_map, variant
    )

    shuffle_generator = torch.Generator(
        device="cpu"
    ).manual_seed(seed + 124)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH,
        shuffle=True,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
        generator=shuffle_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )

    history = []
    best = -1.0
    best_epoch = 0
    bad_epochs = 0
    start_epoch = 1

    latest = remote / "latest_checkpoint.pth"
    summary_path = remote / "training_summary.csv"

    # Completed experiment: verify and skip.
    if summary_path.exists() and summary_path.stat().st_size > 0:
        summary = pd.read_csv(summary_path)

        if (
            len(summary) == 1
            and bool(summary.iloc[0]["complete"])
            and (remote / "best_model.pth").stat().st_size > 0
        ):
            torch.load(
                remote / "best_model.pth",
                map_location="cpu",
                weights_only=True,
            )

            print(
                f"Already completed: {variant}, seed {seed}"
            )
            return

    # Resume an interrupted run.
    if latest.exists() and latest.stat().st_size > 0:
        checkpoint = torch.load(
            latest,
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint["config"] != metadata:
            raise RuntimeError(
                "Checkpoint configuration mismatch."
            )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        history = checkpoint["history"]
        best = checkpoint["best"]
        best_epoch = checkpoint["best_epoch"]
        bad_epochs = checkpoint["bad_epochs"]
        start_epoch = int(checkpoint["epoch"]) + 1

        shuffle_generator.set_state(
            checkpoint["shuffle_gen_state"]
        )

        torch.set_rng_state(
            checkpoint["torch_rng_state"]
        )

        if DEVICE.type == "cuda":
            torch.cuda.set_rng_state_all(
                checkpoint["cuda_rng_states"]
            )

        print(
            f"Resuming {variant} from epoch {start_epoch}"
        )

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for step, (images, masks, ids) in enumerate(train_loader):
            images = images.to(
                DEVICE, non_blocking=True
            )

            masks = masks.to(
                DEVICE, non_blocking=True
            )

            ids = ids.to(
                DEVICE, non_blocking=True
            )

            images, masks = augment_batch(
                images, masks, seed, epoch, step
            )

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images, ids)

            loss = loss_fn(
                outputs,
                masks[:, None],
            )

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Non-finite loss: {variant}, epoch {epoch}"
                )

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        class_dice, mean_dice = validate(
            model, val_loader
        )

        improved = mean_dice > best + 1e-6

        if improved:
            best = mean_dice
            best_epoch = epoch
            bad_epochs = 0

            save_mutable(
                local / "best_model.pth",
                remote / "best_model.pth",
                model.state_dict(),
                "torch",
            )
        else:
            bad_epochs += 1

        history.append({
            "epoch": epoch,
            "training_loss": running_loss / len(train_loader),
            "internal_val_dice_lv": class_dice[0],
            "internal_val_dice_myocardium": class_dice[1],
            "internal_val_dice_rv": class_dice[2],
            "internal_val_mean_dice": mean_dice,
        })

        completed = (
            epoch >= MIN_EPOCHS
            and bad_epochs >= PATIENCE
        ) or epoch == MAX_EPOCHS

        save_mutable(
            local / "training_history.csv",
            remote / "training_history.csv",
            pd.DataFrame(history),
            "csv",
        )

        summary = pd.DataFrame([{
            "model": variant,
            "seed": seed,
            "epochs_completed": epoch,
            "best_epoch": best_epoch,
            "best_dice": best,
            "complete": completed,
        }])

        save_mutable(
            local / "training_summary.csv",
            remote / "training_summary.csv",
            summary,
            "csv",
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best": best,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "history": history,
            "config": metadata,
            "shuffle_gen_state": shuffle_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all()
                if DEVICE.type == "cuda" else []
            ),
        }

        save_mutable(
            local / "latest_checkpoint.pth",
            remote / "latest_checkpoint.pth",
            checkpoint,
            "torch",
        )

        print(
            f"{variant} | seed {seed} | "
            f"epoch {epoch}/{MAX_EPOCHS} | "
            f"loss {running_loss / len(train_loader):.4f} | "
            f"A/B Dice {mean_dice:.6f} | "
            f"best {best:.6f} (epoch {best_epoch})"
        )

        if completed:
            print(
                f"COMPLETED {variant}: "
                f"best internal A/B Dice {best:.6f}"
            )
            return


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()
    seed = args.seed

    if not DRIVE.is_dir():
        raise FileNotFoundError(
            "Mount Google Drive before training."
        )

    required = [
        SPLIT,
        CLUSTER_META,
        TRAIN_CLUSTERS,
        VAL_CLUSTERS,
    ]

    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Required frozen file missing: {path}"
            )

    split = json.loads(SPLIT.read_text())
    cluster_metadata = json.loads(CLUSTER_META.read_text())

    if (
        cluster_metadata["selected_k"] != 2
        or cluster_metadata["vendor_centre_used_for_clustering"] is not False
    ):
        raise ValueError(
            "Expected the original K=2 clustering."
        )

    train_ids = set(map(str, split["train_patients"]))
    val_ids = set(
        map(str, split["internal_validation_patients"])
    )

    if (
        len(train_ids) != 120
        or len(val_ids) != 30
        or train_ids & val_ids
    ):
        raise ValueError("Frozen patient split is invalid.")

    train_map = load_ids(TRAIN_CLUSTERS)
    val_map = load_ids(VAL_CLUSTERS)

    samples = build_dataset(
        split="train",
        root_dir=DATA_ROOT,
    )

    train_samples = [
        s for s in samples
        if str(s["patient"]) in train_ids
    ]

    val_samples = [
        s for s in samples
        if str(s["patient"]) in val_ids
    ]

    def expected_keys(ids):
        return {
            (p, phase)
            for p in ids
            for phase in ("ED", "ES")
        }

    train_keys = {
        (str(s["patient"]), str(s["phase"]))
        for s in train_samples
    }

    val_keys = {
        (str(s["patient"]), str(s["phase"]))
        for s in val_samples
    }

    if (
        len(train_samples) != 240
        or len(val_samples) != 60
        or train_keys != expected_keys(train_ids)
        or val_keys != expected_keys(val_ids)
        or set(train_map) != train_keys
        or set(val_map) != val_keys
    ):
        raise ValueError(
            "Dataset and frozen split/cluster records do not match."
        )

    if any(
        s["vendor"] not in {"A", "B"}
        for s in train_samples + val_samples
    ):
        raise ValueError("Unexpected non-source vendor.")

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
        if DEVICE.type == "cuda" else "CPU",
    )

    print(
        "Training / internal validation patients:",
        len(train_ids), "/", len(val_ids),
    )

    config = {
        "protocol": "Frozen A/B training and internal A/B validation",
        "max_epochs": MAX_EPOCHS,
        "min_epochs": MIN_EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH,
        "learning_rate": LR,
        "prompt_dim": PROMPT_DIM,
        "augmentation": (
            "Rotation +/-10 degrees; scale 0.95-1.05; "
            "translation +/-4%; contrast 0.9-1.1; "
            "brightness +/-0.08; Gaussian noise 0.015"
        ),
        "checkpoint_selection": "Internal A/B pooled foreground Dice",
        "c_d_used_for_training_or_tuning": False,
        "input_sha256": {
            str(p.relative_to(ROOT)): checksum(p)
            for p in required
        },
        "script_sha256": checksum(Path(__file__)),
    }

    transforms = Compose([
        ResizeD(
            keys=["image"],
            spatial_size=(256, 256),
        ),
        ResizeD(
            keys=["mask"],
            spatial_size=(256, 256),
            mode="nearest",
        ),
    ])

    base_train = MNMSDataset(
        train_samples,
        transform=transforms,
    )

    base_val = MNMSDataset(
        val_samples,
        transform=transforms,
    )

    if len(base_train) != 2632 or len(base_val) != 654:
        raise ValueError("Unexpected slice counts.")

    print(
        "Training / validation slices:",
        len(base_train), "/", len(base_val),
    )

    for variant in VARIANTS:
        train_variant(
            variant,
            seed,
            base_train,
            base_val,
            train_map,
            val_map,
            config,
        )

    print(
        "\nALL THREE MATCHED AUGMENTED TRAINING RUNS COMPLETE"
    )


if __name__ == "__main__":
    main()
