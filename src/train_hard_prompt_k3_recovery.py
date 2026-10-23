import os
import json
import random
import hashlib
import shutil
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scipy.stats import skew, kurtosis
from torch.utils.data import Dataset, DataLoader

from monai.transforms import Compose, ResizeD
from monai.networks.nets import UNet
from monai.losses import DiceCELoss

from dataset import build_dataset
from mnms_dataset import MNMSDataset


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_ROOT = "/content/OpenDataset"

SPLIT_PATH = PROJECT_DIR / "strict_ab_cd/source_split.json"

CLUSTER_DIR = (
    PROJECT_DIR / "latent_domain_ablations/k3"
)

LOCAL_OUTPUT = PROJECT_DIR / "hard_prompt_k3_recovery"

DRIVE_OUTPUT = Path(
    "/content/drive/MyDrive/mnms_hard_prompt_k3_recovery_seed42"
)

SEED = 42
K = 3
PROMPT_DIM = 4

BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

def reset_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Verified local -> Drive saving
# ============================================================

def sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(4 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def verified_copy(source, destination):

    source = Path(source)
    destination = Path(destination)

    if not source.is_file():
        raise FileNotFoundError(source)

    if source.stat().st_size == 0:
        raise RuntimeError(
            f"Cannot back up empty file: {source}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary = destination.with_name(
        destination.name + ".copying"
    )

    if temporary.exists():
        raise RuntimeError(
            f"Unfinished backup detected: {temporary}"
        )

    with source.open("rb") as src:
        with temporary.open("wb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())

    if sha256(source) != sha256(temporary):
        raise RuntimeError(
            f"Backup checksum mismatch: {destination}"
        )

    temporary.replace(destination)

    if destination.stat().st_size != source.stat().st_size:
        raise RuntimeError(
            f"Backup size mismatch: {destination}"
        )

    if sha256(source) != sha256(destination):
        raise RuntimeError(
            f"Final backup checksum mismatch: {destination}"
        )


def save_json(obj, filename):

    local = LOCAL_OUTPUT / filename
    temporary = local.with_name(
        local.name + ".tmp"
    )

    with temporary.open("w") as f:
        json.dump(obj, f, indent=2)

    temporary.replace(local)

    verified_copy(
        local,
        DRIVE_OUTPUT / filename
    )


def save_csv(df, filename):

    local = LOCAL_OUTPUT / filename
    temporary = local.with_name(
        local.name + ".tmp"
    )

    df.to_csv(temporary, index=False)
    temporary.replace(local)

    verified_copy(
        local,
        DRIVE_OUTPUT / filename
    )


def save_checkpoint(data, filename):

    local = LOCAL_OUTPUT / filename

    temporary = local.with_name(
        local.name + ".tmp"
    )

    torch.save(data, temporary)
    temporary.replace(local)

    if local.stat().st_size == 0:
        raise RuntimeError(
            f"Local checkpoint is empty: {local}"
        )

    # Verify local serialization before copying.
    torch.load(
        local,
        map_location="cpu",
        weights_only=True
    )

    destination = DRIVE_OUTPUT / filename

    verified_copy(local, destination)

    # Verify that the Drive copy is readable.
    torch.load(
        destination,
        map_location="cpu",
        weights_only=True
    )

    print(
        f"Verified checkpoint backup: {filename} "
        f"({destination.stat().st_size:,} bytes)"
    )


# ============================================================
# Style extraction
# Same feature definitions as frozen latent discovery
# ============================================================

def extract_style_features(volume):

    x = np.asarray(
        volume,
        dtype=np.float32
    )

    x = x[np.isfinite(x)]

    if x.size == 0:
        raise ValueError("Volume contains no finite voxels.")

    nonzero = x[x != 0]

    if nonzero.size > 100:
        x = nonzero

    low, high = np.percentile(
        x, [1, 99]
    )

    x = np.clip(x, low, high)

    median = np.median(x)
    q25, q75 = np.percentile(
        x, [25, 75]
    )

    x = (
        (x - median) / (q75 - q25 + 1e-8)
    ).astype(np.float32)

    p1, p5, p25, _, p75, p95, p99 = np.percentile(
        x, [1, 5, 25, 50, 75, 95, 99]
    )

    hist, _ = np.histogram(
        x,
        bins=64
    )

    probs = (
        hist[hist > 0].astype(np.float64)
        / hist.sum()
    )

    return {
        "intensity_mean": float(np.mean(x)),
        "intensity_std": float(np.std(x)),
        "p01": float(p1),
        "p05": float(p5),
        "p25": float(p25),
        "p75": float(p75),
        "p95": float(p95),
        "p99": float(p99),
        "skewness": float(skew(x)),
        "kurtosis": float(kurtosis(x)),
        "entropy": float(
            -(probs * np.log2(probs)).sum()
        ),
    }


def predict_clusters(
    X,
    scaler,
    pca,
    kmeans,
    n_components
):

    X = np.asarray(
        X,
        dtype=np.float64
    )

    if not np.isfinite(X).all():
        raise ValueError(
            "Non-finite style features."
        )

    X_scaled = scaler.transform(X)

    X_pca = pca.transform(
        X_scaled
    )[:, :n_components]

    return kmeans.predict(
        X_pca
    ).astype(np.int64)


# ============================================================
# Prompt dataset
# ============================================================

class HardPromptDataset(Dataset):

    def __init__(self, base_dataset, prompt_map):
        self.base_dataset = base_dataset
        self.prompt_map = prompt_map

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):

        image, mask, metadata = self.base_dataset[idx]

        key = (
            str(metadata["patient"]),
            str(metadata["phase"])
        )

        return (
            image,
            mask,
            torch.tensor(
                self.prompt_map[key],
                dtype=torch.long
            )
        )


# ============================================================
# Hard-prompt U-Net
# ============================================================

class HardPromptUNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.prompt_embedding = nn.Embedding(
            K,
            PROMPT_DIM
        )

        self.unet = UNet(
            spatial_dims=2,
            in_channels=1 + PROMPT_DIM,
            out_channels=4,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2
        )

    def forward(self, images, prompt_ids):

        prompt = self.prompt_embedding(
            prompt_ids
        )

        prompt = prompt[:, :, None, None]

        prompt = prompt.expand(
            -1,
            -1,
            images.shape[-2],
            images.shape[-1]
        )

        x = torch.cat(
            [images, prompt],
            dim=1
        )

        return self.unet(x)


# ============================================================
# Internal A/B Dice evaluation
# ============================================================

def evaluate(model, loader):

    model.eval()

    intersection = np.zeros(
        3, dtype=np.float64
    )

    prediction = np.zeros(
        3, dtype=np.float64
    )

    ground_truth = np.zeros(
        3, dtype=np.float64
    )

    with torch.inference_mode():

        for images, masks, prompt_ids in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            prompt_ids = prompt_ids.to(DEVICE)

            outputs = model(
                images,
                prompt_ids
            )

            preds = outputs.argmax(dim=1)

            for i, cls in enumerate((1, 2, 3)):

                pred_binary = preds == cls
                gt_binary = masks == cls

                intersection[i] += (
                    (pred_binary & gt_binary)
                    .sum()
                    .item()
                )

                prediction[i] += (
                    pred_binary.sum().item()
                )

                ground_truth[i] += (
                    gt_binary.sum().item()
                )

    dice = (
        2 * intersection + 1e-8
    ) / (
        prediction + ground_truth + 1e-8
    )

    if not np.isfinite(dice).all():
        raise RuntimeError(
            "Non-finite validation Dice."
        )

    return dice, float(dice.mean())


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Preflight: protect previous and partial experiments
    # --------------------------------------------------------

    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError(
            "Google Drive is not mounted."
        )

    required = [
        SPLIT_PATH,
        CLUSTER_DIR / "clustering_metadata.json",
        CLUSTER_DIR / "source_style_features.csv",
        CLUSTER_DIR / "style_scaler.joblib",
        CLUSTER_DIR / "style_pca.joblib",
        CLUSTER_DIR / "style_kmeans.joblib",
    ]

    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Missing or empty input: {path}"
            )

    # These NEW recovery directories must not contain
    # existing experiment files.
    for folder in (LOCAL_OUTPUT, DRIVE_OUTPUT):
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(
                f"Existing recovery files detected: {folder}. "
                "Do not overwrite an existing run."
            )

    LOCAL_OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    DRIVE_OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    reset_seed()

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Frozen patient split
    # --------------------------------------------------------

    with SPLIT_PATH.open() as f:
        split = json.load(f)

    with (
        CLUSTER_DIR / "clustering_metadata.json"
    ).open() as f:
        metadata = json.load(f)

    train_ids = set(
        map(str, split["train_patients"])
    )

    val_ids = set(
        map(str, split["internal_validation_patients"])
    )

    assert len(train_ids) == 120
    assert len(val_ids) == 30
    assert train_ids.isdisjoint(val_ids)

    assert metadata["selected_k"] == K
    assert metadata["vendor_centre_used_for_clustering"] is False

    samples = build_dataset(
        split="train",
        root_dir=DATASET_ROOT
    )

    train_samples = [
        s for s in samples
        if str(s["patient"]) in train_ids
    ]

    val_samples = [
        s for s in samples
        if str(s["patient"]) in val_ids
    ]

    assert len(train_samples) == 240
    assert len(val_samples) == 60

    for group in (train_samples, val_samples):
        assert all(
            s["vendor"] in {"A", "B"}
            for s in group
        )

    print(
        "Training volumes:",
        len(train_samples)
    )

    print(
        "Internal validation volumes:",
        len(val_samples)
    )

    # --------------------------------------------------------
    # Load recovered K=3 pipeline
    # --------------------------------------------------------

    feature_columns = metadata["feature_columns"]

    n_components = int(
        metadata["pca_components_95"]
    )

    scaler = joblib.load(
        CLUSTER_DIR / "style_scaler.joblib"
    )

    pca = joblib.load(
        CLUSTER_DIR / "style_pca.joblib"
    )

    kmeans = joblib.load(
        CLUSTER_DIR / "style_kmeans.joblib"
    )

    assert kmeans.n_clusters == K

    df = pd.read_csv(
        CLUSTER_DIR / "source_style_features.csv",
        dtype={"patient": str, "phase": str}
    )

    train_keys = [
        (
            str(s["patient"]),
            str(s["phase"])
        )
        for s in train_samples
    ]

    feature_keys = list(
        zip(
            df["patient"],
            df["phase"]
        )
    )

    assert len(feature_keys) == 240
    assert len(set(feature_keys)) == 240
    assert set(feature_keys) == set(train_keys)

    X_train = df[
        feature_columns
    ].to_numpy(dtype=np.float64)

    train_labels = predict_clusters(
        X_train,
        scaler,
        pca,
        kmeans,
        n_components
    )

    if not np.array_equal(
        train_labels,
        df["latent_cluster"].to_numpy()
    ):
        raise RuntimeError(
            "Recovered K=3 assignments do not match."
        )

    assert np.array_equal(
        np.bincount(train_labels, minlength=K),
        np.array([95, 80, 65])
    )

    train_map = dict(
        zip(
            feature_keys,
            train_labels.tolist()
        )
    )

    print(
        "Training cluster counts:",
        np.bincount(train_labels, minlength=K)
    )

    # --------------------------------------------------------
    # Validation assignment without refitting
    # --------------------------------------------------------

    print("Extracting internal A/B validation styles...")

    val_keys = []
    val_features = []

    for i, sample in enumerate(val_samples, start=1):

        image_obj = nib.load(
            sample["image_path"]
        )

        image = image_obj.get_fdata(
            dtype=np.float32
        )

        frame = int(
            sample["frame_idx"]
        )

        if image.ndim != 4:
            raise ValueError(
                "Expected a 4D source MRI."
            )

        if not 0 <= frame < image.shape[3]:
            raise ValueError(
                "Invalid MRI frame index."
            )

        features = extract_style_features(
            image[:, :, :, frame]
        )

        val_features.append([
            features[name]
            for name in feature_columns
        ])

        val_keys.append((
            str(sample["patient"]),
            str(sample["phase"])
        ))

        del image, image_obj

        if i % 20 == 0:
            print(f"Validation volumes: {i}/60")

    val_labels = predict_clusters(
        val_features,
        scaler,
        pca,
        kmeans,
        n_components
    )

    val_map = dict(
        zip(
            val_keys,
            val_labels.tolist()
        )
    )

    print(
        "Internal validation cluster counts:",
        np.bincount(val_labels, minlength=K)
    )

    # --------------------------------------------------------
    # Save experimental configuration before training
    # --------------------------------------------------------

    config = {
        "variant": "hard_prompt_k3_recovery",
        "seed": SEED,
        "k": K,
        "routing": "hard",
        "prompt_dim": PROMPT_DIM,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "optimizer": "Adam",
        "loss": "DiceCELoss",
        "pca_components": n_components,
        "feature_columns": feature_columns,
        "training_patients": 120,
        "internal_val_patients": 30,
        "training_volumes": 240,
        "internal_val_volumes": 60,
        "c_d_used_for_training": False,
        "c_d_used_for_model_selection": False,
        "cluster_source": str(CLUSTER_DIR),
    }

    save_json(
        config,
        "experiment_metadata.json"
    )

    verified_copy(
        Path(__file__),
        DRIVE_OUTPUT / Path(__file__).name
    )

    verified_copy(
        SPLIT_PATH,
        DRIVE_OUTPUT / "source_split.json"
    )

    # Save routing assignments.
    for filename, keys, mapping in (
        (
            "train_cluster_assignments.csv",
            train_keys,
            train_map
        ),
        (
            "internal_ab_validation_cluster_assignments.csv",
            val_keys,
            val_map
        )
    ):

        table = pd.DataFrame(
            keys,
            columns=["patient", "phase"]
        )

        table["latent_cluster"] = [
            mapping[key]
            for key in keys
        ]

        save_csv(table, filename)

    # --------------------------------------------------------
    # Dataset preprocessing
    # --------------------------------------------------------

    transforms = Compose([
        ResizeD(
            keys=["image"],
            spatial_size=(256, 256)
        ),
        ResizeD(
            keys=["mask"],
            spatial_size=(256, 256),
            mode="nearest"
        ),
    ])

    train_base = MNMSDataset(
        train_samples,
        transform=transforms
    )

    val_base = MNMSDataset(
        val_samples,
        transform=transforms
    )

    train_dataset = HardPromptDataset(
        train_base,
        train_map
    )

    val_dataset = HardPromptDataset(
        val_base,
        val_map
    )

    print(
        "Training slices:",
        len(train_dataset)
    )

    print(
        "Internal validation slices:",
        len(val_dataset)
    )

    # --------------------------------------------------------
    # Model and optimizer
    # --------------------------------------------------------

    reset_seed()

    model = HardPromptUNet().to(
        DEVICE
    )

    loss_fn = DiceCELoss(
        to_onehot_y=True,
        softmax=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_dice = -1.0
    best_epoch = 0

    history = []

    print("\nTRAINING: K=3 HARD ROUTING RECOVERY")

    for epoch in range(1, EPOCHS + 1):

        model.train()
        running_loss = 0.0

        for images, masks, prompt_ids in train_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            prompt_ids = prompt_ids.to(DEVICE)

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(
                images,
                prompt_ids
            )

            loss = loss_fn(
                outputs,
                masks.unsqueeze(1)
            )

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}"
                )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = (
            running_loss / len(train_loader)
        )

        dice, mean_dice = evaluate(
            model,
            val_loader
        )

        history.append({
            "epoch": epoch,
            "training_loss": avg_loss,
            "internal_val_dice_lv": float(dice[0]),
            "internal_val_dice_myocardium": float(dice[1]),
            "internal_val_dice_rv": float(dice[2]),
            "internal_val_mean_dice": mean_dice,
        })

        is_best = mean_dice > best_dice

        if is_best:

            best_dice = mean_dice
            best_epoch = epoch

            save_checkpoint(
                model.state_dict(),
                "best_model.pth"
            )

        # Save optimizer and model state for recovery.
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_epoch": best_epoch,
                "best_internal_ab_mean_dice": best_dice,
                "config": config,
                "history": history,
            },
            f"checkpoint_epoch_{epoch}.pth"
        )

        summary = {
            "variant": "hard_prompt_k3_recovery",
            "best_epoch": best_epoch,
            "best_internal_ab_mean_dice": best_dice,
            "epochs_completed": epoch,
            "best_model_path": str(
                DRIVE_OUTPUT / "best_model.pth"
            ),
        }

        save_csv(
            pd.DataFrame(history),
            "training_history.csv"
        )

        save_csv(
            pd.DataFrame([summary]),
            "training_summary.csv"
        )

        print(
            f"\nEpoch [{epoch}/{EPOCHS}]"
            f" | Loss: {avg_loss:.4f}"
            f" | Mean Dice: {mean_dice:.6f}"
        )

        print(
            f"LV: {dice[0]:.4f} | "
            f"MYO: {dice[1]:.4f} | "
            f"RV: {dice[2]:.4f}"
        )

        if is_best:
            print(
                f"New best checkpoint: epoch {best_epoch}"
            )

    print("\nK=3 HARD ROUTING RECOVERY COMPLETE")

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best internal A/B mean Dice:",
        f"{best_dice:.6f}"
    )

    print(
        "Verified Drive checkpoint:",
        DRIVE_OUTPUT / "best_model.pth"
    )

    print(
        "Verified training summary:",
        DRIVE_OUTPUT / "training_summary.csv"
    )

    print("No C/D data was used.")


if __name__ == "__main__":
    main()
