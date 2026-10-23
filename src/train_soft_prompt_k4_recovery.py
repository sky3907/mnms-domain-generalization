
import os
import json
import hashlib
import shutil
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, ResizeD
from monai.losses import DiceCELoss

from dataset import build_dataset
from mnms_dataset import MNMSDataset

# Import definitions only. Does not execute hard training.
import train_hard_prompt_k4_recovery as hard


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_ROOT = "/content/OpenDataset"

SPLIT_PATH = (
    PROJECT_DIR / "strict_ab_cd/source_split.json"
)

CLUSTER_DIR = (
    PROJECT_DIR / "latent_domain_ablations/k4"
)

HARD_RECOVERY_DIR = Path(
    "/content/drive/MyDrive/mnms_hard_prompt_k4_recovery_seed42"
)

LOCAL_OUTPUT = (
    PROJECT_DIR / "soft_prompt_k4_recovery"
)

DRIVE_OUTPUT = Path(
    "/content/drive/MyDrive/mnms_soft_prompt_k4_recovery_seed42"
)

SEED = 42
K = 4
PROMPT_DIM = 4

BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3

DEVICE = hard.DEVICE


# ============================================================
# Verified file saving
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

    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(
            f"Missing or empty source file: {source}"
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
            f"Unfinished copy detected: {temporary}"
        )

    with source.open("rb") as src:
        with temporary.open("wb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())

    if sha256(source) != sha256(temporary):
        raise RuntimeError(
            f"Checksum mismatch: {destination}"
        )

    temporary.replace(destination)

    if (
        destination.stat().st_size != source.stat().st_size
        or sha256(destination) != sha256(source)
    ):
        raise RuntimeError(
            f"Final verification failed: {destination}"
        )


def save_json(data, filename):

    local = LOCAL_OUTPUT / filename
    temporary = local.with_name(local.name + ".tmp")

    with temporary.open("w") as f:
        json.dump(data, f, indent=2)

    temporary.replace(local)

    verified_copy(
        local,
        DRIVE_OUTPUT / filename
    )


def save_csv(df, filename):

    local = LOCAL_OUTPUT / filename
    temporary = local.with_name(local.name + ".tmp")

    df.to_csv(temporary, index=False)
    temporary.replace(local)

    verified_copy(
        local,
        DRIVE_OUTPUT / filename
    )


def save_checkpoint(data, filename):

    local = LOCAL_OUTPUT / filename
    temporary = local.with_name(local.name + ".tmp")

    torch.save(data, temporary)
    temporary.replace(local)

    # Confirm local checkpoint is readable.
    torch.load(
        local,
        map_location="cpu",
        weights_only=True
    )

    destination = DRIVE_OUTPUT / filename

    verified_copy(local, destination)

    # Confirm the Drive checkpoint is readable.
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
# Frozen K=4 cluster distances
# ============================================================

def squared_distances(
    X,
    scaler,
    pca,
    kmeans,
    n_components
):

    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError(
            "Invalid style-feature matrix."
        )

    X_scaled = scaler.transform(X)

    X_pca = pca.transform(X_scaled)[
        :, :n_components
    ]

    distances = kmeans.transform(X_pca) ** 2

    if distances.shape != (len(X), K):
        raise ValueError(
            "Unexpected number of cluster distances."
        )

    if not np.isfinite(distances).all():
        raise ValueError(
            "Non-finite cluster distances."
        )

    return distances


def calculate_soft_weights(distances, temperature):

    if (
        not np.isfinite(temperature)
        or temperature <= 1e-8
    ):
        raise ValueError(
            "Invalid soft-routing temperature."
        )

    logits = -distances / temperature

    logits -= logits.max(
        axis=1,
        keepdims=True
    )

    weights = np.exp(logits)

    weights /= weights.sum(
        axis=1,
        keepdims=True
    )

    if not np.isfinite(weights).all():
        raise ValueError(
            "Non-finite soft-routing weights."
        )

    if not np.allclose(
        weights.sum(axis=1), 1.0
    ):
        raise ValueError(
            "Routing weights do not sum to one."
        )

    return weights


# ============================================================
# Restore exact sample order from recovered hard experiment
# ============================================================

def restore_sample_order(samples, saved_csv):

    table = pd.read_csv(
        saved_csv,
        dtype={"patient": str, "phase": str}
    )

    ordered_keys = list(
        zip(
            table["patient"],
            table["phase"]
        )
    )

    sample_map = {
        (
            str(sample["patient"]),
            str(sample["phase"])
        ): sample
        for sample in samples
    }

    if (
        len(ordered_keys) != len(set(ordered_keys))
        or set(ordered_keys) != set(sample_map)
    ):
        raise ValueError(
            f"Sample order mismatch: {saved_csv}"
        )

    return (
        [sample_map[key] for key in ordered_keys],
        ordered_keys
    )


# ============================================================
# Dataset
# ============================================================

class SoftPromptDataset(Dataset):

    def __init__(self, base_dataset, weight_map):

        self.base_dataset = base_dataset
        self.weight_map = weight_map

    def __len__(self):

        return len(self.base_dataset)

    def __getitem__(self, idx):

        image, mask, metadata = self.base_dataset[idx]

        key = (
            str(metadata["patient"]),
            str(metadata["phase"])
        )

        weights = self.weight_map[key]

        return (
            image,
            mask,
            torch.tensor(
                weights,
                dtype=torch.float32
            )
        )


# ============================================================
# Soft Prompt U-Net
#
# Inherits the exact K=4 hard model architecture.
# Only prompt mixing changes.
# ============================================================

class SoftPromptUNet(hard.HardPromptUNet):

    def forward(self, images, weights):

        # weights: [batch_size, 3]
        # embedding matrix: [3, 4]

        prompt = (
            weights @ self.prompt_embedding.weight
        )

        # [B, 4] -> [B, 4, 1, 1]
        prompt = prompt[:, :, None, None]

        # Broadcast over image height and width.
        prompt = prompt.expand(
            -1,
            -1,
            images.shape[-2],
            images.shape[-1]
        )

        prompted_input = torch.cat(
            [images, prompt],
            dim=1
        )

        return self.unet(prompted_input)


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Preflight
    # --------------------------------------------------------

    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError(
            "Mount Google Drive first."
        )

    required = [
        SPLIT_PATH,
        CLUSTER_DIR / "clustering_metadata.json",
        CLUSTER_DIR / "source_style_features.csv",
        CLUSTER_DIR / "style_scaler.joblib",
        CLUSTER_DIR / "style_pca.joblib",
        CLUSTER_DIR / "style_kmeans.joblib",
        HARD_RECOVERY_DIR / "training_summary.csv",
        HARD_RECOVERY_DIR / "experiment_metadata.json",
        HARD_RECOVERY_DIR / "train_cluster_assignments.csv",
        HARD_RECOVERY_DIR / "internal_ab_validation_cluster_assignments.csv",
        HARD_RECOVERY_DIR / "best_model.pth",
    ]

    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Missing or empty file: {path}"
            )

    # Never overwrite an existing or partially completed run.
    for folder in (LOCAL_OUTPUT, DRIVE_OUTPUT):
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(
                f"Existing recovery files detected: {folder}. "
                "Do not delete or overwrite them."
            )

    # --------------------------------------------------------
    # Verify hard recovery
    # --------------------------------------------------------

    hard_summary = pd.read_csv(
        HARD_RECOVERY_DIR / "training_summary.csv"
    )

    if len(hard_summary) != 1:
        raise ValueError(
            "Invalid hard-routing summary."
        )

    record = hard_summary.iloc[0]

    if (
        record["variant"] != "hard_prompt_k4_recovery"
        or int(record["epochs_completed"]) != 10
    ):
        raise ValueError(
            "The K=4 hard reference run is incomplete."
        )

    with (
        HARD_RECOVERY_DIR / "experiment_metadata.json"
    ).open() as f:
        hard_config = json.load(f)

    expected_config = {
        "seed": SEED,
        "k": K,
        "prompt_dim": PROMPT_DIM,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
    }

    for key, value in expected_config.items():
        if hard_config.get(key) != value:
            raise ValueError(
                f"K=4 hard reference mismatch: {key}"
            )

    if (
        hard.K != K
        or hard.PROMPT_DIM != PROMPT_DIM
        or hard.SEED != SEED
    ):
        raise ValueError(
            "Imported hard model settings differ."
        )

    # --------------------------------------------------------
    # Frozen A/B split
    # --------------------------------------------------------

    with SPLIT_PATH.open() as f:
        split = json.load(f)

    with (
        CLUSTER_DIR / "clustering_metadata.json"
    ).open() as f:
        cluster_metadata = json.load(f)

    train_ids = set(
        map(str, split["train_patients"])
    )

    val_ids = set(
        map(str, split["internal_validation_patients"])
    )

    if (
        len(train_ids) != 120
        or len(val_ids) != 30
        or train_ids & val_ids
    ):
        raise ValueError(
            "Invalid frozen A/B patient split."
        )

    if (
        cluster_metadata["selected_k"] != K
        or cluster_metadata[
            "vendor_centre_used_for_clustering"
        ] is not False
    ):
        raise ValueError(
            "Expected source-only K=4 clustering."
        )

    hard.reset_seed()

    all_samples = build_dataset(
        split="train",
        root_dir=DATASET_ROOT
    )

    train_samples = [
        sample for sample in all_samples
        if str(sample["patient"]) in train_ids
    ]

    val_samples = [
        sample for sample in all_samples
        if str(sample["patient"]) in val_ids
    ]

    if len(train_samples) != 240 or len(val_samples) != 60:
        raise ValueError(
            "Unexpected A/B source volume counts."
        )

    train_samples, train_keys = restore_sample_order(
        train_samples,
        HARD_RECOVERY_DIR / "train_cluster_assignments.csv"
    )

    val_samples, val_keys = restore_sample_order(
        val_samples,
        HARD_RECOVERY_DIR
        / "internal_ab_validation_cluster_assignments.csv"
    )

    if any(
        sample["vendor"] not in {"A", "B"}
        for sample in train_samples + val_samples
    ):
        raise ValueError(
            "Unexpected non-source vendor."
        )

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(
        "Training / validation volumes:",
        len(train_samples),
        "/",
        len(val_samples)
    )

    print(
        "Verified frozen split and same sample order "
        "as recovered K=4 hard routing."
    )

    # --------------------------------------------------------
    # Frozen clustering pipeline
    # --------------------------------------------------------

    feature_columns = cluster_metadata[
        "feature_columns"
    ]

    n_components = int(
        cluster_metadata["pca_components_95"]
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

    if kmeans.n_clusters != K:
        raise ValueError(
            "Saved K-means cluster count is incorrect."
        )

    features_df = pd.read_csv(
        CLUSTER_DIR / "source_style_features.csv",
        dtype={"patient": str, "phase": str}
    )

    feature_keys = list(
        zip(
            features_df["patient"],
            features_df["phase"]
        )
    )

    if (
        len(feature_keys) != 240
        or len(set(feature_keys)) != 240
        or set(feature_keys) != set(train_keys)
    ):
        raise ValueError(
            "Training style features do not match."
        )

    X_train = features_df[
        feature_columns
    ].to_numpy(dtype=np.float64)

    distances_train = squared_distances(
        X_train,
        scaler,
        pca,
        kmeans,
        n_components
    )

    if not np.array_equal(
        distances_train.argmin(axis=1),
        features_df["latent_cluster"].to_numpy()
    ):
        raise ValueError(
            "Recovered K=4 assignments differ."
        )

    # --------------------------------------------------------
    # Source-only temperature
    #
    # Median gap between nearest and second-nearest
    # squared distances, calculated from A/B training only.
    # --------------------------------------------------------

    ordered_distances = np.sort(
        distances_train,
        axis=1
    )

    temperature = float(
        np.median(
            ordered_distances[:, 1]
            - ordered_distances[:, 0]
        )
    )

    train_weights = calculate_soft_weights(
        distances_train,
        temperature
    )

    train_weight_map = dict(
        zip(
            feature_keys,
            train_weights.astype(np.float32)
        )
    )

    print("Soft-routing temperature:", temperature)

    print(
        "Mean training routing weights:",
        train_weights.mean(axis=0)
    )

    # --------------------------------------------------------
    # Internal A/B validation routing
    # --------------------------------------------------------

    print(
        "Extracting internal A/B validation styles..."
    )

    val_features = []

    for i, sample in enumerate(val_samples, start=1):

        image_obj = nib.load(
            sample["image_path"]
        )

        image = image_obj.get_fdata(
            dtype=np.float32
        )

        frame = int(sample["frame_idx"])

        if image.ndim != 4 or not 0 <= frame < image.shape[3]:
            raise ValueError(
                "Invalid validation image/frame."
            )

        features = hard.extract_style_features(
            image[:, :, :, frame]
        )

        val_features.append([
            features[name]
            for name in feature_columns
        ])

        del image, image_obj

        if i % 20 == 0:
            print(
                f"Validation volumes: {i}/60"
            )

    distances_val = squared_distances(
        val_features,
        scaler,
        pca,
        kmeans,
        n_components
    )

    val_weights = calculate_soft_weights(
        distances_val,
        temperature
    )

    val_weight_map = dict(
        zip(
            val_keys,
            val_weights.astype(np.float32)
        )
    )

    print(
        "Mean validation routing weights:",
        val_weights.mean(axis=0)
    )

    # --------------------------------------------------------
    # Save configuration and routing records
    # --------------------------------------------------------

    LOCAL_OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    DRIVE_OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    config = {
        **expected_config,
        "variant": "soft_prompt_k4_recovery",
        "routing": (
            "softmax(-squared_centroid_distance / temperature)"
        ),
        "temperature": temperature,
        "temperature_rule": (
            "median(second_nearest_d2 - nearest_d2) "
            "on A/B training volumes"
        ),
        "feature_columns": feature_columns,
        "pca_components": n_components,
        "loss": "DiceCELoss",
        "optimizer": "Adam",
        "initialization": "from scratch",
        "checkpoint_selection": "internal A/B pooled Dice",
        "c_d_used_for_training_or_tuning": False,
        "hard_reference_best_dice": float(
            record["best_internal_ab_mean_dice"]
        ),
    }

    save_json(
        config,
        "experiment_metadata.json"
    )

    for script in (
        Path(__file__),
        Path(hard.__file__)
    ):
        verified_copy(
            script,
            DRIVE_OUTPUT / script.name
        )

    verified_copy(
        SPLIT_PATH,
        DRIVE_OUTPUT / "source_split.json"
    )

    # Preserve the precise fitted clustering pipeline.
    for filename in (
        "style_scaler.joblib",
        "style_pca.joblib",
        "style_kmeans.joblib",
        "clustering_metadata.json"
    ):
        verified_copy(
            CLUSTER_DIR / filename,
            DRIVE_OUTPUT / "clustering" / filename
        )

    for filename, keys, mapping in (
        (
            "train_routing_weights.csv",
            train_keys,
            train_weight_map
        ),
        (
            "internal_ab_validation_routing_weights.csv",
            val_keys,
            val_weight_map
        )
    ):

        df = pd.DataFrame(
            keys,
            columns=["patient", "phase"]
        )

        for cluster in range(K):
            df[f"weight_{cluster}"] = [
                float(mapping[key][cluster])
                for key in keys
            ]

        save_csv(df, filename)

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

    train_dataset = SoftPromptDataset(
        MNMSDataset(
            train_samples,
            transform=transforms
        ),
        train_weight_map
    )

    val_dataset = SoftPromptDataset(
        MNMSDataset(
            val_samples,
            transform=transforms
        ),
        val_weight_map
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

    hard.reset_seed()

    model = SoftPromptUNet().to(DEVICE)

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

    print("\nTRAINING: K=4 SOFT ROUTING RECOVERY")

    for epoch in range(1, EPOCHS + 1):

        model.train()
        running_loss = 0.0

        for images, masks, weights in train_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            weights = weights.to(DEVICE)

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(
                images,
                weights
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

        # Hard evaluator also works with continuous weights
        # because it passes the third batch tensor to model().
        dice, mean_dice = hard.evaluate(
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

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_epoch": best_epoch,
                "best_internal_ab_mean_dice": best_dice,
                "temperature": temperature,
                "config": config,
                "history": history,
            },
            f"checkpoint_epoch_{epoch}.pth"
        )

        summary = {
            "variant": "soft_prompt_k4_recovery",
            "best_epoch": best_epoch,
            "best_internal_ab_mean_dice": best_dice,
            "epochs_completed": epoch,
            "temperature": temperature,
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

    print("\nK=4 SOFT ROUTING RECOVERY COMPLETE")

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
