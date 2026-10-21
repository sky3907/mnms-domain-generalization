"""Train K=2 soft input prompts using only the frozen A/B source split.

Run from /content/mnms-domain-generalization after restoring the frozen
clustering files. Existing model checkpoints are never overwritten by a new run.
"""

import json
import random
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

ROOT_DIR = "/content/OpenDataset"
SPLIT_PATH = Path("strict_ab_cd/source_split.json")
LATENT_DIR = Path("latent_domain_discovery")
OUTPUT_DIR = Path("soft_prompt_k2")
CHECKPOINT_DIR = Path(
    "/content/drive/MyDrive/mnms_soft_prompt_k2_seed42"
)
SEED = 42
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3
PROMPT_DIM = 4
NUM_PROMPTS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reset_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_style_features(volume):
    """Match the feature extraction used by frozen latent-domain discovery."""
    x = np.asarray(volume, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No finite voxels")
    nonzero = x[x != 0]
    if nonzero.size > 100:
        x = nonzero
    low, high = np.percentile(x, [1, 99])
    x = np.clip(x, low, high)
    median = np.median(x)
    q25, q75 = np.percentile(x, [25, 75])
    x = ((x - median) / (q75 - q25 + 1e-8)).astype(np.float32)
    p1, p5, p25, _, p75, p95, p99 = np.percentile(
        x, [1, 5, 25, 50, 75, 95, 99]
    )
    hist, _ = np.histogram(x, bins=64)
    probs = hist[hist > 0].astype(np.float64)
    probs /= hist.sum()
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
        "entropy": float(-(probs * np.log2(probs)).sum()),
    }


def cluster_distances(X, scaler, pca, kmeans, n_components):
    X = np.asarray(X, dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("Non-finite style features")
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)[:, :n_components]
    return kmeans.transform(X_pca) ** 2


def soft_weights(distances, temperature):
    logits = -distances / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


class SoftPromptDataset(Dataset):
    def __init__(self, base_dataset, weight_map):
        self.base_dataset = base_dataset
        self.weight_map = weight_map

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, mask, metadata = self.base_dataset[idx]
        key = (str(metadata["patient"]), str(metadata["phase"]))
        return (
            image,
            mask,
            torch.tensor(self.weight_map[key], dtype=torch.float32),
        )


class SoftPromptUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_embedding = nn.Embedding(NUM_PROMPTS, PROMPT_DIM)
        self.unet = UNet(
            spatial_dims=2,
            in_channels=1 + PROMPT_DIM,
            out_channels=4,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

    def forward(self, images, weights):
        prompt = weights @ self.prompt_embedding.weight
        prompt = prompt[:, :, None, None]
        prompt = prompt.expand(-1, -1, images.shape[-2], images.shape[-1])
        return self.unet(torch.cat([images, prompt], dim=1))


def evaluate(model, loader):
    """Same global per-class resized-slice Dice used for A/B model selection."""
    model.eval()
    intersection = np.zeros(3, dtype=np.float64)
    prediction = np.zeros(3, dtype=np.float64)
    ground_truth = np.zeros(3, dtype=np.float64)
    with torch.inference_mode():
        for images, masks, weights in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            weights = weights.to(DEVICE)
            preds = torch.argmax(model(images, weights), dim=1)
            for i, cls in enumerate([1, 2, 3]):
                pred_binary = preds == cls
                gt_binary = masks == cls
                intersection[i] += (pred_binary & gt_binary).sum().item()
                prediction[i] += pred_binary.sum().item()
                ground_truth[i] += gt_binary.sum().item()
    dice = (2 * intersection + 1e-8) / (prediction + ground_truth + 1e-8)
    return dice, float(dice.mean())


def main():
    reset_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    with open(SPLIT_PATH) as f:
        split = json.load(f)
    train_ids = set(split["train_patients"])
    val_ids = set(split["internal_validation_patients"])
    assert len(train_ids) == 120
    assert len(val_ids) == 30
    assert train_ids.isdisjoint(val_ids)

    # Only the official labeled training partition is loaded.
    all_samples = build_dataset(split="train", root_dir=ROOT_DIR)
    train_samples = [s for s in all_samples if s["patient"] in train_ids]
    val_samples = [s for s in all_samples if s["patient"] in val_ids]
    assert len(train_samples) == 240
    assert len(val_samples) == 60
    assert {s["vendor"] for s in train_samples + val_samples} == {"A", "B"}
    print("Training volumes:", len(train_samples))
    print("Internal validation volumes:", len(val_samples))

    with open(LATENT_DIR / "clustering_metadata.json") as f:
        metadata = json.load(f)
    assert metadata["selected_k"] == 2
    assert metadata["vendor_centre_used_for_clustering"] is False
    feature_columns = metadata["feature_columns"]
    n_components = int(metadata["pca_components_95"])
    scaler = joblib.load(LATENT_DIR / "style_scaler.joblib")
    pca = joblib.load(LATENT_DIR / "style_pca.joblib")
    kmeans = joblib.load(LATENT_DIR / "style_kmeans.joblib")
    assert kmeans.n_clusters == 2

    train_df = pd.read_csv(
        LATENT_DIR / "source_style_features.csv",
        dtype={"patient": str, "phase": str},
    )
    expected_keys = {
        (str(s["patient"]), str(s["phase"])) for s in train_samples
    }
    actual_keys = set(zip(train_df["patient"], train_df["phase"]))
    assert len(train_df) == 240
    assert len(actual_keys) == 240
    assert actual_keys == expected_keys

    X_train = train_df[feature_columns].to_numpy(dtype=np.float64)
    d2_train = cluster_distances(X_train, scaler, pca, kmeans, n_components)
    hard_labels = np.argmin(d2_train, axis=1)
    assert np.array_equal(hard_labels, train_df["latent_cluster"].to_numpy())

    # Fixed source-only temperature heuristic; not tuned on C/D.
    temperature = float(np.median(np.abs(d2_train[:, 0] - d2_train[:, 1])))
    assert np.isfinite(temperature)
    assert temperature > 1e-8
    train_weights = soft_weights(d2_train, temperature)
    assert np.allclose(train_weights.sum(axis=1), 1.0)
    train_weight_map = {
        (str(row.patient), str(row.phase)): weights.astype(np.float32)
        for row, weights in zip(train_df.itertuples(index=False), train_weights)
    }
    print("Soft-routing temperature:", temperature)
    print("Mean training routing weights:", train_weights.mean(axis=0))

    print("Extracting internal A/B validation styles...")
    val_features, val_keys = [], []
    for sample in val_samples:
        image = nib.load(sample["image_path"]).get_fdata(dtype=np.float32)
        volume = image[:, :, :, int(sample["frame_idx"])]
        features = extract_style_features(volume)
        val_features.append([features[name] for name in feature_columns])
        val_keys.append((str(sample["patient"]), str(sample["phase"])))
    assert len(set(val_keys)) == 60
    d2_val = cluster_distances(val_features, scaler, pca, kmeans, n_components)
    val_weights = soft_weights(d2_val, temperature)
    assert np.allclose(val_weights.sum(axis=1), 1.0)
    val_weight_map = {
        key: weights.astype(np.float32)
        for key, weights in zip(val_keys, val_weights)
    }
    print("Mean validation routing weights:", val_weights.mean(axis=0))

    # --------------------------------------------------------
    # Save routing weights and configuration
    # --------------------------------------------------------

    # Stop instead of overwriting an earlier soft-model run.
    if any(CHECKPOINT_DIR.glob("*.pth")):
        raise FileExistsError(
            f"{CHECKPOINT_DIR} already contains checkpoints. "
            "Training stopped to protect them. Do not delete them."
        )

    for filename, keys, weights in [
        (
            "train_routing_weights.csv",
            list(zip(train_df["patient"], train_df["phase"])),
            train_weights,
        ),
        (
            "internal_ab_validation_routing_weights.csv",
            val_keys,
            val_weights,
        ),
    ]:
        routing_df = pd.DataFrame(keys, columns=["patient", "phase"])
        routing_df["prompt_0_weight"] = weights[:, 0]
        routing_df["prompt_1_weight"] = weights[:, 1]
        routing_df["nearest_cluster"] = weights.argmax(axis=1)

        for directory in (OUTPUT_DIR, CHECKPOINT_DIR):
            routing_df.to_csv(directory / filename, index=False)

    config = {
        "experiment": "K=2 soft input-prompt routing",
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "loss": "DiceCELoss",
        "optimizer": "Adam",
        "prompt_dim": PROMPT_DIM,
        "num_prompts": NUM_PROMPTS,
        "routing": "softmax(-squared_centroid_distance / temperature)",
        "temperature": temperature,
        "temperature_rule": "Median absolute squared-distance gap on A/B training",
        "feature_columns": feature_columns,
        "pca_components": n_components,
        "split_path": str(SPLIT_PATH),
        "checkpoint_selection": "Internal A/B global foreground-class mean Dice",
        "c_d_used_for_training_or_tuning": False,
    }

    for directory in (OUTPUT_DIR, CHECKPOINT_DIR):
        (directory / "experiment_metadata.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        (directory / "source_split.json").write_text(
            json.dumps(split, indent=2), encoding="utf-8"
        )

    # Preserve the executed script alongside the Drive checkpoints.
    (CHECKPOINT_DIR / "train_soft_prompt_k2.py").write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )

    # --------------------------------------------------------
    # Same slice preprocessing as the existing hard-prompt model
    # --------------------------------------------------------

    transforms = Compose([
        ResizeD(keys=["image"], spatial_size=(256, 256)),
        ResizeD(keys=["mask"], spatial_size=(256, 256), mode="nearest"),
    ])

    train_base = MNMSDataset(train_samples, transform=transforms)
    val_base = MNMSDataset(val_samples, transform=transforms)

    if len(train_base) == 0 or len(val_base) == 0:
        raise ValueError("Training or internal-validation dataset is empty.")

    print("Train slices:", len(train_base))
    print("Internal validation slices:", len(val_base))

    train_dataset = SoftPromptDataset(train_base, train_weight_map)
    val_dataset = SoftPromptDataset(val_base, val_weight_map)

    # Reset immediately before model initialization, as in prompt training.
    reset_seed()
    model = SoftPromptUNet().to(DEVICE)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )

    def save_checkpoint(payload, destination):
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)

    # --------------------------------------------------------
    # Training and internal A/B checkpoint selection only
    # --------------------------------------------------------

    best_dice = -1.0
    best_epoch = 0
    history = []

    print("\nTRAINING: K=2 SOFT PROMPT ROUTING")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for images, masks, weights in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            weights = weights.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images, weights)
            loss = loss_fn(outputs, masks.unsqueeze(1))

            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        dice, mean_dice = evaluate(model, val_loader)

        if not np.isfinite(dice).all():
            raise RuntimeError(f"Non-finite validation Dice at epoch {epoch}.")

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
            save_checkpoint(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "internal_val_mean_dice": mean_dice,
            "best_internal_ab_mean_dice": best_dice,
            "best_epoch": best_epoch,
            "temperature": temperature,
            "config": config,
            "history": history,
        }
        save_checkpoint(
            checkpoint, CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"
        )

        summary = {
            "variant": "soft_prompt_k2",
            "best_epoch": best_epoch,
            "best_internal_ab_mean_dice": best_dice,
            "epochs_completed": epoch,
            "temperature": temperature,
            "best_model_path": str(CHECKPOINT_DIR / "best_model.pth"),
        }

        # CSVs and metadata persist in Drive, not just the Colab runtime.
        for directory in (OUTPUT_DIR, CHECKPOINT_DIR):
            pd.DataFrame(history).to_csv(
                directory / "training_history.csv", index=False
            )
            pd.DataFrame([summary]).to_csv(
                directory / "training_summary.csv", index=False
            )

        print(f"\nEpoch [{epoch}/{EPOCHS}] | Training loss: {avg_loss:.4f}")
        print(f"Internal A/B LV Dice:         {dice[0]:.4f}")
        print(f"Internal A/B Myocardium Dice: {dice[1]:.4f}")
        print(f"Internal A/B RV Dice:         {dice[2]:.4f}")
        print(f"Internal A/B Mean Dice:       {mean_dice:.4f}")
        if is_best:
            print(f"New best checkpoint: epoch {best_epoch}")

    print("\nK=2 SOFT-PROMPT TRAINING COMPLETE")
    print(f"Best epoch: {best_epoch}")
    print(f"Best internal A/B mean Dice: {best_dice:.4f}")
    print("Best model:", CHECKPOINT_DIR / "best_model.pth")
    print("Persistent history:", CHECKPOINT_DIR / "training_history.csv")
    print("Persistent routing settings:", CHECKPOINT_DIR / "experiment_metadata.json")
    print("C/D was not loaded or used for model selection.")


if __name__ == "__main__":
    main()
