
import json
import hashlib
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


PROJECT_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_DIR / "latent_domain_discovery"
SPLIT_PATH = PROJECT_DIR / "strict_ab_cd/source_split.json"

OUTPUT_ROOT = PROJECT_DIR / "latent_domain_ablations"

DRIVE_ROOT = Path(
    "/content/drive/MyDrive/mnms_latent_domain_ablations"
)

SEED = 42
N_INIT = 20

EXPECTED = {
    3: {
        "sizes": [95, 80, 65],
        "silhouette": 0.3206,
    },
    4: {
        "sizes": [75, 55, 56, 54],
        "silhouette": 0.3208,
    },
}


def sha256(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def main():

    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Mount Google Drive first.")

    required = {
        "features": SOURCE_DIR / "source_style_features.csv",
        "metadata": SOURCE_DIR / "clustering_metadata.json",
        "scaler": SOURCE_DIR / "style_scaler.joblib",
        "pca": SOURCE_DIR / "style_pca.joblib",
        "kmeans_k2": SOURCE_DIR / "style_kmeans.joblib",
        "split": SPLIT_PATH,
    }

    for path in required.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Missing or empty original file: {path}"
            )

    # Protect any existing recovery outputs.
    for k in (3, 4):
        for folder in (
            OUTPUT_ROOT / f"k{k}",
            DRIVE_ROOT / f"k{k}",
        ):
            if folder.exists() and any(folder.iterdir()):
                raise FileExistsError(
                    f"Existing files detected: {folder}. "
                    "Stop and inspect them before proceeding."
                )

    with required["metadata"].open() as f:
        metadata = json.load(f)

    with SPLIT_PATH.open() as f:
        split = json.load(f)

    train_ids = set(
        map(str, split["train_patients"])
    )

    val_ids = set(
        map(str, split["internal_validation_patients"])
    )

    assert len(train_ids) == 120
    assert len(val_ids) == 30
    assert train_ids.isdisjoint(val_ids)
    assert metadata["selected_k"] == 2
    assert metadata["vendor_centre_used_for_clustering"] is False

    feature_columns = metadata["feature_columns"]
    n_components = int(metadata["pca_components_95"])

    df = pd.read_csv(
        required["features"],
        dtype={"patient": str, "phase": str},
    )

    keys = set(zip(df["patient"], df["phase"]))

    expected_keys = {
        (patient, phase)
        for patient in train_ids
        for phase in ("ED", "ES")
    }

    assert len(df) == 240
    assert len(keys) == 240
    assert keys == expected_keys
    assert df["vendor"].isin(["A", "B"]).all()

    X = df[feature_columns].to_numpy(
        dtype=np.float64
    )

    assert np.isfinite(X).all()

    # Load original fitted objects.
    # No new scaler or PCA is fitted.
    scaler = joblib.load(required["scaler"])
    pca = joblib.load(required["pca"])
    kmeans_k2 = joblib.load(required["kmeans_k2"])

    X_pca = pca.transform(
        scaler.transform(X)
    )

    X_cluster = X_pca[:, :n_components]

    # Verify that the original pipeline is unchanged.
    reproduced_k2 = kmeans_k2.predict(
        X_cluster
    )

    if not np.array_equal(
        reproduced_k2,
        df["latent_cluster"].to_numpy(),
    ):
        raise RuntimeError(
            "Original K=2 assignments were not reproduced."
        )

    print("Original K=2 pipeline verified.")
    print("Training volumes:", len(df))
    print("Style features:", len(feature_columns))
    print("PCA components:", n_components)

    source_hashes = {
        name: sha256(path)
        for name, path in required.items()
    }

    # Prepare and verify BOTH new models before writing.
    prepared = {}

    for k in (3, 4):

        model = KMeans(
            n_clusters=k,
            random_state=SEED,
            n_init=N_INIT,
            algorithm="lloyd",
        )

        labels = model.fit_predict(X_cluster)

        sizes = np.bincount(
            labels, minlength=k
        ).tolist()

        silhouette = float(
            silhouette_score(X_cluster, labels)
        )

        db = float(
            davies_bouldin_score(X_cluster, labels)
        )

        print(f"\nK={k}")
        print("Cluster sizes:", sizes)
        print("Silhouette:", round(silhouette, 6))
        print("Davies-Bouldin:", round(db, 6))

        if sizes != EXPECTED[k]["sizes"]:
            raise RuntimeError(
                f"K={k} cluster sizes differ from the "
                "previous experiment. Stop and investigate."
            )

        if abs(
            silhouette - EXPECTED[k]["silhouette"]
        ) > 0.001:
            raise RuntimeError(
                f"K={k} silhouette differs from "
                "the previous experiment."
            )

        prepared[k] = (model, labels, sizes, silhouette, db)

    # Save only after reproducing both original results.
    for k, (
        model, labels, sizes, silhouette, db
    ) in prepared.items():

        result = df.copy()

        result["latent_cluster"] = labels
        result["PC1"] = X_pca[:, 0]
        result["PC2"] = X_pca[:, 1]

        assignments = result[
            [
                "patient",
                "phase",
                "vendor",
                "centre",
                "latent_cluster",
                "PC1",
                "PC2",
            ]
        ]

        same_cluster = int(
            (
                result.groupby("patient")["latent_cluster"]
                .nunique() == 1
            ).sum()
        )

        cluster_metadata = {
            "selected_k": k,
            "seed": SEED,
            "n_init": N_INIT,
            "feature_columns": feature_columns,
            "pca_components_95": n_components,
            "scaler_and_pca_refitted": False,
            "vendor_centre_used_for_clustering": False,
            "silhouette": silhouette,
            "davies_bouldin": db,
            "cluster_sizes": sizes,
            "ed_es_same_cluster_patients": same_cluster,
            "source_input_sha256": source_hashes,
            "sklearn_version": sklearn.__version__,
        }

        local = OUTPUT_ROOT / f"k{k}"
        backup = DRIVE_ROOT / f"k{k}"

        local.mkdir(parents=True, exist_ok=True)
        backup.mkdir(parents=True, exist_ok=True)

        result.to_csv(
            local / "source_style_features.csv",
            index=False,
        )

        assignments.to_csv(
            local / "source_cluster_assignments.csv",
            index=False,
        )

        joblib.dump(
            model,
            local / "style_kmeans.joblib",
        )

        shutil.copy2(
            required["scaler"],
            local / "style_scaler.joblib",
        )

        shutil.copy2(
            required["pca"],
            local / "style_pca.joblib",
        )

        shutil.copy2(
            SPLIT_PATH,
            local / "source_split.json",
        )

        for name in ("vendor", "centre", "phase"):
            composition = pd.crosstab(
                result["latent_cluster"],
                result[name],
            )

            composition.to_csv(
                local / f"cluster_{name}_composition.csv"
            )

        with (
            local / "clustering_metadata.json"
        ).open("w") as f:
            json.dump(cluster_metadata, f, indent=2)

        shutil.copy2(
            Path(__file__),
            local / "recover_cluster_ablations.py",
        )

        # Copy and verify every file in Drive.
        for source_file in sorted(local.iterdir()):

            destination = backup / source_file.name

            shutil.copy2(source_file, destination)

            if destination.stat().st_size == 0:
                raise RuntimeError(
                    f"Empty Drive backup: {destination}"
                )

            if sha256(source_file) != sha256(destination):
                raise RuntimeError(
                    f"Backup checksum mismatch: {destination}"
                )

        # Confirm the saved Drive model can be loaded.
        restored_model = joblib.load(
            backup / "style_kmeans.joblib"
        )

        if not np.array_equal(
            restored_model.predict(X_cluster),
            labels,
        ):
            raise RuntimeError(
                f"K={k} saved K-means model mismatch."
            )

        print(f"\nK={k} RECOVERY COMPLETE")
        print("ED/ES same cluster:", same_cluster, "/120")
        print("Local:", local)
        print("Drive:", backup)
        print("Backup verification: PASSED")

    print("\nALL K=3 AND K=4 CLUSTERING FILES RECOVERED")
    print("Original K=2 files were not modified.")
    print("No segmentation models were trained.")
    print("No C/D data was used.")


if __name__ == "__main__":
    main()
