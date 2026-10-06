
import os
import json
import joblib

import nibabel as nib
import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
)

from dataset import build_dataset


# ============================================================
# Configuration
# ============================================================

ROOT_DIR = "/content/OpenDataset"

# Frozen patient-level A/B source split from strict baseline
SPLIT_PATH = "strict_ab_cd/source_split.json"

OUTPUT_DIR = "latent_domain_discovery"

SEED = 42

# Candidate K values fixed before C/D evaluation
K_VALUES = [2, 3, 4, 5, 6]

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# Load frozen A/B source-training split
# ============================================================

print("Loading frozen source split...")

with open(SPLIT_PATH, "r") as f:
    split_info = json.load(f)

train_patients = set(
    split_info["train_patients"]
)

print(
    "Frozen training patients:",
    len(train_patients),
)


# ============================================================
# Load official training samples
# ============================================================

all_samples = build_dataset(
    split="train",
    root_dir=ROOT_DIR,
)

samples = [
    sample
    for sample in all_samples
    if sample["patient"] in train_patients
]


# ============================================================
# Safety checks
# ============================================================

assert len(samples) == 240, (
    f"Expected 240 patient-phase samples, got {len(samples)}"
)

vendors = sorted(
    set(
        sample["vendor"]
        for sample in samples
    )
)

assert vendors == ["A", "B"], (
    f"Unexpected source vendors: {vendors}"
)

patients = set(
    sample["patient"]
    for sample in samples
)

assert patients == train_patients

print(
    "Source patient-phase volumes:",
    len(samples),
)

print(
    "Source vendors:",
    vendors,
)

print(
    "Centres:",
    sorted(
        set(
            sample["centre"]
            for sample in samples
        )
    ),
)


# ============================================================
# Robust preprocessing
#
# IMPORTANT:
# - Image intensities only
# - No segmentation masks
# - Vendor/centre labels are not used for feature extraction
# ============================================================

def get_robust_voxels(volume):

    volume = np.asarray(
        volume,
        dtype=np.float32,
    )

    finite = volume[
        np.isfinite(volume)
    ]

    if finite.size == 0:
        raise ValueError(
            "Volume contains no finite voxels."
        )

    # Remove zero-valued background where possible.
    nonzero = finite[
        finite != 0
    ]

    if nonzero.size > 100:
        voxels = nonzero
    else:
        voxels = finite

    # Robust clipping.
    low, high = np.percentile(
        voxels,
        [1, 99],
    )

    clipped = np.clip(
        voxels,
        low,
        high,
    )

    # Robust per-volume normalization.
    median = np.median(
        clipped
    )

    q25, q75 = np.percentile(
        clipped,
        [25, 75],
    )

    iqr = q75 - q25

    normalized = (
        clipped - median
    ) / (
        iqr + 1e-8
    )

    return normalized.astype(
        np.float32
    )


# ============================================================
# Entropy
# ============================================================

def compute_entropy(values):

    hist, _ = np.histogram(
        values,
        bins=64,
        density=False,
    )

    probabilities = (
        hist.astype(np.float64)
        / max(hist.sum(), 1)
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    return float(
        -np.sum(
            probabilities
            * np.log2(probabilities)
        )
    )


# ============================================================
# Style feature extraction
#
# Removed:
# - coefficient_variation:
#   unstable after median centering
#
# - p50:
#   forced to approximately zero by median centering
#
# - IQR:
#   forced to approximately one by IQR normalization
# ============================================================

def extract_style_features(volume):

    x = get_robust_voxels(
        volume
    )

    (
        p1,
        p5,
        p25,
        _p50,
        p75,
        p95,
        p99,
    ) = np.percentile(
        x,
        [1, 5, 25, 50, 75, 95, 99],
    )

    mean = float(
        np.mean(x)
    )

    std = float(
        np.std(x)
    )

    feature_dict = {

        "intensity_mean":
            mean,

        "intensity_std":
            std,

        "p01":
            float(p1),

        "p05":
            float(p5),

        "p25":
            float(p25),

        "p75":
            float(p75),

        "p95":
            float(p95),

        "p99":
            float(p99),

        "skewness":
            float(skew(x)),

        "kurtosis":
            float(kurtosis(x)),

        "entropy":
            compute_entropy(x),
    }

    return feature_dict


# ============================================================
# Extract one feature vector per patient-phase volume
# ============================================================

print(
    "\nExtracting style features..."
)

rows = []

for idx, sample in enumerate(samples):

    image_obj = nib.load(
        sample["image_path"]
    )

    image = image_obj.get_fdata(
        dtype=np.float32
    )

    frame_idx = sample[
        "frame_idx"
    ]

    # One volume = one ED or ES frame
    volume = image[
        :,
        :,
        :,
        frame_idx
    ]

    features = extract_style_features(
        volume
    )

    row = {

        "patient":
            sample["patient"],

        "phase":
            sample["phase"],

        # Metadata retained ONLY for post-hoc analysis
        "vendor":
            sample["vendor"],

        "centre":
            sample["centre"],
    }

    row.update(
        features
    )

    rows.append(
        row
    )

    if (
        (idx + 1) % 20 == 0
        or idx + 1 == len(samples)
    ):

        print(
            f"{idx + 1}/{len(samples)} "
            "volumes processed"
        )


df = pd.DataFrame(
    rows
)


# ============================================================
# Feature matrix
# ============================================================

metadata_columns = [
    "patient",
    "phase",
    "vendor",
    "centre",
]

feature_columns = [
    column
    for column in df.columns
    if column not in metadata_columns
]

print(
    "\nFeatures:"
)

for feature in feature_columns:
    print(
        " ",
        feature,
    )


X = df[
    feature_columns
].to_numpy(
    dtype=np.float64
)


# ============================================================
# Numerical safety
# ============================================================

if not np.all(
    np.isfinite(X)
):
    raise ValueError(
        "Non-finite values detected "
        "in style features."
    )


# ============================================================
# Standardize features
#
# Fit ONLY on frozen A/B source-training data.
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# ============================================================
# PCA
#
# Fit ONLY on frozen A/B source-training data.
# ============================================================

pca = PCA()

X_pca_full = pca.fit_transform(
    X_scaled
)

cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

n_components_95 = (
    np.argmax(
        cumulative_variance >= 0.95
    )
    + 1
)

print(
    "\nPCA components for >=95% variance:",
    n_components_95,
)

print(
    "Explained variance:",
    cumulative_variance[
        n_components_95 - 1
    ],
)

X_cluster = X_pca_full[
    :,
    :n_components_95
]


# ============================================================
# Candidate K evaluation
#
# K selection uses only unsupervised source-training metrics.
# Vendor/centre labels are NOT used.
# ============================================================

print(
    "\nEvaluating candidate cluster counts..."
)

cluster_scores = []

models = {}

for k in K_VALUES:

    kmeans = KMeans(
        n_clusters=k,
        random_state=SEED,
        n_init=20,
    )

    labels = kmeans.fit_predict(
        X_cluster
    )

    silhouette = silhouette_score(
        X_cluster,
        labels,
    )

    db_score = davies_bouldin_score(
        X_cluster,
        labels,
    )

    cluster_scores.append(
        {
            "k":
                k,

            "silhouette":
                silhouette,

            "davies_bouldin":
                db_score,

            "inertia":
                kmeans.inertia_,
        }
    )

    models[k] = (
        kmeans,
        labels,
    )

    print(
        f"K={k}: "
        f"silhouette={silhouette:.4f}, "
        f"DB={db_score:.4f}, "
        f"inertia={kmeans.inertia_:.2f}"
    )


score_df = pd.DataFrame(
    cluster_scores
)


# ============================================================
# Fixed K-selection rule
#
# Select K with maximum silhouette score.
# ============================================================

best_k = int(
    score_df.loc[
        score_df[
            "silhouette"
        ].idxmax(),
        "k",
    ]
)

print(
    "\nSelected K:",
    best_k,
)


best_model, cluster_labels = (
    models[best_k]
)


# ============================================================
# Freeze latent-domain assignment pipeline
#
# These objects are fitted ONLY on frozen A/B source-training
# data.
#
# They must later be applied WITHOUT REFITTING to:
# - internal A/B validation
# - final unseen C/D evaluation
# ============================================================

scaler_path = os.path.join(
    OUTPUT_DIR,
    "style_scaler.joblib",
)

pca_path = os.path.join(
    OUTPUT_DIR,
    "style_pca.joblib",
)

kmeans_path = os.path.join(
    OUTPUT_DIR,
    "style_kmeans.joblib",
)


joblib.dump(
    scaler,
    scaler_path,
)

joblib.dump(
    pca,
    pca_path,
)

joblib.dump(
    best_model,
    kmeans_path,
)


print(
    "\nFrozen latent-domain assignment pipeline:"
)

print(
    "Scaler:",
    scaler_path,
)

print(
    "PCA:",
    pca_path,
)

print(
    "K-means:",
    kmeans_path,
)


# ============================================================
# Store cluster assignments
# ============================================================

df["latent_cluster"] = (
    cluster_labels
)


# ============================================================
# PCA coordinates for visualization
# ============================================================

df["PC1"] = (
    X_pca_full[:, 0]
)

df["PC2"] = (
    X_pca_full[:, 1]
)


# ============================================================
# ED/ES consistency sanity check
#
# Analysis only.
# Does NOT influence clustering.
# ============================================================

patient_cluster_counts = (
    df.groupby(
        "patient"
    )["latent_cluster"]
    .nunique()
)

same_cluster = int(
    (
        patient_cluster_counts == 1
    ).sum()
)

different_cluster = int(
    (
        patient_cluster_counts > 1
    ).sum()
)

consistency_percentage = (
    100.0
    * same_cluster
    / len(patient_cluster_counts)
)


print(
    "\n======================================"
)

print(
    "ED/ES CLUSTER CONSISTENCY"
)

print(
    "======================================"
)

print(
    "Patients with ED/ES in same cluster:",
    same_cluster,
)

print(
    "Patients with ED/ES in different clusters:",
    different_cluster,
)

print(
    f"Same-cluster percentage: "
    f"{consistency_percentage:.1f}%"
)


# ============================================================
# Cluster sizes
# ============================================================

print(
    "\n======================================"
)

print(
    "CLUSTER SIZES"
)

print(
    "======================================"
)

print(
    df[
        "latent_cluster"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# Cluster x phase
#
# Post-hoc sanity analysis only.
# ============================================================

print(
    "\n======================================"
)

print(
    "CLUSTER x PHASE"
)

print(
    "======================================"
)

phase_table = pd.crosstab(
    df["latent_cluster"],
    df["phase"],
)

print(
    phase_table
)


# ============================================================
# Post-hoc vendor analysis
#
# Vendor labels are used ONLY here.
# ============================================================

print(
    "\n======================================"
)

print(
    "CLUSTER x VENDOR"
)

print(
    "======================================"
)

vendor_table = pd.crosstab(
    df["latent_cluster"],
    df["vendor"],
)

print(
    vendor_table
)


# ============================================================
# Post-hoc centre analysis
#
# Centre labels are used ONLY here.
# ============================================================

print(
    "\n======================================"
)

print(
    "CLUSTER x CENTRE"
)

print(
    "======================================"
)

centre_table = pd.crosstab(
    df["latent_cluster"],
    df["centre"],
)

print(
    centre_table
)


# ============================================================
# Save result tables
# ============================================================

feature_path = os.path.join(
    OUTPUT_DIR,
    "source_style_features.csv",
)

score_path = os.path.join(
    OUTPUT_DIR,
    "cluster_selection.csv",
)

assignment_path = os.path.join(
    OUTPUT_DIR,
    "source_cluster_assignments.csv",
)


df.to_csv(
    feature_path,
    index=False,
)

score_df.to_csv(
    score_path,
    index=False,
)


df[
    [
        "patient",
        "phase",
        "vendor",
        "centre",
        "latent_cluster",
        "PC1",
        "PC2",
    ]
].to_csv(
    assignment_path,
    index=False,
)


phase_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cluster_phase_composition.csv",
    )
)

vendor_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cluster_vendor_composition.csv",
    )
)

centre_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cluster_centre_composition.csv",
    )
)


# ============================================================
# Save experiment metadata
# ============================================================

metadata = {

    "protocol":
        (
            "Latent-domain discovery on frozen "
            "A/B source-training split"
        ),

    "seed":
        SEED,

    "n_patient_phase_volumes":
        len(df),

    "n_patients":
        len(
            set(
                df["patient"]
            )
        ),

    "candidate_k":
        K_VALUES,

    "selected_k":
        best_k,

    "k_selection_rule":
        "maximum silhouette score",

    "pca_variance_threshold":
        0.95,

    "pca_components_95":
        int(
            n_components_95
        ),

    "pca_explained_variance":
        float(
            cumulative_variance[
                n_components_95 - 1
            ]
        ),

    "feature_columns":
        feature_columns,

    "normalization":
        (
            "1st-99th percentile clipping followed "
            "by per-volume median/IQR normalization"
        ),

    "clustering_unit":
        "patient-phase volume",

    "vendor_centre_used_for_clustering":
        False,

    "ed_es_same_cluster_patients":
        same_cluster,

    "ed_es_different_cluster_patients":
        different_cluster,

    "ed_es_consistency_percentage":
        consistency_percentage,

    "frozen_scaler":
        scaler_path,

    "frozen_pca":
        pca_path,

    "frozen_kmeans":
        kmeans_path,
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "clustering_metadata.json",
    ),
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=4,
    )


print(
    "\n======================================"
)

print(
    "LATENT-DOMAIN DISCOVERY COMPLETE"
)

print(
    "======================================"
)

print(
    "Selected K:",
    best_k,
)

print(
    "PCA components:",
    n_components_95,
)

print(
    f"ED/ES consistency: "
    f"{consistency_percentage:.1f}%"
)

print(
    "\nSaved results to:",
    OUTPUT_DIR,
)
