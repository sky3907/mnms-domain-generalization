
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)


# ============================================================
# Configuration
# ============================================================

INPUT_PATH = (
    "latent_domain_discovery/"
    "source_style_features.csv"
)

OUTPUT_DIR = (
    "latent_domain_discovery/figures"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# Load frozen latent-domain results
# ============================================================

df = pd.read_csv(
    INPUT_PATH
)

print(
    "Loaded volumes:",
    len(df)
)

print(
    "Clusters:",
    sorted(
        df["latent_cluster"].unique()
    )
)

print(
    "Vendors:",
    sorted(
        df["vendor"].unique()
    )
)

print(
    "Centres:",
    sorted(
        df["centre"].unique()
    )
)


# ============================================================
# Post-hoc quantitative analysis
#
# IMPORTANT:
# Vendor and centre labels are used only AFTER clustering.
# They had no role in feature extraction, PCA, K selection,
# or K-means fitting.
# ============================================================

cluster_labels = (
    df["latent_cluster"]
    .to_numpy()
)

vendor_codes = (
    pd.Categorical(
        df["vendor"]
    )
    .codes
)

centre_codes = (
    pd.Categorical(
        df["centre"]
    )
    .codes
)


vendor_ari = adjusted_rand_score(
    vendor_codes,
    cluster_labels,
)

vendor_nmi = normalized_mutual_info_score(
    vendor_codes,
    cluster_labels,
)

centre_ari = adjusted_rand_score(
    centre_codes,
    cluster_labels,
)

centre_nmi = normalized_mutual_info_score(
    centre_codes,
    cluster_labels,
)


print(
    "\n======================================"
)

print(
    "POST-HOC DOMAIN ASSOCIATION"
)

print(
    "======================================"
)

print(
    f"Vendor ARI: {vendor_ari:.4f}"
)

print(
    f"Vendor NMI: {vendor_nmi:.4f}"
)

print(
    f"Centre ARI: {centre_ari:.4f}"
)

print(
    f"Centre NMI: {centre_nmi:.4f}"
)


# ============================================================
# Save post-hoc association metrics
# ============================================================

association_df = pd.DataFrame(
    [
        {
            "metadata_label":
                "vendor",

            "ARI":
                vendor_ari,

            "NMI":
                vendor_nmi,
        },
        {
            "metadata_label":
                "centre",

            "ARI":
                centre_ari,

            "NMI":
                centre_nmi,
        },
    ]
)

association_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "posthoc_domain_association.csv",
    ),
    index=False,
)


# ============================================================
# Plot 1:
# PCA projection coloured by discovered latent cluster
# ============================================================

plt.figure(
    figsize=(7, 6)
)

for cluster in sorted(
    df["latent_cluster"].unique()
):

    subset = df[
        df["latent_cluster"] == cluster
    ]

    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        label=f"Latent cluster {cluster}",
        alpha=0.75,
        s=45,
    )


plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    "Latent-Domain Discovery on A/B Source Training Data"
)

plt.legend()

plt.tight_layout()

cluster_path = os.path.join(
    OUTPUT_DIR,
    "pca_by_latent_cluster.png",
)

plt.savefig(
    cluster_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Plot 2:
# Same PCA coordinates, coloured by vendor
#
# Vendor is shown ONLY for post-hoc interpretation.
# ============================================================

plt.figure(
    figsize=(7, 6)
)

for vendor in sorted(
    df["vendor"].unique()
):

    subset = df[
        df["vendor"] == vendor
    ]

    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        label=f"Vendor {vendor}",
        alpha=0.75,
        s=45,
    )


plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    "Post-hoc Vendor Distribution in Latent Style Space"
)

plt.legend()

plt.tight_layout()

vendor_path = os.path.join(
    OUTPUT_DIR,
    "pca_by_vendor.png",
)

plt.savefig(
    vendor_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Plot 3:
# Same PCA coordinates, coloured by centre
#
# Centre is also used ONLY for post-hoc interpretation.
# ============================================================

plt.figure(
    figsize=(7, 6)
)

for centre in sorted(
    df["centre"].unique()
):

    subset = df[
        df["centre"] == centre
    ]

    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        label=f"Centre {centre}",
        alpha=0.75,
        s=45,
    )


plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    "Post-hoc Centre Distribution in Latent Style Space"
)

plt.legend()

plt.tight_layout()

centre_path = os.path.join(
    OUTPUT_DIR,
    "pca_by_centre.png",
)

plt.savefig(
    centre_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Plot 4:
# Candidate K selection
# ============================================================

score_df = pd.read_csv(
    "latent_domain_discovery/"
    "cluster_selection.csv"
)


plt.figure(
    figsize=(7, 5)
)

plt.plot(
    score_df["k"],
    score_df["silhouette"],
    marker="o",
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Silhouette Score"
)

plt.title(
    "Source-Only Latent Cluster Selection"
)

plt.xticks(
    score_df["k"]
)

plt.tight_layout()

selection_path = os.path.join(
    OUTPUT_DIR,
    "cluster_selection_silhouette.png",
)

plt.savefig(
    selection_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Cluster composition tables
# ============================================================

vendor_counts = pd.crosstab(
    df["latent_cluster"],
    df["vendor"],
)

vendor_percent = pd.crosstab(
    df["latent_cluster"],
    df["vendor"],
    normalize="index",
) * 100


centre_counts = pd.crosstab(
    df["latent_cluster"],
    df["centre"],
)

centre_percent = pd.crosstab(
    df["latent_cluster"],
    df["centre"],
    normalize="index",
) * 100


print(
    "\n======================================"
)

print(
    "CLUSTER x VENDOR COUNTS"
)

print(
    "======================================"
)

print(
    vendor_counts
)


print(
    "\nCLUSTER x VENDOR ROW %"
)

print(
    vendor_percent.round(1)
)


print(
    "\n======================================"
)

print(
    "CLUSTER x CENTRE COUNTS"
)

print(
    "======================================"
)

print(
    centre_counts
)


print(
    "\nCLUSTER x CENTRE ROW %"
)

print(
    centre_percent.round(1)
)


vendor_percent.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cluster_vendor_percent.csv",
    )
)

centre_percent.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cluster_centre_percent.csv",
    )
)


# ============================================================
# Final summary
# ============================================================

print(
    "\nSaved figures:"
)

print(
    cluster_path
)

print(
    vendor_path
)

print(
    centre_path
)

print(
    selection_path
)
