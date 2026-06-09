# -*- coding: utf-8 -*-
"""
Visualize DINOv3 embeddings using t-SNE and UMAP.

Input:
    - dinov3_vitl16_embeddings.npz
    - dinov3_embedding_metadata.csv

Output:
    - tsne_dinov3_cls.png
    - umap_dinov3_cls.png
    - tsne coordinates CSV
    - umap coordinates CSV
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

try:
    import umap
except ImportError:
    raise ImportError(
        "umap-learn is required. Install it using: pip install umap-learn"
    )


LABEL_ORDER = ["normal", "crackle", "wheeze", "both"]


def plot_embedding(df, title, save_path):
    plt.figure(figsize=(8, 6))

    for label in LABEL_ORDER:
        sub = df[df["label"] == label]
        plt.scatter(
            sub["x"],
            sub["y"],
            s=8,
            alpha=0.7,
            label=label,
        )

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved:", save_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embedding_npz",
        type=str,
        required=True,
        help="Path to dinov3_vitl16_embeddings.npz."
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        required=True,
        help="Path to dinov3_embedding_metadata.csv."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save visualization outputs."
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="cls",
        choices=["cls", "mean"],
        help="Embedding type to visualize."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--umap_neighbors",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--umap_min_dist",
        type=float,
        default=0.1,
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    emb = np.load(args.embedding_npz)
    meta = pd.read_csv(args.metadata_csv)

    if args.embedding_type == "cls":
        X = emb["cls_embeddings"]
    else:
        X = emb["mean_token_embeddings"]

    X_scaled = StandardScaler().fit_transform(X)

    print("Running t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate="auto",
        init="pca",
        random_state=args.seed,
    )

    X_tsne = tsne.fit_transform(X_scaled)

    tsne_df = pd.DataFrame({
        "x": X_tsne[:, 0],
        "y": X_tsne[:, 1],
        "label": meta["label"].values,
        "split": meta["split"].values,
        "patient_id": meta["patient_id"].values,
    })

    tsne_csv = os.path.join(
        args.output_dir,
        f"tsne_dinov3_{args.embedding_type}.csv"
    )
    tsne_png = os.path.join(
        args.output_dir,
        f"tsne_dinov3_{args.embedding_type}.png"
    )

    tsne_df.to_csv(tsne_csv, index=False, encoding="utf-8-sig")

    plot_embedding(
        tsne_df,
        f"t-SNE of DINOv3 {args.embedding_type.upper()} Embeddings",
        tsne_png,
    )

    print("Running UMAP...")
    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=args.seed,
    )

    X_umap = reducer.fit_transform(X_scaled)

    umap_df = pd.DataFrame({
        "x": X_umap[:, 0],
        "y": X_umap[:, 1],
        "label": meta["label"].values,
        "split": meta["split"].values,
        "patient_id": meta["patient_id"].values,
    })

    umap_csv = os.path.join(
        args.output_dir,
        f"umap_dinov3_{args.embedding_type}.csv"
    )
    umap_png = os.path.join(
        args.output_dir,
        f"umap_dinov3_{args.embedding_type}.png"
    )

    umap_df.to_csv(umap_csv, index=False, encoding="utf-8-sig")

    plot_embedding(
        umap_df,
        f"UMAP of DINOv3 {args.embedding_type.upper()} Embeddings",
        umap_png,
    )


if __name__ == "__main__":
    main()
