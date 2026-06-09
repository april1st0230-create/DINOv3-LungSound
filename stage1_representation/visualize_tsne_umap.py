import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap


def scatter_plot(df, title, save_path):
    label_order = ["normal", "crackle", "wheeze", "both"]

    plt.figure(figsize=(8, 6))
    for label in label_order:
        sub = df[df["label"] == label]
        plt.scatter(sub["x"], sub["y"], s=8, alpha=0.7, label=label)

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved:", save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="dinov3_outputs")
    parser.add_argument("--embedding_key", type=str, default="cls_embeddings",
                        choices=["cls_embeddings", "mean_token_embeddings"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = os.path.join(args.root, args.out_dir)
    emb_path = os.path.join(out_dir, "dinov3_vitl16_embeddings.npz")
    meta_path = os.path.join(out_dir, "dinov3_embedding_metadata.csv")

    emb = np.load(emb_path)
    meta = pd.read_csv(meta_path)

    X = emb[args.embedding_key]
    X_scaled = StandardScaler().fit_transform(X)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=args.seed
    )
    X_tsne = tsne.fit_transform(X_scaled)

    tsne_df = pd.DataFrame({
        "x": X_tsne[:, 0],
        "y": X_tsne[:, 1],
        "label": meta["label"],
        "split": meta["split"],
        "patient_id": meta["patient_id"]
    })
    tsne_df.to_csv(os.path.join(out_dir, f"tsne_{args.embedding_key}.csv"), index=False, encoding="utf-8-sig")
    scatter_plot(
        tsne_df,
        f"t-SNE of DINOv3 {args.embedding_key}",
        os.path.join(out_dir, f"tsne_{args.embedding_key}.png")
    )

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=args.seed
    )
    X_umap = reducer.fit_transform(X_scaled)

    umap_df = pd.DataFrame({
        "x": X_umap[:, 0],
        "y": X_umap[:, 1],
        "label": meta["label"],
        "split": meta["split"],
        "patient_id": meta["patient_id"]
    })
    umap_df.to_csv(os.path.join(out_dir, f"umap_{args.embedding_key}.csv"), index=False, encoding="utf-8-sig")
    scatter_plot(
        umap_df,
        f"UMAP of DINOv3 {args.embedding_key}",
        os.path.join(out_dir, f"umap_{args.embedding_key}.png")
    )


if __name__ == "__main__":
    main()
