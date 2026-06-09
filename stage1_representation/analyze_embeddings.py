# -*- coding: utf-8 -*-
"""
Analyze DINOv3 embedding space.

This script computes:
    - Silhouette Score
    - k-NN validation/test accuracy
    - Class centroid cosine distance matrix

Input:
    - dinov3_vitl16_embeddings.npz
    - dinov3_embedding_metadata.csv
"""

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances


LABEL_ORDER = ["normal", "crackle", "wheeze", "both"]
LABEL2ID = {
    "normal": 0,
    "crackle": 1,
    "wheeze": 2,
    "both": 3,
}


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
        help="Directory to save analysis outputs."
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="cls",
        choices=["cls", "mean"],
        help="Embedding type to analyze."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of neighbors for k-NN."
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    emb = np.load(args.embedding_npz)
    meta = pd.read_csv(args.metadata_csv)

    if args.embedding_type == "cls":
        X = emb["cls_embeddings"]
    else:
        X = emb["mean_token_embeddings"]

    y = emb["labels"]

    print("Embedding:", X.shape)
    print("Labels:", y.shape)
    print("\nClass distribution:")
    print(meta["label"].value_counts())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sil = silhouette_score(X_scaled, y, metric="euclidean")
    print("\nSilhouette score:", sil)

    train_mask = meta["split"].values == "train"
    val_mask = meta["split"].values == "val"
    test_mask = meta["split"].values == "test"

    X_train = X_scaled[train_mask]
    y_train = y[train_mask]

    X_val = X_scaled[val_mask]
    y_val = y[val_mask]

    X_test = X_scaled[test_mask]
    y_test = y[test_mask]

    knn = KNeighborsClassifier(n_neighbors=args.k, metric="cosine")
    knn.fit(X_train, y_train)

    val_pred = knn.predict(X_val)
    test_pred = knn.predict(X_test)

    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print("kNN validation accuracy:", val_acc)
    print("kNN test accuracy:", test_acc)

    centroids = []

    for label in LABEL_ORDER:
        label_id = LABEL2ID[label]
        centroids.append(X_scaled[y == label_id].mean(axis=0))

    centroids = np.stack(centroids, axis=0)

    dist_mat = pairwise_distances(centroids, metric="cosine")

    centroid_df = pd.DataFrame(
        dist_mat,
        index=LABEL_ORDER,
        columns=LABEL_ORDER,
    )

    summary_df = pd.DataFrame([{
        "embedding_type": args.embedding_type,
        "silhouette_score": sil,
        "knn_k": args.k,
        "knn_val_accuracy": val_acc,
        "knn_test_accuracy": test_acc,
    }])

    centroid_path = os.path.join(
        args.output_dir,
        f"centroid_cosine_distance_{args.embedding_type}.csv"
    )
    summary_path = os.path.join(
        args.output_dir,
        f"representation_summary_{args.embedding_type}.csv"
    )

    centroid_df.to_csv(centroid_path, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nSaved:", centroid_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
