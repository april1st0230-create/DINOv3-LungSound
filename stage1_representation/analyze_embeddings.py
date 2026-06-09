import os
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="dinov3_outputs")
    parser.add_argument("--embedding_key", type=str, default="cls_embeddings",
                        choices=["cls_embeddings", "mean_token_embeddings"])
    args = parser.parse_args()

    out_dir = os.path.join(args.root, args.out_dir)
    emb_path = os.path.join(out_dir, "dinov3_vitl16_embeddings.npz")
    meta_path = os.path.join(out_dir, "dinov3_embedding_metadata.csv")

    emb = np.load(emb_path)
    meta = pd.read_csv(meta_path)

    X = emb[args.embedding_key]
    y = emb["labels"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sil = silhouette_score(X_scaled, y, metric="euclidean")
    print("Silhouette score:", sil)

    train_mask = meta["split"] == "train"
    val_mask = meta["split"] == "val"
    test_mask = meta["split"] == "test"

    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn.fit(X_scaled[train_mask], y[train_mask])

    val_pred = knn.predict(X_scaled[val_mask])
    test_pred = knn.predict(X_scaled[test_mask])

    val_acc = accuracy_score(y[val_mask], val_pred)
    test_acc = accuracy_score(y[test_mask], test_pred)

    print("kNN val accuracy:", val_acc)
    print("kNN test accuracy:", test_acc)

    label_order = ["normal", "crackle", "wheeze", "both"]
    label2id = {"normal": 0, "crackle": 1, "wheeze": 2, "both": 3}

    centroids = []
    for label in label_order:
        label_id = label2id[label]
        centroids.append(X_scaled[y == label_id].mean(axis=0))

    centroids = np.stack(centroids, axis=0)
    dist_mat = pairwise_distances(centroids, metric="cosine")

    centroid_df = pd.DataFrame(dist_mat, index=label_order, columns=label_order)
    centroid_path = os.path.join(out_dir, f"centroid_cosine_distance_{args.embedding_key}.csv")
    centroid_df.to_csv(centroid_path, encoding="utf-8-sig")

    summary = pd.DataFrame([{
        "embedding_key": args.embedding_key,
        "silhouette_score": sil,
        "knn_val_accuracy": val_acc,
        "knn_test_accuracy": test_acc
    }])
    summary_path = os.path.join(out_dir, f"representation_summary_{args.embedding_key}.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("Saved:", centroid_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
