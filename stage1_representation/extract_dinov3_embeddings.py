# -*- coding: utf-8 -*-
"""
Extract DINOv3 embeddings from respiratory sound spectrogram images.

Input:
    - metadata CSV with columns:
        image_path, label, label_id, split, patient_id

Output:
    - dinov3_vitl16_embeddings.npz
    - dinov3_embedding_metadata.csv

Saved arrays:
    - cls_embeddings
    - mean_token_embeddings
    - labels
    - indices
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LungImageDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_root: str = None):
        self.df = dataframe.reset_index(drop=True)
        self.image_root = image_root

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, row):
        path = str(row["image_path"])

        if os.path.exists(path):
            return path

        if self.image_root is not None:
            alt_path = os.path.join(
                self.image_root,
                str(row["label"]),
                os.path.basename(path)
            )
            if os.path.exists(alt_path):
                return alt_path

        raise FileNotFoundError(f"Image not found: {path}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self._resolve_path(row)

        image = Image.open(image_path).convert("RGB")

        return {
            "image": image,
            "label_id": int(row["label_id"]),
            "index": idx,
        }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata_csv",
        type=str,
        required=True,
        help="Metadata CSV containing image_path, label, label_id, split, patient_id."
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Optional root directory of spectrogram images."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save extracted embeddings."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
        help="Hugging Face model name for DINOv3."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for embedding extraction."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    df = pd.read_csv(args.metadata_csv)

    required_cols = ["image_path", "label", "label_id", "split", "patient_id"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    print("Total samples:", len(df))
    print("\nSplit distribution:")
    print(df["split"].value_counts())
    print("\nClass distribution by split:")
    print(pd.crosstab(df["split"], df["label"]))

    dataset = LungImageDataset(df, image_root=args.image_root)

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    def collate_fn(batch):
        images = [item["image"] for item in batch]
        labels = torch.tensor([item["label_id"] for item in batch], dtype=torch.long)
        indices = torch.tensor([item["index"] for item in batch], dtype=torch.long)

        inputs = processor(images=images, return_tensors="pt")
        return inputs, labels, indices

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    all_cls_embeddings = []
    all_mean_token_embeddings = []
    all_labels = []
    all_indices = []

    with torch.no_grad():
        for inputs, labels, indices in tqdm(loader, desc="Extracting DINOv3 embeddings"):
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state

            cls_emb = last_hidden[:, 0, :]
            mean_emb = last_hidden[:, 1:, :].mean(dim=1)

            all_cls_embeddings.append(cls_emb.cpu().numpy())
            all_mean_token_embeddings.append(mean_emb.cpu().numpy())
            all_labels.append(labels.numpy())
            all_indices.append(indices.numpy())

    cls_embeddings = np.concatenate(all_cls_embeddings, axis=0)
    mean_token_embeddings = np.concatenate(all_mean_token_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    indices = np.concatenate(all_indices, axis=0)

    out_emb = os.path.join(args.output_dir, "dinov3_vitl16_embeddings.npz")
    out_meta = os.path.join(args.output_dir, "dinov3_embedding_metadata.csv")

    np.savez_compressed(
        out_emb,
        cls_embeddings=cls_embeddings,
        mean_token_embeddings=mean_token_embeddings,
        labels=labels,
        indices=indices,
    )

    df_out = df.iloc[indices].reset_index(drop=True)
    df_out.to_csv(out_meta, index=False, encoding="utf-8-sig")

    print("\nSaved:", out_emb)
    print("Saved:", out_meta)
    print("cls_embeddings:", cls_embeddings.shape)
    print("mean_token_embeddings:", mean_token_embeddings.shape)


if __name__ == "__main__":
    main()
