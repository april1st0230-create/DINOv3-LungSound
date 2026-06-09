import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import resolve_image_paths, check_missing_images, set_seed


class LungImageDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return {
            "image": image,
            "label_id": int(row["label_id"]),
            "index": idx
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--metadata", type=str, default="dinov3_patient_split_metadata.csv")
    parser.add_argument("--image_dir", type=str, default="logmel_delta_deltadelta_3ch_224")
    parser.add_argument("--out_dir", type=str, default="dinov3_outputs")
    parser.add_argument("--model_name", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    csv_path = os.path.join(args.root, args.metadata)
    image_root = os.path.join(args.root, args.image_dir)
    out_dir = os.path.join(args.root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_emb = os.path.join(out_dir, "dinov3_vitl16_embeddings.npz")
    out_meta = os.path.join(out_dir, "dinov3_embedding_metadata.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device)
    if device == "cuda":
        print(torch.cuda.get_device_name(0))

    df = pd.read_csv(csv_path)
    df = resolve_image_paths(df, image_root)
    check_missing_images(df)

    print("Total samples:", len(df))
    print(df["split"].value_counts())
    print(pd.crosstab(df["split"], df["label"]))

    dataset = LungImageDataset(df)

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
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
        collate_fn=collate_fn
    )

    all_cls, all_mean, all_labels, all_indices = [], [], [], []

    with torch.no_grad():
        for inputs, labels, indices in tqdm(loader):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state

            cls_emb = last_hidden[:, 0, :]
            mean_emb = last_hidden[:, 1:, :].mean(dim=1)

            all_cls.append(cls_emb.cpu().numpy())
            all_mean.append(mean_emb.cpu().numpy())
            all_labels.append(labels.numpy())
            all_indices.append(indices.numpy())

    cls_embeddings = np.concatenate(all_cls, axis=0)
    mean_token_embeddings = np.concatenate(all_mean, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    indices = np.concatenate(all_indices, axis=0)

    np.savez_compressed(
        out_emb,
        cls_embeddings=cls_embeddings,
        mean_token_embeddings=mean_token_embeddings,
        labels=labels,
        indices=indices
    )

    df_out = df.iloc[indices].reset_index(drop=True)
    df_out.to_csv(out_meta, index=False, encoding="utf-8-sig")

    print("Saved:", out_emb)
    print("Saved:", out_meta)
    print("cls_embeddings:", cls_embeddings.shape)
    print("mean_token_embeddings:", mean_token_embeddings.shape)


if __name__ == "__main__":
    main()
