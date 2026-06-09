import os
import copy
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics import classification_report

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import set_seed, resolve_image_paths, check_missing_images, compute_metrics, LABEL_ORDER


class LungImageDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.df = dataframe.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label_id"])
        inputs = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


class DINOv3Classifier(nn.Module):
    def __init__(self, model_name, num_classes=4, dropout=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_emb)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def run_eval(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(pixel_values)
                loss = criterion(logits, labels)

            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)

            total_loss += loss.item() * labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())

    metrics = compute_metrics(np.array(y_true), np.array(y_pred), np.array(y_prob))
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics, np.array(y_true), np.array(y_pred), np.array(y_prob)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--metadata", type=str, default="dinov3_patient_split_metadata.csv")
    parser.add_argument("--image_dir", type=str, default="logmel_delta_deltadelta_3ch_224")
    parser.add_argument("--out_dir", type=str, default="dinov3_focal_finetune_outputs")
    parser.add_argument("--model_name", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device)

    csv_path = os.path.join(args.root, args.metadata)
    image_root = os.path.join(args.root, args.image_dir)
    out_dir = os.path.join(args.root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = resolve_image_paths(df, image_root)
    check_missing_images(df)

    processor = AutoImageProcessor.from_pretrained(args.model_name)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    train_loader = DataLoader(LungImageDataset(train_df, processor), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(LungImageDataset(val_df, processor), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(LungImageDataset(test_df, processor), batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = DINOv3Classifier(args.model_name, args.num_classes).to(device)

    for p in model.backbone.parameters():
        p.requires_grad = False

    for name, p in model.backbone.named_parameters():
        if "encoder.layer.22" in name or "encoder.layer.23" in name:
            p.requires_grad = True

    class_counts = train_df["label_id"].value_counts().sort_index().values
    class_weights = class_counts.sum() / (args.num_classes * class_counts)
    class_weights = np.sqrt(class_weights)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma, reduction="mean")

    head_params = list(model.classifier.parameters())
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": args.lr_head},
            {"params": backbone_params, "lr": args.lr_backbone},
        ],
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_val_score = -1
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        running_loss = 0

        pbar = tqdm(train_loader, desc=f"DINOv3 Focal Epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(pixel_values)
                loss = criterion(logits, labels)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * args.grad_accum
            pbar.set_postfix({"loss": running_loss / (step + 1)})

        scheduler.step()
        val_metrics, _, _, _ = run_eval(model, val_loader, criterion, device)
        val_score = val_metrics["macro_f1"]

        row = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            **{f"val_{k}": v for k, v in val_metrics.items()}
        }
        history.append(row)
        print(row)

        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), os.path.join(out_dir, "best_dinov3_focal_finetuned.pt"))
            bad_epochs = 0
            print("best model saved")
        else:
            bad_epochs += 1
            print("bad_epochs:", bad_epochs)

        if bad_epochs >= args.patience:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(os.path.join(out_dir, "best_dinov3_focal_finetuned.pt"), map_location=device))

    val_metrics, val_true, val_pred, _ = run_eval(model, val_loader, criterion, device)
    test_metrics, test_true, test_pred, _ = run_eval(model, test_loader, criterion, device)

    print("VAL:", val_metrics)
    print("TEST:", test_metrics)
    print("\n===== VAL report =====")
    print(classification_report(val_true, val_pred, target_names=LABEL_ORDER, digits=4, zero_division=0))
    print("\n===== TEST report =====")
    print(classification_report(test_true, test_pred, target_names=LABEL_ORDER, digits=4, zero_division=0))

    result_df = pd.DataFrame([
        {"model": "DINOv3_finetune_last2blocks_focal", "split": "val", **val_metrics},
        {"model": "DINOv3_finetune_last2blocks_focal", "split": "test", **test_metrics},
    ])
    result_df.to_csv(os.path.join(out_dir, "dinov3_focal_finetune_results.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "dinov3_focal_finetune_history.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
