# -*- coding: utf-8 -*-
import os
import copy
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.preprocessing import label_binarize


LABEL_ORDER = ["normal", "crackle", "wheeze", "both"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LungImageDataset(Dataset):
    def __init__(self, dataframe, processor, image_root=None):
        self.df = dataframe.reset_index(drop=True)
        self.processor = processor
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
        image = Image.open(self._resolve_path(row)).convert("RGB")
        label = int(row["label_id"])

        inputs = self.processor(images=image, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


class DINOv3Classifier(nn.Module):
    def __init__(self, model_name, num_classes=4, dropout=0.3):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_emb)
        return logits


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none"
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def build_criterion(loss_name, class_weights, focal_gamma=2.0):
    if loss_name == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)

    if loss_name == "focal":
        return FocalLoss(
            alpha=class_weights,
            gamma=focal_gamma,
            reduction="mean"
        )

    raise ValueError(f"Unsupported loss: {loss_name}")


def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    specificities = []

    for i in range(4):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    macro_sp = np.mean(specificities)
    macro_balacc = (macro_recall + macro_sp) / 2

    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3])

    try:
        macro_auroc = roc_auc_score(
            y_bin,
            y_prob,
            average="macro",
            multi_class="ovr"
        )
    except Exception:
        macro_auroc = np.nan

    return {
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall_sensitivity": macro_recall,
        "macro_specificity": macro_sp,
        "macro_f1": macro_f1,
        "macro_balanced_accuracy": macro_balacc,
        "macro_auroc": macro_auroc,
    }


@torch.no_grad()
def run_eval(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    y_true, y_pred, y_prob = [], [], []

    for batch in tqdm(loader, leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(pixel_values)
            loss = criterion(logits, labels)

        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)

        total_loss += loss.item() * labels.size(0)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        y_prob.extend(prob.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(
        np.array(y_true),
        np.array(y_pred),
        np.array(y_prob)
    )
    metrics["loss"] = avg_loss

    return metrics, np.array(y_true), np.array(y_pred), np.array(y_prob)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/dinov3-vitl16-pretrain-lvd1689m"
    )

    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=["ce", "focal"],
        help="Loss function: ce or focal."
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for focal loss."
    )
    parser.add_argument(
        "--sqrt_class_weight",
        action="store_true",
        help="Apply square root to class weights. Useful for focal loss."
    )

    parser.add_argument(
        "--select_metric",
        type=str,
        default="loss",
        choices=["loss", "macro_f1", "macro_balanced_accuracy", "macro_auroc"],
        help="Validation metric used for best checkpoint selection."
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)

    df = pd.read_csv(args.metadata_csv)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    processor = AutoImageProcessor.from_pretrained(args.model_name)

    train_loader = DataLoader(
        LungImageDataset(train_df, processor, args.image_root),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        LungImageDataset(val_df, processor, args.image_root),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        LungImageDataset(test_df, processor, args.image_root),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = DINOv3Classifier(
        args.model_name,
        args.num_classes
    ).to(device)

    for p in model.backbone.parameters():
        p.requires_grad = False

    for name, p in model.backbone.named_parameters():
        if "encoder.layer.22" in name or "encoder.layer.23" in name:
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print("trainable params:", trainable)
    print("total params:", total)
    print("trainable ratio:", trainable / total)

    class_counts = train_df["label_id"].value_counts().sort_index().values
    class_weights = class_counts.sum() / (args.num_classes * class_counts)

    if args.sqrt_class_weight:
        class_weights = np.sqrt(class_weights)

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    ).to(device)

    print("class_counts:", class_counts)
    print("class_weights:", class_weights.detach().cpu().numpy())

    criterion = build_criterion(
        loss_name=args.loss,
        class_weights=class_weights,
        focal_gamma=args.focal_gamma
    )

    head_params = list(model.classifier.parameters())
    backbone_params = [
        p for p in model.backbone.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": args.lr_head},
            {"params": backbone_params, "lr": args.lr_backbone},
        ],
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    if args.select_metric == "loss":
        best_score = float("inf")
        higher_is_better = False
    else:
        best_score = -float("inf")
        higher_is_better = True

    history = []

    ckpt_name = f"best_dinov3_finetuned_{args.loss}.pt"
    ckpt_path = os.path.join(args.output_dir, ckpt_name)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        running_loss = 0

        pbar = tqdm(
            train_loader,
            desc=f"DINOv3 {args.loss.upper()} Epoch {epoch}/{args.epochs}"
        )

        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
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

        val_metrics, _, _, _ = run_eval(
            model,
            val_loader,
            criterion,
            device
        )

        row = {
            "epoch": epoch,
            "loss_type": args.loss,
            "train_loss": running_loss / len(train_loader),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

        history.append(row)
        print(row)

        current_score = val_metrics[args.select_metric]

        improved = (
            current_score > best_score
            if higher_is_better
            else current_score < best_score
        )

        if improved:
            best_score = current_score
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, ckpt_path)
            print("best model saved:", ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    val_metrics, val_true, val_pred, val_prob = run_eval(
        model,
        val_loader,
        criterion,
        device
    )

    test_metrics, test_true, test_pred, test_prob = run_eval(
        model,
        test_loader,
        criterion,
        device
    )

    print("VAL:", val_metrics)
    print("TEST:", test_metrics)

    print("\n===== VAL report =====")
    print(classification_report(
        val_true,
        val_pred,
        target_names=LABEL_ORDER,
        digits=4,
        zero_division=0
    ))

    print("\n===== TEST report =====")
    print(classification_report(
        test_true,
        test_pred,
        target_names=LABEL_ORDER,
        digits=4,
        zero_division=0
    ))

    model_name_out = f"DINOv3_finetune_last2blocks_{args.loss}"

    result_df = pd.DataFrame([
        {"model": model_name_out, "split": "val", **val_metrics},
        {"model": model_name_out, "split": "test", **test_metrics},
    ])

    result_path = os.path.join(
        args.output_dir,
        f"dinov3_finetune_results_{args.loss}.csv"
    )

    history_path = os.path.join(
        args.output_dir,
        f"dinov3_finetune_history_{args.loss}.csv"
    )

    result_df.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("Saved:", result_path)
    print("Saved:", history_path)


if __name__ == "__main__":
    main()
