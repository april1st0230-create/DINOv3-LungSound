# -*- coding: utf-8 -*-
"""
CNN baseline with handcrafted audio feature fusion.

Supported backbones:
    - vgg11_bn
    - vgg16
    - resnet18
    - resnet34

Input cache files should contain:

Image cache .pt:
    {
        "X": Tensor[N, 3, 224, 224],
        "y": Tensor[N]
    }

Handcrafted feature .pt:
    {
        "X": Tensor[N, AUDIO_DIM]
    }

Model structure:
    spectrogram image -> CNN backbone -> image embedding
    handcrafted audio feature -> normalization
    concat(image embedding, audio feature) -> classification head
"""

import os
import copy
import argparse
import random
import warnings
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import (
    VGG11_BN_Weights,
    VGG16_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
from sklearn.preprocessing import label_binarize


warnings.filterwarnings("ignore", category=UserWarning)

CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split_cache(path_cache: str, path_feat: str, audio_dim: int):
    image_cache = torch.load(path_cache, map_location="cpu")
    X_img = image_cache["X"].float()
    y = image_cache["y"].long()

    feat_cache = torch.load(path_feat, map_location="cpu")
    X_aud = feat_cache["X"].float()

    assert len(X_img) == len(y) == len(X_aud), "Image, label, and audio feature lengths do not match."
    assert X_aud.shape[1] == audio_dim, f"Expected audio_dim={audio_dim}, but got {X_aud.shape[1]}."

    return X_img, y, X_aud


class MelCachedDataset(Dataset):
    def __init__(self, X, A, y, mean_img, std_img, mean_aud, std_aud):
        self.X = X
        self.A = A
        self.y = y

        self.mean_img = mean_img.view(3, 1, 1)
        self.std_img = torch.where(
            std_img == 0,
            torch.ones_like(std_img),
            std_img
        ).view(3, 1, 1)

        self.mean_aud = mean_aud.view(-1)
        self.std_aud = torch.where(
            std_aud == 0,
            torch.ones_like(std_aud),
            std_aud
        ).view(-1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = (self.X[idx] - self.mean_img) / self.std_img
        a = (self.A[idx] - self.mean_aud) / self.std_aud
        y = int(self.y[idx])
        return x, a, y


class AudioConcatBackbone(nn.Module):
    def __init__(self, backbone: str, num_classes: int = 4, audio_dim: int = 5):
        super().__init__()

        self.backbone_name = backbone

        if backbone == "vgg11_bn":
            vgg = models.vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1)
            self.features = vgg.features
            self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.img_dim = 512
            self.is_conv = True

        elif backbone == "vgg16":
            vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            self.features = vgg.features
            self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.img_dim = 512
            self.is_conv = True

        elif backbone == "resnet18":
            res = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            res.fc = nn.Identity()
            self.backbone = res
            self.img_dim = 512
            self.is_conv = False

        elif backbone == "resnet34":
            res = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            res.fc = nn.Identity()
            self.backbone = res
            self.img_dim = 512
            self.is_conv = False

        else:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Choose from vgg11_bn, vgg16, resnet18, resnet34."
            )

        self.drop2d = nn.Dropout2d(p=0.2) if self.is_conv else None
        self.flat = nn.Flatten()

        self.head = nn.Sequential(
            nn.Linear(self.img_dim + audio_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, a):
        if self.backbone_name in ["resnet18", "resnet34"]:
            z = self.backbone(x)
        else:
            z = self.features(x)
            if self.drop2d is not None:
                z = self.drop2d(z)
            z = self.img_pool(z)
            z = self.flat(z)

        z = F.normalize(z, dim=1)
        logits = self.head(torch.cat([z, a], dim=1))
        return logits


def compute_metrics(y_true, y_pred, y_prob, num_classes: int = 4) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(sp)

    macro_specificity = float(np.mean(specificities))
    macro_balanced_accuracy = (macro_recall + macro_specificity) / 2

    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

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
        "macro_specificity": macro_specificity,
        "macro_f1": macro_f1,
        "macro_balanced_accuracy": macro_balanced_accuracy,
        "macro_auroc": macro_auroc,
    }


@torch.no_grad()
def run_eval(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    for images, audio, labels in tqdm(loader, leave=False):
        images = images.to(device)
        audio = audio.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images, audio)
            loss = criterion(logits, labels)

        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)

        total_loss += loss.item() * labels.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        y_prob.extend(prob.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics, y_true, y_pred, y_prob


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=["vgg11_bn", "vgg16", "resnet18", "resnet34"],
        help="CNN backbone to train."
    )

    parser.add_argument("--train_cache", type=str, required=True)
    parser.add_argument("--val_cache", type=str, required=True)
    parser.add_argument("--test_cache", type=str, required=True)

    parser.add_argument("--train_feat", type=str, required=True)
    parser.add_argument("--val_feat", type=str, required=True)
    parser.add_argument("--test_feat", type=str, required=True)

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--audio_dim", type=int, default=5)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=8)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    X_train, y_train, A_train = load_split_cache(
        args.train_cache,
        args.train_feat,
        args.audio_dim
    )
    X_val, y_val, A_val = load_split_cache(
        args.val_cache,
        args.val_feat,
        args.audio_dim
    )
    X_test, y_test, A_test = load_split_cache(
        args.test_cache,
        args.test_feat,
        args.audio_dim
    )

    with torch.no_grad():
        mean_img = X_train.mean(dim=(0, 2, 3))
        std_img = X_train.std(dim=(0, 2, 3))

        mean_aud = A_train.mean(dim=0)
        std_aud = A_train.std(dim=0)

    train_ds = MelCachedDataset(
        X_train, A_train, y_train,
        mean_img, std_img,
        mean_aud, std_aud
    )
    val_ds = MelCachedDataset(
        X_val, A_val, y_val,
        mean_img, std_img,
        mean_aud, std_aud
    )
    test_ds = MelCachedDataset(
        X_test, A_test, y_test,
        mean_img, std_img,
        mean_aud, std_aud
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )

    model = AudioConcatBackbone(
        backbone=args.backbone,
        num_classes=args.num_classes,
        audio_dim=args.audio_dim
    ).to(device)

    class_counts = torch.bincount(
        y_train,
        minlength=args.num_classes
    ).float()

    class_weights = class_counts.sum() / (args.num_classes * class_counts)
    class_weights = class_weights.to(device)

    print("class_counts:", class_counts.numpy())
    print("class_weights:", class_weights.detach().cpu().numpy())

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"{args.backbone} Epoch {epoch}/{args.epochs}"
        )

        for images, audio, labels in pbar:
            images = images.to(device)
            audio = audio.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(images, audio)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({
                "loss": running_loss / len(pbar)
            })

        scheduler.step()

        val_metrics, _, _, _ = run_eval(
            model,
            val_loader,
            criterion,
            device
        )

        row = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            **{f"val_{k}": v for k, v in val_metrics.items()}
        }
        history.append(row)

        print(row)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())

            ckpt_path = os.path.join(
                args.output_dir,
                f"best_{args.backbone}.pt"
            )
            torch.save(best_state, ckpt_path)

            bad_epochs = 0
            print("best model saved:", ckpt_path)

        else:
            bad_epochs += 1
            print("bad_epochs:", bad_epochs)

        if bad_epochs >= args.patience:
            print("Early stopping")
            break

    best_ckpt_path = os.path.join(
        args.output_dir,
        f"best_{args.backbone}.pt"
    )

    model.load_state_dict(
        torch.load(best_ckpt_path, map_location=device)
    )

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

    print("\nVAL:", val_metrics)
    print("TEST:", test_metrics)

    print("\n===== VAL report =====")
    print(classification_report(
        val_true,
        val_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    ))

    print("\n===== TEST report =====")
    print(classification_report(
        test_true,
        test_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    ))

    result_df = pd.DataFrame([
        {
            "model": args.backbone,
            "split": "val",
            **val_metrics
        },
        {
            "model": args.backbone,
            "split": "test",
            **test_metrics
        },
    ])

    result_path = os.path.join(
        args.output_dir,
        f"{args.backbone}_results.csv"
    )

    history_path = os.path.join(
        args.output_dir,
        f"{args.backbone}_history.csv"
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
