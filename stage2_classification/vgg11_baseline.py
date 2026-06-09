# -*- coding: utf-8 -*-
"""
CNN baseline with VGG11-BN backbone and handcrafted audio feature fusion.

Input cache files should contain:
    image cache .pt:
        {"X": Tensor[N,3,224,224], "y": Tensor[N]}
    handcrafted feature .pt:
        {"X": Tensor[N,AUDIO_DIM]}
"""

import os
import argparse
import random
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import VGG11_BN_Weights

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.preprocessing import label_binarize


CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split_cache(path_cache, path_feat, audio_dim):
    b = torch.load(path_cache, map_location="cpu")
    X_img = b["X"].float()
    y = b["y"].long()

    f = torch.load(path_feat, map_location="cpu")
    X_aud = f["X"].float()

    assert len(X_img) == len(y) == len(X_aud)
    assert X_aud.shape[1] == audio_dim

    return X_img, y, X_aud


class MelCachedDataset(Dataset):
    def __init__(self, X, A, y, mean_img, std_img, mean_aud, std_aud):
        self.X = X
        self.A = A
        self.y = y
        self.mean_img = mean_img.view(3, 1, 1)
        self.std_img = torch.where(std_img == 0, torch.ones_like(std_img), std_img).view(3, 1, 1)
        self.mean_aud = mean_aud.view(-1)
        self.std_aud = torch.where(std_aud == 0, torch.ones_like(std_aud), std_aud).view(-1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = (self.X[i] - self.mean_img) / self.std_img
        a = (self.A[i] - self.mean_aud) / self.std_aud
        return x, a, int(self.y[i])


class VGG11AudioConcat(nn.Module):
    def __init__(self, num_classes=4, audio_dim=5):
        super().__init__()

        vgg = models.vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop2d = nn.Dropout2d(p=0.2)
        self.flat = nn.Flatten()

        self.img_dim = 512

        self.head = nn.Sequential(
            nn.Linear(self.img_dim + audio_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, a):
        z = self.features(x)
        z = self.drop2d(z)
        z = self.img_pool(z)
        z = self.flat(z)
        z = F.normalize(z, dim=1)
        return self.head(torch.cat([z, a], dim=1))


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
        macro_auroc = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
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

    metrics = compute_metrics(np.array(y_true), np.array(y_pred), np.array(y_prob))
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics, np.array(y_true), np.array(y_pred), np.array(y_prob)


def main():
    parser = argparse.ArgumentParser()
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

    X_tr, y_tr, A_tr = load_split_cache(args.train_cache, args.train_feat, args.audio_dim)
    X_va, y_va, A_va = load_split_cache(args.val_cache, args.val_feat, args.audio_dim)
    X_te, y_te, A_te = load_split_cache(args.test_cache, args.test_feat, args.audio_dim)

    with torch.no_grad():
        mean_img = X_tr.mean(dim=(0, 2, 3))
        std_img = X_tr.std(dim=(0, 2, 3))
        mean_aud = A_tr.mean(dim=0)
        std_aud = A_tr.std(dim=0)

    train_loader = DataLoader(
        MelCachedDataset(X_tr, A_tr, y_tr, mean_img, std_img, mean_aud, std_aud),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        MelCachedDataset(X_va, A_va, y_va, mean_img, std_img, mean_aud, std_aud),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        MelCachedDataset(X_te, A_te, y_te, mean_img, std_img, mean_aud, std_aud),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = VGG11AudioConcat(
        num_classes=args.num_classes,
        audio_dim=args.audio_dim,
    ).to(device)

    class_counts = torch.bincount(y_tr, minlength=args.num_classes).float()
    class_weights = class_counts.sum() / (args.num_classes * class_counts)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0

        pbar = tqdm(train_loader, desc=f"VGG11-BN Epoch {epoch}/{args.epochs}")

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
            pbar.set_postfix({"loss": running_loss / len(pbar)})

        scheduler.step()

        val_metrics, _, _, _ = run_eval(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(row)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(args.output_dir, "best_vgg11_bn.pt"))
            bad_epochs = 0
            print("best model saved")
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(
        os.path.join(args.output_dir, "best_vgg11_bn.pt"),
        map_location=device,
    ))

    val_metrics, val_true, val_pred, _ = run_eval(model, val_loader, criterion, device)
    test_metrics, test_true, test_pred, _ = run_eval(model, test_loader, criterion, device)

    print("VAL:", val_metrics)
    print("TEST:", test_metrics)

    print("\n===== VAL report =====")
    print(classification_report(val_true, val_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))

    print("\n===== TEST report =====")
    print(classification_report(test_true, test_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))

    pd.DataFrame([
        {"model": "VGG11_BN_AudioConcat", "split": "val", **val_metrics},
        {"model": "VGG11_BN_AudioConcat", "split": "test", **test_metrics},
    ]).to_csv(
        os.path.join(args.output_dir, "vgg11_bn_results.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(history).to_csv(
        os.path.join(args.output_dir, "vgg11_bn_history.csv"),
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
