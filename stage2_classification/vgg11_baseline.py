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
from torchvision import transforms, models
from sklearn.metrics import classification_report

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import set_seed, resolve_image_paths, check_missing_images, compute_metrics, LABEL_ORDER


class LungImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label_id"])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def build_model(model_type, num_classes=4):
    if model_type == "vgg11_bn":
        model = models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_type == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unknown model_type: {model_type}")


def run_eval(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(images)
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


def train_cnn(args):
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

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=args.degrees,
                translate=(args.translate, args.translate),
                scale=(0.95, 1.05)
            )
        ], p=args.affine_p),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    train_loader = DataLoader(LungImageDataset(train_df, train_tf), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(LungImageDataset(val_df, eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(LungImageDataset(test_df, eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.model_type, args.num_classes).to(device)

    class_counts = train_df["label_id"].value_counts().sort_index().values
    class_weights = class_counts.sum() / (args.num_classes * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

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
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_val_loss = float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0

        pbar = tqdm(train_loader, desc=f"{args.model_type} Epoch {epoch}/{args.epochs}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(images)
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
            **{f"val_{k}": v for k, v in val_metrics.items()}
        }
        history.append(row)
        print(row)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), os.path.join(out_dir, f"best_{args.model_type}.pt"))
            bad_epochs = 0
            print("best model saved")
        else:
            bad_epochs += 1
            print("bad_epochs:", bad_epochs)

        if bad_epochs >= args.patience:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(os.path.join(out_dir, f"best_{args.model_type}.pt"), map_location=device))

    val_metrics, val_true, val_pred, _ = run_eval(model, val_loader, criterion, device)
    test_metrics, test_true, test_pred, _ = run_eval(model, test_loader, criterion, device)

    print("VAL:", val_metrics)
    print("TEST:", test_metrics)
    print("\n===== VAL report =====")
    print(classification_report(val_true, val_pred, target_names=LABEL_ORDER, digits=4, zero_division=0))
    print("\n===== TEST report =====")
    print(classification_report(test_true, test_pred, target_names=LABEL_ORDER, digits=4, zero_division=0))

    result_df = pd.DataFrame([
        {"model": args.model_type, "split": "val", **val_metrics},
        {"model": args.model_type, "split": "test", **test_metrics},
    ])
    result_df.to_csv(os.path.join(out_dir, f"{args.model_type}_results.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(os.path.join(out_dir, f"{args.model_type}_history.csv"), index=False, encoding="utf-8-sig")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--metadata", type=str, default="dinov3_patient_split_metadata.csv")
    parser.add_argument("--image_dir", type=str, default="logmel_delta_deltadelta_3ch_224")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="vgg11_bn", choices=["vgg11_bn", "resnet18"])
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--degrees", type=float, default=5)
    parser.add_argument("--translate", type=float, default=0.03)
    parser.add_argument("--affine_p", type=float, default=0.3)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = "cnn_baseline_outputs"

    args.model_type = "vgg11_bn"
    train_cnn(args)
