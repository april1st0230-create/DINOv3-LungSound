import os
import random
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize


LABEL_ORDER = ["normal", "crackle", "wheeze", "both"]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_image_paths(df, image_root):
    df = df.copy()
    df["image_path"] = df.apply(
        lambda row: os.path.join(
            image_root,
            str(row["label"]),
            os.path.basename(str(row["image_path"]))
        ),
        axis=1
    )
    return df


def check_missing_images(df):
    missing = df[~df["image_path"].apply(os.path.exists)]
    if len(missing) > 0:
        print("Missing image count:", len(missing))
        print(missing[["image_path"]].head())
        raise FileNotFoundError("Some image_path values do not exist.")


def compute_metrics(y_true, y_pred, y_prob, num_classes=4):
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

    macro_sp = float(np.mean(specificities))
    macro_balacc = (macro_recall + macro_sp) / 2

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
        "macro_specificity": macro_sp,
        "macro_f1": macro_f1,
        "macro_balanced_accuracy": macro_balacc,
        "macro_auroc": macro_auroc
    }
