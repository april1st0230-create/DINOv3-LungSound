import os
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import LABEL_ORDER


def compute_specificity_per_class(y_true, y_pred, num_classes=4):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(sp)

    return np.array(specificities)


def evaluate_model(name, model, X_data, y_data, split_name):
    pred = model.predict(X_data)
    prob = model.predict_proba(X_data) if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_data, pred)
    macro_f1 = f1_score(y_data, pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_data, pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_data, pred, average="macro", zero_division=0)

    sp_per_class = compute_specificity_per_class(y_data, pred, num_classes=4)
    macro_sp = sp_per_class.mean()
    macro_balacc = (macro_recall + macro_sp) / 2

    if prob is not None:
        y_bin = label_binarize(y_data, classes=[0, 1, 2, 3])
        try:
            macro_auroc = roc_auc_score(y_bin, prob, average="macro", multi_class="ovr")
        except Exception:
            macro_auroc = np.nan
    else:
        macro_auroc = np.nan

    result = {
        "model": name,
        "split": split_name,
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall_sensitivity": macro_recall,
        "macro_specificity": macro_sp,
        "macro_f1": macro_f1,
        "macro_balanced_accuracy": macro_balacc,
        "macro_auroc": macro_auroc
    }

    print(f"\n===== {name} / {split_name} =====")
    print(result)
    print(classification_report(
        y_data,
        pred,
        target_names=LABEL_ORDER,
        digits=4,
        zero_division=0
    ))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="dinov3_outputs")
    parser.add_argument("--embedding_key", type=str, default="cls_embeddings",
                        choices=["cls_embeddings", "mean_token_embeddings"])
    args = parser.parse_args()

    out_dir = os.path.join(args.root, args.out_dir)
    emb = np.load(os.path.join(out_dir, "dinov3_vitl16_embeddings.npz"))
    meta = pd.read_csv(os.path.join(out_dir, "dinov3_embedding_metadata.csv"))

    X = emb[args.embedding_key]
    y = emb["labels"]

    train_mask = meta["split"] == "train"
    val_mask = meta["split"] == "val"
    test_mask = meta["split"] == "test"

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    results = []

    linear_clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=3000,
        solver="lbfgs",
        random_state=42
    )
    linear_clf.fit(X_train_s, y_train)

    for split_name, X_data, y_data in [
        ("val", X_val_s, y_val),
        ("test", X_test_s, y_test)
    ]:
        results.append(evaluate_model(
            f"DINOv3_{args.embedding_key}_Linear",
            linear_clf,
            X_data,
            y_data,
            split_name
        ))

    mlp_clf = MLPClassifier(
        hidden_layer_sizes=(512, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-4,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
        verbose=True
    )
    mlp_clf.fit(X_train_s, y_train)

    for split_name, X_data, y_data in [
        ("val", X_val_s, y_val),
        ("test", X_test_s, y_test)
    ]:
        results.append(evaluate_model(
            f"DINOv3_{args.embedding_key}_MLP",
            mlp_clf,
            X_data,
            y_data,
            split_name
        ))

    result_df = pd.DataFrame(results)
    result_path = os.path.join(out_dir, f"probe_results_{args.embedding_key}.csv")
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    print("Saved:", result_path)


if __name__ == "__main__":
    main()
