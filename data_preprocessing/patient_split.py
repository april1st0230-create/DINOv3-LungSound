# -*- coding: utf-8 -*-
"""
Create patient-level train / validation / test split metadata.

Input metadata CSV should include at least:
    - patient_id
    - image_path
    - label
    - label_id

Output:
    - metadata CSV with an added "split" column

Example:
    python data_preprocessing/patient_split.py ^
      --metadata_csv "C:/path/to/spectrogram_metadata.csv" ^
      --output_csv "C:/path/to/dinov3_patient_split_metadata.csv"
"""

import argparse
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_ORDER = ["normal", "crackle", "wheeze", "both"]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def summarize_split(df: pd.DataFrame):
    print("\n===== Split sample counts =====")
    print(df["split"].value_counts())

    print("\n===== Class distribution by split =====")
    print(pd.crosstab(df["split"], df["label"]))

    print("\n===== Patient counts by split =====")
    print(df.groupby("split")["patient_id"].nunique())


def create_patient_split(
    df: pd.DataFrame,
    train_ratio: float = 0.50,
    val_ratio: float = 0.25,
    test_ratio: float = 0.25,
    seed: int = 42,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio must be 1.0"

    required_cols = ["patient_id", "image_path", "label", "label_id"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    patient_df = (
        df.groupby("patient_id")
        .agg(
            n_samples=("image_path", "count"),
            major_label=("label", lambda x: x.value_counts().idxmax())
        )
        .reset_index()
    )

    patients = patient_df["patient_id"].values
    patient_labels = patient_df["major_label"].values

    temp_ratio = val_ratio + test_ratio

    train_patients, temp_patients, train_y, temp_y = train_test_split(
        patients,
        patient_labels,
        test_size=temp_ratio,
        random_state=seed,
        shuffle=True,
        stratify=patient_labels if len(np.unique(patient_labels)) > 1 else None
    )

    relative_test_ratio = test_ratio / temp_ratio

    try:
        val_patients, test_patients = train_test_split(
            temp_patients,
            test_size=relative_test_ratio,
            random_state=seed,
            shuffle=True,
            stratify=temp_y if len(np.unique(temp_y)) > 1 else None
        )
    except ValueError:
        val_patients, test_patients = train_test_split(
            temp_patients,
            test_size=relative_test_ratio,
            random_state=seed,
            shuffle=True,
            stratify=None
        )

    out = df.copy()
    out["split"] = "none"

    out.loc[out["patient_id"].isin(train_patients), "split"] = "train"
    out.loc[out["patient_id"].isin(val_patients), "split"] = "val"
    out.loc[out["patient_id"].isin(test_patients), "split"] = "test"

    if (out["split"] == "none").any():
        raise RuntimeError("Some samples were not assigned to any split.")

    overlap_train_val = set(train_patients) & set(val_patients)
    overlap_train_test = set(train_patients) & set(test_patients)
    overlap_val_test = set(val_patients) & set(test_patients)

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise RuntimeError("Patient-level leakage detected across splits.")

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata_csv",
        type=str,
        required=True,
        help="Input metadata CSV generated from spectrogram preprocessing."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output metadata CSV with patient-level split column."
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.50,
        help="Train patient ratio."
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.25,
        help="Validation patient ratio."
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.25,
        help="Test patient ratio."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    args = parser.parse_args()

    set_seed(args.seed)

    df = pd.read_csv(args.metadata_csv)

    split_df = create_patient_split(
        df=df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    split_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print("Saved:", args.output_csv)
    summarize_split(split_df)


if __name__ == "__main__":
    main()
