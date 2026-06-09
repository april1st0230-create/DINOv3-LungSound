"""
Patient-level split utility.

This script assumes that a metadata CSV file contains at least:
- patient_id
- label
- image_path

It creates train/val/test split labels without patient leakage.
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--val_size", type=float, default=0.25)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    patients = df["patient_id"].drop_duplicates()

    train_val_patients, test_patients = train_test_split(
        patients,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True
    )

    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=args.val_size,
        random_state=args.seed,
        shuffle=True
    )

    df["split"] = "none"
    df.loc[df["patient_id"].isin(train_patients), "split"] = "train"
    df.loc[df["patient_id"].isin(val_patients), "split"] = "val"
    df.loc[df["patient_id"].isin(test_patients), "split"] = "test"

    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print("Saved:", args.out_csv)
    print(df["split"].value_counts())
    print(pd.crosstab(df["split"], df["label"]))


if __name__ == "__main__":
    main()
