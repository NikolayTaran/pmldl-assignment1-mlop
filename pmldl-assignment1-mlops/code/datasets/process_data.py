#!/usr/bin/env python3
"""Stage 1 — Data Engineering.

Pipeline stage that:
  1. loads the raw Titanic data (data/raw/titanic.csv),
  2. cleans it — imputes missing values, extracts the passenger title,
     drops unusable columns, removes outliers in Fare (IQR rule),
  3. splits into stratified train/test datasets,
  4. saves them to data/processed/.

The script is idempotent: re-running it simply overwrites the outputs,
which makes it safe for the scheduled (every 5 minutes) pipeline.

Usage:
    python code/datasets/process_data.py [--raw ...] [--out ...] [--test-size 0.2]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Survived"

# Columns that carry no generalizable signal for a per-passenger model
DROP_COLUMNS = ["PassengerId", "Name", "Ticket", "Cabin"]


def load_raw(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw data not found at {path}. "
            "Place titanic.csv into data/raw/ (see README for the download link)."
        )
    df = pd.read_csv(path)
    print(f"[stage1] loaded raw data: {df.shape[0]} rows x {df.shape[1]} cols from {path}")
    return df


def extract_title(name: str) -> str:
    """Mr / Mrs / Miss / Master / Rare — a strong feature built from Name."""
    title = name.split(",")[1].split(".")[0].strip().lower() if "," in name and "." in name else ""
    if title in ("mr",):
        return "Mr"
    if title in ("mrs", "mme", "lady", "the countess"):
        return "Mrs"
    if title in ("miss", "mlle", "ms"):
        return "Miss"
    if title in ("master",):
        return "Master"
    return "Rare" if title else "Mr"


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the dataframe; returns (clean df, report dict with cleaning stats)."""
    report: dict = {}
    df = df.copy()

    # ---- missing values overview -------------------------------------------
    report["missing_before"] = {c: int(n) for c, n in df.isna().sum().items() if n > 0}

    # ---- feature engineering before dropping Name ---------------------------
    df["Title"] = df["Name"].apply(extract_title)

    # ---- impute Age with the per-(Pclass, Sex) median ----------------------
    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda s: s.fillna(s.median())
    )
    # any leftovers (a fully-empty group) -> global median
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # ---- impute Embarked with the mode, Fare with the median ---------------
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # ---- drop unusable columns ---------------------------------------------
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    report["missing_after"] = {c: int(n) for c, n in df.isna().sum().items() if n > 0}

    # ---- outlier removal on Fare via the IQR rule --------------------------
    q1, q3 = df["Fare"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    before = len(df)
    df = df[(df["Fare"] >= lower) & (df["Fare"] <= upper)].reset_index(drop=True)
    report["fare_outliers_removed"] = before - len(df)
    report["fare_bounds"] = [float(lower), float(upper)]

    report["rows_after_cleaning"] = int(len(df))
    return df, report


def split_data(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split on the target column."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[TARGET],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 — Data Engineering")
    parser.add_argument("--raw", default="data/raw/titanic.csv")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_path = Path(args.raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(raw_path)
    cleaned, report = clean_data(df)
    train_df, test_df = split_data(cleaned, args.test_size, args.seed)

    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # a small report next to the data — useful artifact for the pipeline logs
    summary = {
        "raw_rows": int(len(df)),
        "clean_rows": int(len(cleaned)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_survival_rate": round(float(train_df[TARGET].mean()), 4),
        "test_survival_rate": round(float(test_df[TARGET].mean()), 4),
        **report,
    }
    report_path = out_dir / "data_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"[stage1] cleaned: {summary['raw_rows']} -> {summary['clean_rows']} rows "
        f"(removed {summary['fare_outliers_removed']} fare outliers)"
    )
    print(
        f"[stage1] split: train={len(train_df)} (survival {summary['train_survival_rate']}), "
        f"test={len(test_df)} (survival {summary['test_survival_rate']})"
    )
    print(f"[stage1] saved: {train_path}, {test_path}, {report_path}")


if __name__ == "__main__":
    main()
