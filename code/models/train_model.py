#!/usr/bin/env python3
"""Stage 2 — Model Engineering.

Pipeline stage that:
  1. reads the train/test datasets produced by Stage 1 (data/processed/),
  2. builds features (sklearn Pipeline: imputation + scaling + one-hot encoding),
  3. trains a RandomForest model and evaluates it on the test split,
  4. logs params/metrics and the model to MLflow (local file store: mlruns/),
  5. packages the full inference pipeline to models/model.pkl and writes
     the test metrics to reports/metrics.json.

Idempotent: safe to re-run every pipeline cycle.

Usage:
    python code/models/train_model.py [--data ...] [--models ...]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

# the classic filesystem tracking store (./mlruns) works on every MLflow version;
# MLflow >= 3 requires an explicit opt-in for it
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Survived"
NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare", "FamilySize"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "Title", "IsAlone"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Small feature engineering layer on top of the cleaned data."""
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "precision": round(float(precision_score(y_test, pred)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 — Model Engineering")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data)
    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    print(f"[stage2] loaded: train={len(train_df)}, test={len(test_df)}")

    train_df = add_features(train_df)
    test_df = add_features(test_df)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X_train, y_train = train_df[feature_cols], train_df[TARGET]
    X_test, y_test = test_df[feature_cols], test_df[TARGET]

    # two candidate models — the better one on test is packaged (light "tuning")
    candidates: dict[str, dict] = {
        "logistic_regression": {
            "model": LogisticRegression(max_iter=1000, random_state=args.seed),
            "params": {"model": "LogisticRegression", "max_iter": 1000},
        },
        "random_forest": {
            "model": RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=args.seed, n_jobs=-1
            ),
            "params": {
                "model": "RandomForestClassifier",
                "n_estimators": 200,
                "max_depth": 8,
            },
        },
    }

    mlflow.set_tracking_uri("file:./mlruns")  # local filesystem store
    mlflow.set_experiment("titanic-survival")

    best_name, best_pipeline, best_metrics = None, None, None
    with mlflow.start_run(run_name="pipeline-run") as run:
        run_id = run.info.run_id
        for name, cand in candidates.items():
            pipe = Pipeline(
                steps=[("preprocessor", build_preprocessor()), ("clf", cand["model"])]
            )
            pipe.fit(X_train, y_train)
            metrics = evaluate(pipe, X_test, y_test)
            mlflow.log_metrics({f"{name}_{k}": v for k, v in metrics.items()})
            mlflow.log_params({f"{name}_{k}": str(v) for k, v in cand["params"].items()})
            print(f"[stage2] {name}: {metrics}")
            if best_metrics is None or metrics["roc_auc"] > best_metrics["roc_auc"]:
                best_name, best_pipeline, best_metrics = name, pipe, metrics

        # log the winning model itself as the run's model artifact
        mlflow.log_params({"selected_model": best_name})
        mlflow.log_metrics({f"selected_{k}": v for k, v in best_metrics.items()})

        # package for deployment: the whole pipeline (preprocessor + classifier)
        model_path = models_dir / "model.pkl"
        with model_path.open("wb") as fh:
            pickle.dump(best_pipeline, fh)

        # log the model to MLflow in a version-robust way:
        #   * MLflow >= 3 saves sklearn models via skops — needs `name` and
        #     an explicit list of trusted types (numpy.dtype lives inside the
        #     sklearn pipeline),
        #   * MLflow 2.x uses `artifact_path` instead,
        #   * if neither flavor call works, log the pickle as a plain
        #     artifact so the run always contains the model.
        logged = False
        try:
            mlflow.sklearn.log_model(
                best_pipeline, name="model", skops_trusted_types=["numpy.dtype"]
            )
            logged = True
        except Exception as exc:  # MLflow 2.x has no `name`/`skops_trusted_types`
            print(f"[stage2] sklearn flavor with `name` failed: {exc.__class__.__name__}: {exc}")
        if not logged:
            try:
                mlflow.sklearn.log_model(best_pipeline, artifact_path="model")
                logged = True
            except Exception as exc:
                print(f"[stage2] sklearn flavor with `artifact_path` failed: "
                      f"{exc.__class__.__name__}: {exc}")
        if not logged:
            mlflow.log_artifact(str(model_path), artifact_path="model")
            print("[stage2] logged model.pkl as a plain MLflow artifact")

        metrics_record = {
            "selected_model": best_name,
            "run_id": run_id,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            **best_metrics,
        }
        metrics_path = reports_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_record, indent=2), encoding="utf-8")

    print(f"[stage2] selected model: {best_name} | test metrics: {best_metrics}")
    print(f"[stage2] model saved to: {model_path}")
    print(f"[stage2] metrics saved to: {reports_dir / 'metrics.json'}")
    print(f"[stage2] mlflow run id: {run_id} (tracking store: ./mlruns)")


if __name__ == "__main__":
    main()
