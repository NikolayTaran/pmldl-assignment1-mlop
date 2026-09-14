"""Stage 3 — Model API (FastAPI).

Serves the Titanic survival model trained by Stage 2.
Endpoints:
    GET  /health   — liveness probe + model metadata
    POST /predict  — single or batch prediction with probability

Runs inside its own Docker container (see Dockerfile).
"""
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

def _default_models_dir() -> str:
    """Outside Docker: repo_root/models (this file lives in code/deployment/api/).
    Inside a bare container (/srv/main.py): fall back to /srv/models."""
    parents = Path(__file__).resolve().parents
    return str(parents[3] / "models") if len(parents) > 3 else "/srv/models"


# the compose file mounts ./models and sets MODEL_DIR=/srv/models
MODEL_PATH = Path(os.environ.get("MODEL_DIR", _default_models_dir())) / "model.pkl"
FEATURE_COLUMNS = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title",
    "FamilySize", "IsAlone",
]

app = FastAPI(
    title="Titanic Survival API",
    description="PMLDL Assignment 1 — model API for Stage 3 (deployment).",
    version="1.0.0",
)


class Passenger(BaseModel):
    """Input fields shown in the web app."""
    Pclass: int = Field(3, ge=1, le=3, description="Ticket class: 1, 2 or 3")
    Sex: str = Field("male", description="'male' or 'female'")
    Age: float = Field(28.0, ge=0.0, le=100.0)
    SibSp: int = Field(0, ge=0, le=10, description="Siblings/spouses aboard")
    Parch: int = Field(0, ge=0, le=10, description="Parents/children aboard")
    Fare: float = Field(14.0, ge=0.0, le=600.0)
    Embarked: str = Field("S", description="Port: C, Q or S")
    Title: str = Field("Mr", description="Mr, Mrs, Miss, Master or Rare")


class PredictionOut(BaseModel):
    survival_probability: float
    survived: int
    model: str
    predicted_at: str


class BatchIn(BaseModel):
    passengers: list[Passenger]


_model = None
_model_loaded_at: "str | None" = None


def get_model():
    """Load the pickle lazily; a fresh container start picks up the latest model."""
    global _model, _model_loaded_at
    if _model is None:
        if not MODEL_PATH.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"Model file not found at {MODEL_PATH}. Run Stage 2 first.",
            )
        with MODEL_PATH.open("rb") as fh:
            _model = pickle.load(fh)
        _model_loaded_at = datetime.now(timezone.utc).isoformat()
    return _model


def _to_frame(items: list[Passenger]) -> pd.DataFrame:
    rows = [p.model_dump() for p in items]
    df = pd.DataFrame(rows)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df[FEATURE_COLUMNS]


@app.get("/health")
def health() -> dict:
    model = get_model()
    return {
        "status": "ok",
        "model_loaded_at": _model_loaded_at,
        "model_type": type(model.named_steps.get("clf")).__name__,
    }


@app.post("/predict", response_model=Union[PredictionOut, list[PredictionOut]])
def predict(payload: Union[Passenger, BatchIn]) -> Union[PredictionOut, list[PredictionOut]]:
    if isinstance(payload, BatchIn):
        if not payload.passengers:
            raise HTTPException(status_code=400, detail="passengers list is empty")
        model = get_model()
        proba = model.predict_proba(_to_frame(payload.passengers))[:, 1]
        return [
            PredictionOut(
                survival_probability=round(float(p), 4),
                survived=int(p >= 0.5),
                model="titanic-survival",
                predicted_at=datetime.now(timezone.utc).isoformat(),
            )
            for p in proba
        ]

    model = get_model()
    proba = float(model.predict_proba(_to_frame([payload]))[0, 1])
    return PredictionOut(
        survival_probability=round(proba, 4),
        survived=int(proba >= 0.5),
        model="titanic-survival",
        predicted_at=datetime.now(timezone.utc).isoformat(),
    )
