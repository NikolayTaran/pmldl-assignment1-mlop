"""Stage 3 — Model API (FastAPI).

Serves the Titanic survival model trained by Stage 2.
Endpoints:
    GET  /         — service info (never "Not Found" again)
    GET  /health   — liveness probe + model metadata + version diagnostics
    POST /predict  — single or batch prediction with probability

Runs inside its own Docker container (see Dockerfile).
"""
import os
import pickle
import platform
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

REBUILD_HINT = (
    "Most common cause: the container was built with an old "
    "code/deployment/api/requirements.txt (pinned scikit-learn), while the model "
    "was trained by the scikit-learn installed on the host. Fix: update "
    "code/deployment/api/requirements.txt to minimum bounds matching your host "
    "packages, then rebuild and restart the containers: "
    "docker compose -f code/deployment/docker-compose.yml up -d --build"
)


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
    version="1.1.0",
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
_model_loaded_at = None
_load_warnings = []


def _versions() -> dict:
    return {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def get_model():
    """Load the pickle lazily; a fresh container start picks up the latest model."""
    global _model, _model_loaded_at, _load_warnings
    if _model is None:
        if not MODEL_PATH.is_file():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model file not found at {MODEL_PATH}. Run Stage 2 first "
                    "(python code/models/train_model.py) or check the ./models "
                    "volume mount in docker-compose.yml."
                ),
            )
        try:
            # sklearn emits InconsistentVersionWarning here when the pickle was
            # created by a different sklearn version than the one running now —
            # capture it and surface it in /health
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with MODEL_PATH.open("rb") as fh:
                    _model = pickle.load(fh)
            _load_warnings = [str(w.message) for w in caught]
        except Exception as exc:
            traceback.print_exc()  # full traceback stays visible in docker logs
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Could not load the model pickle: {exc.__class__.__name__}: {exc}. "
                    f"This container runs scikit-learn {sklearn.__version__}. {REBUILD_HINT}"
                ),
            ) from exc
        _model_loaded_at = datetime.now(timezone.utc).isoformat()
    return _model


def _predict_proba(model, df: pd.DataFrame):
    """predict_proba with a diagnostic error message instead of a raw 500."""
    try:
        return model.predict_proba(df)[:, 1]
    except Exception as exc:
        traceback.print_exc()  # full traceback stays visible in docker logs
        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction failed: {exc.__class__.__name__}: {exc}. "
                f"This container runs scikit-learn {sklearn.__version__} — if it differs "
                "from the version that trained models/model.pkl, the pickle cannot be "
                f"served (check GET /health, field sklearn_version_match). {REBUILD_HINT}"
            ),
        ) from exc


def _to_frame(items: list[Passenger]) -> pd.DataFrame:
    rows = [p.model_dump() for p in items]
    df = pd.DataFrame(rows)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df[FEATURE_COLUMNS]


@app.get("/")
def root() -> dict:
    """Friendly service index — so the root URL never looks like an error."""
    return {
        "service": "Titanic Survival API",
        "description": "PMLDL Assignment 1 — Stage 3 model API",
        "usage": {
            "docs": "/docs",
            "health": "/health",
            "predict": 'POST /predict — a single passenger object or {"passengers": [...]}',
        },
        "versions": _versions(),
    }


@app.get("/health")
def health() -> dict:
    model = get_model()
    version_warnings = [w for w in _load_warnings if "unpickle" in w]
    return {
        "status": "ok",
        "model_loaded_at": _model_loaded_at,
        "model_type": type(model.named_steps.get("clf")).__name__,
        "versions": _versions(),
        "sklearn_version_match": not version_warnings,
        "load_warnings": _load_warnings,
    }


@app.post("/predict", response_model=Union[PredictionOut, list[PredictionOut]])
def predict(payload: Union[Passenger, BatchIn]) -> Union[PredictionOut, list[PredictionOut]]:
    if isinstance(payload, BatchIn):
        if not payload.passengers:
            raise HTTPException(status_code=400, detail="passengers list is empty")
        model = get_model()
        proba = _predict_proba(model, _to_frame(payload.passengers))
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
    proba = float(_predict_proba(model, _to_frame([payload]))[0])
    return PredictionOut(
        survival_probability=round(proba, 4),
        survived=int(proba >= 0.5),
        model="titanic-survival",
        predicted_at=datetime.now(timezone.utc).isoformat(),
    )
