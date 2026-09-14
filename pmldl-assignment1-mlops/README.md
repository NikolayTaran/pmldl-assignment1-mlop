# PMLDL Assignment 1: Automated ML Pipeline (Data → Model → Deployment)

An automated MLOps pipeline with three stages that runs **every 5 minutes**:

| Stage | What it does | Code |
|---|---|---|
| **1. Data Engineering** | loads raw Titanic data, cleans it (imputes missing `Age`/`Embarked`, extracts `Title` from `Name`, removes `Fare` outliers via IQR), splits into stratified train/test | `code/datasets/process_data.py` |
| **2. Model Engineering** | feature engineering (`FamilySize`, `IsAlone`), trains LogisticRegression + RandomForest, picks the better one by ROC-AUC, logs params/metrics/model to **MLflow**, saves `models/model.pkl` and `reports/metrics.json` | `code/models/train_model.py` |
| **3. Deployment** | rebuilds and (re)starts two **separate Docker containers**: FastAPI model API (:8000) and Streamlit web app (:8501) via docker-compose | `code/deployment/` |

The web app has input fields for the passenger, a **Predict** button, and an
area that shows the prediction returned by the API.

## Repository layout

```
├── code/
│   ├── datasets/
│   │   └── process_data.py          # Stage 1
│   ├── models/
│   │   └── train_model.py           # Stage 2 (MLflow)
│   └── deployment/
│       ├── api/                     # Stage 3: FastAPI + Dockerfile
│       │   ├── main.py
│       │   ├── requirements.txt
│       │   └── Dockerfile
│       ├── app/                     # Stage 3: Streamlit + Dockerfile
│       │   ├── streamlit_app.py
│       │   ├── requirements.txt
│       │   └── Dockerfile
│       └── docker-compose.yml
├── data/
│   ├── raw/titanic.csv              # input artifact
│   └── processed/                   # train.csv, test.csv (generated)
├── models/                          # model.pkl (generated)
├── notebooks/                       # optional EDA notebook
├── reports/                         # metrics.json (generated)
├── services/
│   └── scheduler/
│       ├── pipeline.py              # orchestrates stages 1→2→3
│       └── scheduler.py             # runs the pipeline every 5 minutes
├── requirements.txt
└── README.md
```

## Prerequisites

* Python 3.10+
* Docker + Docker Compose (Docker Desktop is enough)

## How to run

### 1. One-time setup

```bash
git clone https://github.com/<your-username>/pmldl-assignment1-mlops.git
cd pmldl-assignment1-mlops

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The raw data is already in `data/raw/titanic.csv`
(source: [datasciencedojo/datasets](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)).

### 2. Run the complete pipeline once

```bash
python services/scheduler/pipeline.py
```

This executes Stage 1 → Stage 2 → Stage 3 and leaves two containers running:

* **Web app**: http://localhost:8501 — fill the fields, press **Predict**
* **Model API**: http://localhost:8000 — interactive docs at http://localhost:8000/docs

### 3. Run the pipeline automatically every 5 minutes

```bash
python services/scheduler/scheduler.py
```

Leave it running — every 5 minutes it re-runs data engineering, retrains the
model, and restarts the containers with the fresh model. Stop with Ctrl+C.

**Alternative — cron** (no terminal window needed):

```cron
*/5 * * * * cd /path/to/pmldl-assignment1-mlops && /path/to/.venv/bin/python services/scheduler/pipeline.py >> reports/pipeline.log 2>&1
```

## Using the API directly

```bash
# health check
curl http://localhost:8000/health

# single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Pclass": 3, "Sex": "female", "Age": 24, "SibSp": 0, "Parch": 2,
       "Fare": 15.5, "Embarked": "S", "Title": "Miss"}'

# batch prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"passengers": [{"Pclass": 1, "Sex": "male", "Age": 40, "SibSp": 1, "Parch": 0,
       "Fare": 90.0, "Embarked": "C", "Title": "Mr"},
      {"Pclass": 3, "Sex": "female", "Age": 6, "SibSp": 0, "Parch": 1,
       "Fare": 12.0, "Embarked": "Q", "Title": "Miss"}]}'
```

Response:

```json
{"survival_probability": 0.7312, "survived": 1, "model": "titanic-survival", "predicted_at": "..."}
```

## MLflow

Stage 2 logs every run to a local MLflow file store (`./mlruns`). Inspect it with:

```bash
mlflow ui --backend-store-uri ./mlruns
# open http://127.0.0.1:5000
```

Metrics are also written in plain JSON to `reports/metrics.json`.

## Notes

* The pipeline is **idempotent** — re-running simply overwrites
  `data/processed/`, `models/model.pkl` and restarts the containers.
* The API container mounts `./models` as a volume, so after each cycle it
  serves the model from the latest run.
* If a full cycle ever takes more than 5 minutes on your machine, start the
  scheduler with a larger interval, e.g. `python services/scheduler/scheduler.py --interval 10`.
