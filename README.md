[README.md](https://github.com/user-attachments/files/32299614/README.md)
# PMLDL Assignment 1: Automated ML Pipeline (Data → Model → Deployment)

An automated MLOps pipeline with three stages that runs **every 5 minutes**:

| Stage | What it does | Code |
|---|---|---|
| **1. Data Engineering** | loads raw Titanic data, cleans it (imputes missin# PMLDL Assignment 1 — Automated ML Pipeline: Data → Model → Deployment

Автоматизированный MLOps-пайплайн для предсказания выживаемости пассажиров
Титаника. Три стадии (данные → модель → деплой), которые запускаются одной
командой и **автоматически повторяются каждые 5 минут**: пайплайн заново чистит
данные, переобучает модель, логирует эксперимент в MLflow и пересобирает два
Docker-контейнера (REST API + веб-приложение).

| Stage | Что делает | Код |
|---|---|---|
| **1. Data Engineering** | загрузка сырых данных, чистка (импутация пропусков, извлечение `Title` из `Name`, удаление выбросов `Fare` по IQR), стратифицированный train/test сплит | `code/datasets/process_data.py` |
| **2. Model Engineering** | фичи (`FamilySize`, `IsAlone`), обучение LogisticRegression + RandomForest, выбор лучшей по ROC-AUC, логирование в **MLflow**, сохранение `models/model.pkl` + `reports/metrics.json` | `code/models/train_model.py` |
| **3. Deployment** | два **отдельных Docker-контейнера**: FastAPI model API (:8000) и Streamlit веб-приложение (:8501) через docker-compose | `code/deployment/` |
| **Оркестрация** | `pipeline.py` соединяет стадии 1→2→3, `scheduler.py` запускает цикл каждые 5 минут | `services/scheduler/` |

## Архитектура

```
                        каждые 5 минут (scheduler.py)
                                      │
 ┌────────────────────────────────────▼─────────────────────────────────────┐
 │                            pipeline.py                                   │
 │                                                                          │
 │  Stage 1: process_data.py          Stage 2: train_model.py               │
 │  data/raw/titanic.csv ──────────►  data/processed/{train,test}.csv       │
 │        (чистка, Title, IQR)      │  2 кандидата → лучший по ROC-AUC      │
 │                                   │  логирование в MLflow (./mlruns)     │
 │                                   ▼                                      │
 │                            models/model.pkl  +  reports/metrics.json     │
 └────────────────────────────────────┬─────────────────────────────────────┘
                                      │ Stage 3: docker compose up -d --build
                                      ▼
                 ┌──────────────────────────────────────────────┐
                 │  Container "titanic-api"   (FastAPI, :8000)   │
                 │  mounts ./models → /srv/models (read-only)    │
                 │  GET /  GET /health  POST /predict            │
                 └──────────────────┬───────────────────────────┘
                                    │ HTTP (docker network, http://api:8000)
                                    ▼
                 ┌──────────────────────────────────────────────┐
                 │  Container "titanic-app"  (Streamlit, :8501)  │
                 │  форма пассажира → кнопка Predict → прогноз   │
                 └──────────────────────────────────────────────┘
```

Ключевая идея деплоя: каталог `./models` монтируется в API-контейнер как
read-only volume, поэтому каждый цикл пайплайна подменяет `model.pkl`, и после
рестарта контейнеров API обслуживает свежую модель — без пересборки образа.

## Структура репозитория

```
├── code/
│   ├── datasets/
│   │   └── process_data.py        # Stage 1 — чистка данных и сплит
│   ├── models/
│   │   └── train_model.py         # Stage 2 — обучение, MLflow, model.pkl
│   └── deployment/
│       ├── api/                   # Stage 3 — model API
│       │   ├── main.py            #   FastAPI: /  /health  /predict
│       │   ├── requirements.txt   #   зависимости контейнера API
│       │   └── Dockerfile         #   python:3.12-slim + uvicorn
│       ├── app/                   # Stage 3 — веб-приложение
│       │   ├── streamlit_app.py   #   форма + Predict + API status
│       │   ├── requirements.txt   #   streamlit, requests
│       │   └── Dockerfile         #   python:3.12-slim + streamlit
│       └── docker-compose.yml     # оба контейнера + volume ./models
├── data/
│   ├── raw/titanic.csv            # исходные данные (891 строка) — единственный вход
│   └── processed/                 # ГЕНЕРИРУЕТСЯ: train.csv, test.csv, data_report.json
├── models/                        # ГЕНЕРИРУЕТСЯ: model.pkl (итоговый sklearn-Pipeline)
├── notebooks/
│   └── eda_titanic.ipynb          # опциональный EDA (пайплайн его не использует)
├── reports/                       # ГЕНЕРИРУЕТСЯ: metrics.json (+ pipeline.log при cron)
├── services/scheduler/
│   ├── pipeline.py                # оркестратор: stage1 → stage2 → docker compose
│   └── scheduler.py               # цикл каждые N минут (по умолчанию 5)
├── mlruns/                        # ГЕНЕРИРУЕТСЯ: локальный MLflow tracking store
├── requirements.txt               # зависимости хоста (stages 1–2 + scheduler)
└── README.md
```

Каталоги `data/processed/`, `models/`, `reports/`, `mlruns/` — артефакты:
генерируются пайплайном, в git не попадают (см. `.gitignore`), пайплайн
идемпотентен — перезапуск просто перезаписывает их.

## Быстрый старт

### 1. Установка (один раз)

```bash
git clone https://github.com/<your-username>/pmldl-assignment1-mlops.git
cd pmldl-assignment1-mlops

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Требования: **Python 3.10+**, **Docker + Docker Compose** (Docker Desktop).
Версии в `requirements.txt` — минимальные границы, не точные пины: pip ставит
самые свежие release с прекомпилированными wheel'ами под вашу версию Python
(работает на 3.10–3.14). Сырые данные уже в репозитории
(`data/raw/titanic.csv`, источник —
[datasciencedojo/datasets](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)).

### 2. Прогнать весь пайплайн один раз

```bash
python services/scheduler/pipeline.py
```

Выполнит Stage 1 → Stage 2 → Stage 3 и оставит два контейнера работающими:

* **Веб-приложение**: http://localhost:8501 — заполнить поля, нажать **Predict**
* **Model API**: http://localhost:8000 — интерактивная документация http://localhost:8000/docs

### 3. Автоматический режим — каждые 5 минут

```bash
python services/scheduler/scheduler.py
```

Оставить работать в терминале; остановка — `Ctrl+C`. Каждые 5 минут: чистка
данных → переобучение → логирование в MLflow → рестарт контейнеров со свежей
моделью. Другой интервал: `--interval 10` (минуты).

**Альтернатива — cron** (не нужно держать терминал открытым):

```cron
*/5 * * * * cd /path/to/pmldl-assignment1-mlops && /path/to/.venv/bin/python services/scheduler/pipeline.py >> reports/pipeline.log 2>&1
```

### 4. Остановить контейнеры

```bash
docker compose -f code/deployment/docker-compose.yml down
```

---

## Stage 1 — Data Engineering (`code/datasets/process_data.py`)

Вход: `data/raw/titanic.csv` (891 пассажир). Выход: `data/processed/train.csv`,
`test.csv` и `data_report.json` (отчёт о чистке).

По шагам:

1. **Извлечение `Title`** из `Name` (до удаления колонки): из строки
   `"Braund, Mr. Owen Harris"` берётся слово между запятой и точкой и
   схлопывается в 5 категорий: `Mr`; `Mrs` (Mrs, Mme, Lady, the Countess);
   `Miss` (Miss, Mlle, Ms); `Master` (мальчики); `Rare` (всё остальное —
   Dr, Rev, Col, Major, Capt, Sir, Don…). Каждый «редкий» титул встречается
   1–8 раз — по отдельности модель на них переобучится, вместе они дают
   полезный сигнал «нестандартный статус».
2. **Импутация пропусков**: `Age` — медиана по группе `(Pclass, Sex)`
   (возраст сильно зависит от класса и пола), остатки — глобальная медиана;
   `Embarked` — мода; `Fare` — медиана.
3. **Удаление мусорных колонок**: `PassengerId`, `Name`, `Ticket`, `Cabin`
   (687 из 891 пропусков — сигнала почти нет).
4. **Удаление выбросов `Fare`** по правилу IQR (1.5×): границы
   [−26.72, 65.63], удалено 116 строк, осталось **775**.
5. **Стратифицированный сплит** 80/20 по целевой `Survived`, `seed=42`:
   **train 620 / test 155** (доля выживших ~34% в обоих).

CLI: `--raw`, `--out`, `--test-size`, `--seed`. Скрипт идемпотентен —
безопасен для запуска каждые 5 минут.

## Stage 2 — Model Engineering (`code/models/train_model.py`)

Вход: `data/processed/{train,test}.csv`. Выход: `models/model.pkl`,
`reports/metrics.json`, run в MLflow.

По шагам:

1. **Feature engineering**: `FamilySize = SibSp + Parch + 1` (размер семьи
   вместе с пассажиром) и `IsAlone = 1 если FamilySize == 1`. Исторически
   лучше всего выживали семьи из 2–4 человек; одиночки и большие семьи — хуже.
2. **Признаки**:
   * числовые: `Age, SibSp, Parch, Fare, FamilySize`
   * категориальные: `Pclass, Sex, Embarked, Title, IsAlone`
3. **Preprocessor** (sklearn `ColumnTransformer`, часть итогового Pipeline):
   * числовые → `SimpleImputer(median)` → `StandardScaler`
   * категориальные → `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")`
     (неизвестная на инференсе категория не уронит API)
4. **Два кандидата**, оба обучаются на train и оцениваются на test:
   * `LogisticRegression(max_iter=1000)`
   * `RandomForestClassifier(n_estimators=200, max_depth=8)`
   * **лучший выбирается по ROC-AUC** (лёгкий авто-отбор вместо ручного тюнинга)
5. **Логирование в MLflow** (локальный file store `./mlruns`, эксперимент
   `titanic-survival`): параметры и метрики **обоих** кандидатов с префиксами
   `logistic_regression_*` / `random_forest_*`, плюс `selected_model` и метрики
   победителя (`selected_*`), и сама модель как артефакт run'а.
6. **Упаковка для деплоя**: целиком `Pipeline(preprocessor + clf)` пиклится в
   `models/model.pkl` — поэтому API на инференсе повторяет всю предобработку
   автоматически; отдельно `reports/metrics.json`.

Текущая модель-победитель (см. `reports/metrics.json`):

| selected_model | accuracy | roc_auc | precision | recall | f1 |
|---|---|---|---|---|---|
| logistic_regression | 0.8258 | **0.8559** | 0.7955 | 0.6604 | 0.7216 |

## Stage 3 — Deployment (`code/deployment/`)

Два сервиса в `docker-compose.yml`:

| Контейнер | Образ | Порт | Назначение |
|---|---|---|---|
| `titanic-api` | `code/deployment/api/Dockerfile` | **8000** | FastAPI, отдаёт прогнозы из `model.pkl` |
| `titanic-app` | `code/deployment/app/Dockerfile` | **8501** | Streamlit UI, ходит в API по `http://api:8000` (внутренняя docker-сеть) |

* `./models` монтируется в API-контейнер (`/srv/models:ro`) — модель обновляется
  без пересборки образа, достаточно рестарта контейнера.
* У каждого образа свой `requirements.txt` (изолированные зависимости) и
  `HEALTHCHECK`.
* `app` зависит от `api` (`depends_on`), оба с `restart: unless-stopped`.

### Model API — справочник эндпоинтов

| Метод | Путь | Что делает |
|---|---|---|
| GET | `/` | визитка сервиса + версии библиотек внутри контейнера |
| GET | `/health` | статус, тип модели, версии, `sklearn_version_match`, предупреждения загрузки pickle |
| POST | `/predict` | один пассажир **или** батч `{"passengers": [...]}` → вероятность выживания + метка |
| GET | `/docs` | интерактивная Swagger-документация |

Валидация входа — pydantic-модель `Passenger` (диапазоны `Pclass 1–3`,
`Age 0–100`, `Fare 0–600` и т.д.). Сервер сам досчитывает `FamilySize` и
`IsAlone` из `SibSp`/`Parch` — клиенту отправлять их не нужно. Модель
загружается ленимо при первом запросе; при несовпадении версии sklearn у
обучившего и обслуживающего контейнеров API вернёт развёрнутое объяснение и
рецепт починки вместо голого 500.

```bash
# здоровье
curl http://localhost:8000/health

# одиночный прогноз
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Pclass": 3, "Sex": "female", "Age": 24, "SibSp": 0, "Parch": 2,
       "Fare": 15.5, "Embarked": "S", "Title": "Miss"}'

# батч
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"passengers": [{"Pclass": 1, "Sex": "male", "Age": 40, "SibSp": 1, "Parch": 0,
       "Fare": 90.0, "Embarked": "C", "Title": "Mr"},
      {"Pclass": 3, "Sex": "female", "Age": 6, "SibSp": 0, "Parch": 1,
       "Fare": 12.0, "Embarked": "Q", "Title": "Miss"}]}'
```

Ответ:

```json
{"survival_probability": 0.7312, "survived": 1, "model": "titanic-survival", "predicted_at": "2025-01-01T12:00:00+00:00"}
```

### Веб-приложение — что значат поля

| Поле | Значение |
|---|---|
| **Ticket class** | класс билета 1/2/3 (1-й — люкс, выживаемость заметно выше) |
| **Sex** | пол — самый сильный предиктор («женщины и дети вперёд») |
| **Age** | возраст, годы (детей спасали в приоритете) |
| **Siblings / spouses aboard** | братья/сёстры и супруг(а) на борту |
| **Parents / children aboard** | родители и дети на борту |
| **Fare (£)** | цена билета (коррелирует с классом) |
| **Port of embarkation** | C — Шербур, Q — Куинстаун, S — Саутгемптон (из C садилось больше богатых) |
| **Title** | титул из имени: `Mr`, `Mrs` (замужние), `Miss` (незамужние/девочки), `Master` (мальчики), `Rare` (Dr, Rev, Col, Sir, … — сгруппированы из-за редкости) |

`FamilySize` и `IsAlone` в форме **нет** — их автоматически вычисляет API.
Раскрывающаяся плашка **API status** показывает живой вывод `/health`; если
`sklearn_version_match: false`, приложение прямо на экране объясняет, что
делать. При 500 от API приложение тоже печатает причину и шаги исправления.

---

## MLflow

Каждый цикл Stage 2 создаёт run в эксперименте `titanic-survival`
(локальный file store `./mlruns`): параметры и метрики обоих кандидатов,
победитель и модель как артефакт. UI:

> **MLflow 3.x:** filesystem store требует явного opt-in
> `MLFLOW_ALLOW_FILE_STORE=true`, без него `mlflow ui` падает; UI стартует
> ~15–20 секунд.

```bash
# Windows (cmd)
set MLFLOW_ALLOW_FILE_STORE=true
mlflow ui --backend-store-uri ./mlruns

# PowerShell
$env:MLFLOW_ALLOW_FILE_STORE = "true"; mlflow ui --backend-store-uri ./mlruns

# Linux / macOS
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri ./mlruns
```

Открыть http://127.0.0.1:5000. Метрики также продублированы в человекочитаемом
`reports/metrics.json`.

## Идемпотентность и надёжность

* **Пайплайн идемпотентен**: повторный запуск просто перезаписывает
  `data/processed/`, `models/model.pkl`, `reports/metrics.json`.
* **Scheduler переживает падения**: исключение одного цикла логируется и не
  останавливает процесс (`except Exception` + sleep до следующего запуска).
* **`pipeline.py` падает громко**: ненулевой код возврата стадии → полный
  stderr и `SystemExit(1)` — удобно для cron.
* **API диагностирует себя**: `/health` показывает версии и предупреждения
  pickle; ошибки `/predict` содержат причину и команду исправления; полный
  traceback — в `docker logs titanic-api`.

## Troubleshooting

**Predict → `500 Server Error`**
(`docker logs titanic-api` → `AttributeError: 'LogisticRegression' object has
no attribute 'multi_class'` или `InconsistentVersionWarning`)
Контейнер собран со старым scikit-learn, а модель обучена новой версией на
хосте — sklearn-пикли гарантированно работают только под версией-создателем.

1. Проверить: `curl http://localhost:8000/health` → `sklearn_version_match`
   должен быть `true`.
2. Обновить `code/deployment/api/requirements.txt` до минимальных границ
   (`scikit-learn>=1.4`).
3. `docker compose -f code/deployment/docker-compose.yml up -d --build`.

**`http://localhost:8000/` показывает `{"detail":"Not Found"}`**
Старый образ (до v1.1): пересобрать контейнеры командой выше. В актуальной
версии `/` отдаёт визитку сервиса, документация — `/docs`, диагностика — `/health`.

**`pip install` падает (`Preparing metadata (pyproject.toml) ... error`, meson)**
Нет прекомпилированного wheel под вашу версию Python, pip пытается собирать из
исходников. `python -m pip install --upgrade pip` и повторить. Если не
помогло — venv на Python 3.12: `py -3.12 -m venv .venv`.

**`ModuleNotFoundError: No module named 'numpy'`**
Не активирован venv или зависимости не установлены: `source .venv/bin/activate`
и `pip install -r requirements.txt`.

**Stage 3: `docker is not installed / not on PATH`**
Установить [Docker Desktop](https://www.docker.com/products/docker-desktop/),
запустить (кит в трее), проверить `docker compose version`. Linux:
`sudo apt install docker.io docker-compose-v2` + `sudo usermod -aG docker $USER`.

**Порты 8000 / 8501 заняты**
Освободить или поменять левую часть маппинга в
`code/deployment/docker-compose.yml` (например, `"8001:8000"`).

**Цикл дольше 5 минут**
Запустить scheduler с большим интервалом:
`python services/scheduler/scheduler.py --interval 10`.

---

## Типовые вопросы на защите

**— Почему модель LogisticRegression, а не «сложнее»?**
Обучаются оба кандидата (LR и RandomForest 200 деревьев, depth 8); по
ROC-AUC на тесте LR победила (0.8559). Выбор автоматический на каждом цикле —
если на новых данных победит RF, пайплайн сам начнёт обслуживать RF.

**— Почему метрики именно такие / что значит ROC-AUC?**
ROC-AUC — вероятность того, что случайный выживший получит больший скор,
чем случайный погибший; 0.856 при базовом уровне ~0.62 (дисбаланс классов
62/38) — сильное разделение. Accuracy 0.826 на test (155 строк).

**— Зачем OneHotEncoder с `handle_unknown="ignore"`?**
Если на инференсе придёт категория, которой не было в train (например,
экзотический порт), encoder передаст нули вместо падения — API не вернёт 500.

**— Почему весь Pipeline (препроцессинг + модель) в одном pickle?**
Гарантия, что трансформации на инференсе **идентичны** обучающим: медианы
импутации, масштабы StandardScaler и one-hot колонки зашиты в сам артефакт.
API не дублирует логику чистки — только досчитывает 2 производные фичи.

**— Зачем Title, если есть Sex и Age?**
Title кодирует сразу три сигнала: пол, возрастную группу (Master = мальчик,
Miss = девочка/незамужняя) и социальный статус (Rare). После удаления
`Name` это единственный след имени в признаках.

**— Почему удалили 116 строк с дорогими билетами (IQR)?**
Длинный хвост Fare (до £512) на 775 строках заставляет модель подгоняться
под считанные экстремальные точки. IQR-обрезка — стандартная защита от
выбросов; модель честно работает в диапазоне до ~£66.

**— Как обновляется модель без пересборки образа?**
`./models` — read-only volume в контейнере; цикл перезаписывает `model.pkl`,
`docker compose up -d --build` пересоздаёт контейнеры, lazy-load в API
подхватывает новый пикль при первом запросе.

**— Что будет, если пайплайн упадёт посреди ночи?**
Scheduler перехватит исключение, залогирует и дождётся следующего цикла;
контейнеры продолжают обслуживать последнюю рабочую модель
(`restart: unless-stopped`).

**— Как проверить, что всё живо?**
`curl http://localhost:8000/health` (API), плашка «API status» в приложении,
`docker ps`, MLflow UI — свежие run'ы каждые 5 минут.
g `Age`/`Embarked`, extracts `Title` from `Name`, removes `Fare` outliers via IQR), splits into stratified train/test | `code/datasets/process_data.py` |
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
