"""Stage 3 — Web application (Streamlit).

Simple UI that talks to the model API (separate container):
  * input fields for the passenger features,
  * a "Predict" button,
  * an area that displays the prediction returned by the API.
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000")

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Predictor")
st.caption(
    "PMLDL Assignment 1 — Stage 3. This app calls the model API "
    f"running in a separate Docker container (`{API_URL}`)."
)

# ---- input fields -----------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Ticket class", [1, 2, 3], index=2, format_func=lambda v: f"{v} ({'1st' if v == 1 else '2nd' if v == 2 else '3rd'})")
    sex = st.radio("Sex", ["male", "female"], horizontal=True)
    age = st.slider("Age", 0.0, 100.0, 28.0, 0.5)
    sibsp = st.number_input("Siblings / spouses aboard", 0, 10, 0)
with col2:
    parch = st.number_input("Parents / children aboard", 0, 10, 0)
    fare = st.slider("Fare (£)", 0.0, 300.0, 14.0, 0.5)
    embarked = st.selectbox("Port of embarkation", ["C", "Q", "S"], index=2,
                            format_func=lambda v: {"C": "C — Cherbourg", "Q": "Q — Queenstown", "S": "S — Southampton"}[v])
    title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"], index=0)


def show_api_troubleshooting(exc: requests.RequestException) -> None:
    """Actionable hints when the API answers with a server error."""
    if not (isinstance(exc, requests.HTTPError) and exc.response is not None
            and exc.response.status_code >= 500):
        return
    detail = ""
    try:
        detail = exc.response.json().get("detail", "")
    except ValueError:
        pass
    if detail:
        st.warning(f"API says: {detail}")
    st.markdown(
        "**How to fix a 500 from the API container** (most common cause — "
        "scikit-learn version mismatch between the host that trained "
        "`models/model.pkl` and the container that serves it):\n"
        "1. Full traceback: `docker logs titanic-api`\n"
        "2. Version check: open `http://localhost:8000/health` — the field "
        "`sklearn_version_match` must be `true`\n"
        "3. Update `code/deployment/api/requirements.txt` (minimum bounds, "
        "same versions as on the host) and rebuild:\n"
        "   `docker compose -f code/deployment/docker-compose.yml up -d --build`"
    )


# ---- predict button ---------------------------------------------------------
pressed = st.button("Predict", type="primary", use_container_width=True)
data = None
if pressed:
    payload = {
        "Pclass": int(pclass),
        "Sex": sex,
        "Age": float(age),
        "SibSp": int(sibsp),
        "Parch": int(parch),
        "Fare": float(fare),
        "Embarked": embarked,
        "Title": title,
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach the model API at {API_URL}: {exc}")
        show_api_troubleshooting(exc)

if data is not None:
    prob = data["survival_probability"]
    survived = bool(data["survived"])

    # ---- prediction display area -------------------------------------------
    st.markdown("---")
    if survived:
        st.success(f"### ✅ Survives — probability {prob:.1%}")
    else:
        st.error(f"### ❌ Does not survive — survival probability {prob:.1%}")

    st.progress(min(prob, 1.0))
    st.caption(
        f"Model: {data['model']} · predicted at {data['predicted_at']} · served by {API_URL}"
    )
elif not pressed:
    st.info("Fill in the passenger details and press **Predict**.")

# ---- API status footer ------------------------------------------------------
with st.expander("API status"):
    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=5)
        health_resp.raise_for_status()
        health = health_resp.json()
        st.json(health)
        if health.get("sklearn_version_match") is False:
            st.error(
                "**scikit-learn version mismatch detected**: the model was trained "
                "with a different scikit-learn than the one inside the API container "
                "(see `load_warnings` above). Update "
                "`code/deployment/api/requirements.txt` to match the host versions "
                "and rebuild: `docker compose -f code/deployment/docker-compose.yml "
                "up -d --build`"
            )
    except requests.RequestException as exc:
        st.error(f"API unreachable: {exc}")
