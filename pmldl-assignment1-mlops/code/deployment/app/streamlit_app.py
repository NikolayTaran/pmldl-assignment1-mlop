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

# ---- predict button ---------------------------------------------------------
if st.button("Predict", type="primary", use_container_width=True):
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
        st.stop()

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
else:
    st.info("Fill in the passenger details and press **Predict**.")

# ---- API status footer ------------------------------------------------------
with st.expander("API status"):
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.json(health)
    except requests.RequestException as exc:
        st.error(f"API unreachable: {exc}")
