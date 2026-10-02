"""Restaurant Tip Amount Predictor — Streamlit App"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# XGBoost model load karne ke liye zaroori
import xgboost  # noqa: F401

st.set_page_config(
    page_title="Restaurant Tip Predictor",
    page_icon="🍽️",
    layout="centered",
)

BUNDLE_PATH = Path(__file__).parent / "tip_model_bundle.pkl"


@st.cache_resource(show_spinner="Loading model...")
def load_bundle():
    # Pehle bundle try karo
    if BUNDLE_PATH.exists():
        return joblib.load(BUNDLE_PATH)

    # Warna purani files se fallback
    return {
        "model": joblib.load("best_tip_regressor.pkl"),
        "scaler": joblib.load("scaler.pkl"),
        "feature_names": list(joblib.load("feature_names.pkl")),
        "te_means": joblib.load("te_means.pkl"),
    }


bundle = load_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
FEATURE_NAMES = list(bundle["feature_names"])
TE_MEANS = bundle["te_means"]


def te_lookup(key, lookup, default=0.0):
    table = TE_MEANS.get(key)
    if table is None:
        return default
    if isinstance(table, dict):
        return float(table.get(lookup, default))
    try:
        return float(table)
    except (TypeError, ValueError):
        return default


def build_features(total_bill, size, sex, smoker, day, time):
    is_weekend = 1 if day in ("Sat", "Sun") else 0
    is_dinner = 1 if time == "Dinner" else 0
    is_male = 1 if sex == "Male" else 0
    is_smoker = 1 if smoker == "Yes" else 0
    is_large_party = 1 if size >= 5 else 0
    bill_per_head = total_bill / size if size > 0 else 0.0
    bill_squared = total_bill ** 2
    log_bill = float(np.log1p(total_bill)) if total_bill >= 0 else 0.0
    day_Sat = 1 if day == "Sat" else 0
    day_Sun = 1 if day == "Sun" else 0
    day_Thur = 1 if day == "Thur" else 0
    time_Lunch = 1 if time == "Lunch" else 0

    row = {
        "total_bill": total_bill,
        "size": size,
        "is_weekend": is_weekend,
        "is_dinner": is_dinner,
        "is_male": is_male,
        "is_smoker": is_smoker,
        "is_large_party": is_large_party,
        "bill_per_head": bill_per_head,
        "bill_squared": bill_squared,
        "log_bill": log_bill,
        "day_Sat": day_Sat,
        "day_Sun": day_Sun,
        "day_Thur": day_Thur,
        "time_Lunch": time_Lunch,
        "size_te": te_lookup("size_te", size),
        "weekend_te": te_lookup("weekend_te", is_weekend),
        "dinner_te": te_lookup("dinner_te", is_dinner),
        "sat_te": te_lookup("sat_te", day_Sat),
        "sun_te": te_lookup("sun_te", day_Sun),
        "thur_te": te_lookup("thur_te", day_Thur),
        "lunch_te": te_lookup("lunch_te", time_Lunch),
        "size_dinner_te": te_lookup("size_dinner_te", (size, is_dinner)),
        "size_weekend_te": te_lookup("size_weekend_te", (size, is_weekend)),
    }
    return pd.DataFrame([row], columns=FEATURE_NAMES)


st.title("🍽️ Restaurant Tip Amount Predictor")
st.write("Enter bill and party details to predict the tip amount.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        total_bill = st.number_input("Total bill ($)", 0.0, 1000.0, 50.0, 1.0)
        size = st.slider("Party size", 1, 10, 2)
        sex = st.selectbox("Sex", ["Male", "Female"])
    with col2:
        smoker = st.selectbox("Smoker", ["No", "Yes"])
        day = st.selectbox("Day", ["Thur", "Fri", "Sat", "Sun"])
        time = st.selectbox("Time", ["Lunch", "Dinner"])
    submitted = st.form_submit_button("💰 Predict tip")

if submitted:
    try:
        X = build_features(total_bill, size, sex, smoker, day, time)
        pred = float(model.predict(scaler.transform(X))[0])
        pred = max(pred, 0.0)
        st.success(f"### Predicted tip: **${pred:.2f}**")
        if total_bill > 0:
            st.caption(f"Tip % of bill: **{(pred / total_bill * 100):.1f}%**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

with st.sidebar:
    st.header("ℹ️ Info")
    st.write(f"**Model:** {type(model).__name__}")
    st.write(f"**Features:** {len(FEATURE_NAMES)}")
