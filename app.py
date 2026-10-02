from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Restaurant Tip Predictor",
    page_icon="🍽️",
    layout="centered",
)

BUNDLE_PATH = Path("tip_model_bundle.pkl")


@st.cache_resource
def load_bundle(uploaded_file=None):
    if uploaded_file is not None:
        return joblib.load(uploaded_file)

    if BUNDLE_PATH.exists():
        return joblib.load(BUNDLE_PATH)

    # Fallback: separate files
    return {
        "model": joblib.load("best_tip_regressor.pkl"),
        "scaler": joblib.load("scaler.pkl"),
        "feature_names": joblib.load("feature_names.pkl"),
        "te_means": joblib.load("te_means.pkl"),
    }


def map_te(te_means, key, lookup, default=0.0):
    val = te_means.get(key, default)
    if isinstance(val, dict):
        return float(val.get(lookup, default))
    return float(val)


def build_features(total_bill, size, sex, smoker, day, time, te_means):
    is_weekend = 1 if day in ("Sat", "Sun") else 0
    is_dinner = 1 if time == "Dinner" else 0
    is_male = 1 if sex == "Male" else 0
    is_smoker = 1 if smoker == "Yes" else 0
    is_large_party = 1 if size >= 5 else 0

    bill_per_head = total_bill / size if size > 0 else 0.0
    bill_squared = total_bill ** 2
    log_bill = np.log1p(total_bill) if total_bill >= 0 else 0.0

    day_Sat = 1 if day == "Sat" else 0
    day_Sun = 1 if day == "Sun" else 0
    day_Thur = 1 if day == "Thur" else 0
    time_Lunch = 1 if time == "Lunch" else 0

    size_te = map_te(te_means, "size_te", size)
    weekend_te = map_te(te_means, "weekend_te", is_weekend)
    dinner_te = map_te(te_means, "dinner_te", is_dinner)
    sat_te = map_te(te_means, "sat_te", day_Sat)
    sun_te = map_te(te_means, "sun_te", day_Sun)
    thur_te = map_te(te_means, "thur_te", day_Thur)
    lunch_te = map_te(te_means, "lunch_te", time_Lunch)
    size_dinner_te = map_te(te_means, "size_dinner_te", (size, is_dinner))
    size_weekend_te = map_te(te_means, "size_weekend_te", (size, is_weekend))

    return {
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
        "size_te": size_te,
        "weekend_te": weekend_te,
        "dinner_te": dinner_te,
        "sat_te": sat_te,
        "sun_te": sun_te,
        "thur_te": thur_te,
        "lunch_te": lunch_te,
        "size_dinner_te": size_dinner_te,
        "size_weekend_te": size_weekend_te,
    }


st.title("🍽️ Restaurant Tip Amount Predictor")
st.write("Enter bill and party details to predict the tip amount.")

uploaded = st.sidebar.file_uploader(
    "Upload tip_model_bundle.pkl (optional)",
    type=["pkl"],
)

bundle = load_bundle(uploaded)

model = bundle["model"]
scaler = bundle["scaler"]
feature_names = list(bundle["feature_names"])
te_means = bundle["te_means"]

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        total_bill = st.number_input(
            "Total bill",
            min_value=0.0,
            max_value=1000.0,
            value=50.0,
            step=1.0,
        )
        size = st.slider("Party size", min_value=1, max_value=10, value=2)
        sex = st.selectbox("Sex", ["Male", "Female"])

    with col2:
        smoker = st.selectbox("Smoker", ["No", "Yes"])
        day = st.selectbox("Day", ["Thur", "Fri", "Sat", "Sun"])
        time = st.selectbox("Time", ["Lunch", "Dinner"])

    submitted = st.form_submit_button("Predict tip")

if submitted:
    features = build_features(total_bill, size, sex, smoker, day, time, te_means)
    X = pd.DataFrame([features], columns=feature_names)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]

    st.success(f"Predicted tip: **${pred:.2f}**")
    st.caption("Model: best_tip_regressor.pkl (StackingRegressor)")
