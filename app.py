"""
🍽️ Restaurant Tip Amount Predictor
====================================
A Machine Learning web app that predicts the tip amount based on
restaurant bill, party size, and customer details.

Model: Stacking Regressor (CV R² ≈ 0.47)
Author: Khanniazi
"""
import os

print("Current folder:", os.getcwd())
print("Files:", os.listdir())

if os.path.exists("saved_model"):
    print("saved_model files:", os.listdir("saved_model"))
else:
    print("saved_model folder NOT FOUND")


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="🍽️ Tip Amount Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS for Attractive UI ----------
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF6B6B, #FFA500, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    /* Prediction card */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 20px 0;
    }
    .prediction-label {
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: 500;
        margin-bottom: 10px;
        opacity: 0.9;
    }
    .prediction-value {
        color: #ffffff;
        font-size: 4rem;
        font-weight: 900;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    }
    /* Info box */
    .info-box {
        background: #f0f2f6;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 15px 0;
    }
    /* Metric styling */
    .metric-box {
        background: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    /* Footer */
    .footer {
        text-align: center;
        color: #999;
        padding: 20px 0;
        font-size: 0.9rem;
        border-top: 1px solid #eee;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)


# ---------- Load Model Artifacts ----------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load all saved model artifacts."""
    base = "saved_model"
    model = joblib.load(os.path.join(base, "best_tip_regressor.pkl"))
    scaler = joblib.load(os.path.join(base, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(base, "feature_names.pkl"))
    te_means = joblib.load(os.path.join(base, "te_means.pkl"))
    return model, scaler, feature_names, te_means


try:
    model, scaler, feature_names, te_means = load_artifacts()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"⚠️ Model load error: {e}")
    st.stop()


# ---------- Prediction Function ----------
def predict_tip(total_bill, size, sex, smoker, day, time):
    """Build feature row, scale it, and predict tip."""
    row = {
        'total_bill':     total_bill,
        'size':           size,
        'is_weekend':     1 if day in ['Sat', 'Sun'] else 0,
        'is_dinner':      1 if time == 'Dinner' else 0,
        'is_male':        1 if sex == 'Male' else 0,
        'is_smoker':      1 if smoker == 'Yes' else 0,
        'is_large_party': 1 if size >= 4 else 0,
        'bill_per_head':  total_bill / size,
        'bill_squared':   total_bill ** 2,
        'log_bill':       np.log1p(total_bill),
        'day_Sat':        1 if day == 'Sat' else 0,
        'day_Sun':        1 if day == 'Sun' else 0,
        'day_Thur':       1 if day == 'Thur' else 0,
        'time_Lunch':     1 if time == 'Lunch' else 0,
    }

    # Fill target-encoded features with stored means
    for c in feature_names:
        if c not in row:
            row[c] = te_means.get(c, 2.99)

    X_new = pd.DataFrame([row]).astype(float)
    X_new = X_new[feature_names]

    X_sc = scaler.transform(X_new)
    pred = model.predict(X_sc)[0]
    return round(float(pred), 2)


# ================================================================
# HEADER
# ================================================================
st.markdown('<h1 class="main-header">🍽️ Restaurant Tip Predictor</h1>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict the tip amount using Machine Learning '
            '— Stacking Regressor (CV R² ≈ 0.47)</p>',
            unsafe_allow_html=True)


# ================================================================
# SIDEBAR — Input Controls
# ================================================================
with st.sidebar:
    st.markdown("## 🎛️ Enter Bill Details")
    st.markdown("---")

    total_bill = st.number_input(
        "💵 Total Bill ($)",
        min_value=1.0, max_value=200.0, value=20.0, step=0.5,
        help="Enter the total bill amount in dollars"
    )

    size = st.slider(
        "👥 Party Size",
        min_value=1, max_value=10, value=2,
        help="Number of people in the party"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        sex = st.radio("👤 Sex", ["Male", "Female"], index=0)
    with col_b:
        smoker = st.radio("🚬 Smoker", ["No", "Yes"], index=0)

    day = st.selectbox(
        "📅 Day of Week",
        ["Sun", "Sat", "Thur", "Fri"], index=0
    )

    time = st.radio(
        "🕐 Time",
        ["Dinner", "Lunch"], index=0, horizontal=True
    )

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Tip", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    - **Algorithm:** Stacking Regressor
    - **Base Models:** Ridge, RF, XGBoost, GB
    - **Test R²:** 0.33
    - **CV R²:** 0.47
    - **Dataset:** 233 restaurant bills
    """)


# ================================================================
# MAIN AREA
# ================================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📋 Your Input Summary")

    summary_df = pd.DataFrame({
        "Feature": ["💵 Total Bill", "👥 Party Size", "👤 Sex",
                    "🚬 Smoker", "📅 Day", "🕐 Time"],
        "Value": [f"${total_bill:.2f}", size, sex, smoker, day, time]
    })
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    # Extra insights
    bill_per_head = total_bill / size
    st.markdown(f"""
    <div class="info-box">
        <b>💡 Insights:</b><br>
        • Bill per head: <b>${bill_per_head:.2f}</b><br>
        • Weekend: <b>{"Yes" if day in ["Sat","Sun"] else "No"}</b><br>
        • Large party (≥4): <b>{"Yes" if size >= 4 else "No"}</b>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### 🎯 Prediction Result")

    if predict_btn:
        with st.spinner("Calculating..."):
            tip = predict_tip(total_bill, size, sex, smoker, day, time)
            tip_pct = (tip / total_bill) * 100
            total_with_tip = total_bill + tip

        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label">Predicted Tip Amount</div>
            <div class="prediction-value">${tip}</div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Tip %", f"{tip_pct:.1f}%")
        m2.metric("Total + Tip", f"${total_with_tip:.2f}")
        m3.metric("Per Person", f"${total_with_tip/size:.2f}")

    else:
        st.info("👈 Enter your details and click **🔮 Predict Tip** to get the prediction.")


# ================================================================
# EXAMPLES SECTION
# ================================================================
st.markdown("---")
st.markdown("### 🧪 Try These Examples")

example_df = pd.DataFrame({
    "Scenario": ["☕ Casual Lunch", "🍷 Weekend Dinner", "🎉 Big Group",
                 "🍔 Quick Bite", "💰 Splurge Night"],
    "Bill": ["$15", "$35", "$50", "$10", "$70"],
    "Size": [2, 4, 6, 1, 5],
    "Day": ["Thur", "Sat", "Sat", "Fri", "Sun"],
    "Time": ["Lunch", "Dinner", "Dinner", "Lunch", "Dinner"],
})
st.dataframe(example_df, hide_index=True, use_container_width=True)


# ================================================================
# FOOTER
# ================================================================
st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit & Scikit-learn<br>
    © 2026 — Restaurant Tip Predictor
</div>
""", unsafe_allow_html=True)
