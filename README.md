# 🍽️ Restaurant Tip Amount Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge)

A Machine Learning web application that predicts the **tip amount** for a restaurant bill based on bill total, party size, and customer details.

[🚀 **Try the Live App**](https://your-streamlit-app-url.streamlit.app)

</div>

---

## 📌 Overview

This project uses a **Stacking Regressor ensemble** (Ridge + Random Forest + XGBoost + Gradient Boosting) to predict tip amounts. The model was trained on the classic restaurant tips dataset.

### 🎯 Model Performance

| Metric | Score |
|--------|-------|
| **Test R²** | 0.33 |
| **CV R²** | **0.47** ⭐ |
| **MAE** | $0.59 |
| **RMSE** | $0.74 |

---

## ✨ Features

- 🎨 **Beautiful gradient UI** with custom CSS styling
- 📊 **Real-time predictions** from interactive sidebar controls
- 💡 **Insight panel** showing bill-per-head and party details
- 🧪 **Example scenarios** for quick testing
- ⚡ **Cached model loading** for fast inference
- 📱 **Fully responsive** layout

---

## 🧠 Machine Learning Pipeline

1. **Data Cleaning** — Removed duplicates, invalid entries, extreme outliers (k=3.0 IQR)
2. **Feature Engineering** — Created 14 features: binary flags, ratios, non-linear transforms
3. **K-Fold Target Encoding** — Captured group patterns (size, day, time) with 10-fold CV
4. **Train/Test Split** — 80/20 split with StandardScaler
5. **Hyperparameter Tuning** — RandomizedSearchCV across 4 base models
6. **Stacking Ensemble** — Ridge meta-learner on top of tuned base models

---

## 📂 Project Structure
