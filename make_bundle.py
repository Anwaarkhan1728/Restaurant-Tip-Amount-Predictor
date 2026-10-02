import joblib

bundle = {
    "model": joblib.load("best_tip_regressor.pkl"),
    "scaler": joblib.load("scaler.pkl"),
    "feature_names": joblib.load("feature_names.pkl"),
    "te_means": joblib.load("te_means.pkl"),
}

joblib.dump(bundle, "tip_model_bundle.pkl")
print("Created tip_model_bundle.pkl")
