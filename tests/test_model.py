
import joblib
import numpy as np
import pandas as pd
import joblib
# Tests pour le modèle XGBoost

def test_xgboost_prediction():
   
    MODEL_PATH = "best_model.pkl"
    def load_model():
        return joblib.load(MODEL_PATH)

    best_model = load_model()

      # Données factices
    X = pd.DataFrame({
        "ps2_mean": np.random.rand(20),
        "ps2_std": np.random.rand(20),
        "ps2_max": np.random.rand(20),
        "fs1_mean": np.random.rand(20),
        "fs1_std": np.random.rand(20),
        "fs1_max": np.random.rand(20)
    })

    sample = X.iloc[[0]]  # Une ligne de test
    proba = best_model.predict_proba(sample)[0][1]
    pred = best_model.predict(sample)[0]
    assert 0.0 <= proba <= 1.0
    assert pred in [0, 1]

