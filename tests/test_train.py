
import numpy as np
import pandas as pd
# from src.preprocessing import preprocess_data
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# from src.model import get_model

def preprocess_data(X, model_type="logistic"):
    # Colonnes ordinales et binaires
    ordinal_cols = ["cooler_condition", "pump_leakage", "accumulator_pressure"]
    binary_cols = ["stable_flag"]

    # Ordre des catégories
    ordinal_categories = [
        [3, 20, 100],  # Cooler
        [0, 1, 2],     # Pump leakage
        [90, 100, 115, 130]  # Accumulator pressure
    ]

    # Colonnes numériques
    numeric_cols = [col for col in X.columns if col not in ordinal_cols + binary_cols]

    # Préprocesseur
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("ord", OrdinalEncoder(categories=ordinal_categories), ordinal_cols),
        ("bin", "passthrough", binary_cols)
    ])

    # Pipeline complet avec modèle choisi
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", get_model(model_type))
    ])

    return pipeline


def get_model(model_type="logistic"):
    """
    Retourne un modèle en fonction du type choisi.
    model_type : 'logistic', 'random_forest', 'xgboost'
    """
    if model_type == "logistic":
        return LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "xgboost":
        return XGBClassifier(eval_metric='logloss', random_state=42)
    else:
        raise ValueError(f"Type de modèle inconnu : {model_type}")


def test_train_pipeline_metrics():
    # Données factices
    X_fake = pd.DataFrame({
        "ps2_mean": np.random.rand(20),
        "ps2_std": np.random.rand(20),
        "ps2_max": np.random.rand(20),
        "fs1_mean": np.random.rand(20),
        "fs1_std": np.random.rand(20),
        "fs1_max": np.random.rand(20),
        "cooler_condition": np.random.choice([3, 20, 100], size=20),
        "pump_leakage": np.random.choice([0, 1, 2], size=20),
        "accumulator_pressure": np.random.choice([90, 100, 115, 130], size=20),
        "stable_flag": np.random.choice([0, 1], size=20)
    })
    y_fake = np.random.randint(0, 2, size=20)

    pipeline = preprocess_data(X_fake, model_type="random_forest")
    pipeline.fit(X_fake, y_fake)

    y_pred = pipeline.predict(X_fake)
    acc = accuracy_score(y_fake, y_pred)

    assert 0 <= acc <= 1
