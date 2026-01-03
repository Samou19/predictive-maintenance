
import numpy as np
import pandas as pd
import pytest
#from src.preprocessing import preprocess_data
#from src.model import get_model

from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

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


def test_pipeline_creation():
    # Données factices avec colonnes attendues
    X_fake = pd.DataFrame({
        "ps2_mean": np.random.rand(5),
        "ps2_std": np.random.rand(5),
        "ps2_max": np.random.rand(5),
        "fs1_mean": np.random.rand(5),
        "fs1_std": np.random.rand(5),
        "fs1_max": np.random.rand(5),
        "cooler_condition": np.random.choice([3, 20, 100], size=5),
        "pump_leakage": np.random.choice([0, 1, 2], size=5),
        "accumulator_pressure": np.random.choice([90, 100, 115, 130], size=5),
        "stable_flag": np.random.choice([0, 1], size=5)
    })
    y_fake = np.array([0, 1, 0, 1, 0])

    pipeline = preprocess_data(X_fake, model_type="logistic")
    assert pipeline is not None

    # Entraînement et prédiction
    pipeline.fit(X_fake, y_fake)
    pred = pipeline.predict(X_fake.iloc[[0]])[0]
    assert pred in [0, 1]

def test_pipeline_with_empty_data():
    X_empty = pd.DataFrame(columns=[
        "ps2_mean", "ps2_std", "ps2_max",
        "fs1_mean", "fs1_std", "fs1_max",
        "cooler_condition", "pump_leakage",
        "accumulator_pressure", "stable_flag"
    ])
    y_empty = pd.Series(dtype=int)

    pipeline = preprocess_data(X_empty, model_type="logistic")
    with pytest.raises(ValueError):
        pipeline.fit(X_empty, y_empty)
