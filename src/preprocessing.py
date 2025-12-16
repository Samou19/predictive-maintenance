
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
#from src.model import get_model  # Import de la fonction

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

def load_and_preprocess(ps2_file, fs1_file, profile_file):
    # Trouver le dossier racine (parent de src)
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"

    # Construire les chemins complets
    ps2_path = data_dir / ps2_file
    fs1_path = data_dir / fs1_file
    profile_path = data_dir / profile_file

    # Vérifier existence
    for path in [ps2_path, fs1_path, profile_path]:
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

    # Charger les fichiers capteurs
    ps2 = pd.read_csv(ps2_path, sep="\t", header=None)
    fs1 = pd.read_csv(fs1_path, sep="\t", header=None)

    # Charger le fichier profile
    profile = pd.read_csv(profile_path, sep="\t", header=None)
    profile.columns = ["cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure", "stable_flag"]

    # Créer des features agrégées pour PS2 et FS1
    ps2_features = ps2.apply([np.mean, np.std, np.min, np.max], axis=1)
    fs1_features = fs1.apply([np.mean, np.std, np.min, np.max], axis=1)

    # Renommer les colonnes
    ps2_features.columns = ["ps2_mean", "ps2_std", "ps2_min", "ps2_max"]
    fs1_features.columns = ["fs1_mean", "fs1_std", "fs1_min", "fs1_max"]

    # Fusionner tout
    features = pd.concat([ps2_features, fs1_features, profile], axis=1)

    # Ajouter la cible (valve optimale ou non)
    features["target"] = (features["valve_condition"] == 100).astype(int)

    # Séparer X et y
    # Colonnes ordinales
    features["cooler_condition"] = pd.Categorical(features["cooler_condition"], categories=[3, 20, 100], ordered=True)
    features["pump_leakage"] = pd.Categorical(features["pump_leakage"], categories=[0, 1, 2], ordered=True)
    features["accumulator_pressure"] = pd.Categorical(features["accumulator_pressure"], categories=[90, 100, 115, 130], ordered=True)
    features["stable_flag"] = pd.Categorical(features["stable_flag"], categories=[0, 1], ordered=False)

    X = features.drop(columns=["target", "valve_condition", "fs1_min", "ps2_min"])  # On garde les colonnes utiles
    y = features["target"]

    return X, y

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
