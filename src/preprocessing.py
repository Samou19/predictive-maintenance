
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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
    X = features.drop(columns=["cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure", "stable_flag", "target"])
    y = features["target"]

    return X, y


def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

