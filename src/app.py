
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from pathlib import Path
import numpy as np

def get_model(model_type="logistic"):
    if model_type == "logistic":
        return LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "xgboost":
        return XGBClassifier(eval_metric='logloss', random_state=42)
    else:
        raise ValueError(f"Type de modèle inconnu : {model_type}")


def read_window_csv(path, first=None, end=None, **kwargs):
    """
    Lit deux portions spécifiques d'un fichier CSV sans entête :
    - Les 'first' premières lignes
    - Les 'end' dernières lignes
    
    :param path: Chemin du fichier CSV
    :param first: Nombre de premières lignes à lire
    :param end: Nombre de dernières lignes à lire
    :param kwargs: Arguments supplémentaires pour pd.read_csv
    :return: DataFrame concaténé
    """
    dfs = []
    
    # Options pour fichier sans entête
    kwargs.setdefault('header', None)
    
    # Lire les premières lignes
    if first is not None and first > 0:
        df = pd.read_csv(path, nrows=first, **kwargs)
    
    # Lire les dernières lignes
    if end is not None and end > 0:
        # Compter le nombre total de lignes
        total_rows = sum(1 for _ in open(path))
        skip_start = total_rows - end
        df = pd.read_csv(path, skiprows=range(skip_start), **kwargs)
    
    return df

def load_and_preprocess(ps2_file, fs1_file, profile_file, first=2000, end=None):
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
    ps2 = read_window_csv(path=ps2_path, sep="\t", header=None, first=first, end=end)
    fs1 = read_window_csv(path=fs1_path, sep="\t", header=None, first=first, end=end)

    # Charger le fichier profile
    profile = read_window_csv(path=profile_path, sep="\t", header=None, first=first, end=end)
    profile.columns = ["cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure", "stable_flag"]

    # Créer des features agrégées pour PS2 et FS1
    ps2_features = ps2.apply([np.mean, np.std, np.min, np.max], axis=1)
    fs1_features = fs1.apply([np.mean, np.std, np.min, np.max], axis=1)

    # Renommer les colonnes
    ps2_features.columns = ["ps2_mean", "ps2_std", "ps2_min", "ps2_max"]
    fs1_features.columns = ["fs1_mean", "fs1_std", "fs1_min", "fs1_max"]

    # Fusionner tout
    features = pd.concat([ps2_features, fs1_features, profile], axis=1)

    # Ajouter la cible (valve optimale ou non = 1 si valve_condition != 100)
    features["target"] = (features["valve_condition"] != 100).astype(int)

    # Prendre les 2000 premiers cycles
    #features = features.head(2000)

    # Séparer X et y
    # Colonnes ordinales
    #features["cooler_condition"] = pd.Categorical(features["cooler_condition"], categories=[3, 20, 100], ordered=True)
    #features["pump_leakage"] = pd.Categorical(features["pump_leakage"], categories=[0, 1, 2], ordered=True)
    #features["accumulator_pressure"] = pd.Categorical(features["accumulator_pressure"], categories=[90, 100, 115, 130], ordered=True)
    features["stable_flag"] = pd.Categorical(features["stable_flag"], categories=[0, 1], ordered=False)

    X = features.drop(columns=["target", "valve_condition", "fs1_min", "ps2_min",
                               "cooler_condition", "pump_leakage", "accumulator_pressure",
                               "stable_flag"])  # On garde les colonnes utiles
    y = features["target"]

    return X, y

app = FastAPI(title="API Maintenance Prédictive", description="Prédiction de la condition de la valve", version="1.0")

pipeline = joblib.load("pipeline.pkl")
X, y = load_and_preprocess("PS2.txt", "FS1.txt", "profile.txt", first=2000)

class CycleRequest(BaseModel):
    cycle: int = Field(..., description="Numéro du cycle à prédire", example=12)


class PredictionOutput(BaseModel):
    numero_cycle_analyse: int = Field(..., description="Numéro du cycle analysé", example=13)
    etat_valve: str = Field(..., description="État prédit de la valve (Optimal, À surveiller, Critique)", example="Optimal")
    probabilite_prediction: float = Field(..., description="Probabilité associée à la prédiction", example=0.987)

@app.get("/", summary="Accueil", description="Page d'accueil de l'API")
def home():
    return {"message": "Bienvenue sur l'API de maintenance prédictive"}

@app.post("/predict", response_model=PredictionOutput, summary="Prédiction", description="Prédit l'état de la valve pour un cycle donné")
def predict(request: CycleRequest) -> PredictionOutput:
    cycle_num = request.cycle

    if cycle_num < 1 or cycle_num > X.shape[0]:
        raise HTTPException(status_code=400, detail=f"Numéro de cycle invalide. Choisissez entre 1 et {X.shape[0]}")

    idx = cycle_num - 1
    cycle_features = pd.DataFrame([X.iloc[idx]], columns=X.columns)

    prediction = int(pipeline.predict(cycle_features)[0])
    proba = float(pipeline.predict_proba(cycle_features)[0][1])

    return PredictionOutput(
        numero_cycle_analyse=cycle_num,
        etat_valve="Non optimal" if prediction == 1 else "Optimal",
        probabilite_prediction=round(proba, 3)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)