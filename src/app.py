
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

def load_and_preprocess(ps2_file, fs1_file, profile_file):
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"

    ps2_path = data_dir / ps2_file
    fs1_path = data_dir / fs1_file
    profile_path = data_dir / profile_file

    for path in [ps2_path, fs1_path, profile_path]:
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

    ps2 = pd.read_csv(ps2_path, sep="\t", header=None)
    fs1 = pd.read_csv(fs1_path, sep="\t", header=None)
    profile = pd.read_csv(profile_path, sep="\t", header=None)
    profile.columns = ["cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure", "stable_flag"]

    ps2_features = ps2.apply([np.mean, np.std, np.min, np.max], axis=1)
    fs1_features = fs1.apply([np.mean, np.std, np.min, np.max], axis=1)

    ps2_features.columns = ["ps2_mean", "ps2_std", "ps2_min", "ps2_max"]
    fs1_features.columns = ["fs1_mean", "fs1_std", "fs1_min", "fs1_max"]

    features = pd.concat([ps2_features, fs1_features, profile], axis=1)
    features["target"] = (features["valve_condition"] == 100).astype(int)

    features["cooler_condition"] = pd.Categorical(features["cooler_condition"], categories=[3, 20, 100], ordered=True)
    features["pump_leakage"] = pd.Categorical(features["pump_leakage"], categories=[0, 1, 2], ordered=True)
    features["accumulator_pressure"] = pd.Categorical(features["accumulator_pressure"], categories=[90, 100, 115, 130], ordered=True)
    features["stable_flag"] = pd.Categorical(features["stable_flag"], categories=[0, 1], ordered=False)

    X = features.drop(columns=["target", "valve_condition", "fs1_min", "ps2_min"])
    y = features["target"]

    return X, y

app = FastAPI(title="API Maintenance Prédictive", description="Prédiction de la condition de la valve", version="1.0")

pipeline = joblib.load("pipeline.pkl")
X, y = load_and_preprocess("PS2.txt", "FS1.txt", "profile.txt")

class CycleRequest(BaseModel):
    cycle: int = Field(..., description="Numéro du cycle à prédire", example=12)

class PredictionOutput(BaseModel):
    Numéro_du_cycle: int = Field(..., description="Numéro du cycle analysé", example=12)
    prédiction: str = Field(..., description="Statut prédit (Optimal, À surveiller, Critique)", example="Optimal")
    probabilité: float = Field(..., description="Probabilité associée à la prédiction", example=0.997)

@app.get("/", summary="Accueil", description="Page d'accueil de l'API")
def home():
    return {"message": "Bienvenue sur l'API de maintenance prédictive"}

@app.post("/predict", response_model=PredictionOutput, summary="Prédiction", description="Prédit l'état de la machine")
def predict(request: CycleRequest) -> PredictionOutput:
    cycle_num = request.cycle

    if cycle_num < 1 or cycle_num > X.shape[0]:
        raise HTTPException(status_code=400, detail=f"Numéro de cycle invalide. Choisissez entre 1 et {X.shape[0]}")

    idx = cycle_num - 1
    cycle_features = pd.DataFrame([X.iloc[idx]], columns=X.columns)

    prediction = int(pipeline.predict(cycle_features)[0])
    proba = float(pipeline.predict_proba(cycle_features)[0][1])

    return PredictionOutput(
        Numéro_du_cycle=cycle_num,
        prédiction="Optimal" if prediction == 1 else "Non optimal",
        probabilité=round(proba, 3)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)