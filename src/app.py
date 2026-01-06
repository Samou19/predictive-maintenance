# API FastAPI (main.py)
from fastapi import FastAPI
import pandas as pd

#from src.schemas import CycleInput
# Schéma des données (schemas.py)
from pydantic import BaseModel, Field

class CycleInput(BaseModel):
    ps2_mean: float = Field(..., alias="Pression PS2 moyenne (bar)")
    ps2_std: float = Field(..., alias="Écart-type pression PS2 (bar)")
    ps2_max: float = Field(..., alias="Pression PS2 maximale (bar)")
    fs1_mean: float = Field(..., alias="Débit FS1 moyen (l/min)")
    fs1_std: float = Field(..., alias="Écart-type débit FS1 (l/min)")
    fs1_max: float = Field(..., alias="Débit FS1 maximal (l/min)")

class Config:
    populate_by_name = True  # Permet d'utiliser soit le nom, soit l'alias

class PredictionOutput(BaseModel):
    prediction: int = Field(..., description="La classe prédite", example=1)
    etat_valve: str = Field(..., description="État prédit de la valve (Optimal, Non optimal)", example="Optimal")
    probabilite_prediction: float = Field(..., description="Probabilité associée à la prédiction", example=0.987)


#from src.model import load_model
# Chargement du modèle (model.py)
import joblib

MODEL_PATH = "best_model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)


app = FastAPI(
    title="Predictive Maintenance API",
    description="Prédiction de la condition de la valve",
    version="1.0"
)

model = load_model()

@app.get("/", summary="Accueil", description="Page d'accueil de l'API")
def home():
    return {"message": "Bienvenue sur l'API de maintenance prédictive"}

@app.get("/health", summary="Vérification de l'état de santé de l'API", description="Vérifie si l'API est opérationnelle")
def health_check():
    return {"status": "API is running"}


@app.post("/predict")
def predict(cycle: CycleInput):
    try:
        FEATURES = ["ps2_mean", "ps2_std", "ps2_max", "fs1_mean", "fs1_std", "fs1_max"]
        X = pd.DataFrame([[cycle.model_dump()[f] for f in FEATURES]], columns=FEATURES)

        proba = float(model.predict_proba(X)[0][1])  # Conversion explicite en float
        prediction = int(proba >= 0.5)

        return PredictionOutput(
            prediction=prediction,
            etat_valve="NON OPTIMAL" if prediction == 1 else "OPTIMAL",
            probabilite_prediction=round(proba, 3)
        )
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)