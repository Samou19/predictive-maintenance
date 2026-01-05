# API FastAPI (main.py)
from fastapi import FastAPI
import pandas as pd

#from src.schemas import CycleInput
# Schéma des données (schemas.py)
from pydantic import BaseModel

class CycleInput(BaseModel):
    ps2_mean: float
    ps2_std: float
    ps2_max: float
    fs1_mean: float
    fs1_std: float
    fs1_max: float

#from src.model import load_model
# Chargement du modèle (model.py)
import joblib

MODEL_PATH = "./src/best_model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)


app = FastAPI(
    title="Predictive Maintenance API",
    description="Prédiction de la condition de la valve",
    version="1.0"
)

model = load_model()

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict(cycle: CycleInput):
    X = pd.DataFrame([cycle.dict()])

    proba = model.predict_proba(X)[0][1]
    prediction = int(proba >= 0.5)

    return {
        "prediction": prediction,
        "label": "NON_OPTIMAL" if prediction == 1 else "OPTIMAL",
        "failure_probability": round(proba, 3)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)