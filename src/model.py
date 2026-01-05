# Chargement du modèle (model.py)
import joblib

MODEL_PATH = "best_model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)
