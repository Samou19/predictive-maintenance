
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_api_predict():
    payload = {
        "Pression PS2 moyenne (bar)": 12.0,
        "Écart-type pression PS2 (bar)": 4.0,
        "Pression PS2 maximale (bar)": 19.0,
        "Débit FS1 moyen (l/min)": 5.0,
        "Écart-type débit FS1 (l/min)": 8.0,
        "Débit FS1 maximal (l/min)": 3.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Vérification des clés
    assert "prediction" in data
    assert "etat_valve" in data
    assert "probabilite_prediction" in data

    # Vérification des types et valeurs
    assert isinstance(data["prediction"], int)
    assert data["prediction"] in [0, 1] 
    assert isinstance(data["probabilite_prediction"], float)
    assert data["etat_valve"] in ["OPTIMAL", "NON OPTIMAL"]


