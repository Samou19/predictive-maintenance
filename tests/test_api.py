
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_api_predict():
    payload = {
        "ps2_mean": 12.0,
        "ps2_std": 4.0,
        "ps2_max": 19.0,
        "fs1_mean": 5.0,
        "fs1_std": 8.0,
        "fs1_max": 3.0
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


