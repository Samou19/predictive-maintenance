
from flask import Flask, request, jsonify
import numpy as np
from model import load_model, predict_cycle
from preprocessing import load_and_preprocess, preprocess_data

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # ✅ Permet d'afficher les accents correctement

model = load_model("model.pkl")
X, y = load_and_preprocess("PS2.txt", "FS1.txt", "profile.txt")
X_scaled = preprocess_data(X)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    cycle_num = int(data["cycle"])
    
    if cycle_num < 0 or cycle_num >= X_scaled.shape[0]:
        return jsonify({"erreur": "Numero de cycle invalide, veuillez entrer un numero de cycle valide"}), 400
    
    features = X_scaled[cycle_num]
    prediction = predict_cycle(model, features)
    
    return jsonify({
        "numero_cycle": cycle_num,
        "prediction": "1 = Optimal" if prediction == 1 else "0 = Non optimal"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
