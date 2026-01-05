import os
import pandas as pd
import pathlib

# Chemins des fichiers
ps2_file = "PS2.txt"
fs1_file = "FS1.txt"
profile_file = "profile.txt"

# Trouver le dossier racine (parent de src)
base_dir = pathlib.Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"

# Construire les chemins complets
ps2_path = data_dir / ps2_file
fs1_path = data_dir / fs1_file
profile_path = data_dir / profile_file

# Vérifier existence
for path in [ps2_path, fs1_path, profile_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

# Charger les fichiers capteurs
ps2 = pd.read_csv(ps2_path, sep="\t", header=None)
fs1 = pd.read_csv(fs1_path, sep="\t", header=None)

# Charger le fichier profile
profile = pd.read_csv(profile_path, sep="\t", header=None)
profile.columns = ["cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure", "stable_flag"]

# 1100% = optimal (0) vs Non optimal (1)
y = (profile["valve_condition"] != 100).astype(int)

import numpy as np

def extract_features(df, df_name):
    return pd.DataFrame({
        f"{df_name}_mean": df.mean(axis=1),
        f"{df_name}_std": df.std(axis=1),
        f"{df_name}_min": df.min(axis=1),
        f"{df_name}_max": df.max(axis=1)
    })

X_ps2 = extract_features(ps2, "ps2")
X_fs1 = extract_features(fs1, "fs1")

XX = pd.concat([X_ps2, X_fs1], axis=1)

X = XX.drop(columns=[ "ps2_min","fs1_min"])

# les 2000 premiers cycles
X_train = X.iloc[:2000] 
y_train = y.iloc[:2000]

# les 205 derniers cycles
X_test = X.iloc[2000:] 
y_test = y.iloc[2000:]

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ]),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42)
}

from xgboost import XGBClassifier

models["XGBoost"] = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)


from sklearn.model_selection import cross_validate

results = {}

for name, model in models.items():
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        scoring=["accuracy", "f1", "recall", "precision"]
    )
    results[name] = {
        "accuracy": scores["test_accuracy"].mean(),
        "f1-score": scores["test_f1"].mean(),
        "recall": scores["test_recall"].mean(),
        "precision": scores["test_precision"].mean()
    }

pd.DataFrame(results).T

best_model_name = "XGBoost"
best_model = models[best_model_name]

best_model.fit(X_train, y_train)

# Sauvegarde du modèle (en amont)
import joblib

joblib.dump(best_model, "best_model.pkl")
