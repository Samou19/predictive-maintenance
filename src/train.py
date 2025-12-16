
from preprocessing import load_and_preprocess, preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import joblib
import pandas as pd

# Chargement des données
X, y = load_and_preprocess("PS2.txt", "FS1.txt", "profile.txt")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2000, test_size=205, stratify=y, random_state=42)

# Choix du modèle : 'logistic', 'random_forest', 'xgboost'
pipeline = preprocess_data(X_train, model_type="xgboost")

# Entraînement
pipeline.fit(X_train, y_train)

# Sauvegarde
joblib.dump(pipeline, "pipeline.pkl")

# Évaluation
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]


# Calcul des métriques
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "ROC AUC": roc_auc_score(y_test, y_proba)
}

# Conversion en DataFrame avec arrondi à 3 décimales
df_metrics = pd.DataFrame([metrics]).round(3)

print("\nÉvaluation du modèle :")
print(df_metrics)
