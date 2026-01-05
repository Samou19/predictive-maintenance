
from preprocessing import read_window_csv, load_and_preprocess, preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import joblib
import pandas as pd

# Chargement des données
X, y = load_and_preprocess("PS2.txt", "FS1.txt", "profile.txt", first=2000)
print(f"Données chargées : {X.shape[0]} cycles, {X.shape[1]} features,\n cible équilibrée : {y.value_counts().round(2)}, \n Liste des features : {X.columns.tolist()}")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Choix du modèle : 'logistic', 'random_forest', 'xgboost'
pipeline = preprocess_data(X_train, model_type="random_forest")

# Entraînement
pipeline.fit(X_train, y_train)


# Récupérer le modèle Random Forest à l'intérieur du pipeline
rf_model = pipeline.named_steps['model']  # 'model' est le nom donné dans votre Pipeline

# Récupérer les importances
importances = rf_model.feature_importances_

# Récupérer les noms des features après transformation
feature_names = pipeline.named_steps['preprocess'].get_feature_names_out()

# Créer un DataFrame pour afficher
import pandas as pd
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)


print(importance_df)

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

print("\nÉvaluation du modèle sur les 2000 premiers cycles:")
print(df_metrics)

################################################################    
# Evaluation sur 205 derniers cycles
################################################################    
# Chargement des données
X_test_final, y_test_final = load_and_preprocess("PS2.txt", "FS1.txt", "profile.txt", end=205)
print(f"Données chargées : {X_test_final.shape[0]} cycles, {X_test_final.shape[1]} features,\n cible équilibrée : {y_test_final.value_counts().round(2)}")


# Évaluation
y_pred_final = pipeline.predict(X_test_final)
y_proba_final = pipeline.predict_proba(X_test_final)[:, 1]
# Calcul des métriques
metrics = {
    "Accuracy": accuracy_score(y_test_final, y_pred_final),
    "F1 Score": f1_score(y_test_final, y_pred_final),
    "Precision": precision_score(y_test_final, y_pred_final),
    "Recall": recall_score(y_test_final, y_pred_final),
    "ROC AUC": roc_auc_score(y_test_final, y_proba_final)
}

# Conversion en DataFrame avec arrondi à 3 décimales
df_metrics = pd.DataFrame([metrics]).round(3)

print("\nÉvaluation du modèle sur les 205 derniers cycles:")
print(df_metrics)
