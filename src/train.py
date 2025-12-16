
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from preprocessing import preprocess_data
from model import train_model

# Simulation de données (à remplacer par vos vraies données)
# np.random.seed(42)
# X = np.random.rand(2205, 5)  # 2205 cycles, 5 features
# y = np.random.randint(0, 2, size=2205)  # 0 = non optimal, 1 = optimal
# Chargement des données réelles

from preprocessing import load_and_preprocess

X, y = load_and_preprocess("PS2.txt", "FS1.txt", "profile.txt")

print("Données chargées :", X.shape, y.shape)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2000, test_size=205, stratify=y, random_state=42)

# Prétraitement
X_train_scaled = preprocess_data(X_train)
X_test_scaled = preprocess_data(X_test)

# Entraînement
model = train_model(X_train_scaled, y_train)

# Évaluation
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("Évaluation du modèle :")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
