
# Maintenance Prédictive -  Condition de la Valve

## Contexte
Ce projet vise à mettre en place un système de maintenance prédictive pour des machines industrielles. L’objectif est de prédire si la condition de la valve est optimale (=100%) ou non pour chaque cycle de production, afin d’anticiper les défaillances et optimiser la maintenance.

## Objectif
Prédire si la condition de la valve est optimale (100%) ou non pour chaque cycle.
- Construire un modèle de Machine Learning pour prédire la condition de la valve.
- Utiliser les 2000 premiers cycles pour l’entraînement et le reste pour le test.
- Déployer une API FastAPI permettant de faire une prédiction pour un cycle donné.

## Données
Les données proviennent du dataset UCI :
https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems

Fichiers utilisés :
- PS2.txt : Pression (bar) échantillonnée à 100 Hz
- FS1.txt : Débit volumique (l/min) échantillonné à 10 Hz
- profile.txt : Variables de profil dont la condition de la valve

## Structure
- `data/` : Fichiers sources (FS1, PS2, profile)
- `src/` : code source (prétraitement, modèle, API)
- `tests/` : tests unitaires
- `Dockerfile` : containerisation
- `requirements.txt` : dépendances


## Modèle utilisé: XGBoost
Après avoir testé et comparé les modèles XGBoost, régression logistique et Random Forest, XGBoost offre les meilleures performances. Les résultats obtenus sont présentés ci-dessous.
### Évaluation du modèle XGBoost:
- `Accuracy` : 0.87
- `Precision` : 1.0 
- `Recall` : 0.80
- `F1 Score` : 0.89

## Installation et exécution

```bash
git clone <URL_DU_REPO>
cd <nom_du_projet>

pip install -r requirements.txt

uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```
## Accéder à la documentation interactive :
http://localhost:8000/docs

## Construire une image <NOM-IMAGE>

```bash
docker build -t <NOM-IMAGE> .
docker run -d -p 8000:8000 <NOM-IMAGE>:latest
```
## Exemple de requête API pour le modèle XGBoost
```bash
curl -X 'POST' \
  'https://predictive-maintenance-qy5w.onrender.com/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "Pression PS2 (moyenne)": 23,
  "Pression PS2 (écart type)": 5,
  "Pression PS2 (maximum)": 10,
  "Débit FS1 (moyenne)": 8,
  "Débit FS1 (écart type)": 3.9,
  "Débit FS1 (maximum)": 1.8
}'
```
