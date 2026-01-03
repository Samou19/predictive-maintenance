
# Image de base
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source et le modèle
COPY src/ /app/src/
COPY pipeline.pkl /app/
COPY tests/ /app/tests/

# Copier les fichiers .txt du dossier /data
COPY data/*.txt /app/data/

# Exposer le port pour l'API
EXPOSE 8000


# Commande par défaut : lancer l'API FastAPI
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]

# OK