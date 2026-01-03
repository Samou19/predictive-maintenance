
FROM python:3.10-slim

WORKDIR /app

# Installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code et les fichiers nécessaires
COPY src/ /app/src/
COPY pipeline.pkl /app/
COPY tests/ /app/tests/
COPY data/*.txt /app/data/

# Exposer le port
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
