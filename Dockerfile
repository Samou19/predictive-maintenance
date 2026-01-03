FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY pipeline.pkl /app/
COPY tests/ /app/tests/
COPY data/*.txt /app/data/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
