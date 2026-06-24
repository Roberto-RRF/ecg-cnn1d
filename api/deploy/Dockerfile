FROM python:3.12-slim

WORKDIR /app

# libgomp1 lo requiere PyTorch en runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Rutas dentro del contenedor y carpeta escribible para matplotlib
ENV ECG_DATA_DIR=/app/data \
    ECG_MODELS_DIR=/app/models \
    ECG_ARTIFACTS=/app/artifacts.json \
    MPLCONFIGDIR=/tmp/mpl

# Hugging Face Spaces espera el puerto 7860
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
