# Containerfile (estándar OCI: funciona igual con Podman o Docker)
# API REST del modelo de demanda de MiBici (P1) — TC5061 Equipo 32 / Rolando Calderón.
#
#   1. python -m mibici.api.exportar          # exporta el champion del registry
#   2. podman build -t mibici-api .
#   3. podman run --rm -p 8000:8000 mibici-api

FROM docker.io/library/python:3.14-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELO_DIR=/app/modelo

# Dependencias primero: esta capa se cachea mientras no cambie el archivo.
COPY requirements_api.txt ./
RUN pip install --no-cache-dir -r requirements_api.txt

# Solo el código que la API necesita (config y features son stdlib/pandas).
COPY mibici/__init__.py mibici/config.py mibici/features.py mibici/
COPY mibici/api/ mibici/api/

# Modelo champion exportado del Model Registry (paso 1) con su linaje.
COPY models/api/p1/ /app/modelo/

# El servicio no necesita privilegios de root.
RUN useradd --create-home apiuser
USER apiuser

EXPOSE 8000
CMD ["uvicorn", "mibici.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
