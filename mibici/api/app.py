"""API REST del modelo de demanda de MiBici (P1) con FastAPI.

Sirve el modelo champion de ``models:/mibici-equipo32-p1-demanda`` empaquetado
en la imagen por ``mibici/api/exportar.py`` (estrategia documentada en el
README). El directorio del modelo se toma de la variable de entorno
``MODELO_DIR`` (por defecto ``/app/modelo``, la ruta dentro del contenedor).

Ejecución local (fuera del contenedor, tras exportar el modelo):

    MODELO_DIR=models/api/p1 uvicorn mibici.api.app:app --port 8000
"""

import json
import os
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI

from mibici.api.esquemas import Linaje, ObservacionDemanda, Prediccion
from mibici.features import P1_FEATURES

MODELO_DIR = Path(os.getenv("MODELO_DIR", "/app/modelo"))

# El modelo se carga una sola vez al arrancar el proceso, no por petición.
_modelo = mlflow.sklearn.load_model(str(MODELO_DIR / "modelo"))
_linaje = Linaje(**json.loads((MODELO_DIR / "linaje.json").read_text(encoding="utf-8")))

app = FastAPI(
    title="MiBici — API de demanda (P1)",
    description=(
        "Predice los viajes que iniciarán en una estación de MiBici al día "
        "siguiente, con el modelo champion del Model Registry del Equipo 32 "
        "(TC5061, repo individual de Rolando Calderón)."
    ),
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    """Prueba de vida del servicio y del modelo cargado."""
    return {"status": "ok", "modelo_cargado": _modelo is not None}


@app.get("/info", response_model=Linaje)
def info() -> Linaje:
    """Linaje del modelo servido: registry, versión, run, commit y datos."""
    return _linaje


@app.post("/predict", response_model=Prediccion)
def predecir(obs: ObservacionDemanda) -> Prediccion:
    """Predice la demanda del día siguiente para una estación.

    Pydantic valida tipos, rangos y coherencia (p. ej. ``fin_semana`` vs
    ``dia_semana``) antes de que la observación llegue al modelo; una entrada
    inválida responde 422 con el detalle del error.
    """
    entrada = pd.DataFrame([obs.model_dump()])[P1_FEATURES].astype("float64")
    prediccion = float(_modelo.predict(entrada)[0])
    return Prediccion(
        viajes_predichos=round(prediccion, 2),
        modelo=_linaje.nombre,
        version=_linaje.version,
    )
