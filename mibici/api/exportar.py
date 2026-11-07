"""Exporta el modelo champion de P1 desde el Model Registry a ``models/api/p1``.

Recupera ``models:/mibici-equipo32-p1-demanda@champion`` por nombre y alias,
descarga sus artefactos tal cual están en el registry y escribe
``linaje.json`` con la trazabilidad (versión, run, git_commit, data_version).
El Containerfile copia ese directorio dentro de la imagen, de modo que el
contenedor es autocontenido y reproducible sin depender del servidor de
MLflow en tiempo de ejecución.

Requiere el servidor de tracking arriba (o acceso al backend sqlite):

    python -m mibici.api.exportar
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient
import typer

from mibici.config import MODELS_DIR
from mibici.modeling.registrar import NOMBRES
from mibici.modeling.train import configurar_tracking

app = typer.Typer()


def _localizar_mlmodel(raiz: Path) -> Path:
    """Devuelve el directorio que contiene el archivo MLmodel descargado."""
    for ruta in [raiz, *raiz.rglob("*")]:
        if ruta.is_dir() and (ruta / "MLmodel").exists():
            return ruta
    raise FileNotFoundError(f"No se encontró MLmodel bajo {raiz}")


@app.command()
def main(
    question: str = typer.Option("p1", "--question"),
    alias: str = typer.Option("champion", "--alias"),
):
    configurar_tracking()
    cliente = MlflowClient()
    nombre = NOMBRES[question.lower()]

    version = cliente.get_model_version_by_alias(nombre, alias)
    run = cliente.get_run(version.run_id)
    logger.info(f"Recuperando models:/{nombre}@{alias} (versión {version.version})")

    destino = MODELS_DIR / "api" / question.lower()
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)

    descarga = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{nombre}@{alias}", dst_path=str(destino / "_descarga")
        )
    )
    origen = _localizar_mlmodel(descarga)
    shutil.move(str(origen), str(destino / "modelo"))
    shutil.rmtree(destino / "_descarga", ignore_errors=True)

    linaje = {
        "nombre": nombre,
        "alias": alias,
        "version": str(version.version),
        "run_id": version.run_id,
        "git_commit": run.data.tags.get("git_commit", "?"),
        "data_version": run.data.tags.get("data_version", "?"),
        "test_mae": run.data.metrics.get("test_mae"),
        "exportado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (destino / "linaje.json").write_text(
        json.dumps(linaje, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.success(f"Modelo exportado a {destino} (run {version.run_id[:8]}…)")


if __name__ == "__main__":
    app()
