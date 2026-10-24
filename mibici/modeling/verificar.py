"""Punto de entrada ``verify``: comprueba que el modelo en producción carga por alias.

Carga ``models:/<nombre>@champion`` y reporta clase, versión, corrida de origen
y linaje. Es la verificación que pide el paso 8 de la actividad.

    mlflow run . -e verify -P question=p1 --env-manager local
"""

import mlflow.sklearn
import typer
from loguru import logger
from mlflow.tracking import MlflowClient

from mibici.modeling.registrar import NOMBRES
from mibici.modeling.train import configurar_tracking

app = typer.Typer()


@app.command()
def main(question: str = typer.Option("p1", "--question")):
    configurar_tracking()
    cliente = MlflowClient()
    nombre = NOMBRES[question.lower()]
    uri = f"models:/{nombre}@champion"

    modelo = mlflow.sklearn.load_model(uri)
    version = cliente.get_model_version_by_alias(nombre, "champion")
    logger.success(f"Cargado {uri}")
    logger.info(f"Clase: {type(modelo).__name__}")
    logger.info(f"Versión: {version.version} | run de origen: {version.run_id}")
    logger.info(f"Descripción: {version.description}")


if __name__ == "__main__":
    app()
