"""Punto de entrada ``register``: lleva el mejor modelo de una pregunta al Model Registry.

Selecciona la mejor corrida candidata del experimento según la métrica de la
pregunta, registra su modelo con nombre claro, documenta el linaje (corrida,
commit, versión de datos) en la descripción y asigna los aliases
``champion`` (mejor) y ``challenger`` (segunda mejor), el esquema vigente de
ciclo de vida de MLflow.

    mlflow run . -e register -P question=p1 --env-manager local
"""

import mlflow
import typer
from loguru import logger
from mlflow.tracking import MlflowClient

from mibici.config import TEAM_ID
from mibici.modeling.train import configurar_tracking

app = typer.Typer()

NOMBRES = {
    "p1": f"mibici-{TEAM_ID}-p1-demanda",
    "p2": f"mibici-{TEAM_ID}-p2-riesgo",
    "p3": f"mibici-{TEAM_ID}-p3-grupos",
}
SELECCION = {
    "p1": ("metrics.test_mae", "ASC"),
    "p2": ("metrics.test_f1", "DESC"),
    "p3": ("metrics.test_silhouette", "DESC"),
}
DESCRIPCION = {
    "p1": "Demanda de viajes por estación y día (regresión). Métrica de selección: test_mae.",
    "p2": "Riesgo de quedarse sin bicicletas o anclajes a 2 horas (clasificación). Métrica: test_f1.",
    "p3": "Grupos de estaciones por patrón de uso (clustering). Métrica: test_silhouette.",
}


@app.command()
def main(question: str = typer.Option("p1", "--question")):
    configurar_tracking()
    cliente = MlflowClient()
    question = question.lower()
    nombre = NOMBRES[question]
    metrica, orden = SELECCION[question]

    exp = mlflow.get_experiment_by_name(f"mibici-{TEAM_ID}")
    corridas = cliente.search_runs(
        [exp.experiment_id],
        filter_string=(
            f"tags.question = '{question.upper()}' and tags.stage_of_work = 'candidato' "
            "and attributes.status = 'FINISHED'"
        ),
        order_by=[f"{metrica} {orden}"],
        max_results=10,
    )
    corridas = [c for c in corridas if c.data.metrics.get(metrica.split(".")[1]) is not None]
    if not corridas:
        logger.error(f"No hay corridas candidatas con {metrica} para {question}")
        raise typer.Exit(1)

    def registrar(run, alias: str):
        mv = mlflow.register_model(f"runs:/{run.info.run_id}/modelo", nombre)
        valor = run.data.metrics[metrica.split(".")[1]]
        cliente.update_model_version(
            name=nombre,
            version=mv.version,
            description=(
                f"{DESCRIPCION[question]} | run: {run.info.run_name} ({run.info.run_id}) | "
                f"{metrica.split('.')[1]}={valor:.4f} | git_commit: "
                f"{run.data.tags.get('git_commit', '?')[:12]} | data_version: "
                f"{run.data.tags.get('data_version', '?')} | experimento: mibici-{TEAM_ID}"
            ),
        )
        cliente.set_registered_model_alias(nombre, alias, mv.version)
        logger.success(
            f"{nombre} v{mv.version} <- {run.info.run_name} "
            f"({metrica.split('.')[1]}={valor:.4f}) alias=@{alias}"
        )
        return mv

    registrar(corridas[0], "champion")
    if len(corridas) > 1:
        registrar(corridas[1], "challenger")
    cliente.update_registered_model(name=nombre, description=DESCRIPCION[question])


if __name__ == "__main__":
    app()
