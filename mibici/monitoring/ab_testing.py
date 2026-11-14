"""A/B testing entre dos versiones del modelo de demanda (P1).

Compara, sobre el mismo conjunto de evaluación, la versión en producción
contra una versión reentrenada con datos más recientes (*continuous
training* simulado):

* **v1 (A)** — el champion actual del Model Registry
  (``models:/mibici-equipo32-p1-demanda@champion``), entrenado con
  TRAIN = ene–sep 2025.
* **v2 (B)** — el mismo pipeline (regresión Poisson) reentrenado con
  ene–nov 2025 (TRAIN + VAL + noviembre), registrado como **nueva versión**
  del mismo modelo en el Registry con el alias ``challenger``.

Ambas versiones se evalúan sobre **diciembre 2025**, un tramo que ninguna
de las dos vio durante su entrenamiento, de modo que la comparación sea
justa. El resultado queda en consola, en MLflow (run del reentrenamiento)
y en ``reports/monitoreo/ab_test_p1.json``. Ejecutable:

    mlflow run . -e ab_test --env-manager local
    # o directamente:
    python -m mibici.monitoring.ab_testing
"""

from datetime import datetime, timezone
import json

import mlflow
import mlflow.sklearn
from loguru import logger
from mlflow.tracking import MlflowClient
import typer

from mibici.config import (
    ESTACIONES_PARQUET,
    P1_FERIADOS,
    P1_HORIZONTE_DIAS,
    P1_LAGS_DIAS,
    P1_TEST_END,
    P1_TRAIN_START,
    P1_VENTANAS_MEDIA_DIAS,
    REPORTS_DIR,
    VIAJES_PARQUET,
)
from mibici.features import P1_FEATURES, p1_agregar_features, p1_construir_panel_diario
from mibici.modeling import p1_regresion as p1
from mibici.modeling.registrar import NOMBRES
from mibici.modeling.train import configurar_tracking, contexto_comun

app = typer.Typer()

MONITOREO_DIR = REPORTS_DIR / "monitoreo"
V2_TRAIN_END = "2025-12-01"  # v2 entrena con ene–nov
EVAL_START = "2025-12-01"  # ambos se evalúan sobre diciembre
METRICAS_TABLA = ["mae", "rmse", "wape", "sesgo", "mape_total_dia"]


def _preparar_datos():
    """Panel diario completo de P1, cortado en entrenamiento de v2 y evaluación."""
    viajes = p1.cargar_viajes(VIAJES_PARQUET)
    capacidad = p1.cargar_capacidad(ESTACIONES_PARQUET)
    panel = p1_construir_panel_diario(viajes, P1_TRAIN_START, P1_TEST_END)
    df = p1_agregar_features(
        panel, capacidad, P1_FERIADOS, P1_LAGS_DIAS, P1_VENTANAS_MEDIA_DIAS, P1_HORIZONTE_DIAS
    )
    train_v2 = p1.cortar(df, P1_TRAIN_START, V2_TRAIN_END).dropna(subset=P1_FEATURES)
    evaluacion = p1.cortar(df, EVAL_START, P1_TEST_END).dropna(subset=P1_FEATURES)
    return train_v2, evaluacion


def _x(df):
    return df[P1_FEATURES].astype("float64")


@app.command()
def main():
    configurar_tracking()
    cliente = MlflowClient()
    nombre = NOMBRES["p1"]
    MONITOREO_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Preparando datos (v2: ene–nov | evaluación A/B: diciembre)…")
    train_v2, evaluacion = _preparar_datos()
    logger.info(f"Entrenamiento v2: {len(train_v2):,} filas | Evaluación: {len(evaluacion):,} filas")

    # ---- A: champion vigente desde el Registry (por nombre y alias) ----
    version_v1 = cliente.get_model_version_by_alias(nombre, "champion")
    modelo_v1 = mlflow.sklearn.load_model(f"models:/{nombre}@champion")
    logger.info(f"v1 = {nombre} v{version_v1.version} (@champion, run {version_v1.run_id[:8]}…)")

    # ---- B: mismo pipeline reentrenado con datos recientes ----
    contexto = contexto_comun()
    modelo_v2 = p1.construir_lineal()
    modelo_v2.fit(_x(train_v2), train_v2["viajes"])

    metricas_v1 = p1.evaluar(evaluacion["viajes"], modelo_v1.predict(_x(evaluacion)), "dic_v1")
    metricas_v2 = p1.evaluar(evaluacion["viajes"], modelo_v2.predict(_x(evaluacion)), "dic_v2")

    with mlflow.start_run(run_name="p1_poisson_v2_reentrenado") as run:
        mlflow.set_tags(
            {
                "question": "P1",
                "model_family": "poisson",
                "stage_of_work": "reentrenamiento",
                "git_commit": contexto["git_commit"],
                "data_version": contexto["data_version"],
            }
        )
        mlflow.log_params(
            {
                "ventana_entrenamiento": f"{P1_TRAIN_START} a {V2_TRAIN_END}",
                "ventana_evaluacion": f"{EVAL_START} a {P1_TEST_END}",
                "hp_modelo": "poisson_lineal (mismo pipeline que el champion)",
            }
        )
        mlflow.log_metrics({**metricas_v1, **metricas_v2})
        mlflow.sklearn.log_model(
            modelo_v2,
            name="modelo",
            input_example=_x(train_v2).head(3),
        )
        mv2 = mlflow.register_model(f"runs:/{run.info.run_id}/modelo", nombre)

    cliente.update_model_version(
        name=nombre,
        version=mv2.version,
        description=(
            f"v2 reentrenada (continuous training): ene–nov 2025. "
            f"A/B sobre diciembre vs v{version_v1.version}. "
            f"git_commit: {contexto['git_commit'][:12]} | data_version: {contexto['data_version']}"
        ),
    )
    cliente.set_registered_model_alias(nombre, "challenger", mv2.version)
    logger.success(f"v2 registrada como {nombre} v{mv2.version} (alias @challenger)")

    # ---- Tabla A/B ----
    logger.info("A/B testing sobre diciembre 2025 (ningún modelo lo vio al entrenar):")
    logger.info(f"  {'métrica':<16} {'v1 champion':>12} {'v2 reentrenada':>14} {'mejora':>9}")
    for met in METRICAS_TABLA:
        a, b = metricas_v1[f"dic_v1_{met}"], metricas_v2[f"dic_v2_{met}"]
        mejora = (a - b) / abs(a) * 100 if a else 0.0
        logger.info(f"  {met:<16} {a:>12.4f} {b:>14.4f} {mejora:>8.1f}%")

    gana_v2 = metricas_v2["dic_v2_mae"] < metricas_v1["dic_v1_mae"]
    ganador = f"v{mv2.version} (reentrenada)" if gana_v2 else f"v{version_v1.version} (champion)"
    logger.success(f"Ganador por MAE en diciembre: {ganador}")
    logger.info(
        "El alias @champion NO se mueve automáticamente: la promoción es la "
        "recomendación del reporte de la actividad y se aplica con "
        "set_registered_model_alias tras revisión."
    )

    resultado = {
        "modelo": nombre,
        "v1": {"version": str(version_v1.version), "alias": "champion", "entrenamiento": f"{P1_TRAIN_START} a 2025-10-01"},
        "v2": {"version": str(mv2.version), "alias": "challenger", "entrenamiento": f"{P1_TRAIN_START} a {V2_TRAIN_END}"},
        "evaluacion": f"{EVAL_START} a {P1_TEST_END}",
        "metricas_v1": metricas_v1,
        "metricas_v2": metricas_v2,
        "ganador_por_mae": ganador,
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    ruta = MONITOREO_DIR / "ab_test_p1.json"
    ruta.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success(f"Resultado del A/B guardado en {ruta}")


if __name__ == "__main__":
    app()
