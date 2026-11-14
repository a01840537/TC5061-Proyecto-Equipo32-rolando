"""Reporte de drift de datos de P1 con Evidently.

Compara el periodo de **referencia** (TRAIN: ene–sep 2025, los datos con los
que se entrenó el modelo champion) contra el periodo **actual** (TEST:
nov–dic 2025), que simula el comportamiento en producción: es un periodo
posterior de los mismos datos abiertos de MiBici que el modelo nunca vio.

Genera con Evidently un reporte de deriva por variable + resumen de calidad
de los datos, lo guarda como HTML en ``reports/monitoreo/`` y resume en
consola qué variables derivaron. Ejecutable como entry point del MLflow
Project:

    mlflow run . -e monitor --env-manager local
    # o directamente:
    python -m mibici.monitoring.drift
"""

import json
import re

from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from loguru import logger
import typer

from mibici.config import REPORTS_DIR
from mibici.features import P1_FEATURES
from mibici.modeling.train import preparar_p1

app = typer.Typer()

MONITOREO_DIR = REPORTS_DIR / "monitoreo"
COLUMNAS = P1_FEATURES + ["viajes"]


def _resumir_drift(snapshot) -> int:
    """Imprime el resumen por variable y regresa cuántas columnas derivaron."""
    datos = json.loads(snapshot.json())
    derivadas = 0
    for metrica in datos.get("metrics", []):
        nombre = metrica.get("metric_name", "")
        if nombre.startswith("DriftedColumnsCount"):
            valor = metrica.get("value", {})
            derivadas = int(valor.get("count", 0))
            logger.info(
                f"Columnas con drift: {derivadas} de {len(COLUMNAS)} "
                f"(proporción {valor.get('share', 0):.2f})"
            )
        elif nombre.startswith("ValueDrift("):
            m = re.search(r"column=([^,]+),method=([^,]+),threshold=([^)]+)", nombre)
            if not m:
                continue
            columna, metodo, umbral = m.groups()
            valor = metrica.get("value")
            logger.info(f"  {columna:<22} {metodo:<16} score={valor:.4f} (umbral {umbral})")
    return derivadas


@app.command()
def main():
    MONITOREO_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Preparando particiones de P1 (referencia=TRAIN, actual=TEST)…")
    train, _val, test = preparar_p1()
    # Evidently 0.7 no acepta int8 como etiqueta categórica: todo a float64
    referencia = train[COLUMNAS].reset_index(drop=True).astype("float64")
    actual = test[COLUMNAS].reset_index(drop=True).astype("float64")
    logger.info(f"Referencia: {len(referencia):,} filas | Actual: {len(actual):,} filas")

    reporte = Report([DataDriftPreset(), DataSummaryPreset()])
    snapshot = reporte.run(reference_data=referencia, current_data=actual)

    ruta_html = MONITOREO_DIR / "drift_p1.html"
    snapshot.save_html(str(ruta_html))
    (MONITOREO_DIR / "drift_p1.json").write_text(snapshot.json(), encoding="utf-8")

    derivadas = _resumir_drift(snapshot)
    logger.success(f"Reporte de Evidently guardado en {ruta_html}")
    if derivadas:
        logger.warning(
            f"{derivadas} variable(s) con drift: revisar el HTML e interpretar "
            "si afecta el desempeño del modelo (ver README y reporte de la actividad)."
        )
    else:
        logger.success("Sin drift significativo entre referencia y actual.")


if __name__ == "__main__":
    app()
