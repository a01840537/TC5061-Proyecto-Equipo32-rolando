"""Chequeos de calidad de datos con DeepChecks sobre la tabla de modelado de P1.

Corre dos suites de DeepChecks sobre el panel diario estación × día
(la tabla con la que se entrena P1):

* ``data_integrity`` sobre TRAIN: duplicados, nulos, tipos mezclados,
  columnas constantes, correlación feature-etiqueta, valores atípicos.
* ``train_test_validation`` entre TRAIN y TEST: deriva (drift) de
  features y de etiqueta, features nuevas o faltantes, fuga de filas
  duplicadas entre particiones.

Guarda los reportes HTML en ``reports/validacion/`` y resume en consola
qué chequeos pasaron y cuáles no. Ejecutable como entry point del
MLflow Project:

    mlflow run . -e validate --env-manager local
    # o directamente:
    python -m mibici.data_validation
"""

from pathlib import Path

# --- Parches de compatibilidad ------------------------------------------------
# DeepChecks 0.19.1 se escribió para scikit-learn < 1.8 y NumPy < 2; nuestro
# entorno es más nuevo. Dos ajustes ANTES de importar deepchecks:
# 1) reinscribir el scorer 'max_error' (sklearn >= 1.8 lo quitó del catálogo,
#    aunque la función max_error sigue existiendo);
# 2) restaurar el alias np.Inf (NumPy 2 lo eliminó en favor de np.inf).
import numpy as np
from sklearn.metrics import get_scorer_names, make_scorer, max_error

if "max_error" not in get_scorer_names():
    from sklearn.metrics import _scorer

    _scorer._SCORERS["max_error"] = make_scorer(max_error, greater_is_better=False)
if not hasattr(np, "Inf"):
    np.Inf = np.inf
# ------------------------------------------------------------------------------

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, train_test_validation
from loguru import logger
import typer

from mibici.config import REPORTS_DIR, SEED
from mibici.features import P1_FEATURES
from mibici.modeling.train import preparar_p1

app = typer.Typer()

VALIDACION_DIR = REPORTS_DIR / "validacion"
CATEGORICAS = ["dia_semana", "fin_semana", "feriado"]


def _dataset(df, muestra: int) -> Dataset:
    """Envuelve una partición en un Dataset de DeepChecks (con muestreo)."""
    cols = P1_FEATURES + ["viajes"]
    if muestra and len(df) > muestra:
        df = df.sample(muestra, random_state=SEED)
    return Dataset(
        df[cols].reset_index(drop=True),
        label="viajes",
        cat_features=CATEGORICAS,
    )


def _resumir(nombre: str, resultado, ruta: Path) -> int:
    """Imprime el resumen de una suite y regresa cuántas condiciones fallaron."""
    resultado.save_as_html(str(ruta), as_widget=False)
    fallidos = resultado.get_not_passed_checks(fail_if_warning=False)
    total = len(resultado.results)
    logger.info(f"[{nombre}] {total} chequeos ejecutados -> reporte: {ruta}")
    if not fallidos:
        logger.success(f"[{nombre}] todos los chequeos con condición pasaron")
        return 0
    for check in fallidos:
        condiciones = "; ".join(
            f"{c.name}: {c.details}" for c in check.conditions_results if not c.is_pass()
        )
        logger.warning(f"[{nombre}] FALLÓ {check.get_header()} -> {condiciones}")
    return len(fallidos)


@app.command()
def main(
    muestra: int = typer.Option(
        50_000, "--sample", help="Filas máximas por partición (0 = todas)."
    ),
):
    VALIDACION_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Preparando particiones de P1 (train/val/test)…")
    train, _val, test = preparar_p1()
    ds_train = _dataset(train, muestra)
    ds_test = _dataset(test, muestra)

    logger.info("Suite 1/2: integridad de datos sobre TRAIN…")
    integridad = data_integrity().run(ds_train)
    fallas = _resumir("integridad", integridad, VALIDACION_DIR / "integridad_datos.html")

    logger.info("Suite 2/2: validación TRAIN vs TEST (deriva y fuga)…")
    deriva = train_test_validation().run(train_dataset=ds_train, test_dataset=ds_test)
    fallas += _resumir("train-vs-test", deriva, VALIDACION_DIR / "deriva_train_test.html")

    if fallas:
        logger.warning(f"{fallas} chequeo(s) con condiciones no aprobadas: revisar los HTML.")
    else:
        logger.success("Validación de datos completa sin fallas.")


if __name__ == "__main__":
    app()
