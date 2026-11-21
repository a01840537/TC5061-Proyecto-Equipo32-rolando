"""Modelo de humo para el CI (Actividad 9).

El champion real se exporta del Model Registry local y NO vive en git
(models/ está en .gitignore). Para que el pipeline de CI pueda construir la
imagen de la API y validar su levantamiento de punta a punta, este script
entrena el MISMO pipeline de P1 (regresión Poisson, mibici.modeling.
p1_regresion.construir_lineal) sobre datos sintéticos y lo deja en
models/api/p1/ con el mismo formato que produce mibici/api/exportar.py.

    python scripts/modelo_stub.py
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

# Ejecutable como `python scripts/modelo_stub.py` desde la raíz del repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow.sklearn
import numpy as np
import pandas as pd

from mibici.config import SEED
from mibici.features import P1_FEATURES
from mibici.modeling.p1_regresion import construir_lineal

DESTINO = Path("models/api/p1")
N = 500


def main():
    rng = np.random.default_rng(SEED)
    columnas = {}
    for col in P1_FEATURES:
        if col == "dia_semana":
            columnas[col] = rng.integers(0, 7, N)
        elif col in ("fin_semana", "feriado"):
            columnas[col] = rng.integers(0, 2, N)
        elif col.startswith(("lag_", "media_")):
            columnas[col] = rng.poisson(40, N).astype("float64")
        else:  # dias_activa, capacidad
            columnas[col] = rng.integers(5, 40, N)
    x = pd.DataFrame(columnas)
    x["fin_semana"] = (x["dia_semana"] >= 5).astype(int)
    x = x[P1_FEATURES]
    y = rng.poisson(1 + 0.5 * x["lag_1"])

    modelo = construir_lineal().fit(x, y)

    shutil.rmtree(DESTINO, ignore_errors=True)
    DESTINO.mkdir(parents=True)
    mlflow.sklearn.save_model(modelo, str(DESTINO / "modelo"))
    (DESTINO / "linaje.json").write_text(
        json.dumps(
            {
                "nombre": "mibici-equipo32-p1-demanda",
                "alias": "stub-ci",
                "version": "0",
                "run_id": "ci-smoke-test",
                "git_commit": "ci",
                "data_version": "sintetico",
                "test_mae": None,
                "exportado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Modelo de humo listo en {DESTINO}")


if __name__ == "__main__":
    main()
