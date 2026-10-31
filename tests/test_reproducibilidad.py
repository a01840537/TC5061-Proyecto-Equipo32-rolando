"""Pruebas de reproducibilidad: misma semilla => mismas predicciones.

Construye dos veces cada modelo con las fábricas reales del proyecto
(mibici.modeling.train), los entrena sobre los mismos datos sintéticos
y exige predicciones *idénticas* (no solo parecidas). Si alguien quita
``random_state=SEED`` de un modelo, estas pruebas fallan.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

from mibici.config import SEED
from mibici.features import P1_FEATURES
from mibici.modeling.train import construir_p1, construir_p2

N = 400


@pytest.fixture(scope="module")
def datos_p1():
    """Tabla sintética con el esquema de P1 (conteos >= 0 para log1p)."""
    rng = np.random.default_rng(SEED)
    columnas = {}
    for col in P1_FEATURES:
        if col == "dia_semana":
            columnas[col] = rng.integers(0, 7, N)
        elif col in ("fin_semana", "feriado"):
            columnas[col] = rng.integers(0, 2, N)
        elif col.startswith(("lag_", "media_")):
            columnas[col] = rng.poisson(20, N).astype("float64")
        else:  # dias_activa, capacidad
            columnas[col] = rng.integers(5, 40, N)
    x = pd.DataFrame(columnas)
    y = rng.poisson(lam=1 + 0.2 * x["lag_1"])
    return x, pd.Series(y, name="viajes")


@pytest.fixture(scope="module")
def datos_p2():
    """Matriz numérica y etiqueta binaria para el clasificador de P2."""
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(N, 5))
    y = (x[:, 0] + 0.5 * rng.normal(size=N) > 0).astype(int)
    return x, y


def _dos_entrenamientos(fabrica, x, y):
    """Construye el modelo dos veces, entrena con los mismos datos y predice."""
    preds = []
    for _ in range(2):
        modelo, _stage = fabrica()
        modelo.fit(x, y)
        preds.append(modelo.predict(x))
    return preds


def test_p1_hgb_es_reproducible(datos_p1):
    x, y = datos_p1
    a, b = _dos_entrenamientos(lambda: construir_p1("hgb", 50, -1, -1), x, y)
    assert np.array_equal(a, b), "P1 HGB: misma semilla debe dar predicciones idénticas"


def test_p1_poisson_es_reproducible(datos_p1):
    x, y = datos_p1
    a, b = _dos_entrenamientos(lambda: construir_p1("poisson", -1, -1, -1), x, y)
    assert np.array_equal(a, b)


def test_p2_hgb_probabilidades_reproducibles(datos_p2):
    x, y = datos_p2
    probas = []
    for _ in range(2):
        modelo, _stage = construir_p2("hgb", 50, -1, -1)
        modelo.fit(x, y)
        probas.append(modelo.predict_proba(x)[:, 1])
    assert np.array_equal(probas[0], probas[1])


def test_p3_kmeans_etiquetas_reproducibles(datos_p2):
    """Clustering con la SEED del proyecto: mismas etiquetas en dos ajustes."""
    x, _ = datos_p2
    etiquetas = [
        KMeans(n_clusters=3, random_state=SEED, n_init=10).fit_predict(x) for _ in range(2)
    ]
    assert np.array_equal(etiquetas[0], etiquetas[1])


def test_semilla_distinta_puede_cambiar_el_clustering(datos_p2):
    """Control negativo: la semilla sí participa (inicializaciones distintas)."""
    x, _ = datos_p2
    a = KMeans(n_clusters=3, random_state=SEED, n_init=1, init="random").fit(x)
    b = KMeans(n_clusters=3, random_state=SEED + 1, n_init=1, init="random").fit(x)
    # Los centros iniciales dependen de la semilla; los finales casi siempre difieren.
    assert not np.allclose(np.sort(a.cluster_centers_, axis=0), np.sort(b.cluster_centers_, axis=0))
