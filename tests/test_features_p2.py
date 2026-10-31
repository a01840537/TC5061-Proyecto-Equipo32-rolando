"""Pruebas unitarias de las features de P2 (riesgo de estación).

Con un conjunto sintético y pequeño de viajes se verifica que la
etiqueta de riesgo se construye como la define el diseño: ``sin_bicis``
cuando el flujo neto futuro es muy negativo, ``sin_anclajes`` cuando es
muy positivo, y ``normal`` en horas tranquilas. Casos límite: estaciones
con capacidad 0 (se descartan), filas sin ventana futura completa
(se descartan) y horizontes inválidos (deben lanzar ValueError para
evitar fuga de información, ver p2_agregar_features).
"""

import pandas as pd
import pytest

from mibici.features import p2_agregar_etiqueta, p2_agregar_features, p2_construir_panel

INICIO = "2025-01-01"
FIN = "2025-01-01 10:00"  # horas 00..09 (inclusive='left')
H = pd.Timestamp  # abreviatura


def _viaje(inicio, fin, origen, destino):
    return {
        "inicio_viaje": H(inicio),
        "fin_viaje": H(fin),
        "origen_id": origen,
        "destino_id": destino,
    }


@pytest.fixture()
def viajes():
    """Dos estaciones activas (1 y 2) y una con capacidad 0 (3).

    Hora 00: actividad inicial leve. Hora 03: ráfaga de 8 viajes 1 -> 2,
    que vacía la estación 1 y llena la 2.
    """
    filas = [
        _viaje("2025-01-01 00:05", "2025-01-01 00:20", 1, 2),
        _viaje("2025-01-01 00:10", "2025-01-01 00:25", 3, 3),
    ]
    filas += [
        _viaje(f"2025-01-01 03:{m:02d}", f"2025-01-01 03:{m + 10:02d}", 1, 2)
        for m in range(5, 45, 5)  # 8 viajes dentro de la hora 03
    ]
    return pd.DataFrame(filas)


CAPACIDAD = pd.Series({1: 10, 2: 10, 3: 0})


def test_panel_rellena_horas_sin_movimiento_con_cero(viajes):
    panel = p2_construir_panel(viajes, INICIO, FIN)
    tranquila = panel.loc[(1, H("2025-01-01 05:00"))]
    assert tranquila["salidas"] == 0 and tranquila["llegadas"] == 0 and tranquila["neto"] == 0
    # La ráfaga quedó asignada a la hora correcta
    assert panel.loc[(1, H("2025-01-01 03:00")), "salidas"] == 8
    assert panel.loc[(2, H("2025-01-01 03:00")), "llegadas"] == 8


def test_etiqueta_sin_bicis_y_sin_anclajes(viajes):
    """H=2, fraccion=0.4 -> umbral 4 bicis. La ráfaga de 8 debe marcar riesgo."""
    panel = p2_construir_panel(viajes, INICIO, FIN)
    etiqueta = p2_agregar_etiqueta(panel, horizonte=2, capacidad=CAPACIDAD, fraccion=0.4)

    # En t=01:00 la ventana futura (02:00 y 03:00) contiene la ráfaga
    e1 = etiqueta.loc[(1, H("2025-01-01 01:00"))]
    assert e1["riesgo"] == 1 and e1["tipo_riesgo"] == "sin_bicis"
    assert e1["neto_futuro"] == -8

    e2 = etiqueta.loc[(2, H("2025-01-01 01:00"))]
    assert e2["riesgo"] == 1 and e2["tipo_riesgo"] == "sin_anclajes"
    assert e2["neto_futuro"] == 8

    # Una hora tranquila no marca riesgo
    calma = etiqueta.loc[(1, H("2025-01-01 05:00"))]
    assert calma["riesgo"] == 0 and calma["tipo_riesgo"] == "normal"


def test_etiqueta_descarta_capacidad_cero(viajes):
    """Caso límite: una estación con capacidad 0 no puede generar etiqueta."""
    panel = p2_construir_panel(viajes, INICIO, FIN)
    etiqueta = p2_agregar_etiqueta(panel, horizonte=2, capacidad=CAPACIDAD, fraccion=0.4)
    assert 3 not in etiqueta.index.get_level_values("estacion")


def test_etiqueta_descarta_filas_sin_ventana_futura_completa(viajes):
    """Caso límite: las últimas H horas no tienen futuro completo y se van."""
    panel = p2_construir_panel(viajes, INICIO, FIN)
    etiqueta = p2_agregar_etiqueta(panel, horizonte=2, capacidad=CAPACIDAD, fraccion=0.4)
    ultima = etiqueta.index.get_level_values("hora").max()
    assert ultima == H("2025-01-01 07:00")  # 09:00 y 08:00 descartadas


@pytest.mark.parametrize("horizonte", [0, 24, 48])
def test_horizonte_invalido_lanza_error(viajes, horizonte):
    """H fuera de [1, 24) rompería la garantía de no-fuga de los rezagos."""
    panel = p2_construir_panel(viajes, INICIO, FIN)
    etiqueta = p2_agregar_etiqueta(panel, horizonte=2, capacidad=CAPACIDAD, fraccion=0.4)
    with pytest.raises(ValueError):
        p2_agregar_features(etiqueta, horizonte=horizonte)
