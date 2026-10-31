"""Pruebas unitarias de la ingesta (mibici/dataset.py).

Cubren los tres riesgos reales de los archivos de MiBici: columnas con
acentos y nombres cambiantes entre años, codificaciones mixtas
(utf-8 / latin-1) y el versionado de datos por hash. Casos límite
incluidos: variantes de columna, archivo no-utf8 y sensibilidad del
hash a un solo byte.
"""

import pandas as pd
import pytest

from mibici.dataset import (
    COLUMNAS_PERSONALES,
    COLUMNAS_VIAJES,
    _normalizar_nombre,
    _sha256,
    leer_csv,
    normalizar_viajes,
)


# ---------- Normalización de nombres de columna ----------


@pytest.mark.parametrize(
    ("original", "esperado"),
    [
        ("Año_de_nacimiento", "ano_de_nacimiento"),
        ("Inicio del viaje", "inicio_del_viaje"),
        ("Género", "genero"),
        ("Origen Id", "origen_id"),
        ("  Destino_Id  ", "destino_id"),
        ("A?o_de_nacimiento", "a_o_de_nacimiento"),  # caracter corrupto -> '_'
    ],
)
def test_normalizar_nombre(original, esperado):
    assert _normalizar_nombre(original) == esperado


def test_mapeo_cubre_las_variantes_historicas_de_anio():
    """Las tres formas en que MiBici ha escrito 'año de nacimiento'."""
    variantes = ("ano_de_nacimiento", "anio_de_nacimiento", "a_o_de_nacimiento")
    for v in variantes:
        assert COLUMNAS_VIAJES[v] == "anio_nacimiento"


def test_normalizar_viajes_unifica_esquema_y_minimiza_datos_personales():
    df = pd.DataFrame(
        {
            "Viaje_Id": [1, 2],
            "Usuario_Id": [10, 20],
            "Género": ["M", "F"],
            "Año_de_nacimiento": [1990, 1985],
            "Inicio_del_viaje": ["2025-01-01 08:00:00", "2025-01-01 09:30:00"],
            "Fin_del_viaje": ["2025-01-01 08:15:00", "no-es-fecha"],
            "Origen_Id": [100, 200],
            "Destino_Id": [200, 100],
        }
    )
    out = normalizar_viajes(df, origen="2025_01")

    # Esquema unificado y mes de origen trazable
    assert {"viaje_id", "inicio_viaje", "fin_viaje", "origen_id", "destino_id"} <= set(out.columns)
    assert (out["mes_archivo"] == "2025_01").all()

    # Fechas tipadas; la fecha inválida queda NaT en vez de romper la ingesta
    assert pd.api.types.is_datetime64_any_dtype(out["inicio_viaje"])
    assert pd.isna(out.loc[1, "fin_viaje"])

    # Minimización de datos personales: ninguna columna personal sobrevive
    assert not set(COLUMNAS_PERSONALES) & set(out.columns)


# ---------- Codificación ----------


def test_leer_csv_cae_a_latin1_cuando_no_es_utf8(tmp_path):
    """Los CSV viejos de MiBici vienen en latin-1; la lectura no debe romper."""
    ruta = tmp_path / "viajes_latin1.csv"
    ruta.write_bytes("Año,Género\n1990,M\n".encode("latin-1"))

    df = leer_csv(ruta)
    assert list(df.columns) == ["Año", "Género"]
    assert df.iloc[0, 0] == 1990


def test_leer_csv_utf8_normal(tmp_path):
    ruta = tmp_path / "viajes_utf8.csv"
    ruta.write_text("Año,Viaje_Id\n1990,1\n", encoding="utf-8")
    assert list(leer_csv(ruta).columns) == ["Año", "Viaje_Id"]


# ---------- Versionado de datos (hash del manifiesto) ----------


def test_sha256_es_determinista_y_sensible_a_un_byte(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_bytes(b"viaje_id,origen_id\n1,100\n")
    b.write_bytes(b"viaje_id,origen_id\n1,100\n")

    # Mismo contenido -> mismo hash (data_version reproducible)
    assert _sha256(a) == _sha256(b)

    # Un byte distinto -> hash distinto (detecta datos alterados)
    b.write_bytes(b"viaje_id,origen_id\n1,101\n")
    assert _sha256(a) != _sha256(b)
