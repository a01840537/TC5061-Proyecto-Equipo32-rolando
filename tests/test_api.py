"""Pruebas de los esquemas de la API de demanda (P1).

No requieren el modelo exportado ni el contenedor: validan que el contrato
de entrada de la API esté alineado con las variables reales del modelo y
que las validaciones rechacen datos imposibles.
"""

import pydantic
import pytest

from mibici.api.esquemas import ObservacionDemanda
from mibici.features import P1_FEATURES

EJEMPLO = ObservacionDemanda.model_config["json_schema_extra"]["examples"][0]


def test_esquema_alineado_con_las_features_del_modelo():
    """Los campos del esquema son exactamente P1_FEATURES, en el mismo orden."""
    assert list(ObservacionDemanda.model_fields) == P1_FEATURES


def test_ejemplo_documentado_es_valido():
    obs = ObservacionDemanda(**EJEMPLO)
    assert obs.capacidad == 24


@pytest.mark.parametrize(
    "cambio",
    [
        {"dia_semana": 7},          # fuera de rango 0-6
        {"capacidad": 0},           # la capacidad debe ser > 0
        {"lag_1": -5},              # un conteo no puede ser negativo
        {"fin_semana": 1},          # inconsistente: dia_semana=2 no es fin de semana
        {"dias_activa": 0},         # al menos un día de actividad
    ],
)
def test_entradas_invalidas_se_rechazan(cambio):
    datos = {**EJEMPLO, **cambio}
    with pytest.raises(pydantic.ValidationError):
        ObservacionDemanda(**datos)


def test_fin_semana_coherente_en_sabado():
    datos = {**EJEMPLO, "dia_semana": 5, "fin_semana": 1}
    assert ObservacionDemanda(**datos).fin_semana == 1
