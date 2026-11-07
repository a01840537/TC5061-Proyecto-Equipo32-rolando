"""Esquemas Pydantic de entrada y salida de la API de demanda (P1).

Los campos de :class:`ObservacionDemanda` son exactamente las variables del
modelo (``P1_FEATURES``), en el mismo orden; la prueba
``tests/test_api.py`` verifica que no se desalineen. Las restricciones de
rango rechazan datos imposibles antes de llegar al modelo.
"""

from pydantic import BaseModel, Field, model_validator


class ObservacionDemanda(BaseModel):
    """Una estación × día con las variables disponibles la víspera."""

    dia_semana: int = Field(ge=0, le=6, description="Día de la semana (0=lunes … 6=domingo)")
    fin_semana: int = Field(ge=0, le=1, description="1 si es sábado o domingo")
    feriado: int = Field(ge=0, le=1, description="1 si el día es feriado oficial")
    lag_1: float = Field(ge=0, description="Viajes iniciados hace 1 día")
    lag_2: float = Field(ge=0, description="Viajes iniciados hace 2 días")
    lag_7: float = Field(ge=0, description="Viajes iniciados hace 7 días")
    lag_14: float = Field(ge=0, description="Viajes iniciados hace 14 días")
    media_7d: float = Field(ge=0, description="Media móvil de 7 días")
    media_28d: float = Field(ge=0, description="Media móvil de 28 días")
    media_mismo_dia_4sem: float = Field(ge=0, description="Media del mismo día de las últimas 4 semanas")
    dias_activa: int = Field(ge=1, description="Días desde la primera actividad de la estación")
    capacidad: int = Field(gt=0, description="Anclajes de la estación (> 0)")

    @model_validator(mode="after")
    def _fin_semana_coherente(self):
        """fin_semana debe corresponder al día: 1 solo para sábado (5) o domingo (6)."""
        if self.fin_semana != int(self.dia_semana >= 5):
            raise ValueError(
                "fin_semana inconsistente con dia_semana "
                f"(dia_semana={self.dia_semana} implica fin_semana={int(self.dia_semana >= 5)})"
            )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "dia_semana": 2,
                    "fin_semana": 0,
                    "feriado": 0,
                    "lag_1": 48,
                    "lag_2": 51,
                    "lag_7": 45,
                    "lag_14": 50,
                    "media_7d": 47.3,
                    "media_28d": 44.9,
                    "media_mismo_dia_4sem": 46.5,
                    "dias_activa": 180,
                    "capacidad": 24,
                }
            ]
        }
    }


class Prediccion(BaseModel):
    """Respuesta del endpoint de predicción."""

    viajes_predichos: float = Field(description="Viajes iniciados esperados para el día siguiente")
    modelo: str = Field(description="Nombre del modelo registrado que respondió")
    version: str = Field(description="Versión del modelo en el Model Registry")


class Linaje(BaseModel):
    """Trazabilidad del modelo servido hacia el Model Registry."""

    nombre: str
    alias: str
    version: str
    run_id: str
    git_commit: str
    data_version: str
    test_mae: float | None = None
    exportado_en: str
