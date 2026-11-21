"""Prueba de carga de la API de demanda (P1) con Locust — Actividad 9.

Simula usuarios consultando la predicción de demanda de estaciones de
MiBici: cada usuario envía observaciones válidas (coherentes con las
validaciones Pydantic de la API) con valores aleatorios realistas, y
ocasionalmente consulta /health, como lo haría un balanceador.

Ejecución sin interfaz, con reporte HTML y CSV:

    locust -f scripts/locustfile.py --headless -u 50 -r 5 -t 3m ^
        --host http://127.0.0.1:8000 ^
        --html reports/carga/locust_u50.html --csv reports/carga/locust_u50
"""

import random

from locust import HttpUser, between, task


def observacion_aleatoria() -> dict:
    """Una estación × día plausible y válida para el esquema de la API."""
    dia_semana = random.randint(0, 6)
    base = random.uniform(5, 120)  # nivel de demanda de la estación
    ruido = lambda: max(0.0, random.gauss(base, base * 0.15))
    return {
        "dia_semana": dia_semana,
        "fin_semana": int(dia_semana >= 5),
        "feriado": int(random.random() < 0.05),
        "lag_1": round(ruido(), 1),
        "lag_2": round(ruido(), 1),
        "lag_7": round(ruido(), 1),
        "lag_14": round(ruido(), 1),
        "media_7d": round(ruido(), 1),
        "media_28d": round(ruido(), 1),
        "media_mismo_dia_4sem": round(ruido(), 1),
        "dias_activa": random.randint(30, 400),
        "capacidad": random.randint(10, 40),
    }


class UsuarioDemanda(HttpUser):
    """Cliente que pide predicciones de demanda del día siguiente."""

    wait_time = between(0.5, 2)

    @task(10)
    def predecir(self):
        self.client.post("/predict", json=observacion_aleatoria())

    @task(1)
    def health(self):
        self.client.get("/health")
