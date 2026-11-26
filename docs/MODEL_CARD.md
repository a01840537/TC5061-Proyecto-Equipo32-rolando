# Model Card — Demanda diaria de viajes por estación (MiBici P1)

**Modelo:** `mibici-equipo32-p1-demanda` · versión 1 · alias `@champion`
**Tipo:** Regresión Poisson lineal (scikit-learn Pipeline: one-hot de día de semana + log1p/escalado de conteos)
**Autor:** Rolando Calderón Carballo (A01840537) — Equipo 32, TC5061 (MNA)
**Fecha de entrenamiento:** octubre de 2026 · **Última revisión:** noviembre de 2026
**Linaje:** run `aed1d7da…` · git_commit `b8675e0a…` · data_version `7aa5e436d887`

## Uso previsto

Predecir el número de **viajes que iniciarán en cada estación de MiBici
(Guadalajara) al día siguiente**, para apoyar la planeación operativa:
rebalanceo de bicicletas, mantenimiento y dimensionamiento por estación.

- **Usuarios previstos:** equipo de operación del sistema de bicicletas.
- **Horizonte:** 1 día; granularidad estación × día.
- **Modo de servir recomendado:** inferencia por lotes nocturna (~300
  estaciones/día); la API en tiempo real existe para integración y demo
  (análisis FinOps en `docs/TC5061_Operacion_FinOps_Equipo32.pdf`).
- **Usos NO previstos:** decisiones sobre personas u usuarios individuales;
  horizontes mayores a 1 día; otras ciudades o sistemas sin reentrenar;
  cualquier uso que interprete la predicción como relación causal.

## Datos de entrenamiento

- **Fuente:** datos abiertos de MiBici (https://www.mibici.net/es/datos-abiertos/),
  enero–diciembre 2025: 4,532,032 viajes, 484 estaciones.
- **Privacidad:** los campos personales de la fuente (pseudónimo de usuario,
  género, año de nacimiento) **se eliminan en la ingesta** y no participan en
  ninguna variable del modelo.
- **Versionado:** cada archivo fuente con hash SHA-256 en `data/manifest.json`
  (`data_version 7aa5e436d887`); la ingesta es reproducible con un comando.
- **Particiones:** TRAIN ene–sep · VAL oct · TEST nov–dic (cortes temporales,
  sin fuga: las 12 variables usan solo información anterior al día predicho).

## Desempeño (TEST: nov–dic 2025, nunca visto en entrenamiento)

| Métrica | Valor | Referencia (baseline naive semanal) |
|---|---|---|
| MAE (viajes por estación-día) | **5.41** | 7.22 |
| Lectura | ~25% mejor que repetir el mismo día de la semana anterior | |

En el A/B de la semana 9 (evaluación sobre diciembre), esta versión superó a
una v2 reentrenada con datos más recientes en las 5 métricas evaluadas
(detalle en `docs/TC5061_Drift_ABTest_Equipo32.pdf`).

## Limitaciones conocidas

- **Extrapolación en `dias_activa`:** la variable deriva por construcción
  (Wasserstein 2.59 vs TRAIN); en producción siempre está fuera del rango de
  entrenamiento. Vigilada por el monitoreo; si degrada el desempeño, la
  corrección prevista es de feature engineering, no reentrenos frecuentes.
- **Eventos atípicos:** clima extremo, eventos masivos o cambios de
  infraestructura (estaciones nuevas/cerradas) no están representados; las
  estaciones nuevas carecen de historia y reciben predicciones débiles.
- **Sesgo estacional:** entrenado con un solo año; patrones interanuales
  (p. ej. diciembre) se capturan de forma limitada.
- **Sobrepredicción leve:** sesgo medio +2.6 viajes en la evaluación de
  diciembre; considerar al usarlo para dimensionar a la baja.

## Gobernanza y mantenimiento

- **Ciclo de vida:** versiones en el Model Registry de MLflow con aliases;
  producción apunta a `@champion` y la promoción exige ganar un A/B sobre un
  periodo que ninguna versión vio (procedimiento en `mibici/monitoring/`).
- **Monitoreo:** drift de datos con Evidently (entry point `monitor`);
  reentrenamiento **disparado por evidencia** (drift en variables de decisión
  o degradación del MAE), no por calendario.
- **Verificación:** 34 pruebas automatizadas (incluida reproducibilidad por
  semilla) y pipeline de CI que valida la imagen del servicio en cada cambio.
- **Contacto:** Rolando Calderón Carballo — repositorio
  `TC5061-Proyecto-Equipo32-rolando` (issues).
