# TC5061-Proyecto-Equipo32-rolando

> **Continuación individual** del proyecto integrador, a cargo de Rolando Calderón Carballo
> (A01840537). Parte del repositorio del equipo
> [TC5061-Proyecto-Equipo32](https://github.com/a01840537/TC5061-Proyecto-Equipo32)
> (primera entrega, calificada), conservando su historial completo; de aquí en adelante el
> desarrollo en este repositorio es individual.

Proyecto integrador de TC5061 (Operaciones de aprendizaje automático): propuesta de solución de ML
sobre los datos abiertos de [MiBici](https://mibici.net/es/datos-abiertos/), el sistema de bicicleta
pública del Área Metropolitana de Guadalajara. Datos: ene–dic 2025 (4,532,032 viajes, 484 estaciones).

El proyecto responde tres preguntas de negocio:

| # | Pregunta | Tipo de tarea | Usuario |
|---|----------|---------------|---------|
| P1 | ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | Regresión | Operador del sistema (planeación de flota) |
| P2 | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes? | Clasificación | Equipo de rebalanceo |
| P3 | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican? | Clustering | Planeación urbana y expansión |

El ML Canvas y el diagrama de arquitectura se encuentran en `references/`.

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mibici and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mibici   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mibici a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── p1_regresion.py     <- P1: viajes diarios por estación (baseline + 2 candidatos, MLflow)
    │   ├── p2_clasificacion.py <- P2: riesgo de quedarse sin bicis/anclajes (MLflow)
    │   ├── p3_clustering.py    <- P3: clustering de estaciones por patrón de uso (MLflow)
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------


## Reproducibilidad y MLOps

El proyecto utiliza una estructura basada en Cookiecutter Data Science. La lógica final se mantiene dentro del paquete `mibici`; los notebooks se reservan para exploración.

El entorno fue probado con Python 3.14.4. Las dependencias directas se encuentran fijadas por versión en `requirements.txt`.
También se verificó la ejecución completa (ingesta, pruebas, P1, P2 y P3) con **Python 3.12.13** desde un clon limpio.

### Instalación

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt


```

La consistencia del entorno puede verificarse con:

```bash
python -m pip check
```

### Datos

La ingesta se ejecuta desde el módulo:

```bash
python -m mibici.dataset
```

`dataset.py` descarga los datos públicos de MiBici definidos en `config.py`, genera los archivos procesados y crea `data/manifest.json`.

El manifiesto incluye una `data_version` calculada a partir de los archivos de entrada. Esta versión se registra posteriormente en MLflow para mantener trazabilidad entre datos y experimentos.

Los datos crudos, intermedios y procesados no se versionan en Git.

## Reproducción completa desde un clon limpio

```bash
git clone https://github.com/a01840537/TC5061-Proyecto-Equipo32-rolando.git
cd TC5061-Proyecto-Equipo32-rolando
python -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Servidor de tracking (terminal aparte, desde la raíz del repositorio)
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000   # PowerShell: $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

# Entrega 2: todo se ejecuta vía MLflow Project (en Windows con --env-manager local)
mlflow run . -e ingest --env-manager local --experiment-name mibici-equipo32    # 1. ingesta
python -m pytest tests                                                          # 2. pruebas de no fuga (P1)
mlflow run . -e train  -P question=p1 -P model=naive       --env-manager local --experiment-name mibici-equipo32
mlflow run . -e train  -P question=p1 -P model=poisson     --env-manager local --experiment-name mibici-equipo32
mlflow run . -e search -P question=p1 -P model=hgb         --env-manager local --experiment-name mibici-equipo32
mlflow run . -e train  -P question=p2 -P model=logreg      --env-manager local --experiment-name mibici-equipo32
mlflow run . -e search -P question=p2 -P model=hgb         --env-manager local --experiment-name mibici-equipo32
mlflow run . -e train  -P question=p3 -P model=kmeans2     --env-manager local --experiment-name mibici-equipo32
mlflow run . -e train  -P question=p3 -P model=agglo -P k=4 --env-manager local --experiment-name mibici-equipo32
mlflow run . -e search -P question=p3 -P model=kmeans      --env-manager local --experiment-name mibici-equipo32
mlflow run . -e register -P question=p1 --env-manager local   # ídem para p2 y p3
mlflow run . -e verify   -P question=p1 --env-manager local   # ídem para p2 y p3
```

La ingesta debe producir `data_version = 7aa5e436d887`; si el valor cambia, los archivos de MiBici
descargados no son los mismos que se usaron en los experimentos.

> Flujo de la entrega 1 (legado): `make train` ejecuta directamente los módulos
> `p1_regresion`, `p2_clasificacion` y `p3_clustering`. Sigue funcionando, pero la
> entrega 2 lanza todo mediante `mlflow run` como se documenta arriba.

**Privacidad.** Los CSV de MiBici incluyen un pseudónimo de usuario, el género y el año de nacimiento.
Como ninguna pregunta los necesita, `dataset.py` los elimina antes de escribir los Parquet procesados
(minimización de datos). Los datos se usan con fines académicos citando la fuente (portal de datos
abiertos de MiBici) y no se redistribuyen en este repositorio.

## P1 — Regresión de viajes diarios por estación

**Pregunta:** ¿cuántos viajes se iniciarán en cada estación el día siguiente? Usuario: operador del
sistema (planeación de flota y turnos). Granularidad: estación × día; horizonte: 1 día (la predicción
del día D se emite al cierre de D-1).

```bash
python -m mibici.modeling.p1_regresion
```

- **Partición temporal:** TRAIN ene-sep 2025, VAL oct 2025 (selección del modelo), TEST nov-dic 2025.
- **Variables** (`mibici/features.py`, funciones `p1_*`): calendario del día objetivo (día de la semana,
  fin de semana, feriado), rezagos de 1, 2, 7 y 14 días, medias móviles de 7 y 28 días que terminan en
  D-1, promedio del mismo día de las 4 semanas previas, antigüedad y capacidad de la estación.
  `p1_agregar_features` lanza `LeakageError` si se pide un rezago menor que el horizonte.
- **Modelos:** baseline ingenuo semanal (`t − 7 días`), candidato lineal `PoissonRegressor`
  (conteos en `log1p`) y candidato `HistGradientBoostingRegressor` con pérdida Poisson.
- **Métricas:** `train_/val_/test_` + `mae`, `rmse`, `wape`, `r2`, `sesgo` y `mape_total_dia`.
- **Runs:** `p1_baseline_naive_semanal`, `p1_candidato_poisson_lineal`, `p1_candidato_hgb_poisson`.

## P2 — Clasificación de riesgo de disponibilidad

**Pregunta:** ¿una estación estará en riesgo de quedarse sin bicicletas o sin anclajes en las próximas
2 horas? Usuario: equipo de rebalanceo.

```bash
python -m mibici.modeling.p2_clasificacion
```

- **Etiqueta:** riesgo si `|neto(t+1 … t+2)| >= 0.4 × capacidad` (`p2_agregar_etiqueta`).
- **Partición temporal:** TRAIN ene-sep, VAL oct (umbral de decisión que maximiza F1), TEST nov-dic 2025.
- **Modelos:** baseline regresión logística balanceada y candidato `HistGradientBoostingClassifier`.
- **Runs:** `p2_baseline_logreg`, `p2_candidato_hgb`.

## P3 — Clustering de estaciones

La pregunta P3 busca agrupar estaciones de MiBici de acuerdo con sus patrones de uso y evaluar si esos perfiles se mantienen estables en el tiempo.

La configuración de P3 se encuentra centralizada en `mibici/config.py`. El pipeline final se ejecuta con:

```bash
python -m mibici.modeling.p3_clustering
```

### Partición temporal

Para evitar fuga de información se utiliza una separación temporal sin mezcla aleatoria:

- TRAIN: 1 de enero al 19 de octubre de 2025.
- TEST: 20 de octubre al 31 de diciembre de 2025.
- Ventana histórica reducida: 1 de abril al 19 de octubre de 2025.

Los viajes con duración menor a 60 segundos o mayor a 24 horas se excluyen. También se eliminan viajes que cruzan los límites temporales entre TRAIN y TEST.

### Construcción de características

El feature engineering se encuentra en `mibici/features.py`.

Una estación se considera elegible cuando:

- tiene al menos 30 días de historial;
- presenta menos de 60 días de inactividad al cierre del periodo.

Para cada estación se generan 12 características de comportamiento relacionadas con intensidad, balance, uso en fin de semana y distribución de salidas y llegadas por bloque horario.

Las variables de intensidad se transforman con `log1p` y posteriormente se aplica `StandardScaler`. El escalador se ajusta exclusivamente con TRAIN.

### Comparación de modelos

El pipeline reproduce tres alternativas:

- Baseline: K-Means con k=2.
- Candidato: Agglomerative Clustering con enlace Ward y k=4.
- Modelo final: K-Means con k=4.

La comparación utiliza Silhouette, Davies-Bouldin y Calinski-Harabasz. Para el modelo final también se evalúa estabilidad temporal mediante ARI, NMI y proporción de estaciones que conservan su cluster.

### MLflow Tracking

Todos los runs del proyecto se registran en el experimento único `mibici-equipo32`;
la pregunta atendida se distingue mediante tags (`problema: P1/P2/P3`).

Los runs tienen nombres descriptivos:

- `p3_baseline_kmeans_k2`
- `p3_candidato_agglomerative_k4`
- `p3_kmeans_final`

Además de los tres runs de P3, el experimento contiene los runs de P1 (`p1_baseline_naive_semanal`,
`p1_candidato_poisson_lineal`, `p1_candidato_hgb_poisson`) y de P2 (`p2_baseline_logreg`,
`p2_candidato_hgb`). Todos llevan los tags `equipo = equipo32`, `problema` (P1/P2/P3), `pregunta`,
`etapa` (baseline/candidato), `git_commit` y `data_version`.

Cada run registra, según corresponda:

- versión de los datos (`data_version`);
- commit de Git (`git_commit`);
- pregunta atendida;
- rango temporal;
- semilla y modelo;
- hiperparámetros;
- reglas de filtrado y elegibilidad;
- número de características;
- transformación y escalamiento;
- métricas diferenciadas por conjunto, como `train_silhouette` y `test_silhouette`.

El run final registra como artefactos el modelo serializado, la visualización PCA, el mapa interactivo, el resultado geográfico y la tabla resumen de clusters.

La interfaz local de MLflow puede iniciarse desde la raíz del repositorio con:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

y consultarse en `http://localhost:5000`.

## Git y control de versiones

Los datos de MiBici, modelos serializados, base de datos de MLflow, artefactos de MLflow y resultados generados por P3 se excluyen mediante `.gitignore`.

El código se desarrolla en ramas independientes y todo cambio hacia `main` debe incorporarse mediante Pull Request revisado por otro integrante del equipo.

Convenciones: ramas `feat/`, `fix/`, `docs/` + descripción corta; mensajes de commit
siguiendo [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/). La rama
`main` está protegida; cada integrante abre al menos un PR y revisa al menos uno.


## Equipo 32 (autores de la base)

| Integrante | Matrícula | GitHub | Rol |
|------------|-----------|--------|-----|
| Rolando Calderón Carballo | A01840537 | `a01840537` | Infraestructura: repo, ingesta reproducible, versionado de datos, configuración |
| Leydy Johanna Camargo Medina | A01840540 | `JohannaCamargo2805` | P1 — Regresión: demanda de viajes por estación |
| Manuel Alejandro Cardona Esquivel | A01795954 | `alexcard33` | P2 — Clasificación: riesgo de disponibilidad por estación |
| Gerardo Iván Conde Reynoso | A01840602 | `GIvanConder` | P3 — Clustering: patrones de uso de estaciones |

El detalle de qué hizo cada integrante (commits, Pull Requests y revisiones) está en
[`CONTRIBUCIONES.md`](CONTRIBUCIONES.md).

## Entrega 2 — Servidor de tracking, MLflow Project y Model Registry

### Servidor de tracking local

El Model Registry requiere un backend respaldado por base de datos. El servidor se
levanta desde la raíz del repositorio, en una terminal dedicada:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

En la terminal de trabajo, todos los procesos se apuntan al servidor mediante la
variable de entorno:

```bash
# Windows PowerShell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
# Linux / macOS
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

La interfaz queda en http://127.0.0.1:5000. La base (`mlflow.db`), `mlruns/` y
`mlartifacts/` están excluidos del repositorio vía `.gitignore`.

### Ejecución como MLflow Project

El entorno está declarado en `python_env.yaml`; en Windows las corridas se lanzan
con el entorno activo (`--env-manager local`). Todas las corridas de la entrega 2
se ejecutan vía `mlflow run`, nunca desde notebooks:

```bash
# Ingesta reproducible
mlflow run . -e ingest --env-manager local --experiment-name mibici-equipo32

# Entrenamiento (pregunta, modelo e hiperparámetros; -1 = default de config.py)
mlflow run . -e train -P question=p2 -P model=hgb -P learning_rate=0.05 --env-manager local --experiment-name mibici-equipo32

# Búsqueda de hiperparámetros con nested runs
mlflow run . -e search -P question=p2 -P model=hgb --env-manager local --experiment-name mibici-equipo32

# Registro del mejor modelo y verificación de carga por alias
mlflow run . -e register -P question=p2 --env-manager local
mlflow run . -e verify -P question=p2 --env-manager local
```

### Esquema del Model Registry

Un modelo registrado por pregunta: `mibici-equipo32-p1-demanda`,
`mibici-equipo32-p2-riesgo` y `mibici-equipo32-p3-grupos`. El ciclo de vida usa el
esquema vigente de **aliases**: `champion` apunta a la versión en producción y
`challenger` a la segunda mejor. Se eligió aliases sobre las etapas clásicas porque
es el esquema recomendado por MLflow (las etapas están deprecadas), permite más de
un rol por modelo sin transiciones destructivas, y el modelo en producción se carga
por nombre estable: `models:/mibici-equipo32-p1-demanda@champion`. Cada versión
registrada documenta su linaje en la descripción: corrida de origen, commit de Git
y versión de los datos (`data_version`).

## Actividad 7 — Pruebas y chequeos de seguridad

Esta sección documenta las pruebas automatizadas y los chequeos de seguridad
integrados al repositorio (trabajo individual: Rolando Calderón Carballo, A01840537).

### Pruebas (pytest)

```powershell
python -m pytest tests -v
```

**Resultado: 26 pruebas en verde** en 4 suites:

| Suite | Qué cubre |
|---|---|
| `tests/test_features_p1.py` | No-fuga temporal de los rezagos de P1 (3 pruebas, entrega 1) |
| `tests/test_dataset.py` | Ingesta: normalización de columnas con acentos y variantes históricas, fechas inválidas → NaT, minimización de datos personales, fallback utf-8 → latin-1, hash SHA-256 determinista y sensible a 1 byte (10 pruebas) |
| `tests/test_features_p2.py` | Etiqueta de riesgo de P2 con viajes sintéticos: `sin_bicis` / `sin_anclajes` / `normal`; casos límite: capacidad 0 descartada, filas sin ventana futura descartadas, horizonte fuera de [1, 24) lanza `ValueError` (6 pruebas) |
| `tests/test_reproducibilidad.py` | Misma semilla ⇒ predicciones idénticas en P1 (HGB y Poisson), P2 (probabilidades) y P3 (KMeans), más control negativo con semilla distinta (5 pruebas) |

### Validación de datos (DeepChecks)

```powershell
mlflow run . -e validate --env-manager local
# equivalente: python -m mibici.data_validation --sample 50000
```

Corre dos suites sobre la tabla de modelado de P1 y deja reportes HTML en
`reports/validacion/` (excluidos de git por tamaño):

- **`data_integrity` sobre TRAIN: 12/12 chequeos aprobados** (sin duplicados,
  nulos mezclados, columnas constantes ni conflictos de tipo).
- **`train_test_validation` TRAIN vs TEST: 11/12.** El único fallo es
  *Feature Drift* en `dias_activa` (KS ≈ 0.99), y es esperado: la feature
  cuenta días desde la primera actividad de cada estación, por lo que en TEST
  (meses posteriores) su distribución se desplaza por construcción. Es deriva
  determinística de una feature temporal, no un problema de calidad de datos.

Notas de compatibilidad (documentadas en `mibici/data_validation.py`):
DeepChecks 0.19.1 requiere tres ajustes con scikit-learn ≥ 1.8 / NumPy 2 /
Python ≥ 3.12 — reinscribir el scorer `max_error`, restaurar el alias
`np.Inf` y `setuptools<81` (provee `pkg_resources`).

### Auditoría de dependencias (pip-audit)

```powershell
pip-audit
```

- **Antes:** 86 vulnerabilidades conocidas en 12 paquetes (todas transitivas,
  salvo mlflow 3.17.1 → CVE-2026-53995).
- **Remediación:** actualización de mlflow a 3.17.2 y de las dependencias
  transitivas afectadas (aiohttp, certifi, filelock, gitpython, multidict,
  opentelemetry, pillow, pyasn1, tornado, werkzeug).
- **Después: 3 hallazgos, todos en setuptools 80.10.2 — riesgo aceptado.**
  El fix pide setuptools ≥ 83, pero DeepChecks 0.19.1 depende de
  `pkg_resources`, retirado de setuptools ≥ 81. Al ser una herramienta de
  desarrollo local sin servicios expuestos, se acepta el riesgo y se
  re-evaluará cuando DeepChecks publique soporte para setuptools nuevos.

### Análisis estático de seguridad (bandit)

```powershell
bandit -r mibici
```

**Resultado: 0 hallazgos Medium/High y 13 Low en 2,926 líneas.** Los 13 son el
mismo patrón en `p1_regresion.py`, `p2_clasificacion.py` y `p3_clustering.py`:
las llamadas a `git rev-parse HEAD` / `git status` que capturan el commit para
el linaje de MLflow (reglas B404/B603/B607, CWE-78). Se aceptan sin cambios:
el comando es una lista fija de argumentos, sin entrada del usuario y sin
`shell=True`, que es exactamente la forma segura que bandit recomienda.

### Higiene del repositorio

`.gitignore` excluye credenciales y datos (`.env`, `data/`, `mlflow.db`,
`mlruns/`, `mlartifacts/`, modelos serializados) y los reportes pesados de
validación (`reports/validacion/`). Las versiones de todas las herramientas
quedan pineadas en `requirements.txt`.