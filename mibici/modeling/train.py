"""Punto de entrada ``train`` del MLflow Project (entrega 2).

Entrena una corrida para una pregunta (p1/p2/p3) con un modelo y sus
hiperparámetros, reutilizando la preparación de datos, la evaluación y las
gráficas de los pipelines de la entrega 1, y registra en MLflow:

- tags para filtrar (question, model_family, stage_of_work, git_commit, data_version);
- parámetros reales del modelo (``get_params``) y de la partición;
- el modelo con firma y ejemplo de entrada (``log_model``);
- la versión de los datos: manifiesto como artefacto y ``mlflow.log_input``;
- gráficas y configuración organizadas en subcarpetas (plots/, data/, config/).

Convención del centinela: un hiperparámetro en -1 significa "usar el valor
por defecto de config.py / del modelo". Así el MLproject no repite valores.

Ejecución (siempre vía MLflow Project)::

    mlflow run . -e train -P question=p2 -P model=hgb -P learning_rate=0.05 \
        --env-manager local --experiment-name mibici-equipo32
"""

import json
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from mibici.config import (
    ESTACIONES_PARQUET,
    MANIFEST_PATH,
    P1_BASELINE_LAG_DIAS,
    P1_FERIADOS,
    P1_HORIZONTE_DIAS,
    P1_LAGS_DIAS,
    P1_MLFLOW_EXPERIMENT,
    P1_OUTPUT_DIR,
    P1_TEST_END,
    P1_TEST_START,
    P1_TRAIN_START,
    P1_VAL_START,
    P1_VENTANAS_MEDIA_DIAS,
    P2_FRACCION_CAPACIDAD,
    P2_HORIZONTE_HORAS,
    P2_OUTPUT_DIR,
    P2_TEST_END,
    P2_TEST_START,
    P2_TRAIN_START,
    P2_VAL_START,
    P3_MAX_INACTIVITY_DAYS,
    P3_MIN_HISTORY_DAYS,
    P3_N_CLUSTERS,
    P3_N_INIT,
    P3_OUTPUT_DIR,
    P3_TEST_END,
    P3_TEST_START,
    P3_TRAIN_END,
    P3_TRAIN_START,
    PROJ_ROOT,
    SEED,
    TEAM_ID,
    VIAJES_PARQUET,
)
from mibici.features import (
    P1_FEATURES,
    P2_FEATURES,
    P3_MODEL_FEATURES,
    p1_agregar_features,
    p1_construir_panel_diario,
    p2_agregar_etiqueta,
    p2_agregar_features,
    p2_ajustar_historico,
    p2_aplicar_historico,
    p2_construir_panel,
    p2_historico_pasado,
    p3_construir_features,
    p3_filtrar_viajes,
)
from mibici.modeling import p1_regresion as p1
from mibici.modeling import p2_clasificacion as p2
from mibici.modeling.p3_clustering import preparar_features as p3_preparar_features

app = typer.Typer()

SKOPS_TRUSTED = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]

# etiquetas y familias por pregunta
BASELINES = {"p1": "naive", "p2": "logreg", "p3": "kmeans2"}


def configurar_tracking() -> None:
    """URI desde la variable de entorno (servidor local); respaldo: SQLite directo."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{PROJ_ROOT / 'mlflow.db'}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(P1_MLFLOW_EXPERIMENT)  # mibici-equipo32, común a las 3 preguntas


def contexto_comun() -> dict:
    commit, sucio = p1.obtener_git_commit()
    if sucio:
        logger.warning("Hay cambios sin commit: el run no será reproducible desde git_commit")
    return {
        "git_commit": commit,
        "git_sucio": sucio,
        "data_version": json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["data_version"],
    }


def _sentinela(valor: float, defecto):
    """-1 => usa el valor por defecto."""
    return defecto if valor is None or float(valor) < 0 else valor


def _params_limpios(modelo) -> dict:
    """get_params() del estimador final, serializable y sin ruido."""
    est = modelo
    if hasattr(modelo, "named_steps"):  # Pipeline: reporta el estimador final
        est = list(modelo.named_steps.values())[-1]
    out = {}
    for k, v in est.get_params(deep=False).items():
        out[k] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
    return out


def registrar_run(
    *,
    run_name: str,
    question: str,
    model_family: str,
    stage: str,
    params: dict,
    metricas: dict,
    artefactos_plots: list,
    modelo=None,
    input_example=None,
    train_df: pd.DataFrame | None = None,
    contexto: dict,
    nested: bool = False,
) -> str:
    """Un run de MLflow con tags, params, métricas, artefactos y modelo con firma."""
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        mlflow.set_tags(
            {
                "equipo": TEAM_ID,
                "question": question.upper(),
                "problema": question.upper(),
                "model_family": model_family,
                "stage_of_work": stage,
                "etapa": stage,
                "git_commit": contexto["git_commit"],
                "git_sucio": str(contexto["git_sucio"]),
                "data_version": contexto["data_version"],
            }
        )
        mlflow.log_params({**params, "data_version": contexto["data_version"], "seed": SEED})
        mlflow.log_metrics(metricas)

        # datos: manifiesto siempre; log_input del conjunto de entrenamiento
        mlflow.log_artifact(str(MANIFEST_PATH), artifact_path="data")
        if train_df is not None:
            muestra = train_df if len(train_df) <= 200_000 else train_df.sample(
                200_000, random_state=SEED
            )
            ds = mlflow.data.from_pandas(
                muestra, source=str(VIAJES_PARQUET), name=f"{question}_train"
            )
            mlflow.log_input(ds, context="training")

        # configuración con la que corrió
        cfg = {
            "run_name": run_name,
            "question": question,
            "model_family": model_family,
            "stage_of_work": stage,
            "params": params,
            "seed": SEED,
            "data_version": contexto["data_version"],
            "git_commit": contexto["git_commit"],
        }
        ruta_cfg = PROJ_ROOT / "reports" / f"{run_name}_config.json"
        ruta_cfg.parent.mkdir(parents=True, exist_ok=True)
        ruta_cfg.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifact(str(ruta_cfg), artifact_path="config")

        for ruta in artefactos_plots:
            mlflow.log_artifact(str(ruta), artifact_path="plots")

        if modelo is not None:
            mlflow.sklearn.log_model(
                modelo,
                name="modelo",
                input_example=input_example,
                skops_trusted_types=SKOPS_TRUSTED,
            )
        return run.info.run_id


# ---------------------------------------------------------------- P1
def preparar_p1():
    viajes = p1.cargar_viajes(VIAJES_PARQUET)
    capacidad = p1.cargar_capacidad(ESTACIONES_PARQUET)
    panel = p1_construir_panel_diario(viajes, P1_TRAIN_START, P1_TEST_END)
    df = p1_agregar_features(
        panel, capacidad, P1_FERIADOS, P1_LAGS_DIAS, P1_VENTANAS_MEDIA_DIAS, P1_HORIZONTE_DIAS
    )
    train = p1.cortar(df, P1_TRAIN_START, P1_VAL_START).dropna(subset=P1_FEATURES)
    val = p1.cortar(df, P1_VAL_START, P1_TEST_START).dropna(subset=P1_FEATURES)
    test = p1.cortar(df, P1_TEST_START, P1_TEST_END).dropna(subset=P1_FEATURES)
    return train, val, test


def construir_p1(model: str, max_iter, learning_rate, max_depth):
    from sklearn.ensemble import HistGradientBoostingRegressor

    if model == "naive":
        return p1.BaselineNaiveSemanal(f"lag_{P1_BASELINE_LAG_DIAS}"), "baseline"
    if model == "poisson":
        return p1.construir_lineal(), "candidato"
    if model == "hgb":
        return (
            HistGradientBoostingRegressor(
                loss="poisson",
                max_iter=int(_sentinela(max_iter, 300)),
                learning_rate=float(_sentinela(learning_rate, 0.06)),
                max_depth=None if _sentinela(max_depth, -1) == -1 else int(max_depth),
                random_state=SEED,
            ),
            "candidato",
        )
    raise typer.BadParameter(f"Modelo P1 desconocido: {model} (naive|poisson|hgb)")


def correr_p1(model, max_iter, learning_rate, max_depth, contexto, nested=False, run_name=None):
    P1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test = preparar_p1()
    modelo, stage = construir_p1(model, max_iter, learning_rate, max_depth)
    x = lambda d: d[P1_FEATURES].astype("float64")
    modelo.fit(x(train), train["viajes"])
    pred = {n: modelo.predict(x(d)) for n, d in (("train", train), ("val", val), ("test", test))}
    metricas = {
        **p1.evaluar(train["viajes"], pred["train"], "train"),
        **p1.evaluar(val["viajes"], pred["val"], "val"),
        **p1.evaluar(test["viajes"], pred["test"], "test"),
    }
    base_test = test[f"lag_{P1_BASELINE_LAG_DIAS}"].to_numpy(dtype="float64")
    nombre = run_name or f"p1_{model}"
    plots = p1.guardar_graficas(nombre, test, pred["test"], base_test)
    es_baseline = model == BASELINES["p1"]
    run_id = registrar_run(
        run_name=nombre,
        question="p1",
        model_family=model,
        stage="baseline" if es_baseline else "candidato",
        params={
            **(
                _params_limpios(modelo)
                if model != "naive"
                else {"lag_dias": P1_BASELINE_LAG_DIAS}
            ),
            "modelo": type(modelo).__name__,
            "train_inicio": P1_TRAIN_START,
            "val_inicio": P1_VAL_START,
            "test_inicio": P1_TEST_START,
            "filas_train": len(train),
        },
        metricas=metricas,
        artefactos_plots=plots,
        modelo=None if model == "naive" else modelo,
        input_example=x(test).head(5),
        train_df=train[P1_FEATURES + ["viajes"]],
        contexto=contexto,
        nested=nested,
    )
    logger.success(f"{nombre}: test_mae={metricas['test_mae']:.3f} run_id={run_id}")
    return run_id, metricas


# ---------------------------------------------------------------- P2
def preparar_p2():
    viajes = p2.cargar_viajes(VIAJES_PARQUET)
    capacidad = p2.cargar_capacidad(ESTACIONES_PARQUET)
    panel = p2_construir_panel(viajes, P2_TRAIN_START, P2_TEST_END)
    df = p2_agregar_etiqueta(panel, P2_HORIZONTE_HORAS, capacidad, P2_FRACCION_CAPACIDAD)
    df = p2_agregar_features(df, P2_HORIZONTE_HORAS)
    train = p2.cortar(df, P2_TRAIN_START, P2_VAL_START)
    val = p2.cortar(df, P2_VAL_START, P2_TEST_START)
    test = p2.cortar(df, P2_TEST_START, P2_TEST_END)
    viajes_train = viajes.loc[viajes["inicio_viaje"] < pd.Timestamp(P2_VAL_START)]
    estaciones_train = pd.Index(viajes_train["origen_id"]).union(viajes_train["destino_id"])
    val = val.loc[val.index.get_level_values("estacion").isin(estaciones_train)]
    test = test.loc[test.index.get_level_values("estacion").isin(estaciones_train)]
    historico = p2_ajustar_historico(train)
    train = p2_historico_pasado(train)
    val, test = (p2_aplicar_historico(d, historico) for d in (val, test))
    train, val, test = (d.dropna(subset=P2_FEATURES) for d in (train, val, test))
    return train, val, test


def construir_p2(model: str, max_iter, learning_rate, max_depth):
    from sklearn.ensemble import HistGradientBoostingClassifier

    if model == "logreg":
        return p2.construir_baseline(), "baseline"
    if model == "hgb":
        return (
            HistGradientBoostingClassifier(
                class_weight="balanced",
                max_iter=int(_sentinela(max_iter, 200)),
                learning_rate=float(_sentinela(learning_rate, 0.1)),
                max_depth=None if _sentinela(max_depth, -1) == -1 else int(max_depth),
                random_state=SEED,
            ),
            "candidato",
        )
    raise typer.BadParameter(f"Modelo P2 desconocido: {model} (logreg|hgb)")


def correr_p2(model, max_iter, learning_rate, max_depth, contexto, nested=False, run_name=None):
    P2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test = preparar_p2()
    modelo, _ = construir_p2(model, max_iter, learning_rate, max_depth)
    x = lambda d: d[P2_FEATURES].astype("float64")
    modelo.fit(x(train), train["riesgo"])
    prob = {n: modelo.predict_proba(x(d))[:, 1] for n, d in (("train", train), ("val", val), ("test", test))}
    umbral = p2.elegir_umbral(val["riesgo"], prob["val"])
    metricas = {
        **p2.evaluar(train["riesgo"], prob["train"], umbral, "train", train["tipo_riesgo"]),
        **p2.evaluar(val["riesgo"], prob["val"], umbral, "val", val["tipo_riesgo"]),
        **p2.evaluar(test["riesgo"], prob["test"], umbral, "test", test["tipo_riesgo"]),
    }
    nombre = run_name or f"p2_{model}"
    plots = p2.guardar_graficas(nombre, test["riesgo"], prob["test"], umbral)
    es_baseline = model == BASELINES["p2"]
    run_id = registrar_run(
        run_name=nombre,
        question="p2",
        model_family=model,
        stage="baseline" if es_baseline else "candidato",
        params={
            **_params_limpios(modelo),
            "modelo": type(modelo).__name__,
            "umbral_decision": round(float(umbral), 4),
            "horizonte_horas": P2_HORIZONTE_HORAS,
            "fraccion_capacidad": P2_FRACCION_CAPACIDAD,
            "train_inicio": P2_TRAIN_START,
            "val_inicio": P2_VAL_START,
            "test_inicio": P2_TEST_START,
            "filas_train": len(train),
        },
        metricas=metricas,
        artefactos_plots=plots,
        modelo=modelo,
        input_example=x(test).head(5),
        train_df=train[P2_FEATURES + ["riesgo"]],
        contexto=contexto,
        nested=nested,
    )
    logger.success(f"{nombre}: test_f1={metricas['test_f1']:.3f} run_id={run_id}")
    return run_id, metricas


# ---------------------------------------------------------------- P3
def preparar_p3():
    viajes = pd.read_parquet(VIAJES_PARQUET)
    viajes["inicio_viaje"] = pd.to_datetime(viajes["inicio_viaje"])
    viajes["fin_viaje"] = pd.to_datetime(viajes["fin_viaje"])
    viajes = p3_filtrar_viajes(viajes, min_duration_seconds=60, max_duration_seconds=86400)
    train = viajes.loc[
        (viajes["inicio_viaje"] >= pd.Timestamp(P3_TRAIN_START))
        & (viajes["inicio_viaje"] <= pd.Timestamp(P3_TRAIN_END))
        & (viajes["fin_viaje"] < pd.Timestamp(P3_TEST_START))
    ]
    test = viajes.loc[
        (viajes["inicio_viaje"] >= pd.Timestamp(P3_TEST_START))
        & (viajes["fin_viaje"] < pd.Timestamp(P3_TEST_END))
    ]
    f_train, _ = p3_construir_features(
        train, fin_periodo=pd.Timestamp(P3_TRAIN_END),
        max_inactivity_days=P3_MAX_INACTIVITY_DAYS, min_history_days=P3_MIN_HISTORY_DAYS,
    )
    f_test, _ = p3_construir_features(
        test, fin_periodo=pd.Timestamp(P3_TEST_END) - pd.Timedelta(seconds=1),
        max_inactivity_days=P3_MAX_INACTIVITY_DAYS, min_history_days=P3_MIN_HISTORY_DAYS,
    )
    x_train, scaler = p3_preparar_features(f_train)
    x_test, _ = p3_preparar_features(f_test, scaler=scaler)
    return x_train, x_test


def grafica_p3(nombre: str, x_train: pd.DataFrame, labels: np.ndarray):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    P3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comp = PCA(n_components=2, random_state=SEED).fit_transform(x_train)
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(comp[:, 0], comp[:, 1], c=labels, cmap="tab10", s=22)
    ax.set_title(f"P3 · Estaciones en PCA 2D — {nombre}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(sc, label="cluster")
    ruta = P3_OUTPUT_DIR / f"{nombre}_pca.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    tam = pd.Series(labels).value_counts().sort_index().rename("estaciones")
    ruta_csv = P3_OUTPUT_DIR / f"{nombre}_tamanos.csv"
    tam.to_csv(ruta_csv)
    return [ruta, ruta_csv]


def correr_p3(model, k, contexto, nested=False, run_name=None):
    x_train, x_test = preparar_p3()
    k_ = int(_sentinela(k, P3_N_CLUSTERS))
    if model == "kmeans2":
        modelo, k_ = KMeans(n_clusters=2, random_state=SEED, n_init=P3_N_INIT), 2
    elif model == "kmeans":
        modelo = KMeans(n_clusters=k_, random_state=SEED, n_init=P3_N_INIT)
    elif model == "agglo":
        modelo = AgglomerativeClustering(n_clusters=k_, linkage="ward")
    else:
        raise typer.BadParameter(f"Modelo P3 desconocido: {model} (kmeans2|kmeans|agglo)")

    labels = modelo.fit_predict(x_train)
    metricas = {
        "train_silhouette": float(silhouette_score(x_train, labels)),
        "train_calinski_harabasz": float(calinski_harabasz_score(x_train, labels)),
        "train_davies_bouldin": float(davies_bouldin_score(x_train, labels)),
    }
    if hasattr(modelo, "predict"):  # estabilidad temporal, solo KMeans
        labels_test = modelo.predict(x_test)
        metricas["test_silhouette"] = float(silhouette_score(x_test, labels_test))
        comunes = x_train.index.intersection(x_test.index)
        if len(comunes) > 1:
            lt = pd.Series(labels, index=x_train.index).loc[comunes]
            le = pd.Series(labels_test, index=x_test.index).loc[comunes]
            metricas["estabilidad_ari"] = float(adjusted_rand_score(lt, le))

    nombre = run_name or (f"p3_{model}_k{k_}" if model != "kmeans2" else "p3_kmeans2")
    plots = grafica_p3(nombre, x_train, labels)
    es_baseline = model == BASELINES["p3"]
    run_id = registrar_run(
        run_name=nombre,
        question="p3",
        model_family=model,
        stage="baseline" if es_baseline else "candidato",
        params={
            **_params_limpios(modelo),
            "modelo": type(modelo).__name__,
            "k": k_,
            "estaciones_train": len(x_train),
            "train_inicio": P3_TRAIN_START[:10],
            "train_fin": P3_TRAIN_END[:10],
        },
        metricas=metricas,
        artefactos_plots=plots,
        modelo=modelo if hasattr(modelo, "predict") else None,
        input_example=x_train.head(5),
        train_df=x_train,
        contexto=contexto,
        nested=nested,
    )
    logger.success(f"{nombre}: train_silhouette={metricas['train_silhouette']:.3f} run_id={run_id}")
    return run_id, metricas


@app.command()
def main(
    question: str = typer.Option("p1", "--question"),
    model: str = typer.Option("baseline", "--model"),
    max_iter: float = typer.Option(-1, "--max-iter"),
    learning_rate: float = typer.Option(-1, "--learning-rate"),
    max_depth: float = typer.Option(-1, "--max-depth"),
    k: float = typer.Option(-1, "--k"),
):
    configurar_tracking()
    contexto = contexto_comun()
    question = question.lower()
    if model == "baseline":
        model = BASELINES[question]
    if question == "p1":
        correr_p1(model, max_iter, learning_rate, max_depth, contexto)
    elif question == "p2":
        correr_p2(model, max_iter, learning_rate, max_depth, contexto)
    elif question == "p3":
        correr_p3(model, k, contexto)
    else:
        raise typer.BadParameter("question debe ser p1, p2 o p3")


if __name__ == "__main__":
    app()
