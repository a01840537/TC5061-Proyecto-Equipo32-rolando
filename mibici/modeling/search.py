"""Punto de entrada ``search``: búsqueda de hiperparámetros con *nested runs*.

Una corrida padre por búsqueda (stage_of_work=search) y una corrida hija por
combinación, todas en el experimento común. La rejilla por pregunta es corta y
deliberada: el objetivo es la comparación organizada, no una búsqueda masiva.

    mlflow run . -e search -P question=p2 -P model=hgb \
        --env-manager local --experiment-name mibici-equipo32
"""

import mlflow
import typer
from loguru import logger

from mibici.modeling.train import (
    configurar_tracking,
    contexto_comun,
    correr_p1,
    correr_p2,
    correr_p3,
)
from mibici.config import TEAM_ID

app = typer.Typer()

# rejillas por (pregunta, modelo): lista de dicts de hiperparámetros
REJILLAS = {
    ("p1", "hgb"): [
        {"max_iter": 150, "learning_rate": 0.1},
        {"max_iter": 300, "learning_rate": 0.06},
        {"max_iter": 500, "learning_rate": 0.03},
        {"max_iter": 300, "learning_rate": 0.06, "max_depth": 6},
    ],
    ("p2", "hgb"): [
        {"max_iter": 100, "learning_rate": 0.1},
        {"max_iter": 200, "learning_rate": 0.1},
        {"max_iter": 200, "learning_rate": 0.05, "max_depth": 8},
        {"max_iter": 400, "learning_rate": 0.05},
    ],
    ("p3", "kmeans"): [{"k": 3}, {"k": 4}, {"k": 5}, {"k": 6}],
}

# métrica que ordena cada búsqueda y su sentido
SELECCION = {"p1": ("test_mae", min), "p2": ("test_f1", max), "p3": ("test_silhouette", max)}


@app.command()
def main(
    question: str = typer.Option("p2", "--question"),
    model: str = typer.Option("hgb", "--model"),
):
    configurar_tracking()
    contexto = contexto_comun()
    question = question.lower()
    rejilla = REJILLAS.get((question, model))
    if rejilla is None:
        raise typer.BadParameter(f"No hay rejilla para ({question}, {model})")
    metrica, mejor_fn = SELECCION[question]

    with mlflow.start_run(run_name=f"{question}_search_{model}") as padre:
        mlflow.set_tags(
            {
                "equipo": TEAM_ID,
                "question": question.upper(),
                "problema": question.upper(),
                "model_family": model,
                "stage_of_work": "search",
                "etapa": "search",
                "git_commit": contexto["git_commit"],
                "data_version": contexto["data_version"],
            }
        )
        mlflow.log_param("combinaciones", len(rejilla))
        resultados = []
        for i, combo in enumerate(rejilla, start=1):
            nombre = f"{question}_{model}_" + "_".join(f"{k}{v}" for k, v in combo.items())
            logger.info(f"[{i}/{len(rejilla)}] {nombre}")
            if question == "p1":
                run_id, m = correr_p1(
                    model, combo.get("max_iter", -1), combo.get("learning_rate", -1),
                    combo.get("max_depth", -1), contexto, nested=True, run_name=nombre,
                )
            elif question == "p2":
                run_id, m = correr_p2(
                    model, combo.get("max_iter", -1), combo.get("learning_rate", -1),
                    combo.get("max_depth", -1), contexto, nested=True, run_name=nombre,
                )
            else:
                run_id, m = correr_p3(model, combo.get("k", -1), contexto, nested=True, run_name=nombre)
            resultados.append((run_id, combo, m.get(metrica)))

        validos = [r for r in resultados if r[2] is not None]
        mejor = mejor_fn(validos, key=lambda r: r[2])
        mlflow.log_params({f"mejor_{k}": v for k, v in mejor[1].items()})
        mlflow.log_metric(f"mejor_{metrica}", mejor[2])
        mlflow.set_tag("mejor_run_id", mejor[0])
        logger.success(f"Mejor combinación: {mejor[1]} con {metrica}={mejor[2]:.4f} ({mejor[0]})")


if __name__ == "__main__":
    app()
