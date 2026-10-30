"""
### Airflow Lab 3 – Wine clustering with parallel training and a quality gate

1. **load_and_validate**: read `wine.csv` and fail fast on missing columns, nulls or too few rows
2. **preprocess**: standardise the 13 chemical features
3. **candidate_ks**: build the list of k values from the run's params
4. **train_kmeans**: one mapped task **per k**, all running in parallel
5. **select_k**: elbow method on the inertia curve (falls back to best silhouette)
6. **evaluate**: silhouette score plus agreement with the true cultivars (ARI)
7. **quality_gate**: branches on the silhouette threshold
    - pass → **promote_model**: becomes the production model; the previous one is archived
    - fail → **flag_for_review**: written to `review/`; production is left untouched
8. **score_new_samples**: clusters the held-out wines with whatever model is in production
9. **write_run_report**: one JSON summary per run

Trigger with **Trigger DAG ▸ Advanced** to change `k_min`, `k_max` or `silhouette_threshold`.
"""
from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.sdk import Param, dag, get_current_context, task

from src import clustering as cl

DATA_DIR = Path(__file__).resolve().parent / "data"
WORK_DIR = Path(os.environ.get("LAB3_WORKING_DIR", "/opt/airflow/working_data"))
MODELS_DIR = WORK_DIR / "models"
REVIEW_DIR = WORK_DIR / "review"
REPORTS_DIR = WORK_DIR / "reports"


def _run_id() -> str:
    return get_current_context()["run_id"]


def _params() -> dict:
    return get_current_context()["params"]


@dag(
    dag_id="Airflow_Lab3_Wine_Clustering",
    description="Parallel KMeans training on the Wine dataset with an elbow-based k and a quality gate",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 10, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "Utkarsh Saraogi",
        "retries": 1,
        "retry_delay": timedelta(seconds=30),
    },
    params={
        "k_min": Param(1, type="integer", minimum=1, description="Smallest k to try"),
        "k_max": Param(10, type="integer", minimum=2, maximum=20, description="Largest k to try"),
        "silhouette_threshold": Param(
            0.25, type="number", minimum=-1, maximum=1,
            description="Minimum silhouette score needed to promote the model",
        ),
    },
    tags=["lab3", "clustering", "wine"],
    doc_md=__doc__,
)
def wine_clustering():
    @task
    def load_and_validate() -> dict:
        run_dir = WORK_DIR / "runs" / cl.safe_name(_run_id())
        return cl.load_and_validate(DATA_DIR / "wine.csv", run_dir)

    @task
    def preprocess(data: dict) -> dict:
        return cl.preprocess(data["csv_path"], data["run_dir"])

    @task
    def candidate_ks() -> list[int]:
        p = _params()
        return cl.candidate_ks(int(p["k_min"]), int(p["k_max"]))

    # map_index_template labels each mapped task "k=3" etc. in the UI instead of 0, 1, 2...
    @task(map_index_template="{{ k_label }}")
    def train_kmeans(prep: dict, k: int) -> dict:
        get_current_context()["k_label"] = f"k={k}"
        return cl.fit_kmeans(prep["scaled_path"], k, prep["run_dir"])

    @task
    def select_k(results: list[dict]) -> dict:
        return cl.select_k(list(results))

    @task
    def evaluate(choice: dict, data: dict, prep: dict) -> dict:
        result = cl.evaluate(choice, data["csv_path"], prep["scaled_path"])
        print(json.dumps(result, indent=2))
        return result

    @task.branch
    def quality_gate(evaluation: dict) -> str:
        threshold = float(_params()["silhouette_threshold"])
        if evaluation["silhouette"] is not None and evaluation["silhouette"] >= threshold:
            print(f"PASS: silhouette {evaluation['silhouette']:.3f} >= {threshold}")
            return "promote_model"
        print(f"FAIL: silhouette {evaluation['silhouette']} < {threshold}")
        return "flag_for_review"

    @task
    def promote_model(evaluation: dict, prep: dict) -> dict:
        return cl.promote(evaluation, prep["scaler_path"], MODELS_DIR, _run_id())

    @task
    def flag_for_review(evaluation: dict) -> dict:
        return cl.flag_for_review(evaluation, float(_params()["silhouette_threshold"]), REVIEW_DIR, _run_id())

    # Runs after either branch. Uses the production model on disk, so a rejected
    # model never replaces it: new data is still scored with the last good model.
    @task(trigger_rule="none_failed_min_one_success")
    def score_new_samples(data: dict) -> dict | None:
        out = Path(data["run_dir"]) / "new_sample_predictions.csv"
        result = cl.score_new_samples(MODELS_DIR, DATA_DIR / "new_samples.csv", out)
        if result is None:
            print("No production model yet (first run was rejected); nothing to score.")
        return result

    @task(trigger_rule="none_failed_min_one_success")
    def write_run_report(evaluation: dict, scored: dict | None) -> str:
        ti = get_current_context()["ti"]
        promoted = ti.xcom_pull(task_ids="promote_model")
        flagged = ti.xcom_pull(task_ids="flag_for_review")
        report = {
            "run_id": _run_id(),
            "params": dict(_params()),
            "decision": "promoted" if promoted else "flagged_for_review",
            "evaluation": evaluation,
            "new_samples": scored,
        }
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORTS_DIR / f"{cl.safe_name(_run_id())}.json"
        out.write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps(report, indent=2, default=str))
        if flagged:
            print(f"Model flagged for review: {flagged['review_file']}")
        return str(out)

    data = load_and_validate()
    prep = preprocess(data)
    results = train_kmeans.partial(prep=prep).expand(k=candidate_ks())
    choice = select_k(results)
    evaluation = evaluate(choice, data, prep)

    gate = quality_gate(evaluation)
    promoted = promote_model(evaluation, prep)
    flagged = flag_for_review(evaluation)
    gate >> [promoted, flagged]

    scored = score_new_samples(data)
    [promoted, flagged] >> scored
    write_run_report(evaluation, scored)


wine_clustering()
