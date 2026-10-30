"""DAG integrity tests: does the DAG load, and is it wired the way we expect?

Needs Airflow, so run these inside the container:
    docker compose run --rm airflow-cli bash -c "cd /opt/airflow && pytest tests -v"
"""
import pytest

pytest.importorskip("airflow")

from conftest import DAGS_DIR  # noqa: E402

DAG_ID = "Airflow_Lab3_Wine_Clustering"


@pytest.fixture(scope="module")
def dagbag():
    from airflow.dag_processing.dagbag import DagBag

    # Pointing at dags/ only; example DAGs are already off via the compose file
    return DagBag(dag_folder=str(DAGS_DIR))


@pytest.fixture(scope="module")
def dag(dagbag):
    return dagbag.get_dag(DAG_ID)


def test_no_import_errors(dagbag):
    assert dagbag.import_errors == {}


def test_dag_is_loaded(dag):
    assert dag is not None


def test_all_tasks_present(dag):
    assert set(dag.task_ids) == {
        "load_and_validate", "preprocess", "candidate_ks", "train_kmeans", "select_k",
        "evaluate", "quality_gate", "promote_model", "flag_for_review",
        "score_new_samples", "write_run_report",
    }


def test_main_chain(dag):
    def downstream(task_id):
        return set(dag.get_task(task_id).downstream_task_ids)

    assert "preprocess" in downstream("load_and_validate")
    assert "train_kmeans" in downstream("preprocess")
    assert "train_kmeans" in downstream("candidate_ks")
    assert "select_k" in downstream("train_kmeans")
    assert "evaluate" in downstream("select_k")
    assert "quality_gate" in downstream("evaluate")


def test_quality_gate_branches_to_promote_or_review(dag):
    gate = dag.get_task("quality_gate")
    assert {"promote_model", "flag_for_review"} <= set(gate.downstream_task_ids)
    assert {"promote_model", "flag_for_review"} <= set(dag.get_task("score_new_samples").upstream_task_ids)


def test_tasks_after_the_branch_run_whichever_side_was_taken(dag):
    for task_id in ("score_new_samples", "write_run_report"):
        assert dag.get_task(task_id).trigger_rule == "none_failed_min_one_success"


def test_train_kmeans_is_dynamically_mapped(dag):
    task = dag.get_task("train_kmeans")
    # @task + .expand() gives a DecoratedMappedOperator, a MappedOperator subclass
    assert "MappedOperator" in {cls.__name__ for cls in type(task).__mro__}


def test_schedule_retries_and_params(dag):
    assert dag.catchup is False
    assert dag.max_active_runs == 1
    assert dag.default_args["retries"] == 1
    assert dag.params["k_min"] == 1
    assert dag.params["k_max"] == 10
    assert dag.params["silhouette_threshold"] == 0.25
