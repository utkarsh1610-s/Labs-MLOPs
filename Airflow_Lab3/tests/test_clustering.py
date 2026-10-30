"""Unit tests for the clustering steps. No Airflow needed, so these run anywhere."""
import json

import pandas as pd
import pytest

from conftest import DAGS_DIR
from src import clustering as cl

WINE_CSV = DAGS_DIR / "data" / "wine.csv"
NEW_CSV = DAGS_DIR / "data" / "new_samples.csv"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Run the steps up to evaluation once and share the results across tests."""
    run_dir = tmp_path_factory.mktemp("run")
    data = cl.load_and_validate(WINE_CSV, run_dir)
    prep = cl.preprocess(data["csv_path"], data["run_dir"])
    results = [cl.fit_kmeans(prep["scaled_path"], k, run_dir) for k in cl.candidate_ks(1, 10)]
    choice = cl.select_k(results)
    evaluation = cl.evaluate(choice, data["csv_path"], prep["scaled_path"])
    return {"data": data, "prep": prep, "results": results, "choice": choice, "evaluation": evaluation}


# ---------- data validation ----------

def test_dataset_files_are_as_expected():
    wine, new = pd.read_csv(WINE_CSV), pd.read_csv(NEW_CSV)
    assert len(wine) == 168 and len(new) == 10
    assert set(cl.FEATURES) <= set(wine.columns)
    assert set(wine[cl.LABEL]) == {0, 1, 2}


def test_validation_accepts_good_data(tmp_path):
    info = cl.load_and_validate(WINE_CSV, tmp_path / "run")
    assert info["n_rows"] == 168
    assert info["n_features"] == 13
    assert (tmp_path / "run").is_dir()


@pytest.mark.parametrize(
    "corrupt, message",
    [
        (lambda df: df.drop(columns=["alcohol"]), "Missing columns"),
        (lambda df: df.assign(alcohol=[None] + df["alcohol"].tolist()[1:]), "Null values"),
        (lambda df: df.head(10), "need at least"),
        (lambda df: df.assign(hue="high"), "Non-numeric"),
    ],
)
def test_validation_rejects_bad_data(tmp_path, corrupt, message):
    bad = tmp_path / "bad.csv"
    corrupt(pd.read_csv(WINE_CSV)).to_csv(bad, index=False)
    with pytest.raises(ValueError, match=message):
        cl.load_and_validate(bad, tmp_path / "run")


# ---------- k range and training ----------

def test_candidate_ks():
    assert cl.candidate_ks(1, 5) == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        cl.candidate_ks(5, 5)
    with pytest.raises(ValueError):
        cl.candidate_ks(0, 5)


def test_fit_kmeans_outputs(trained):
    by_k = {r["k"]: r for r in trained["results"]}
    assert by_k[1]["silhouette"] is None  # undefined for one cluster
    assert all(-1 <= r["silhouette"] <= 1 for k, r in by_k.items() if k > 1)
    # inertia always drops as k grows
    inertias = [by_k[k]["inertia"] for k in sorted(by_k)]
    assert inertias == sorted(inertias, reverse=True)


# ---------- selection and evaluation ----------

def test_elbow_finds_the_three_cultivars(trained):
    assert trained["choice"]["selection_method"] == "elbow"
    assert trained["choice"]["k"] == 3


def test_select_k_falls_back_to_silhouette_without_an_elbow():
    # a perfectly straight inertia line has no elbow
    results = [
        {"k": k, "inertia": 100.0 - 10 * k, "silhouette": s, "model_path": f"m{k}"}
        for k, s in [(2, 0.2), (3, 0.5), (4, 0.3), (5, 0.1)]
    ]
    choice = cl.select_k(results)
    assert choice["selection_method"] == "silhouette_fallback"
    assert choice["k"] == 3


def test_evaluation_matches_true_cultivars(trained):
    ev = trained["evaluation"]
    assert ev["silhouette"] > 0.25
    assert ev["adjusted_rand_index"] > 0.8
    assert sum(ev["cluster_sizes"].values()) == 168


# ---------- promote / review / scoring ----------

def test_promote_then_score_and_archive(trained, tmp_path):
    models = tmp_path / "models"
    assert cl.score_new_samples(models, NEW_CSV, tmp_path / "p.csv") is None  # nothing in production yet

    cl.promote(trained["evaluation"], trained["prep"]["scaler_path"], models, "run_1")
    scored = cl.score_new_samples(models, NEW_CSV, tmp_path / "p.csv")
    assert scored["model_run_id"] == "run_1"
    assert scored["n_samples"] == 10
    assert len(pd.read_csv(tmp_path / "p.csv")) == 10
    assert scored["adjusted_rand_index"] > 0.7

    # promoting again keeps the old production model in the archive
    cl.promote(trained["evaluation"], trained["prep"]["scaler_path"], models, "run_2")
    assert (models / "archive" / "run_1" / cl.PRODUCTION_MODEL).exists()
    assert json.loads((models / cl.PRODUCTION_META).read_text())["run_id"] == "run_2"


def test_flag_for_review_leaves_production_alone(trained, tmp_path):
    models = tmp_path / "models"
    record = cl.flag_for_review(trained["evaluation"], 0.9, tmp_path / "review", "run_x")
    assert "below threshold 0.9" in record["reason"]
    assert (tmp_path / "review" / "run_x.json").exists()
    assert not models.exists()


def test_safe_name():
    assert cl.safe_name("manual__2026-10-29T10:00:00+00:00") == "manual__2026-10-29T10_00_00_00_00"
