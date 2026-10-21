import json

import numpy as np
import pytest

from calibrate_model import calibrate
from data import N_FEATURES, make_dataset, metrics_path, model_path, seed_from_timestamp
from evaluate_model import evaluate
from train_model import train

TIMESTAMP = "20261020120000"


def test_dataset_is_deterministic_per_seed():
    a = make_dataset(seed_from_timestamp(TIMESTAMP))
    b = make_dataset(seed_from_timestamp(TIMESTAMP))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("timestamp", ["20260101000000", "20261020120000", "20271231235959"])
def test_dataset_shapes_and_split(timestamp):
    X_train, X_calib, X_test, y_train, y_calib, y_test = make_dataset(seed_from_timestamp(timestamp))
    total = len(y_train) + len(y_calib) + len(y_test)
    assert 500 <= total <= 2000
    assert X_train.shape[1] == X_calib.shape[1] == X_test.shape[1] == N_FEATURES
    # roughly 60 / 20 / 20
    assert abs(len(y_train) / total - 0.6) < 0.02
    assert abs(len(y_test) / total - 0.2) < 0.02
    # both classes present in every split
    for y in (y_train, y_calib, y_test):
        assert set(np.unique(y)) == {0, 1}


def test_full_pipeline(tmp_path):
    models_dir, metrics_dir = tmp_path / "models", tmp_path / "metrics"

    train(TIMESTAMP, models_dir)
    assert model_path(TIMESTAMP, models_dir).exists()

    metrics = evaluate(TIMESTAMP, models_dir, metrics_dir)
    assert 0.0 <= metrics["model"]["f1_score"] <= 1.0
    # sanity floor: the synthetic task is easy, a working model should beat chance comfortably
    assert metrics["model"]["accuracy"] > 0.7

    result = calibrate(TIMESTAMP, "sigmoid", models_dir, metrics_dir)
    assert model_path(TIMESTAMP, models_dir, calibrated=True).exists()
    assert 0.0 <= result["brier_after"] <= 1.0

    saved = json.loads(metrics_path(TIMESTAMP, metrics_dir).read_text())
    assert {"model", "calibration"} <= saved.keys()


def test_evaluate_without_model_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate(TIMESTAMP, tmp_path / "models", tmp_path / "metrics")
