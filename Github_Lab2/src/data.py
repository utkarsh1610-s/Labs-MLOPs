"""Synthetic dataset shared by the train, evaluate and calibrate steps.

The dataset is regenerated from a seed derived from the run timestamp, so every
script in one pipeline run sees exactly the same train / calibration / test
split without having to pass data files between steps.
"""
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

LAB_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = LAB_DIR / "models"
METRICS_DIR = LAB_DIR / "metrics"

N_FEATURES = 6


def seed_from_timestamp(timestamp: str) -> int:
    """Turn a run timestamp like '20261020153000' into a valid NumPy seed."""
    return int(timestamp) % (2**32)


def make_dataset(seed: int):
    """Return (X_train, X_calib, X_test, y_train, y_calib, y_test).

    Split is 60% train, 20% calibration, 20% test. The sample count varies
    between runs (500-2000) to mimic new data arriving, but it never drops
    low enough to break training, unlike the original randint(0, 2000).
    """
    rng = np.random.default_rng(seed)
    n_samples = int(rng.integers(500, 2001))

    X, y = make_classification(
        n_samples=n_samples,
        n_features=N_FEATURES,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        flip_y=0.05,
        random_state=seed,
        shuffle=True,
    )

    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest
    )
    return X_train, X_calib, X_test, y_train, y_calib, y_test


def model_path(timestamp: str, models_dir: Path = MODELS_DIR, calibrated: bool = False) -> Path:
    suffix = "_calibrated" if calibrated else ""
    return Path(models_dir) / f"model_{timestamp}{suffix}.joblib"


def metrics_path(timestamp: str, metrics_dir: Path = METRICS_DIR) -> Path:
    return Path(metrics_dir) / f"{timestamp}_metrics.json"
