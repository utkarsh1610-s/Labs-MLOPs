"""Train a RandomForest on the synthetic dataset and save a timestamp-versioned model."""
import argparse
import os
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from data import LAB_DIR, MODELS_DIR, make_dataset, model_path, seed_from_timestamp

PARAMS = {"n_estimators": 100, "max_depth": 8, "random_state": 0}


def log_to_mlflow(timestamp: str, n_train: int, train_metrics: dict) -> None:
    """Record the run in a local MLflow store. Skipped when MLFLOW_DISABLE=1 (used by tests)."""
    if os.environ.get("MLFLOW_DISABLE") == "1":
        return
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{LAB_DIR / 'mlflow.db'}")
    mlflow.set_experiment("github_lab2_random_forest")
    with mlflow.start_run(run_name=f"train_{timestamp}"):
        mlflow.log_params({**PARAMS, "timestamp": timestamp, "n_train": n_train})
        mlflow.log_metrics(train_metrics)


def train(timestamp: str, models_dir: Path = MODELS_DIR) -> Path:
    X_train, _, _, y_train, _, _ = make_dataset(seed_from_timestamp(timestamp))

    model = RandomForestClassifier(**PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    train_metrics = {
        "train_accuracy": accuracy_score(y_train, y_pred),
        "train_f1": f1_score(y_train, y_pred),
    }
    log_to_mlflow(timestamp, len(y_train), train_metrics)

    out = model_path(timestamp, models_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f"Trained on {len(y_train)} samples, saved model to {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Run timestamp from GitHub Actions")
    args = parser.parse_args()
    train(args.timestamp)
