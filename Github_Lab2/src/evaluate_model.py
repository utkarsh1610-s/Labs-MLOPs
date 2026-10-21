"""Evaluate a trained model on the held-out test split and write a metrics JSON."""
import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score

from data import METRICS_DIR, MODELS_DIR, make_dataset, metrics_path, model_path, seed_from_timestamp


def score(model, X, y) -> dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "f1_score": round(f1_score(y, y_pred), 4),
        "roc_auc": round(roc_auc_score(y, y_prob), 4),
        "brier_score": round(brier_score_loss(y, y_prob), 4),
    }


def evaluate(timestamp: str, models_dir: Path = MODELS_DIR, metrics_dir: Path = METRICS_DIR) -> dict:
    src = model_path(timestamp, models_dir)
    if not src.exists():
        raise FileNotFoundError(f"No model for timestamp {timestamp} at {src}; run train_model.py first")
    model = joblib.load(src)

    _, _, X_test, _, _, y_test = make_dataset(seed_from_timestamp(timestamp))
    metrics = {"timestamp": timestamp, "n_test": len(y_test), "model": score(model, X_test, y_test)}

    out = metrics_path(timestamp, metrics_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=4))
    print(f"Test metrics: {metrics['model']} -> {out}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Run timestamp from GitHub Actions")
    args = parser.parse_args()
    evaluate(args.timestamp)
