"""Calibrate a trained model's probabilities and record the before/after Brier score.

Calibration is fit on the calibration split (never seen in training) and judged
on the test split, so the improvement reported is honest.
"""
import argparse
import json
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from data import METRICS_DIR, MODELS_DIR, make_dataset, metrics_path, model_path, seed_from_timestamp
from evaluate_model import score


def calibrate(
    timestamp: str,
    method: str = "sigmoid",
    models_dir: Path = MODELS_DIR,
    metrics_dir: Path = METRICS_DIR,
) -> dict:
    model = joblib.load(model_path(timestamp, models_dir))
    _, X_calib, X_test, _, y_calib, y_test = make_dataset(seed_from_timestamp(timestamp))

    # FrozenEstimator keeps the already-trained forest as-is; only the
    # probability mapping (Platt sigmoid or isotonic) is learned here.
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    calibrated.fit(X_calib, y_calib)

    out_model = model_path(timestamp, models_dir, calibrated=True)
    joblib.dump(calibrated, out_model)

    before = score(model, X_test, y_test)
    after = score(calibrated, X_test, y_test)
    result = {
        "method": method,
        "n_calib": len(y_calib),
        "brier_before": before["brier_score"],
        "brier_after": after["brier_score"],
        "brier_improvement": round(before["brier_score"] - after["brier_score"], 4),
        "calibrated_model": after,
    }

    # Append to the metrics file written by evaluate_model.py
    out_metrics = metrics_path(timestamp, metrics_dir)
    metrics = json.loads(out_metrics.read_text()) if out_metrics.exists() else {"timestamp": timestamp}
    metrics["calibration"] = result
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, indent=4))

    print(f"Calibrated ({method}): Brier {before['brier_score']} -> {after['brier_score']}, saved {out_model}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Run timestamp from GitHub Actions")
    parser.add_argument("--method", choices=["sigmoid", "isotonic"], default="sigmoid")
    args = parser.parse_args()
    calibrate(args.timestamp, args.method)
