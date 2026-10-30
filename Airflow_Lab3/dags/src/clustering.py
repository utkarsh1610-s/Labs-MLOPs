"""Wine clustering steps used by the Airflow DAG.

Every function takes and returns plain, JSON-safe values (paths, numbers,
dicts), so Airflow can pass them between tasks through XCom without pickling.
Large objects (data, models) are written to the run's working directory and
only their paths travel between tasks.

Nothing here imports Airflow, so the logic can be unit-tested on its own.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315", "proline",
]
LABEL = "cultivar"  # ground truth, never used for fitting
MIN_ROWS = 30
KMEANS_KWARGS = {"n_init": 10, "max_iter": 300, "random_state": 42}

PRODUCTION_MODEL = "wine_kmeans.joblib"
PRODUCTION_META = "wine_kmeans.json"


def load_and_validate(csv_path: str | Path, run_dir: str | Path) -> dict:
    """Read the CSV and fail fast if it isn't fit for training."""
    df = pd.read_csv(csv_path)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if len(df) < MIN_ROWS:
        raise ValueError(f"Only {len(df)} rows; need at least {MIN_ROWS}")
    nulls = df[FEATURES].isna().sum()
    if nulls.any():
        raise ValueError(f"Null values found: {nulls[nulls > 0].to_dict()}")
    non_numeric = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric columns: {non_numeric}")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "csv_path": str(csv_path),
        "run_dir": str(run_dir),
        "n_rows": len(df),
        "n_features": len(FEATURES),
        "has_labels": LABEL in df.columns,
    }


def preprocess(csv_path: str | Path, run_dir: str | Path) -> dict:
    """Standardise the features (wine measurements range from ~0.1 to ~1600)."""
    df = pd.read_csv(csv_path)
    scaler = StandardScaler().fit(df[FEATURES])

    run_dir = Path(run_dir)
    scaled_path = run_dir / "scaled.npy"
    scaler_path = run_dir / "scaler.joblib"
    np.save(scaled_path, scaler.transform(df[FEATURES]))
    joblib.dump(scaler, scaler_path)
    return {"scaled_path": str(scaled_path), "scaler_path": str(scaler_path), "run_dir": str(run_dir)}


def candidate_ks(k_min: int, k_max: int) -> list[int]:
    if not 1 <= k_min < k_max:
        raise ValueError(f"Need 1 <= k_min < k_max, got k_min={k_min}, k_max={k_max}")
    return list(range(k_min, k_max + 1))


def fit_kmeans(scaled_path: str | Path, k: int, run_dir: str | Path) -> dict:
    """Fit one KMeans model. The DAG runs one of these per k, in parallel."""
    X = np.load(scaled_path)
    model = KMeans(n_clusters=k, **KMEANS_KWARGS).fit(X)

    model_path = Path(run_dir) / f"kmeans_k{k}.joblib"
    joblib.dump(model, model_path)
    return {
        "k": k,
        "inertia": float(model.inertia_),
        # silhouette is undefined for a single cluster
        "silhouette": float(silhouette_score(X, model.labels_)) if k > 1 else None,
        "model_path": str(model_path),
    }


def select_k(results: list[dict]) -> dict:
    """Pick k with the elbow method; fall back to best silhouette if there's no clear elbow."""
    results = sorted(results, key=lambda r: r["k"])
    ks = [r["k"] for r in results]
    inertias = [r["inertia"] for r in results]

    elbow = KneeLocator(ks, inertias, curve="convex", direction="decreasing").elbow if len(ks) >= 3 else None
    if elbow is not None:
        method, k = "elbow", int(elbow)
    else:
        scored = [r for r in results if r["silhouette"] is not None]
        if not scored:
            raise ValueError("No elbow found and no silhouette scores to fall back on")
        method, k = "silhouette_fallback", max(scored, key=lambda r: r["silhouette"])["k"]

    chosen = next(r for r in results if r["k"] == k)
    return {**chosen, "selection_method": method, "inertia_by_k": dict(zip(map(str, ks), inertias))}


def evaluate(choice: dict, csv_path: str | Path, scaled_path: str | Path) -> dict:
    """Score the chosen model. ARI compares clusters to the true cultivars (1.0 = perfect match)."""
    model = joblib.load(choice["model_path"])
    labels = model.predict(np.load(scaled_path))
    df = pd.read_csv(csv_path)

    evaluation = {
        "k": choice["k"],
        "selection_method": choice["selection_method"],
        "silhouette": choice["silhouette"],
        "inertia": choice["inertia"],
        "cluster_sizes": {str(c): int(n) for c, n in zip(*np.unique(labels, return_counts=True))},
        "model_path": choice["model_path"],
    }
    if LABEL in df.columns:
        evaluation["adjusted_rand_index"] = float(adjusted_rand_score(df[LABEL], labels))
    return evaluation


def promote(evaluation: dict, scaler_path: str | Path, models_dir: str | Path, run_id: str) -> dict:
    """Bundle scaler + chosen KMeans into one pipeline and make it the production model.

    The previous production model is kept as a timestamped backup.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path, meta_path = models_dir / PRODUCTION_MODEL, models_dir / PRODUCTION_META

    if model_path.exists() and meta_path.exists():
        old = json.loads(meta_path.read_text())
        archive = models_dir / "archive" / safe_name(old.get("run_id", "unknown"))
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, archive / PRODUCTION_MODEL)
        shutil.copy2(meta_path, archive / PRODUCTION_META)

    pipeline = Pipeline([("scaler", joblib.load(scaler_path)), ("kmeans", joblib.load(evaluation["model_path"]))])
    joblib.dump(pipeline, model_path)
    meta = {"run_id": run_id, **{k: v for k, v in evaluation.items() if k != "model_path"}}
    meta_path.write_text(json.dumps(meta, indent=2))
    return {"production_model": str(model_path), **meta}


def flag_for_review(evaluation: dict, threshold: float, review_dir: str | Path, run_id: str) -> dict:
    """Record a rejected model so someone can look at it; production is left untouched."""
    review_dir = Path(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    out = review_dir / f"{safe_name(run_id)}.json"
    record = {
        "run_id": run_id,
        "reason": f"silhouette {evaluation['silhouette']:.3f} below threshold {threshold}",
        **evaluation,
    }
    out.write_text(json.dumps(record, indent=2))
    return {"review_file": str(out), **record}


def score_new_samples(models_dir: str | Path, new_csv: str | Path, out_path: str | Path) -> dict | None:
    """Assign new wines to clusters with whatever model is currently in production."""
    model_path = Path(models_dir) / PRODUCTION_MODEL
    if not model_path.exists():
        return None
    pipeline = joblib.load(model_path)
    meta = json.loads((Path(models_dir) / PRODUCTION_META).read_text())

    df = pd.read_csv(new_csv)
    df["cluster"] = pipeline.predict(df[FEATURES])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    result = {
        "model_run_id": meta["run_id"],
        "n_samples": len(df),
        "predictions_path": str(out_path),
        "cluster_counts": {str(c): int(n) for c, n in df["cluster"].value_counts().sort_index().items()},
    }
    if LABEL in df.columns:
        result["adjusted_rand_index"] = float(adjusted_rand_score(df[LABEL], df["cluster"]))
    return result


def safe_name(run_id: str) -> str:
    """Airflow run ids contain ':' and '+', which are awkward in file names."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in run_id)
