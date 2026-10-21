# Github Lab 2 – Model Training, Versioning and Calibration with GitHub Actions

## Overview

This lab automates a small machine-learning pipeline with **GitHub Actions**. Every push to `main` that touches this lab triggers a workflow that:

1. runs the test suite, and stops if any test fails
2. trains a Random Forest classifier
3. evaluates it on held-out data
4. calibrates its predicted probabilities
5. commits the new model and its metrics back into the repo, versioned by timestamp

It is based on `Labs/Github_Labs/Lab2` from the MLOps (IE-7374) course repo. The changes I made, mainly to the CI/CD workflow, are listed in [Changes I made](#changes-i-made).

**Where things are**

| What | Path |
|---|---|
| CI/CD workflow | [`.github/workflows/github_lab2_retrain.yml`](../.github/workflows/github_lab2_retrain.yml), at the repo root because GitHub only reads workflows from there |
| Lab code, tests, outputs | `Github_Lab2/` (this folder) |

---

## 1. Prerequisites

| Tool | Version | Why it matters |
|---|---|---|
| Python | **3.11** (what CI uses, and what this lab was tested on) | Must be **3.11 or newer**: the current scikit-learn and numpy releases that `pip` installs no longer support 3.10 or older |
| pip | ships with Python | installs the dependencies |
| git | any recent version (tested with 2.50) | to clone the repo and push |
| GitHub account | – | only needed to run the CI/CD part (section 4) |

Python packages, installed in step 2 from `requirements.txt`:

| Package | Tested version | Note |
|---|---|---|
| scikit-learn | 1.9.1 | **must be 1.6 or newer**; `sklearn.frozen.FrozenEstimator` used for calibration was added in 1.6 |
| numpy | 2.4.6 | |
| joblib | 1.6.0 | saves and loads models |
| mlflow | 3.17.1 | local experiment tracking during training |
| pytest | 9.1.1 | test suite |

**Installing Python 3.11 if you don't have it**
- **macOS:** download the 3.11 installer from [python.org/downloads](https://www.python.org/downloads/), or `brew install python@3.11`
- **Windows:** download the 3.11 installer from [python.org/downloads](https://www.python.org/downloads/) and tick **"Add python.exe to PATH"**
- **Linux (Ubuntu/Debian):** `sudo apt install python3.11 python3.11-venv`

Check it worked:
```bash
python3.11 --version     # Windows: py -3.11 --version
git --version
```

---

## 2. Set up the environment

```bash
git clone https://github.com/utkarsh1610-s/Labs-MLOPs.git
cd Labs-MLOPs/Github_Lab2
```

Create and activate a virtual environment:

**macOS / Linux**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies (this takes a minute or two, mostly for MLflow):
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

All commands below assume you are inside `Github_Lab2/` with the virtual environment active. You should see `(.venv)` at the start of your prompt.

---

## 3. Run it locally

### 3.1 Run the tests

```bash
python -m pytest test -v
```

**Success looks like:**
```
test/test_pipeline.py::test_dataset_is_deterministic_per_seed PASSED          [ 16%]
test/test_pipeline.py::test_dataset_shapes_and_split[20260101000000] PASSED   [ 33%]
test/test_pipeline.py::test_dataset_shapes_and_split[20261020120000] PASSED   [ 50%]
test/test_pipeline.py::test_dataset_shapes_and_split[20271231235959] PASSED   [ 66%]
test/test_pipeline.py::test_full_pipeline PASSED                              [ 83%]
test/test_pipeline.py::test_evaluate_without_model_fails PASSED               [100%]
============================== 6 passed ==============================
```

### 3.2 Run the pipeline

The timestamp is the model version. Use this fixed one so your numbers match the ones below exactly; the data is generated from it.

**macOS / Linux**
```bash
ts=20261004000000
python src/train_model.py --timestamp $ts
python src/evaluate_model.py --timestamp $ts
python src/calibrate_model.py --timestamp $ts --method sigmoid
```

**Windows (PowerShell)**
```powershell
$ts = "20261004000000"
python src/train_model.py --timestamp $ts
python src/evaluate_model.py --timestamp $ts
python src/calibrate_model.py --timestamp $ts --method sigmoid
```

For a new version instead, use the current time: `ts=$(date '+%Y%m%d%H%M%S')` (PowerShell: `$ts = Get-Date -Format yyyyMMddHHmmss`).

**Success looks like** (paths shortened; MLflow may also print a few `INFO` lines):
```
Trained on 718 samples, saved model to .../Github_Lab2/models/model_20261004000000.joblib
Test metrics: {'accuracy': 0.9042, 'f1_score': 0.9004, 'roc_auc': 0.9553, 'brier_score': 0.084} -> .../metrics/20261004000000_metrics.json
Calibrated (sigmoid): Brier 0.084 -> 0.073, saved .../models/model_20261004000000_calibrated.joblib
```

You should now have these files:
```
models/model_20261004000000.joblib
models/model_20261004000000_calibrated.joblib
metrics/20261004000000_metrics.json
mlflow.db                       (local MLflow store, git-ignored)
```

`metrics/20261004000000_metrics.json`:
```json
{
    "timestamp": "20261004000000",
    "n_test": 240,
    "model": { "accuracy": 0.9042, "f1_score": 0.9004, "roc_auc": 0.9553, "brier_score": 0.084 },
    "calibration": {
        "method": "sigmoid",
        "n_calib": 240,
        "brier_before": 0.084,
        "brier_after": 0.073,
        "brier_improvement": 0.011,
        "calibrated_model": { "accuracy": 0.9, "f1_score": 0.8966, "roc_auc": 0.9553, "brier_score": 0.073 }
    }
}
```

### 3.3 (Optional) Browse training runs in MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open <http://127.0.0.1:5000> and pick the experiment **github_lab2_random_forest**. Press `Ctrl+C` to stop it.

---

## 4. Run it on GitHub Actions (the CI/CD part)

### 4.1 One-time setup
1. Get the code into a GitHub repo you own: fork this repo, or push your clone to a new repo.
2. In that repo go to **Settings → Actions → General**:
   - **Actions permissions:** allow all actions.
   - **Workflow permissions:** select **Read and write permissions**. The workflow already asks for `contents: write`, but some accounts default to read-only, and then the final push step fails with a `403` error.
3. Open the **Actions** tab. If GitHub asks, click **"I understand my workflows, go ahead and enable them"**.

No secrets are needed. The workflow pushes with the built-in `GITHUB_TOKEN`.

### 4.2 Trigger the pipeline
Do either of these:
- **Push a change** to anything under `Github_Lab2/src/`, `Github_Lab2/test/`, `Github_Lab2/requirements.txt` or the workflow file, on `main`:
  ```bash
  git add .
  git commit -m "Trigger Lab 2 pipeline"
  git push origin main
  ```
  Pushes that only touch other folders, such as `DATA LABELLING/`, will **not** trigger it.
- **Run it manually:** **Actions → "Github Lab 2 - Retrain, Evaluate and Calibrate Model" → Run workflow**, and choose `sigmoid` or `isotonic`.

### 4.3 What a successful run looks like
1. **Actions tab:** the run shows a green ✓ after about 2–3 minutes. Every step is green:
   `Run tests → Generate timestamp version → Train model → Evaluate model → Calibrate model → Add metrics to job summary → Commit and push new model version`
2. **Run summary page:** click the run. Under **Summary** you see `Model version <timestamp>` followed by the metrics JSON.
3. **New commit on `main`:** authored by **github-actions[bot]**, with the message
   `Lab 2: add model version <timestamp> [skip ci]`. It adds three files:
   ```
   Github_Lab2/models/model_<timestamp>.joblib
   Github_Lab2/models/model_<timestamp>_calibrated.joblib
   Github_Lab2/metrics/<timestamp>_metrics.json
   ```
4. Each new run adds a new version alongside the old ones, so `models/` and `metrics/` keep the full history.

Run `git pull` afterwards to get the bot's commit locally.

**If a run fails**
| Symptom | Cause / fix |
|---|---|
| Fails at **Run tests** | A test broke. Nothing was trained or committed. Open the step log to see which test. |
| Fails at **Commit and push** with `403` / `Permission denied` | Workflow permissions are read-only. See step 4.1.2. |
| Workflow doesn't start after a push | The push didn't touch any of the `paths:` listed in 4.2, or Actions isn't enabled (4.1.3). |

### 4.4 Proof of a successful run
<!-- After your first green run, add screenshots here, e.g.:
![Green workflow run](docs/actions_run.png)
![Bot commit](docs/bot_commit.png)
-->
_Screenshots of the green Actions run and the bot commit go here._

---

## 5. How the pipeline works

```
push to main (Github_Lab2/** or the workflow file)   — or —   manual "Run workflow"
        │
        ▼
  pytest ──fail──▶ stop, nothing is trained or committed
        │
        ▼
  timestamp (UTC, e.g. 20261020153000)   ← model version
        │
        ▼
  train_model.py      → models/model_<ts>.joblib
        │
        ▼
  evaluate_model.py   → metrics/<ts>_metrics.json   (accuracy, F1, ROC AUC, Brier)
        │
        ▼
  calibrate_model.py  → models/model_<ts>_calibrated.joblib
                        + "calibration" section added to the metrics JSON
        │
        ▼
  metrics posted to the run's Summary page
        │
        ▼
  github-actions[bot] commits models/ + metrics/ back to main  [skip ci]
```

### Folder structure

```
Labs-MLOPs/
├── .github/workflows/
│   └── github_lab2_retrain.yml   # the CI/CD pipeline
└── Github_Lab2/
    ├── src/
    │   ├── data.py               # shared dataset, splits and file naming
    │   ├── train_model.py        # trains RandomForest, logs run to MLflow
    │   ├── evaluate_model.py     # scores the model on the held-out test split
    │   └── calibrate_model.py    # Platt (sigmoid) or isotonic calibration
    ├── test/
    │   ├── conftest.py
    │   └── test_pipeline.py      # runs before training in CI
    ├── models/                   # created by the pipeline: versioned models
    ├── metrics/                  # created by the pipeline: one JSON per version
    ├── requirements.txt
    └── README.md
```

### Data and splits

The data is **synthetic**, generated fresh each run by scikit-learn's `make_classification`: 6 numeric features (3 of them informative), 2 classes, and 5% of labels flipped at random as noise. No dataset file is downloaded or stored. The row count varies from 500 to 2000 per run, to mimic new data arriving.

The run timestamp seeds the generator. Train, evaluate and calibrate run as separate scripts, but each one regenerates **exactly the same** data, so no data files have to be passed between steps. This is also why the fixed timestamp in step 3.2 always gives the same numbers.

| Split | Share | Used by |
|---|---|---|
| train | 60% | `train_model.py` |
| calibration | 20% | `calibrate_model.py` (fits the probability mapping) |
| test | 20% | `evaluate_model.py` and the before/after calibration comparison |

### Calibration

A classifier is *calibrated* when its predicted probabilities match how often the event actually happens: of all the samples given 0.8, about 80% should really be positive. `calibrate_model.py` wraps the trained forest in `CalibratedClassifierCV(FrozenEstimator(model))`. The forest itself stays fixed, and only a probability mapping is learned on the calibration split:

- **sigmoid** (Platt scaling, default): fits a logistic curve. Works with small calibration sets.
- **isotonic**: fits a free-form increasing step function. More flexible, but needs more data.

The metric is the **Brier score**, the mean squared error of the predicted probabilities (lower is better). It is measured before and after calibration on the test split.

**Results over 8 runs** (timestamps `20261001000000` to `20261008000000`):

| | Mean Brier score | Improved on the uncalibrated model in |
|---|---|---|
| Uncalibrated | 0.1048 | – |
| **sigmoid** | **0.1010** | **7 of 8 runs** |
| isotonic | 0.1056 | 2 of 8 runs |

Sigmoid is the default because it helps consistently here. Isotonic usually does worse, because the calibration split is small (roughly 100–400 rows) and isotonic overfits. Calibration hardly changes accuracy or ROC AUC; it only makes the probabilities more trustworthy.

---

## 6. Changes I made

### 6.1 CI/CD workflow changes (core of this lab)

| Original workflow | My workflow | What it does |
|---|---|---|
| No tests before training | **`Run tests` step (pytest) runs first** | A broken change fails the run before any model is trained or committed |
| Train → evaluate → commit | **New `Calibrate model` step** | Produces a calibrated model alongside each version, and records before/after Brier |
| Push-only trigger | **`workflow_dispatch` with a `calibration_method` choice** (`sigmoid` / `isotonic`) | Lets you rerun on demand and compare methods from the Actions UI |
| Any push to `main` retrained, even unrelated folders | **`paths:` filter** | Only changes to this lab or its workflow trigger it |
| Relied on manual repo permission settings | **`permissions: contents: write`** declared in the file | The job can push the new model back with the built-in token |
| Two quick pushes could run at once and clash | **`concurrency` group** | Runs queue one at a time, so their commits can't collide |
| Metrics only visible by opening files | **Metrics written to `$GITHUB_STEP_SUMMARY`** | Each run's results show up on its Summary page |
| Committed under a personal name/email | **Commits as `github-actions[bot]`, message tagged `[skip ci]`** | Clear which commits are automated; never re-triggers itself |
| `git push` could fail if `main` moved during the run | **`git pull --rebase` before pushing** | The push still succeeds when someone pushed during the run |
| Paths hardcoded to `Labs/Github_Labs/Lab2/`; outputs moved with `mv` | **`working-directory: Github_Lab2`**; scripts write straight to `models/` and `metrics/` | Simpler steps, fewer places to break |
| Python 3.9, `actions/setup-python@v4`, no dependency caching | **Python 3.11, `actions/setup-python@v5` with pip cache** keyed on `requirements.txt` | Faster repeat runs; matches the Python version the code is tested on |

### 6.2 ML code fixes

| Original code | My code |
|---|---|
| `n_samples=random.randint(0, 2000)` could produce 0 or a handful of samples and crash training | 500–2000 samples, seeded from the timestamp |
| Model scored on its own training data, then `evaluate_model.py` regenerated a *different* random dataset, so the F1 score had no real meaning | Proper 60/20/20 train / calibration / test split; evaluation only on unseen test data |
| Calibration described in the README but never implemented | `calibrate_model.py` with sigmoid / isotonic, Brier before vs after |
| Only F1 reported | Accuracy, F1, ROC AUC and Brier score |
| No tests | `test/test_pipeline.py`: determinism, split sizes, full end-to-end run, and a missing-model error case |
| Scripts depended on the current directory | Paths resolved from each file's location, so the scripts run from anywhere |
| MLflow experiment mislabelled "Reuters Corpus Volume" | Experiment named `github_lab2_random_forest`, logging the model's parameters and training metrics |
