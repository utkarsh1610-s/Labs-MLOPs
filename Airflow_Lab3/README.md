# Airflow Lab 3 – Wine Clustering Pipeline with Parallel Training and a Quality Gate

## Overview

An **Apache Airflow 3.3.2** pipeline, running in Docker. It clusters wines by their chemistry, picks the number of clusters automatically, and only replaces the production model when the new one is good enough.

Each run:
1. **validates** the input data and stops if it is broken
2. trains **one KMeans model per k (1–10) in parallel**, using Airflow's dynamic task mapping
3. picks k with the **elbow method**
4. **evaluates** the chosen model
5. goes through a **quality gate** that branches: if the model passes it is **promoted** to production; if it fails it is **flagged for review**, and the current production model stays in place
6. assigns 10 held-out "new" wines to clusters using the current production model
7. writes a JSON report

It is based on `Airflow_Labs/Lab_1` from the MLOps (IE-7374) course repo, a KMeans + elbow-method DAG on credit-card data. What I changed, mainly the DAG itself, is listed in [Changes I made](#7-changes-i-made).

---

## 1. Prerequisites

| Tool | Version | Why it matters |
|---|---|---|
| **Docker Desktop** (Mac/Windows) or Docker Engine + Compose plugin (Linux) | Docker 24+ with Compose v2 (tested with Docker 29.9, Compose 5.6) | Everything runs in containers; you do **not** need Python or Airflow installed |
| Memory for Docker | **at least 4 GB, 8 GB recommended** | Airflow runs 7 containers. Docker Desktop → Settings → Resources → Memory |
| Disk | ~10 GB free | Docker images (the built Airflow image is ~3.2 GB) plus the Postgres database |
| git | any recent version | to clone the repo |
| Free port | **8080** (or any port you choose, see [Troubleshooting](#6-troubleshooting)) | the Airflow web UI |
| *(optional)* Python | 3.13 | only to run the unit tests outside Docker (section 5.2) |

**Installing Docker**
- **macOS / Windows:** install [Docker Desktop](https://www.docker.com/products/docker-desktop/), open it once and wait until it says *Engine running*. On Windows, use the WSL 2 backend.
- **Linux:** follow [Install Docker Engine](https://docs.docker.com/engine/install/), which includes the `docker compose` plugin.

Check it worked:
```bash
docker --version
docker compose version
docker run --rm hello-world
```

Everything inside the containers is pinned by the image:

| Inside the Airflow image | Version |
|---|---|
| apache-airflow | 3.3.2 |
| Python | 3.13 |
| scikit-learn | 1.9.1 |
| pandas | 3.0.5 |
| numpy | 2.5.3 |
| kneed (added by our `Dockerfile`) | 0.9.0 |
| pytest (added by our `Dockerfile`) | 9.2.0 |

---

## 2. Set up

```bash
git clone https://github.com/utkarsh1610-s/Labs-MLOPs.git
cd Labs-MLOPs/Airflow_Lab3
```

**macOS / Linux**
```bash
./setup.sh
```

**Windows:** run this in WSL 2 or Git Bash, or run the same steps by hand in PowerShell:
```powershell
mkdir logs, plugins, config, working_data -Force
"AIRFLOW_UID=50000`nFERNET_KEY=" | Out-File -Encoding ascii .env
docker compose build
docker compose up airflow-init
```

`setup.sh` does four things:
1. creates the `logs/`, `plugins/`, `config/` and `working_data/` folders that Docker mounts
2. writes a `.env` file with your user id, so files the containers create belong to you
3. **builds the image**: Airflow 3.3.2 plus `kneed` and `pytest`. This takes 1–3 minutes the first time.
4. runs `airflow-init`, which creates the database and the admin user

**Success looks like** this, at the end of the output:
```
airflow-init-1 exited with code 0

Setup done. Start Airflow with:  docker compose up -d
```

---

## 3. Run the pipeline

### 3.1 Start Airflow
```bash
docker compose up -d
```
Wait about a minute, then check that everything is up:
```bash
docker compose ps
```
**Success looks like** this: every service is `Up`, and they all reach `(healthy)` within ~1–2 minutes.
```
airflow-apiserver       Up (healthy)
airflow-dag-processor   Up (healthy)
airflow-scheduler       Up (healthy)
airflow-triggerer       Up (healthy)
airflow-worker          Up (healthy)
postgres                Up (healthy)
redis                   Up (healthy)
```

### 3.2 Open the UI
Go to **<http://localhost:8080>** and log in with **username `airflow`, password `airflow`**. These are the default local-only credentials from the official Airflow Compose file.

The home page should show all four health checks green: MetaDatabase, Scheduler, Triggerer, Dag Processor.

### 3.3 The first run happens by itself
The DAG starts **unpaused** and is scheduled `@daily`, so as soon as Airflow is up the scheduler runs it once for today's date. You don't have to do anything.

Open **Dags → Airflow_Lab3_Wine_Clustering** to watch it. A run takes about **50 seconds**.

### 3.4 Trigger it yourself (optional)
In the DAG page, click **Trigger**. Under **Advanced** you can change the run's parameters:

| Param | Default | Meaning |
|---|---|---|
| `k_min` | 1 | smallest k to try |
| `k_max` | 10 | largest k to try (max 20) |
| `silhouette_threshold` | 0.25 | minimum silhouette score needed to promote the model |

Or from the terminal:
```bash
docker compose exec airflow-scheduler airflow dags trigger Airflow_Lab3_Wine_Clustering
```

### 3.5 See the reject branch
Trigger a run with a threshold the model can't meet:
```bash
docker compose exec airflow-scheduler airflow dags trigger Airflow_Lab3_Wine_Clustering \
  --run-id manual_strict_threshold -c '{"silhouette_threshold": 0.9}'
```
This time `promote_model` is **skipped** and `flag_for_review` runs. The production model is not touched, and the new wines are still scored with the last good model.

---

## 4. What a successful run looks like

### 4.1 In the UI (Graph view of the run)
```
load_and_validate ─┬─▶ preprocess ──┐
                   └─▶ candidate_ks ─┴─▶ train_kmeans [10] ─▶ select_k ─▶ evaluate ─▶ quality_gate ─┬─▶ promote_model ──┐
                                                                                                 └─▶ flag_for_review ─┴─▶ score_new_samples ─▶ write_run_report
```
- With the default params, every task is **green except `flag_for_review`**, which is pink (*skipped*). That is correct: the model passed the gate.
- `train_kmeans [10]` is one box with 10 mapped tasks. Click it and the **Mapped Tasks** tab lists them as `k=1` … `k=10`. Their start times are all within about a second of each other, because they ran in parallel.
- Open the `evaluate` or `write_run_report` task → **Logs** to see the results printed.

### 4.2 Numbers you should get
The data and the KMeans seed are fixed, so every run with the default params gives the same result:

| Result | Value | Meaning |
|---|---|---|
| chosen k | **3** (method `elbow`) | matches the 3 real grape cultivars in the data |
| silhouette | **0.288** | passes the 0.25 threshold → **promoted** |
| adjusted Rand index (ARI) vs true cultivars | **0.89** | 1.0 = clusters exactly match the real cultivars, 0 = random |
| cluster sizes | 61 / 48 / 59 | |
| held-out new wines, ARI | **1.0** | all 10 new wines were put with their own cultivar |

### 4.3 Files on your machine
Everything the pipeline writes lands in `Airflow_Lab3/working_data/`, which is mounted from the containers:
```
working_data/
├── models/
│   ├── wine_kmeans.joblib        # production model: scaler + KMeans in one sklearn Pipeline
│   ├── wine_kmeans.json          # which run produced it, k, scores
│   └── archive/<run_id>/         # previous production models (appears after the 2nd promotion)
├── review/<run_id>.json          # only for runs that failed the quality gate
├── reports/<run_id>.json         # one summary per run
└── runs/<run_id>/
    ├── scaled.npy, scaler.joblib
    ├── kmeans_k1.joblib … kmeans_k10.joblib
    └── new_sample_predictions.csv
```

Example `reports/<run_id>.json` (trimmed):
```json
{
  "run_id": "scheduled__2026-10-29T00:00:00+00:00",
  "params": { "k_min": 1, "k_max": 10, "silhouette_threshold": 0.25 },
  "decision": "promoted",
  "evaluation": {
    "k": 3,
    "selection_method": "elbow",
    "silhouette": 0.2883,
    "adjusted_rand_index": 0.8916,
    "cluster_sizes": { "0": 61, "1": 48, "2": 59 }
  },
  "new_samples": {
    "n_samples": 10,
    "model_run_id": "scheduled__2026-10-29T00:00:00+00:00",
    "cluster_counts": { "0": 4, "1": 3, "2": 3 },
    "adjusted_rand_index": 1.0
  }
}
```
For the strict-threshold run in 3.5, `decision` is `"flagged_for_review"`, `review/manual_strict_threshold.json` says `"silhouette 0.288 below threshold 0.9"`, and `new_samples.model_run_id` still points at the earlier, promoted run.

---

## 5. Tests

### 5.1 Inside Docker (all 22 tests, no local Python needed)
```bash
docker compose run --rm airflow-cli bash -c "cd /opt/airflow && pytest tests -v"
```
**Success looks like:** `22 passed`.

- `tests/test_clustering.py` (14 tests) covers:
  - data validation, including rejecting missing columns, nulls, too few rows and text in numeric columns
  - the k range
  - the elbow method picking 3, and the silhouette fallback when there is no elbow
  - evaluation scores
  - promoting and archiving
  - the review path leaving production untouched
- `tests/test_dag.py` (8 tests) checks that:
  - the DAG loads with no import errors
  - all 11 tasks exist and are wired correctly
  - `quality_gate` branches to both sides
  - the tasks after the branch use the `none_failed_min_one_success` trigger rule
  - `train_kmeans` is dynamically mapped
  - the params, retries and catchup settings are right

### 5.2 Outside Docker (unit tests only, optional)
```bash
python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest tests -v
```
**Success looks like:** `14 passed, 1 skipped`. The skipped item is `test_dag.py`, which needs Airflow and only runs in Docker.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `Bind for 0.0.0.0:8080 failed: port is already allocated` | Another app uses 8080. Add `AIRFLOW_HOST_PORT=8081` to `.env`, run `docker compose up -d` again, and open <http://localhost:8081> |
| `airflow-init` warns *Not enough memory* | Give Docker at least 4 GB (Docker Desktop → Settings → Resources) |
| DAG not listed in the UI | Wait ~30 s for the DAG processor, then check `docker compose exec airflow-scheduler airflow dags list-import-errors` |
| Permission errors writing `logs/` or `working_data/` (Linux) | Make sure `.env` contains your `AIRFLOW_UID` (`id -u`), then rerun `./setup.sh` |

**Stop Airflow:**
```bash
docker compose down        # stop, keep the database and run history
docker compose down -v     # stop and delete the database (start completely fresh)
```

---

## 7. Changes I made

### 7.1 DAG / orchestration changes (core of this lab)

| Original Lab 1 DAG | My DAG | What it does |
|---|---|---|
| 4 `PythonOperator`s in a straight line | **11 tasks written with the TaskFlow API** (`@task`, `@task.branch`) | Clearer code; values pass between tasks as normal function arguments |
| One task looped over k = 1…49 internally | **Dynamic task mapping:** `train_kmeans.partial(...).expand(k=...)` creates one task per k at run time | The k values train **in parallel** across Celery workers; each k has its own logs and retries, and is labelled `k=…` in the UI through `map_index_template` |
| Fixed k range hard-coded in Python | **DAG `Param`s** `k_min`, `k_max`, `silhouette_threshold`, with types and limits, editable in the Trigger form | Change the experiment without editing code; bad values are rejected before the run starts |
| No quality check: whatever was trained was used | **`@task.branch` quality gate** → `promote_model` **or** `flag_for_review` | A weak model never replaces the production one |
| No model management | **Promote with archive:** the production model is replaced only on pass, and the previous one is copied to `models/archive/<run_id>/` | Keeps a history and makes rollback possible |
| Final task always used the just-trained model | **`score_new_samples` joins both branches** with trigger rule `none_failed_min_one_success` and always uses the production model on disk | New data keeps being served by the last good model even when a run is rejected |
| No input checks | **`load_and_validate` task** fails the run on missing columns, nulls, non-numeric data or fewer than 30 rows | Bad data stops the pipeline at the first step |
| Data passed through XCom as **base64-encoded pickles** | Tasks pass **small JSON values** (paths, scores); data and models go to a per-run folder `working_data/runs/<run_id>/` | Smaller XComs, no pickle in the database, and runs never overwrite each other |
| No schedule (manual only), `retries: 0` | **`@daily` schedule**, `catchup=False`, `max_active_runs=1`, **1 retry after 30 s** | Runs on its own, never backfills, never overlaps |
| No run summary | **`write_run_report`** writes `reports/<run_id>.json`, and the DAG has its own documentation (`doc_md`) shown in the UI | One file shows what happened in each run |
| No tests | **22 tests**, including 8 DAG integrity tests that run against the real Airflow inside the container | Catches import errors and wiring mistakes before they reach the scheduler |

### 7.2 Docker / environment changes

| Original setup | My setup |
|---|---|
| Fetch an old (2.5.1) `docker-compose.yaml` and install packages at every container start via `_PIP_ADDITIONAL_REQUIREMENTS` | Official **Airflow 3.3.2** compose file plus a **`Dockerfile`** that installs `kneed` and `pytest` once, at build time |
| ~80 example DAGs loaded | `LOAD_EXAMPLES: 'false'`, so only this lab's DAG appears |
| DAGs paused when first loaded | Starts **unpaused**, so the schedule works right away |
| Fixed port 8080 | `AIRFLOW_HOST_PORT` setting for machines where 8080 is taken |
| — | `working_data/` and `tests/` mounted into the containers; `setup.sh` does the whole first-time setup |

### 7.3 Data and ML changes

| Original | Mine |
|---|---|
| Credit-card customer data (8,950 rows, 3 of 17 columns used), with no way to check if the clusters are meaningful | **Wine dataset** (scikit-learn's built-in UCI Wine): 178 wines, 13 chemical measurements, 3 known grape cultivars. The cultivar is **never** used for training, only to check the result afterwards (ARI). `scripts/make_dataset.py` regenerates the CSVs. |
| **Saved the k=49 model** (the last one in the loop), so the elbow k was computed and then ignored | The **elbow-selected model** is the one evaluated and promoted. If no elbow exists, the best silhouette score is used. |
| Predicted on **unscaled** `test.csv` values (e.g. `CREDIT_LIMIT = 241514`) with a model trained on data scaled to 0–1, so the prediction was meaningless | The scaler and KMeans are saved together as **one sklearn `Pipeline`**, so new data is always scaled the same way |
| Only "optimal k" printed | Silhouette score, ARI against the true cultivars, and cluster sizes |
| A single test row | 10 stratified held-out wines (`new_samples.csv`) scored at the end of every run |

---

## 8. Project structure

```
Airflow_Lab3/
├── dags/
│   ├── wine_clustering_dag.py     # the DAG (TaskFlow, mapping, branching, params)
│   ├── src/clustering.py          # the ML steps: plain Python, no Airflow imports
│   └── data/
│       ├── wine.csv               # 168 wines for training
│       └── new_samples.csv        # 10 held-out wines scored at the end of each run
├── tests/
│   ├── test_clustering.py         # unit tests (run anywhere)
│   └── test_dag.py                # DAG integrity tests (need Airflow → run in Docker)
├── scripts/make_dataset.py        # rebuilds the two CSVs from scikit-learn's Wine dataset
├── Dockerfile                     # Airflow 3.3.2 + kneed + pytest
├── docker-compose.yaml            # official Airflow 3.3.2 Celery setup, lab changes marked "Lab 3"
├── setup.sh                       # one-time setup
├── requirements.txt               # packages added to the image
└── requirements-dev.txt           # local test environment, same versions as the image
```

`logs/`, `config/`, `plugins/`, `working_data/` and `.env` are created at runtime and are git-ignored.

### Why the ML code is separate from the DAG
`src/clustering.py` has no Airflow imports, and every function takes and returns plain values (paths, numbers, dicts). That means:
- it can be unit-tested without Airflow
- each DAG task is a thin wrapper of two or three lines
- what travels through XCom is small and JSON-safe
