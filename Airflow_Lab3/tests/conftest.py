import sys
from pathlib import Path

# Airflow puts the dags/ folder on sys.path, which is how the DAG can do
# `from src import clustering`. Do the same for pytest.
DAGS_DIR = Path(__file__).resolve().parents[1] / "dags"
sys.path.insert(0, str(DAGS_DIR))
