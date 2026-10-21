import os
import sys
from pathlib import Path

# The scripts in src/ import each other as top-level modules (that's how they
# run under `python src/train_model.py`), so put src/ on the path for pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Keep test runs out of the MLflow store.
os.environ["MLFLOW_DISABLE"] = "1"
