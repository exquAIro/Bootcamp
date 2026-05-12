import warnings
from pathlib import Path

from optuna.exceptions import ExperimentalWarning

# Prevent experimental warnings for Optuna
warnings.filterwarnings("ignore", category=ExperimentalWarning)

# # MLflow
MLFLOW_HOST = "localhost"
MLFLOW_PORT = "5000"

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "Data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
