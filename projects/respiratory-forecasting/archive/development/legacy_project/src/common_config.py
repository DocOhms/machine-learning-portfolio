"""Shared configuration for IEEE respiratory forecasting experiments."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_FRACTION_FOR_ENSEMBLE_WEIGHTS = 0.20

LAGS = [1, 2, 3]
ROLLING_WINDOW = 3
SEASONAL_LAG = 52

N_CLUSTERS = 4
KMEANS_N_INIT = 10
SILHOUETTE_SAMPLE_SIZE = 5000

NSSP_COMPONENT_COLUMNS = [
    "percent_visits_covid",
    "percent_visits_influenza",
    "percent_visits_rsv",
]
ILINET_REPORTING_COLUMNS = ["NUM. OF PROVIDERS"]

MODEL_PARAMS = {
    "RandomForest": {
        "n_estimators": 150,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
    },
    "GradientBoosting": {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 3,
        "random_state": RANDOM_STATE,
    },
    "MLP": {
        "hidden_layer_sizes": (32, 16),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "learning_rate_init": 0.001,
        "max_iter": 300,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "random_state": RANDOM_STATE,
        "tol": 1e-4,
        "n_iter_no_change": 20,
    },
}
