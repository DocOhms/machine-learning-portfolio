# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common_features import add_time_series_features, chronological_split  # noqa: E402
from common_metrics import evaluate_predictions  # noqa: E402


class FeatureBoundaryTests(unittest.TestCase):
    # Calculation provenance:
    #   add_time_series_features -> common_features.add_time_series_features (src/common_features.py)
    #   pd.DataFrame -> pandas.DataFrame (third-party package)
    def test_target_and_covariate_features_are_lagged(self) -> None:
        frame = pd.DataFrame(
            {
                "group": ["A"] * 6,
                "epi_order": [202401, 202402, 202403, 202404, 202405, 202406],
                "WEEK": [1, 2, 3, 4, 5, 6],
                "target": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "aux": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        featured = add_time_series_features(
            frame,
            target="target",
            group_cols=["group"],
            lagged_covariates={"aux": [1]},
        )
        row = featured.loc[featured["epi_order"].eq(202404)].iloc[0]
        self.assertEqual(row["lag_1"], 30.0)
        self.assertEqual(row["lag_2"], 20.0)
        self.assertEqual(row["lag_3"], 10.0)
        self.assertEqual(row["rolling_mean_3"], 20.0)
        self.assertEqual(row["aux_lag_1"], 3.0)
        self.assertNotEqual(row["lag_1"], row["target"])

    # Calculation provenance:
    #   chronological_split -> common_features.chronological_split (src/common_features.py)
    #   np.arange -> numpy.arange (third-party package)
    #   np.repeat -> numpy.repeat (third-party package)
    #   pd.DataFrame -> pandas.DataFrame (third-party package)
    def test_chronological_split_keeps_weeks_disjoint(self) -> None:
        frame = pd.DataFrame(
            {
                "epi_order": np.repeat([202401, 202402, 202403, 202404, 202405], 2),
                "value": np.arange(10),
            }
        )
        train, test, cutoff = chronological_split(frame, test_size=0.4)
        self.assertTrue((train["epi_order"] < cutoff).all())
        self.assertTrue((test["epi_order"] >= cutoff).all())
        self.assertTrue(set(train["epi_order"]).isdisjoint(set(test["epi_order"])))


class RepositoryArtifactTests(unittest.TestCase):
    # Calculation provenance:
    #   json.loads -> json.loads (Python standard library)
    def test_data_manifest_has_expected_files(self) -> None:
        manifest = json.loads(
            (PROJECT_ROOT / "data" / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            {item["name"] for item in manifest["files"]},
            {"NSSP.csv", "ILINet.csv", "ILINet_2016_2021.csv"},
        )
        self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest["files"]))

    # Calculation provenance:
    #   pd.read_csv -> pandas.read_csv (third-party package)
    #   pd.read_csv(path).set_index -> pandas.read_csv(path).set_index (third-party package)
    def test_primary_reference_metrics_are_preserved(self) -> None:
        expected = {
            "experiment_1_nssp_primary": (47534, 0.803473143311848),
            "experiment_2_ilinet_contemporary": (1774, 0.7160871996239156),
            "experiment_3_ilinet_pandemic_stress_test": (2808, 0.8334582539677647),
        }
        for name, (rows, rmse) in expected.items():
            path = (
                PROJECT_ROOT
                / "results"
                / "reference"
                / "hybrid"
                / name
                / "metrics"
                / f"{name}_metrics.csv"
            )
            metrics = pd.read_csv(path).set_index("Model")
            hybrid = metrics.loc["Hybrid_InvMSE"]
            self.assertEqual(int(hybrid["N"]), rows)
            self.assertAlmostEqual(float(hybrid["RMSE"]), rmse, places=12)

    # Calculation provenance:
    #   evaluate_predictions -> common_metrics.evaluate_predictions (src/common_metrics.py)
    #   np.array -> numpy.array (third-party package)
    def test_metric_evaluation_orders_by_rmse(self) -> None:
        metrics = evaluate_predictions(
            np.array([1.0, 2.0, 3.0]),
            {
                "bad": np.array([0.0, 0.0, 0.0]),
                "perfect": np.array([1.0, 2.0, 3.0]),
            },
        )
        self.assertEqual(metrics.iloc[0]["Model"], "perfect")
        self.assertEqual(metrics.iloc[0]["RMSE"], 0.0)


if __name__ == "__main__":
    unittest.main()
