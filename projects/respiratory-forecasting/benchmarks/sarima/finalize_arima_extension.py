# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Combine the audited ARIMA extension with the completed full NSSP benchmark."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sarima_pilot import OUTPUT_ROOT


ROOT = OUTPUT_ROOT
EXT = ROOT / "arima_extension_exp2_exp3"


# Calculation provenance:
#   pd.DataFrame -> pandas.DataFrame (third-party package)
#   pd.read_csv -> pandas.read_csv (third-party package)
#   pd.read_csv(EXT / 'ARIMA_EXTENSION_DECISION_DATA.csv').set_index -> pandas.read_csv(EXT / 'ARIMA_EXTENSION_DECISION_DATA.csv').set_index (third-party package)
#   pd.read_csv(ROOT / 'nssp_full' / 'FINAL_SAME_ROW_COMPARISON.csv').set_index -> pandas.read_csv(ROOT / 'nssp_full' / 'FINAL_SAME_ROW_COMPARISON.csv').set_index (third-party package)
def main() -> None:
    nssp = pd.read_csv(ROOT / "nssp_full" / "FINAL_SAME_ROW_COMPARISON.csv").set_index("Model")
    ilinet = pd.read_csv(EXT / "ARIMA_EXTENSION_DECISION_DATA.csv").set_index("Experiment")
    rows = [
        {
            "Experiment": "Experiment 1: NSSP",
            "Test_Rows": int(nssp.loc["AR1", "N"]),
            "Nonseasonal_Model": "ARIMA(1,0,0), equivalent to AR(1)",
            "ARIMA_RMSE": nssp.loc["AR1", "RMSE"],
            "ARIMA_MAE": nssp.loc["AR1", "MAE"],
            "ARIMA_R2": nssp.loc["AR1", "R2"],
            "Seasonal_Model": "SARIMA(1,0,0)x(1,0,0)[52]",
            "SARIMA_RMSE": nssp.loc["SARIMA_100x100_52", "RMSE"],
            "SARIMA_MAE": nssp.loc["SARIMA_100x100_52", "MAE"],
            "SARIMA_R2": nssp.loc["SARIMA_100x100_52", "R2"],
            "Hybrid_RMSE": nssp.loc["Hybrid_InvMSE", "RMSE"],
            "Hybrid_MAE": nssp.loc["Hybrid_InvMSE", "MAE"],
            "Hybrid_R2": nssp.loc["Hybrid_InvMSE", "R2"],
        },
        {
            "Experiment": "Experiment 2: contemporary ILINet",
            "Test_Rows": 1774,
            "Nonseasonal_Model": "ARIMA(0,1,1)",
            "ARIMA_RMSE": ilinet.loc["exp2", "ARIMA_RMSE"],
            "ARIMA_MAE": ilinet.loc["exp2", "ARIMA_MAE"],
            "ARIMA_R2": ilinet.loc["exp2", "ARIMA_R2"],
            "Seasonal_Model": "SARIMA(1,0,1)x(1,0,0)[52]",
            "SARIMA_RMSE": ilinet.loc["exp2", "SARIMA_RMSE"],
            "SARIMA_MAE": ilinet.loc["exp2", "SARIMA_MAE"],
            "SARIMA_R2": ilinet.loc["exp2", "SARIMA_R2"],
            "Hybrid_RMSE": ilinet.loc["exp2", "Hybrid_RMSE"],
            "Hybrid_MAE": ilinet.loc["exp2", "Hybrid_MAE"],
            "Hybrid_R2": ilinet.loc["exp2", "Hybrid_R2"],
        },
        {
            "Experiment": "Experiment 3: pandemic ILINet",
            "Test_Rows": 2808,
            "Nonseasonal_Model": "ARIMA(1,1,0)",
            "ARIMA_RMSE": ilinet.loc["exp3", "ARIMA_RMSE"],
            "ARIMA_MAE": ilinet.loc["exp3", "ARIMA_MAE"],
            "ARIMA_R2": ilinet.loc["exp3", "ARIMA_R2"],
            "Seasonal_Model": "SARIMA(2,0,0)x(1,0,0)[52]",
            "SARIMA_RMSE": ilinet.loc["exp3", "SARIMA_RMSE"],
            "SARIMA_MAE": ilinet.loc["exp3", "SARIMA_MAE"],
            "SARIMA_R2": ilinet.loc["exp3", "SARIMA_R2"],
            "Hybrid_RMSE": ilinet.loc["exp3", "Hybrid_RMSE"],
            "Hybrid_MAE": ilinet.loc["exp3", "Hybrid_MAE"],
            "Hybrid_R2": ilinet.loc["exp3", "Hybrid_R2"],
        },
    ]
    frame = pd.DataFrame(rows)
    frame["Hybrid_vs_ARIMA_RMSE_Reduction"] = (frame["ARIMA_RMSE"] - frame["Hybrid_RMSE"]) / frame["ARIMA_RMSE"]
    frame["Hybrid_vs_SARIMA_RMSE_Reduction"] = (frame["SARIMA_RMSE"] - frame["Hybrid_RMSE"]) / frame["SARIMA_RMSE"]
    frame.to_csv(EXT / "CROSS_EXPERIMENT_ARIMA_SARIMA_HYBRID.csv", index=False)

    lines = [
        "# Cross-experiment ARIMA/SARIMA retention decision",
        "",
        "| Experiment | Selected nonseasonal model | ARIMA RMSE | Selected annual model | SARIMA RMSE | Hybrid RMSE |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.Experiment} | {row.Nonseasonal_Model} | {row.ARIMA_RMSE:.6f} | "
            f"{row.Seasonal_Model} | {row.SARIMA_RMSE:.6f} | {row.Hybrid_RMSE:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "1. Retain one validation-selected nonseasonal ARIMA reference and one validation-selected annual SARIMA reference in every experiment.",
            "2. Keep both classical models outside the RF/GBR/MLP inverse-MSE ensemble; they are comparators, not ensemble members.",
            "3. Report that annual seasonality was dataset-dependent: it harmed NSSP performance relative to ARIMA(1,0,0), but improved performance over nonseasonal ARIMA in both ILINet experiments.",
            "4. Report that the hybrid had the lowest RMSE in all three experiments, while avoiding a claim that SARIMA is universally weak.",
            "5. Retain the seasonal-differenced D=1 candidates only as NSSP sensitivity/convergence evidence; do not promote them to full cross-experiment benchmarks.",
            "",
            "This scope answers the classical-benchmark concern without changing the locked hybrid framework or using final-test performance for order selection.",
        ]
    )
    (EXT / "CROSS_EXPERIMENT_RETENTION_DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
