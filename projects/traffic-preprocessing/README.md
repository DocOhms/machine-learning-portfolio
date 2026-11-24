# Honolulu traffic data preprocessing

Two exploratory notebooks clean and describe the included Honolulu traffic snapshots. They demonstrate feature engineering and visualization. They do not train or evaluate a prediction model.

| Notebook | Included snapshot | Main steps |
| --- | --- | --- |
| [Traffic camera locations](Traffic_Camera_Locations.ipynb) | 253 camera-location records | Parse latitude and longitude, create a road-name keyword flag and longitude bins, standardize coordinates, and plot locations |
| [Traffic incidents](Traffic_Incidents.ipynb) | 194 reports dated January 23–24, 2026 | Parse timestamps, create time and category features, and plot incident counts |

The two CSVs are in [`data/`](data/). The notebooks describe them as City and County of Honolulu camera data and Honolulu traffic incident reports, respectively. Their original download URLs, update times, and redistribution terms were not recorded in this project. Treat them as fixed example snapshots, not current or complete citywide data.

## Run locally

Use Python 3.12. From the portfolio repository root:

```sh
cd projects/traffic-preprocessing
python -m venv .venv
```

Activate `.venv` with the command for your operating system, then:

```sh
python -m pip install -r requirements.txt
python -m jupyter lab
```

Open either notebook and use **Run All**. Start Jupyter from this project folder so the notebook paths `data/Traffic_Camera_Locations.csv` and `data/Traffic_Incidents.csv` resolve. Requirements list the direct package versions used to refresh the notebook outputs.

## Interpretation limits

- The camera road flag matches text keywords in descriptions. Longitude bins are rough geographic groupings, not validated neighborhood boundaries.
- The incident severity field is a type-based proxy. The top-ten-area field is calculated on this same 194-report snapshot, so it is descriptive and would need a training-only fit for predictive use.
- The category encoding assigns integers for exploration; those integers do not imply an ordinal relationship. Coordinate scaling also uses the full snapshot. Any future model needs a train/test design and preprocessing fitted only on training data.
- Two days of incidents are too limited to establish a general hotspot pattern or a forecasting result.

For the separate, evaluated forecasting study in this portfolio, see [respiratory surveillance forecasting](../respiratory-forecasting/README.md).
