# ARIMA/SARIMA benchmark workflow

This directory contains the preserved classical time-series benchmark workflow adapted to the clean repository layout. It reads:

- raw inputs from `data/raw/`;
- locked hybrid rows and metadata from `results/reference/hybrid/`;
- candidate definitions from `benchmarks/sarima/pilot_config.json`.

It writes new artifacts only below `results/reproduced/classical/`.

For this public copy, unpack the checksum-verified `data/reference_calculations.zip`
at the repository root before running these scripts. Its `results/reference/`
members supply the locked comparison rows; the default calculation notebook can
read the archive directly, but these original scripts expect CSVs on disk.
The notebook's final cell contains the safe, hash-checked extraction routine:
set only `EXTRACT_REFERENCE = True`, leaving all `RUN_*` switches false.
Exact raw data and scientific dependencies are also required. Do not
launch these expensive runs merely to review the saved calculations.

## Method

Candidate orders are selected on the chronological validation interval. The selected model is refitted on pre-test history and scored using rolling one-week-ahead predictions on the exact locked final-test rows. Final-test outcomes are not used for order selection.

This describes the within-run procedure, not an untouched development history.
All three final-test windows were inspected during auxiliary-lag development.

The day-one profile contains nonseasonal ARIMA references and annual seasonal autoregressive SARIMA candidates. Seasonal-differenced candidates remain in the broader grid because the archived NSSP sensitivity work found optimizer-convergence problems for those specifications.

## Commands

Quick development run:

```powershell
python benchmarks/sarima/sarima_pilot.py --experiments exp2 --quick-candidates 4
```

Preserved day-one profile:

```powershell
python benchmarks/sarima/sarima_pilot.py --experiments exp2 exp3 exp1 --candidate-profile day1
python benchmarks/sarima/summarize_pilot.py
python benchmarks/sarima/validate_pilot.py
```

ILINet nonseasonal extension after the pilot outputs exist:

```powershell
python benchmarks/sarima/run_ilinet_arima_extension.py
python benchmarks/sarima/validate_ilinet_arima_extension.py
```

The full NSSP runner is resumable and can create many batch files. Launch it only after reviewing its CLI help and confirming adequate compute and storage:

```powershell
python benchmarks/sarima/nssp_full_runner.py --help
```

The preserved completed evidence is under `results/reference/classical/`; it is not modified by these commands.
