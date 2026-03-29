# taxI_trip_distance

Autonomous experimentation scaffold for the MachineHack Taxi Trip Distance challenge.

This repo adapts the structure of [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) to a tabular regression workflow:

- `train.py` is the only Python file the agent should edit.
- There is intentionally no `prepare.py`; preprocessing and feature engineering live inside `train.py`.
- `program.md` is the main control file for the agent. It defines the endless experiment loop and the ground rules.

## Goal

Predict `trip_distance_miles` from the provided taxi trip features and minimize validation RMSE on a fixed 80/20 split of `Train.csv`.

## Repo structure

```text
train.py        # baseline training/evaluation/prediction pipeline
program.md      # autonomous research instructions
pyproject.toml  # dependencies
```

## Data expectations

`train.py` automatically looks for:

- `Train.csv`
- `Test.csv`
- `Submission.csv`

It checks these locations in order:

1. The explicitly provided CLI path, if any
2. The current working directory
3. The repo root
4. The repo parent directory

That means you can keep the competition CSVs either in the repo root or in the parent dataset folder.

## Quick start

```bash
uv sync
uv run train.py
```

Or with explicit dataset paths:

```bash
uv run train.py --train-path ../Train.csv --test-path ../Test.csv --submission-path ../Submission.csv
```

The script writes all generated artifacts into `outputs/` and prints a stable summary block including `val_rmse`, which is the number the autonomous loop should optimize.
