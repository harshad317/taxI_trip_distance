# taxI_trip_distance

Autonomous experimentation scaffold for the MachineHack Taxi Trip Distance challenge.

This repo adapts the structure of [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) to a tabular regression workflow:

- `train.py` is the only Python file the agent should edit.
- There is intentionally no `prepare.py`; preprocessing and feature engineering live inside `train.py`.
- `program.md` is the main control file for the agent. It defines the endless experiment loop, model roles, and ground rules.

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

`uv run train.py` is a single evaluation run. It writes generated artifacts into `outputs/` and prints a stable summary block including `val_rmse`.

## Recommended Autoresearch Setup

This repo is intended to be operated in the same style as `autoresearch`:

- lead agent: `gpt-5.2`
- research/planning: `gpt-5.2`
- coding / implementation / git push worker: `gpt-5.3-codex` with `high` reasoning

The lead agent should open and follow [program.md](/Users/hpu4454/Desktop/Compititions/Dataset/taxI_trip_distance/program.md). It should keep running until manually interrupted, and it should only persist and push changes when validation RMSE improves.

## Optional Built-in Search

To keep searching until you manually stop it:

```bash
uv run train.py --autoloop > run.log 2>&1
```

What `--autoloop` does:

- runs the current baseline once
- proposes new hyperparameter and feature mutations
- evaluates each candidate on the same fixed 80/20 split
- discards non-improving candidates automatically
- persists improved configurations back into `train.py`
- commits improved configurations
- pushes improved commits to the current git branch unless you pass `--no-push`

Useful variants:

```bash
# run forever in the foreground and watch logs
uv run train.py --autoloop | tee run.log

# test the loop without committing or pushing
uv run train.py --autoloop --max-runs 3 --no-commit --no-push
```

The built-in loop is a fallback. The preferred workflow is still the external `program.md`-driven agent loop.
