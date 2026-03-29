# taxi autoresearch

This repo adapts the `autoresearch` workflow to the Taxi Trip Distance challenge.

## Setup

To set up a new autonomous run, work with the user to:

1. Agree on a run tag. Use a short tag based on the date, for example `mar29`.
2. Create a fresh branch such as `autoresearch/<tag>`.
3. Read the in-scope files for full context:
   - `README.md`
   - `train.py`
   - `program.md`
   - `pyproject.toml`
4. Verify the dataset exists. `train.py` auto-discovers `Train.csv`, `Test.csv`, and `Submission.csv` from:
   - the working directory
   - the repo root
   - the repo parent directory
   If they are missing, ask the human to place them or pass explicit paths.
5. Initialize `results.tsv` with a header row only. Keep it untracked.
6. Run the baseline once before changing anything.

## Scope

The repo is intentionally small:

- `train.py` is the only Python file you should edit.
- `program.md` is the human-owned control file.
- There is no `prepare.py`; all preprocessing, feature engineering, training, validation, and submission export live in `train.py`.

## Fixed evaluation protocol

These rules exist so experiments stay comparable:

- Target column: `trip_distance_miles`
- Validation split: 20 percent of `Train.csv`
- Split seed: `42`
- Metric: RMSE on the validation split
- Lower `val_rmse` is better

Do not change the evaluation protocol unless the human explicitly asks. You may improve features, model choice, hyperparameters, ensembling, regularization, or training logic inside `train.py`, but keep the validation metric and split stable.

## What you can change

- Feature engineering inside `train.py`
- Model selection and hyperparameters
- Training logic
- Submission generation logic
- Logging details, as long as the required summary keys remain intact

## What you should not change

- The dataset itself
- The target column
- The 80/20 validation protocol
- The `val_rmse:` output line format
- Dependencies in `pyproject.toml`, unless the human explicitly asks

## Baseline run

Launch a single experiment with:

```bash
uv run train.py > run.log 2>&1
```

The script prints a summary block like this:

```text
---
val_rmse:          0.737727
training_seconds:  12.3
total_seconds:     13.1
best_iteration:    1120
train_rows:        11759
validation_rows:   2940
submission_file:   outputs/submission_predictions.csv
```

Extract the key metric with:

```bash
grep "^val_rmse:\|^best_iteration:" run.log
```

## Logging results

Keep `results.tsv` as a tab-separated file with this schema:

```text
commit	val_rmse	best_iteration	status	description
```

Column rules:

1. Short git commit hash
2. Validation RMSE, or `0.000000` for crashes
3. Best iteration, or `0` for crashes
4. `keep`, `discard`, or `crash`
5. Short description of the experiment

Example:

```text
commit	val_rmse	best_iteration	status	description
a1b2c3d	0.737727	1120	keep	baseline catboost with engineered time and geo features
b2c3d4e	0.731400	980	keep	increase depth and add borough interaction feature
c3d4e5f	0.744100	1400	discard	switch to weaker regularization
d4e5f6g	0.000000	0	crash	invalid feature transform
```

## Built-in experiment loop

The endless loop is implemented directly inside `train.py`.

Launch it with:

```bash
uv run train.py --autoloop > run.log 2>&1
```

Its behavior is:

1. Run the current baseline configuration once.
2. Generate a mutated candidate configuration.
3. Evaluate the candidate on the fixed validation split.
4. Log the result to `results.tsv`.
5. If the candidate improves `val_rmse`, persist it into `train.py`.
6. Commit the improved `train.py`.
7. Push the improved commit to the current branch, unless `--no-push` is used.
8. If the candidate does not improve, discard it and try another mutation.
9. Repeat forever until interrupted.

Useful flags:

- `--max-runs N` for a bounded test run
- `--no-commit` to test the loop without changing git history
- `--no-push` to keep improvements local
- `--allow-dirty` to bypass the clean-tree check

## Operating principles

- Favor simple changes with measurable gains.
- Avoid test leakage. Never use `Test.csv` labels because they do not exist.
- Prefer stable, reproducible experiments over noisy hacks.
- If you run out of ideas, think harder and keep iterating.

## Never stop

Once the experiment loop begins, do not pause to ask the human whether to continue. Do not ask if this is a good stopping point. Continue running experiments until the human manually interrupts you.
