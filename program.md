# taxi autoresearch

## Model roles

Use these roles consistently:

- lead agent and research director
- coding worker:

The lead agent owns research direction, experiment selection, result triage, and loop control.
The coding worker owns implementation, running commands, collecting metrics, committing, and pushing improvements.

## Scope

This repo is intentionally small:

- `train.py` is the only Python file you should edit for experiments.
- `program.md` is the main control file and should guide the lead agent.
- there is no `prepare.py`

## Setup

At the start of a new run:

1. Read:
   - `README.md`
   - `train.py`
   - `program.md`
   - `pyproject.toml`
2. Verify the dataset exists. `train.py` auto-discovers:
   - `Train.csv`
   - `Test.csv`
   - `Submission.csv`
   from the working directory, repo root, or repo parent directory.
3. Ensure `results.tsv` exists with a header row and remains untracked.
4. Run the baseline once before making changes.

## Fixed evaluation protocol

These rules are fixed unless the human explicitly changes them:

- target: `trip_distance_miles`
- validation split: 20 percent of `Train.csv`
- split seed: `42`
- metric: validation RMSE
- lower `val_rmse` is better

## Baseline run

Run:

```bash
uv run train.py > run.log 2>&1
```

Then extract:

```bash
grep "^val_rmse:\|^best_iteration:" run.log
```

## Lead agent workflow

The lead agent should:

1. Study the current baseline, results, and recent failures.
2. Research the next promising idea.
3. Form exactly one bounded experiment hypothesis at a time.
4. Spawn a coding worker.
5. Give that worker clear ownership of:
   - `train.py`
   - optionally `README.md` or `program.md` if documentation needs to be updated
6. Review the returned metric and code changes.
7. Keep and push only true improvements.
8. Discard regressions or crashes and continue immediately.

## Coding worker workflow

The coding worker should:

1. Implement only the assigned experiment.
2. Run the experiment and collect:
   - `val_rmse`
   - `best_iteration`
   - any crash details if applicable
3. If the run improves the best known `val_rmse`:
   - keep the code changes
   - update `results.tsv`
   - Run the model on `Test.csv` and save the output as shown in `Submission.csv` file in `Predictions` folder.
   - commit the improvement
   - push to the current branch
4. If the run does not improve:
   - do not keep the experiment
   - log it as `discard` or `crash`
   - return control to the lead agent for the next idea

## Results logging

Keep `results.tsv` tab-separated with this schema:

```text
timestamp	run	status	val_rmse	best_iteration	commit	branch	description	model_config	feature_config
```

`results.tsv` must not be committed.
Always keep the predictions from the model in a saperate folder called: `Predictions`. Keep the file updting with best results. Make sure the file is named as `prediction_taxi.csv` and is always updated with latest results.

## What can change

- feature engineering inside `train.py`
- model hyperparameters
- training logic
- submission generation
- repo docs when the operating workflow changes

## What must not change casually

- dataset contents
- target column
- fixed validation protocol
- `val_rmse:` summary line

## Fallback mode

`train.py --autoloop` exists as a non-LLM fallback search loop. It is not the primary workflow. The preferred workflow is still:

- `gpt-5.2` for research/control
- `gpt-5.3-codex` at `high` reasoning for coding, committing, and pushing

## Never stop

Once the experiment loop begins, do not ask whether to continue. Keep researching, delegating, evaluating, and only preserving genuine improvements until the user manually stops the run.
