#!/usr/bin/env python3
"""Single-file baseline for autonomous taxi trip distance experiments."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

try:
    from catboost import CatBoostRegressor
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "CatBoost is required. Install dependencies first with `uv sync`."
    ) from exc


TARGET_COLUMN = "trip_distance_miles"
VALIDATION_FRACTION = 0.20
RANDOM_STATE = 42


@dataclass
class ExperimentConfig:
    iterations: int = 2500
    learning_rate: float = 0.03
    depth: int = 8
    l2_leaf_reg: float = 5.0
    min_data_in_leaf: int = 10
    random_strength: float = 1.0
    subsample: float = 0.85
    bootstrap_type: str = "Bernoulli"
    early_stopping_rounds: int = 200


MODEL_CONFIG = ExperimentConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a taxi trip distance baseline."
    )
    parser.add_argument("--train-path", default=None, help="Optional explicit path to Train.csv")
    parser.add_argument("--test-path", default=None, help="Optional explicit path to Test.csv")
    parser.add_argument(
        "--submission-path",
        default=None,
        help="Optional explicit path to Submission.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where predictions and metrics will be saved.",
    )
    return parser.parse_args()


def resolve_input_path(explicit_path: str | None, filename: str) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Could not find explicit path: {path}")
        return path

    repo_root = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / filename,
        repo_root / filename,
        repo_root.parent / filename,
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {filename}. Searched:\n{searched}")


def haversine_distance_miles(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> np.ndarray:
    earth_radius_miles = 3958.7613
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(
        np.radians,
        (lat1.to_numpy(), lon1.to_numpy(), lat2.to_numpy(), lon2.to_numpy()),
    )
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2.0) ** 2
    )
    return 2 * earth_radius_miles * np.arcsin(np.sqrt(a))


def make_regression_bins(target: pd.Series, max_bins: int = 10) -> pd.Series | None:
    unique_values = target.nunique(dropna=False)
    bin_count = min(max_bins, unique_values)
    if bin_count < 2:
        return None

    bins = pd.qcut(target, q=bin_count, labels=False, duplicates="drop")
    if pd.Series(bins).nunique(dropna=False) < 2:
        return None
    return bins


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.copy()

    parsed_datetimes: dict[str, pd.Series] = {}
    for column in ("pickup_datetime", "dropoff_datetime"):
        datetime_values = pd.to_datetime(
            features[column],
            format="%m/%d/%y %H:%M",
            errors="coerce",
        )
        parsed_datetimes[column] = datetime_values
        prefix = column.replace("_datetime", "")
        features[f"{prefix}_month"] = datetime_values.dt.month
        features[f"{prefix}_day"] = datetime_values.dt.day
        features[f"{prefix}_dayofweek"] = datetime_values.dt.dayofweek
        features[f"{prefix}_hour"] = datetime_values.dt.hour
        features[f"{prefix}_minute"] = datetime_values.dt.minute
        features[f"{prefix}_is_weekend"] = (datetime_values.dt.dayofweek >= 5).astype(int)

    duration_minutes = (
        parsed_datetimes["dropoff_datetime"] - parsed_datetimes["pickup_datetime"]
    ).dt.total_seconds() / 60.0
    features["trip_duration_minutes"] = duration_minutes.clip(lower=0)

    features["straight_line_distance_miles"] = haversine_distance_miles(
        features["pickup_latitude"],
        features["pickup_longitude"],
        features["dropoff_latitude"],
        features["dropoff_longitude"],
    )
    features["lat_diff"] = features["dropoff_latitude"] - features["pickup_latitude"]
    features["lon_diff"] = features["dropoff_longitude"] - features["pickup_longitude"]
    features["abs_lat_diff"] = features["lat_diff"].abs()
    features["abs_lon_diff"] = features["lon_diff"].abs()

    features["estimated_speed_mph"] = np.where(
        features["trip_duration_minutes"] > 0,
        features["straight_line_distance_miles"]
        / features["trip_duration_minutes"]
        * 60.0,
        0.0,
    )

    passenger_count = features["passenger_count"].replace(0, 1)
    straight_line = features["straight_line_distance_miles"].replace(0, np.nan)
    trip_duration = features["trip_duration_minutes"].replace(0, np.nan)
    fare_estimate = features["fare_estimate"].replace(0, np.nan)

    features["total_estimated_cost"] = (
        features["fare_estimate"] + features["tip_amount"] + features["tolls_amount"]
    )
    features["fare_per_passenger"] = features["fare_estimate"] / passenger_count
    features["tip_fraction_of_fare"] = features["tip_amount"] / fare_estimate
    features["fare_per_straight_mile"] = features["fare_estimate"] / straight_line
    features["fare_per_minute"] = features["fare_estimate"] / trip_duration
    features["tolls_per_straight_mile"] = features["tolls_amount"] / straight_line
    features = features.replace([np.inf, -np.inf], np.nan)

    for column in features.select_dtypes(include="object").columns:
        features[column] = features[column].fillna("missing").astype(str)

    return features


def build_model(random_state: int, iterations: int | None = None) -> CatBoostRegressor:
    effective_iterations = MODEL_CONFIG.iterations if iterations is None else iterations
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=effective_iterations,
        learning_rate=MODEL_CONFIG.learning_rate,
        depth=MODEL_CONFIG.depth,
        l2_leaf_reg=MODEL_CONFIG.l2_leaf_reg,
        min_data_in_leaf=MODEL_CONFIG.min_data_in_leaf,
        random_strength=MODEL_CONFIG.random_strength,
        subsample=MODEL_CONFIG.subsample,
        bootstrap_type=MODEL_CONFIG.bootstrap_type,
        random_seed=random_state,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )


def fit_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    random_state: int,
) -> CatBoostRegressor:
    categorical_columns = x_train.select_dtypes(include="object").columns.tolist()
    categorical_indices = [x_train.columns.get_loc(column) for column in categorical_columns]
    model = build_model(random_state=random_state)
    model.fit(
        x_train,
        y_train,
        eval_set=(x_valid, y_valid),
        cat_features=categorical_indices,
        use_best_model=True,
        early_stopping_rounds=MODEL_CONFIG.early_stopping_rounds,
    )
    return model


def fit_final_model(
    x_full: pd.DataFrame,
    y_full: pd.Series,
    random_state: int,
    iterations: int,
) -> CatBoostRegressor:
    categorical_columns = x_full.select_dtypes(include="object").columns.tolist()
    categorical_indices = [x_full.columns.get_loc(column) for column in categorical_columns]
    model = build_model(random_state=random_state, iterations=iterations)
    model.fit(x_full, y_full, cat_features=categorical_indices)
    return model


def save_submission(
    submission_template_path: Path | None,
    predictions: np.ndarray,
    output_path: Path,
) -> None:
    if submission_template_path is not None and submission_template_path.exists():
        submission = pd.read_csv(submission_template_path)
    else:
        submission = pd.DataFrame(columns=[TARGET_COLUMN])

    if len(submission) not in (0, len(predictions)):
        raise ValueError(
            f"Submission template has {len(submission)} rows but predictions have "
            f"{len(predictions)} rows."
        )

    if TARGET_COLUMN not in submission.columns or len(submission) == 0:
        submission = pd.DataFrame({TARGET_COLUMN: predictions})
    else:
        submission[TARGET_COLUMN] = predictions

    submission.to_csv(output_path, index=False)


def print_summary(metrics: dict[str, str | float | int]) -> None:
    print("---")
    for key, value in metrics.items():
        print(f"{key + ':':<19} {value}")


def main() -> None:
    total_start = time.time()
    args = parse_args()

    train_path = resolve_input_path(args.train_path, "Train.csv")
    test_path = resolve_input_path(args.test_path, "Test.csv")
    submission_path = (
        resolve_input_path(args.submission_path, "Submission.csv")
        if args.submission_path is not None
        or (Path.cwd() / "Submission.csv").exists()
        or (Path(__file__).resolve().parent / "Submission.csv").exists()
        or (Path(__file__).resolve().parent.parent / "Submission.csv").exists()
        else None
    )

    output_dir = (Path(__file__).resolve().parent / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"Training data must contain `{TARGET_COLUMN}`.")

    split_bins = make_regression_bins(train_df[TARGET_COLUMN])
    train_split_df, valid_split_df = train_test_split(
        train_df,
        test_size=VALIDATION_FRACTION,
        random_state=RANDOM_STATE,
        stratify=split_bins,
    )
    train_split_df = train_split_df.reset_index(drop=True)
    valid_split_df = valid_split_df.reset_index(drop=True)

    train_split_path = output_dir / "train_split.csv"
    valid_split_path = output_dir / "validation_split.csv"
    train_split_df.to_csv(train_split_path, index=False)
    valid_split_df.to_csv(valid_split_path, index=False)

    x_train = engineer_features(train_split_df.drop(columns=[TARGET_COLUMN]))
    y_train = train_split_df[TARGET_COLUMN]
    x_valid = engineer_features(valid_split_df.drop(columns=[TARGET_COLUMN]))
    y_valid = valid_split_df[TARGET_COLUMN]

    x_test = engineer_features(test_df.drop(columns=[TARGET_COLUMN], errors="ignore"))
    x_test = x_test[x_train.columns]

    train_start = time.time()
    model = fit_model(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        random_state=RANDOM_STATE,
    )
    training_seconds = time.time() - train_start

    validation_predictions = np.clip(model.predict(x_valid), a_min=0.0, a_max=None)
    val_rmse = math.sqrt(mean_squared_error(y_valid, validation_predictions))

    validation_output = valid_split_df.copy()
    validation_output["predicted_trip_distance_miles"] = validation_predictions
    validation_output["absolute_error"] = (
        validation_output["predicted_trip_distance_miles"] - validation_output[TARGET_COLUMN]
    ).abs()
    validation_predictions_path = output_dir / "validation_predictions.csv"
    validation_output.to_csv(validation_predictions_path, index=False)

    full_train_features = engineer_features(train_df.drop(columns=[TARGET_COLUMN]))
    full_train_target = train_df[TARGET_COLUMN]
    final_iterations = max(200, model.tree_count_)
    final_model = fit_final_model(
        x_full=full_train_features,
        y_full=full_train_target,
        random_state=RANDOM_STATE,
        iterations=final_iterations,
    )

    test_predictions = np.clip(final_model.predict(x_test), a_min=0.0, a_max=None)
    submission_output_path = output_dir / "submission_predictions.csv"
    save_submission(submission_path, test_predictions, submission_output_path)

    metrics = {
        "val_rmse": f"{val_rmse:.6f}",
        "training_seconds": f"{training_seconds:.1f}",
        "total_seconds": f"{time.time() - total_start:.1f}",
        "best_iteration": model.get_best_iteration(),
        "train_rows": len(train_split_df),
        "validation_rows": len(valid_split_df),
        "num_features": x_train.shape[1],
        "submission_file": str(submission_output_path.relative_to(Path(__file__).resolve().parent)),
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print_summary(metrics)


if __name__ == "__main__":
    main()
