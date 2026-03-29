#!/usr/bin/env python3
"""Single-file baseline and endless search loop for taxi trip distance experiments."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
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
    bagging_temperature: float = 1.0
    early_stopping_rounds: int = 200


@dataclass
class FeatureConfig:
    keep_raw_datetime_strings: bool = True
    add_borough_pair: bool = False
    add_same_borough_flag: bool = False
    add_peak_period_features: bool = False
    add_cyclical_time_features: bool = False
    add_coordinate_center_features: bool = False
    add_manhattan_distance_feature: bool = False
    add_coordinate_bins: bool = False


# AUTOTUNE_CONFIG_START
MODEL_CONFIG = ExperimentConfig(
    iterations=2500,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=10.0,
    min_data_in_leaf=22,
    random_strength=1.0,
    subsample=0.85,
    bootstrap_type='Bernoulli',
    bagging_temperature=1.0,
    early_stopping_rounds=200,
)
FEATURE_CONFIG = FeatureConfig(
    keep_raw_datetime_strings=True,
    add_borough_pair=False,
    add_same_borough_flag=True,
    add_peak_period_features=False,
    add_cyclical_time_features=False,
    add_coordinate_center_features=False,
    add_manhattan_distance_feature=False,
    add_coordinate_bins=False,
)
# AUTOTUNE_CONFIG_END


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and search taxi trip distance models."
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
    parser.add_argument(
        "--autoloop",
        action="store_true",
        help="Run endless search: keep only improvements and continue until interrupted.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional limit for autoloop candidate runs. Omit for endless search.",
    )
    parser.add_argument(
        "--results-path",
        default="results.tsv",
        help="TSV log file used by autoloop mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for candidate generation in autoloop mode.",
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=1e-6,
        help="Minimum RMSE decrease required to keep a candidate.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow autoloop to start with a dirty git tree.",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Do not push improved commits during autoloop.",
    )
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help="Do not persist or commit improved configurations during autoloop.",
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


def resolve_optional_submission_path(explicit_path: str | None) -> Path | None:
    if explicit_path is not None:
        return resolve_input_path(explicit_path, "Submission.csv")

    repo_root = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / "Submission.csv",
        repo_root / "Submission.csv",
        repo_root.parent / "Submission.csv",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


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


def get_categorical_columns(frame: pd.DataFrame) -> list[str]:
    return frame.select_dtypes(include=["object", "string"]).columns.tolist()


def engineer_features(frame: pd.DataFrame, feature_config: FeatureConfig) -> pd.DataFrame:
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

        if feature_config.add_cyclical_time_features:
            minute_of_day = datetime_values.dt.hour.fillna(0) * 60 + datetime_values.dt.minute.fillna(0)
            angle = 2.0 * np.pi * minute_of_day / (24.0 * 60.0)
            features[f"{prefix}_time_sin"] = np.sin(angle)
            features[f"{prefix}_time_cos"] = np.cos(angle)

        if feature_config.add_peak_period_features:
            features[f"{prefix}_is_peak_hour"] = datetime_values.dt.hour.isin(
                [7, 8, 9, 16, 17, 18, 19]
            ).astype(int)

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

    if feature_config.add_coordinate_center_features:
        features["route_center_latitude"] = (
            features["pickup_latitude"] + features["dropoff_latitude"]
        ) / 2.0
        features["route_center_longitude"] = (
            features["pickup_longitude"] + features["dropoff_longitude"]
        ) / 2.0

    if feature_config.add_manhattan_distance_feature:
        features["manhattan_distance_proxy"] = (
            features["abs_lat_diff"] * 69.0 + features["abs_lon_diff"] * 52.0
        )

    if feature_config.add_same_borough_flag:
        features["same_borough_trip"] = (
            features["pickup_borough"].astype(str) == features["dropoff_borough"].astype(str)
        ).astype(int)

    if feature_config.add_borough_pair:
        features["borough_route"] = (
            features["pickup_borough"].astype(str) + "->" + features["dropoff_borough"].astype(str)
        )
        features["traffic_route_profile"] = (
            features["traffic_congestion_level"].astype(str) + "|" + features["borough_route"].astype(str)
        )

    if feature_config.add_coordinate_bins:
        pickup_lat_bin = features["pickup_latitude"].round(2).astype(str)
        pickup_lon_bin = features["pickup_longitude"].round(2).astype(str)
        dropoff_lat_bin = features["dropoff_latitude"].round(2).astype(str)
        dropoff_lon_bin = features["dropoff_longitude"].round(2).astype(str)
        features["pickup_geo_bin"] = pickup_lat_bin + "|" + pickup_lon_bin
        features["dropoff_geo_bin"] = dropoff_lat_bin + "|" + dropoff_lon_bin
        features["route_geo_bin"] = features["pickup_geo_bin"] + "->" + features["dropoff_geo_bin"]

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

    if not feature_config.keep_raw_datetime_strings:
        features = features.drop(columns=["pickup_datetime", "dropoff_datetime"])

    for column in get_categorical_columns(features):
        features[column] = features[column].fillna("missing").astype(str)

    return features


def build_model(model_config: ExperimentConfig, random_state: int, iterations: int | None = None) -> CatBoostRegressor:
    effective_iterations = model_config.iterations if iterations is None else iterations
    params: dict[str, float | int | str | bool] = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": effective_iterations,
        "learning_rate": model_config.learning_rate,
        "depth": model_config.depth,
        "l2_leaf_reg": model_config.l2_leaf_reg,
        "min_data_in_leaf": model_config.min_data_in_leaf,
        "random_strength": model_config.random_strength,
        "bootstrap_type": model_config.bootstrap_type,
        "random_seed": random_state,
        "thread_count": -1,
        "verbose": False,
        "allow_writing_files": False,
    }
    if model_config.bootstrap_type == "Bayesian":
        params["bagging_temperature"] = model_config.bagging_temperature
    else:
        params["subsample"] = model_config.subsample
    return CatBoostRegressor(**params)


def fit_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    model_config: ExperimentConfig,
    random_state: int,
) -> CatBoostRegressor:
    categorical_columns = get_categorical_columns(x_train)
    categorical_indices = [x_train.columns.get_loc(column) for column in categorical_columns]
    model = build_model(model_config=model_config, random_state=random_state)
    model.fit(
        x_train,
        y_train,
        eval_set=(x_valid, y_valid),
        cat_features=categorical_indices,
        use_best_model=True,
        early_stopping_rounds=model_config.early_stopping_rounds,
    )
    return model


def fit_final_model(
    x_full: pd.DataFrame,
    y_full: pd.Series,
    model_config: ExperimentConfig,
    random_state: int,
    iterations: int,
) -> CatBoostRegressor:
    categorical_columns = get_categorical_columns(x_full)
    categorical_indices = [x_full.columns.get_loc(column) for column in categorical_columns]
    model = build_model(model_config=model_config, random_state=random_state, iterations=iterations)
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


def run_experiment(
    train_path: Path,
    test_path: Path,
    submission_path: Path | None,
    output_dir: Path,
    model_config: ExperimentConfig,
    feature_config: FeatureConfig,
    generate_submission: bool,
    print_metrics: bool,
) -> dict[str, str | float | int | dict[str, float | int | bool | str]]:
    total_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
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

    x_train = engineer_features(train_split_df.drop(columns=[TARGET_COLUMN]), feature_config)
    y_train = train_split_df[TARGET_COLUMN]
    x_valid = engineer_features(valid_split_df.drop(columns=[TARGET_COLUMN]), feature_config)
    y_valid = valid_split_df[TARGET_COLUMN]

    train_start = time.time()
    model = fit_model(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        model_config=model_config,
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

    submission_file = "not_generated"
    final_iterations = max(200, int(model.tree_count_))

    if generate_submission:
        test_df = pd.read_csv(test_path)
        x_test = engineer_features(test_df.drop(columns=[TARGET_COLUMN], errors="ignore"), feature_config)
        x_test = x_test[x_train.columns]

        full_train_features = engineer_features(train_df.drop(columns=[TARGET_COLUMN]), feature_config)
        full_train_target = train_df[TARGET_COLUMN]
        final_model = fit_final_model(
            x_full=full_train_features,
            y_full=full_train_target,
            model_config=model_config,
            random_state=RANDOM_STATE,
            iterations=final_iterations,
        )

        test_predictions = np.clip(final_model.predict(x_test), a_min=0.0, a_max=None)
        submission_output_path = output_dir / "submission_predictions.csv"
        save_submission(submission_path, test_predictions, submission_output_path)
        submission_file = str(submission_output_path.relative_to(Path(__file__).resolve().parent))

    metrics = {
        "val_rmse": f"{val_rmse:.6f}",
        "training_seconds": f"{training_seconds:.1f}",
        "total_seconds": f"{time.time() - total_start:.1f}",
        "best_iteration": int(model.get_best_iteration()),
        "train_rows": len(train_split_df),
        "validation_rows": len(valid_split_df),
        "num_features": x_train.shape[1],
        "submission_file": submission_file,
        "model_config": asdict(model_config),
        "feature_config": asdict(feature_config),
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    if print_metrics:
        summary_metrics = {
            key: value
            for key, value in metrics.items()
            if key not in {"model_config", "feature_config"}
        }
        print_summary(summary_metrics)

    return metrics


def format_dataclass_assignment(name: str, class_name: str, instance: object) -> str:
    lines = [f"{name} = {class_name}("]
    for key, value in asdict(instance).items():
        lines.append(f"    {key}={value!r},")
    lines.append(")")
    return "\n".join(lines)


def render_autotune_block(model_config: ExperimentConfig, feature_config: FeatureConfig) -> str:
    return (
        "# AUTOTUNE_CONFIG_START\n"
        + format_dataclass_assignment("MODEL_CONFIG", "ExperimentConfig", model_config)
        + "\n"
        + format_dataclass_assignment("FEATURE_CONFIG", "FeatureConfig", feature_config)
        + "\n# AUTOTUNE_CONFIG_END"
    )


def persist_best_config(
    source_file: Path,
    model_config: ExperimentConfig,
    feature_config: FeatureConfig,
) -> None:
    original_text = source_file.read_text(encoding="utf-8")
    pattern = re.compile(
        r"# AUTOTUNE_CONFIG_START\n.*?\n# AUTOTUNE_CONFIG_END",
        re.DOTALL,
    )
    replacement = render_autotune_block(model_config, feature_config)
    updated_text, replacements = pattern.subn(replacement, original_text, count=1)
    if replacements != 1:
        raise RuntimeError("Failed to update the persisted config block in train.py")
    source_file.write_text(updated_text, encoding="utf-8")


def run_git(repo_root: Path, args: list[str], capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def ensure_clean_git_tree(repo_root: Path) -> None:
    status = run_git(repo_root, ["status", "--short"]).stdout.strip()
    if status:
        raise RuntimeError(
            "Autoloop requires a clean git working tree. Commit or stash current changes first."
        )


def init_results_file(results_path: Path) -> None:
    if results_path.exists():
        return
    header = (
        "timestamp\trun\tstatus\tval_rmse\tbest_iteration\tcommit\tbranch\t"
        "description\tmodel_config\tfeature_config\n"
    )
    results_path.write_text(header, encoding="utf-8")


def append_result(
    results_path: Path,
    run_index: int,
    status: str,
    val_rmse: str,
    best_iteration: int,
    commit_hash: str,
    branch: str,
    description: str,
    model_config: ExperimentConfig,
    feature_config: FeatureConfig,
) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    model_summary = json.dumps(asdict(model_config), sort_keys=True, separators=(",", ":"))
    feature_summary = json.dumps(asdict(feature_config), sort_keys=True, separators=(",", ":"))
    row = (
        f"{timestamp}\t{run_index}\t{status}\t{val_rmse}\t{best_iteration}\t{commit_hash}\t"
        f"{branch}\t{description}\t{model_summary}\t{feature_summary}\n"
    )
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(row)


def get_current_branch(repo_root: Path) -> str:
    return run_git(repo_root, ["branch", "--show-current"]).stdout.strip()


def get_head_commit(repo_root: Path) -> str:
    return run_git(repo_root, ["rev-parse", "--short", "HEAD"]).stdout.strip()


def commit_current_config(repo_root: Path, message: str) -> str:
    run_git(repo_root, ["add", "train.py"], capture_output=False)
    run_git(repo_root, ["commit", "-m", message], capture_output=False)
    return get_head_commit(repo_root)


def push_current_branch(repo_root: Path) -> None:
    branch = get_current_branch(repo_root)
    run_git(repo_root, ["push", "origin", branch], capture_output=False)


def update_best_outputs(latest_dir: Path, best_dir: Path) -> None:
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(latest_dir, best_dir)


def clamp_float(value: float, lower: float, upper: float, digits: int = 5) -> float:
    return round(max(lower, min(upper, value)), digits)


def clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def mutate_model_config(model_config: ExperimentConfig, rng: random.Random) -> None:
    mutation = rng.choice(
        [
            "iterations",
            "learning_rate",
            "depth",
            "l2_leaf_reg",
            "min_data_in_leaf",
            "random_strength",
            "bootstrap_type",
            "subsample",
            "bagging_temperature",
            "early_stopping_rounds",
        ]
    )

    if mutation == "iterations":
        factor = rng.choice([0.75, 0.85, 1.15, 1.30, 1.45])
        model_config.iterations = clamp_int(int(round(model_config.iterations * factor / 50.0) * 50), 800, 5000)
    elif mutation == "learning_rate":
        factor = rng.choice([0.7, 0.85, 1.15, 1.3, 1.45])
        model_config.learning_rate = clamp_float(model_config.learning_rate * factor, 0.008, 0.15)
    elif mutation == "depth":
        model_config.depth = clamp_int(model_config.depth + rng.choice([-2, -1, 1, 2]), 4, 12)
    elif mutation == "l2_leaf_reg":
        factor = rng.choice([0.6, 0.8, 1.25, 1.5, 2.0])
        model_config.l2_leaf_reg = clamp_float(model_config.l2_leaf_reg * factor, 1.0, 30.0, digits=3)
    elif mutation == "min_data_in_leaf":
        model_config.min_data_in_leaf = clamp_int(
            model_config.min_data_in_leaf + rng.choice([-8, -5, -2, 2, 5, 8, 12]),
            1,
            64,
        )
    elif mutation == "random_strength":
        model_config.random_strength = clamp_float(
            model_config.random_strength + rng.choice([-0.5, -0.25, 0.25, 0.5, 1.0]),
            0.0,
            5.0,
            digits=3,
        )
    elif mutation == "bootstrap_type":
        model_config.bootstrap_type = rng.choice(["Bernoulli", "Bayesian", "MVS"])
    elif mutation == "subsample":
        model_config.subsample = clamp_float(
            model_config.subsample + rng.choice([-0.15, -0.1, -0.05, 0.05, 0.1, 0.15]),
            0.55,
            1.0,
            digits=3,
        )
    elif mutation == "bagging_temperature":
        model_config.bagging_temperature = clamp_float(
            model_config.bagging_temperature + rng.choice([-0.5, -0.25, 0.25, 0.5, 1.0]),
            0.0,
            10.0,
            digits=3,
        )
    elif mutation == "early_stopping_rounds":
        model_config.early_stopping_rounds = clamp_int(
            model_config.early_stopping_rounds + rng.choice([-50, -25, 25, 50, 100]),
            50,
            500,
        )


def mutate_feature_config(feature_config: FeatureConfig, rng: random.Random) -> None:
    field_name = rng.choice(list(asdict(feature_config).keys()))
    setattr(feature_config, field_name, not getattr(feature_config, field_name))


def propose_candidate(
    best_model_config: ExperimentConfig,
    best_feature_config: FeatureConfig,
    rng: random.Random,
) -> tuple[ExperimentConfig, FeatureConfig]:
    while True:
        candidate_model_config = copy.deepcopy(best_model_config)
        candidate_feature_config = copy.deepcopy(best_feature_config)
        for _ in range(rng.randint(1, 4)):
            if rng.random() < 0.7:
                mutate_model_config(candidate_model_config, rng)
            else:
                mutate_feature_config(candidate_feature_config, rng)
        if (
            candidate_model_config != best_model_config
            or candidate_feature_config != best_feature_config
        ):
            return candidate_model_config, candidate_feature_config


def describe_candidate_changes(
    base_model_config: ExperimentConfig,
    base_feature_config: FeatureConfig,
    candidate_model_config: ExperimentConfig,
    candidate_feature_config: FeatureConfig,
) -> str:
    changes: list[str] = []
    for key, value in asdict(candidate_model_config).items():
        if value != getattr(base_model_config, key):
            changes.append(f"{key}={value}")
    for key, value in asdict(candidate_feature_config).items():
        if value != getattr(base_feature_config, key):
            changes.append(f"{key}={value}")
    return ", ".join(changes[:6]) if changes else "no visible changes"


def run_autoloop(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parent
    source_file = Path(__file__).resolve()
    if not args.allow_dirty:
        ensure_clean_git_tree(repo_root)

    train_path = resolve_input_path(args.train_path, "Train.csv")
    test_path = resolve_input_path(args.test_path, "Test.csv")
    submission_path = resolve_optional_submission_path(args.submission_path)

    output_root = (repo_root / args.output_dir).resolve()
    latest_dir = output_root / "latest"
    best_dir = output_root / "best"
    results_path = (repo_root / args.results_path).resolve()
    init_results_file(results_path)

    branch = get_current_branch(repo_root)
    rng = random.Random(args.seed)
    best_model_config = copy.deepcopy(MODEL_CONFIG)
    best_feature_config = copy.deepcopy(FEATURE_CONFIG)

    print("[autoloop] running baseline evaluation")
    baseline_metrics = run_experiment(
        train_path=train_path,
        test_path=test_path,
        submission_path=submission_path,
        output_dir=latest_dir,
        model_config=best_model_config,
        feature_config=best_feature_config,
        generate_submission=False,
        print_metrics=True,
    )
    best_rmse = float(baseline_metrics["val_rmse"])
    best_iteration = int(baseline_metrics["best_iteration"])
    update_best_outputs(latest_dir, best_dir)
    baseline_full_metrics = run_experiment(
        train_path=train_path,
        test_path=test_path,
        submission_path=submission_path,
        output_dir=best_dir,
        model_config=best_model_config,
        feature_config=best_feature_config,
        generate_submission=True,
        print_metrics=False,
    )
    append_result(
        results_path=results_path,
        run_index=0,
        status="keep",
        val_rmse=str(baseline_metrics["val_rmse"]),
        best_iteration=best_iteration,
        commit_hash=get_head_commit(repo_root),
        branch=branch,
        description="baseline",
        model_config=best_model_config,
        feature_config=best_feature_config,
    )
    print(
        f"[autoloop] baseline best val_rmse={best_rmse:.6f}, "
        f"submission={baseline_full_metrics['submission_file']}"
    )

    run_index = 0
    while args.max_runs is None or run_index < args.max_runs:
        run_index += 1
        candidate_model_config, candidate_feature_config = propose_candidate(
            best_model_config=best_model_config,
            best_feature_config=best_feature_config,
            rng=rng,
        )
        description = describe_candidate_changes(
            base_model_config=best_model_config,
            base_feature_config=best_feature_config,
            candidate_model_config=candidate_model_config,
            candidate_feature_config=candidate_feature_config,
        )
        print(f"[autoloop] run {run_index}: testing {description}")

        try:
            candidate_metrics = run_experiment(
                train_path=train_path,
                test_path=test_path,
                submission_path=submission_path,
                output_dir=latest_dir,
                model_config=candidate_model_config,
                feature_config=candidate_feature_config,
                generate_submission=False,
                print_metrics=True,
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # pragma: no cover
            print(f"[autoloop] run {run_index}: crash - {exc}")
            append_result(
                results_path=results_path,
                run_index=run_index,
                status="crash",
                val_rmse="0.000000",
                best_iteration=0,
                commit_hash=get_head_commit(repo_root),
                branch=branch,
                description=f"{description}; crash={exc}",
                model_config=candidate_model_config,
                feature_config=candidate_feature_config,
            )
            continue

        candidate_rmse = float(candidate_metrics["val_rmse"])
        candidate_iteration = int(candidate_metrics["best_iteration"])
        improved = candidate_rmse < best_rmse - args.improvement_threshold

        if improved:
            best_model_config = candidate_model_config
            best_feature_config = candidate_feature_config
            best_rmse = candidate_rmse

            commit_hash = get_head_commit(repo_root)
            if not args.no_commit:
                persist_best_config(source_file, best_model_config, best_feature_config)
                commit_message = f"Improve val_rmse to {best_rmse:.6f}"
                commit_hash = commit_current_config(repo_root, commit_message)
                if not args.no_push:
                    try:
                        push_current_branch(repo_root)
                    except Exception as exc:  # pragma: no cover
                        print(f"[autoloop] push failed after improvement: {exc}")

            update_best_outputs(latest_dir, best_dir)
            run_experiment(
                train_path=train_path,
                test_path=test_path,
                submission_path=submission_path,
                output_dir=best_dir,
                model_config=best_model_config,
                feature_config=best_feature_config,
                generate_submission=True,
                print_metrics=False,
            )
            append_result(
                results_path=results_path,
                run_index=run_index,
                status="keep",
                val_rmse=str(candidate_metrics["val_rmse"]),
                best_iteration=candidate_iteration,
                commit_hash=commit_hash,
                branch=branch,
                description=description,
                model_config=best_model_config,
                feature_config=best_feature_config,
            )
            print(f"[autoloop] run {run_index}: improved best val_rmse to {best_rmse:.6f}")
        else:
            append_result(
                results_path=results_path,
                run_index=run_index,
                status="discard",
                val_rmse=str(candidate_metrics["val_rmse"]),
                best_iteration=candidate_iteration,
                commit_hash=get_head_commit(repo_root),
                branch=branch,
                description=description,
                model_config=candidate_model_config,
                feature_config=candidate_feature_config,
            )
            print(
                f"[autoloop] run {run_index}: discarded "
                f"candidate {candidate_rmse:.6f} >= best {best_rmse:.6f}"
            )


def main() -> None:
    args = parse_args()
    if args.autoloop:
        try:
            run_autoloop(args)
        except KeyboardInterrupt:
            print("\n[autoloop] stopped by user")
        return

    train_path = resolve_input_path(args.train_path, "Train.csv")
    test_path = resolve_input_path(args.test_path, "Test.csv")
    submission_path = resolve_optional_submission_path(args.submission_path)
    output_dir = (Path(__file__).resolve().parent / args.output_dir).resolve()

    run_experiment(
        train_path=train_path,
        test_path=test_path,
        submission_path=submission_path,
        output_dir=output_dir,
        model_config=MODEL_CONFIG,
        feature_config=FEATURE_CONFIG,
        generate_submission=True,
        print_metrics=True,
    )


if __name__ == "__main__":
    main()
