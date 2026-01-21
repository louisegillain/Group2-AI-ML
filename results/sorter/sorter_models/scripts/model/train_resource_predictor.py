

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

from scripts.data import PreProcessing


TARGET_COLUMNS = [
    "peak_ram_kb",
    "avg_ram_kb",
]


def parse_time_ram_file(details_path: Path) -> Optional[Dict[str, float]]:
    """Read wall-clock and RAM stats from time_RAM_details.txt."""
    mapping = {
        "Wall-clock time in seconds": "wall_clock_seconds",
        "Peak RAM usage in KB": "peak_ram_kb",
        "Average RAM usage in KB": "avg_ram_kb",
        "Minimum RAM usage in KB": "min_ram_kb",
    }
    stats: Dict[str, float] = {}
    try:
        for line in details_path.read_text().splitlines():
            if "=" not in line:
                continue
            key, value = [chunk.strip() for chunk in line.split("=", 1)]
            if key in mapping:
                try:
                    stats[mapping[key]] = float(value)
                except ValueError:
                    continue
        return stats if TARGET_COLUMNS[0] in stats else None
    except FileNotFoundError:
        return None


def _maybe_numeric(value):
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, bool):
        return int(value)
    return value


LOG_TARGETS = {"peak_ram_kb", "avg_ram_kb"}
ENV_BASELINE = 1.0
SMALL_BUFFER_THRESHOLD = 10240


class PerTargetRegressor:
    """Wraps target-specific regressors with buffer-regime gating."""

    def __init__(
        self,
        models: Dict[str, Dict[str, RandomForestRegressor] | RandomForestRegressor],
        buffer_feature_index: Optional[int],
        buffer_mean: Optional[float],
        buffer_scale: Optional[float],
        env_feature_index: Optional[int],
        env_mean: Optional[float],
        env_scale: Optional[float],
        env_baseline: float = ENV_BASELINE,
        scale_ram_by_env: bool = False,
        threshold: float = SMALL_BUFFER_THRESHOLD,
    ):
        self.models = models
        self.buffer_feature_index = buffer_feature_index
        self.buffer_mean = buffer_mean
        self.buffer_scale = buffer_scale
        self.env_feature_index = env_feature_index
        self.env_mean = env_mean
        self.env_scale = env_scale
        self.env_baseline = env_baseline
        self.scale_ram_by_env = scale_ram_by_env
        self.threshold = threshold

    def _buffer_values(self, X: np.ndarray) -> Optional[np.ndarray]:
        if (
            self.buffer_feature_index is None
            or self.buffer_mean is None
            or self.buffer_scale is None
        ):
            return None
        return (
            X[:, self.buffer_feature_index] * self.buffer_scale
            + self.buffer_mean
        )

    def _env_values(self, X: np.ndarray) -> Optional[np.ndarray]:
        if (
            self.env_feature_index is None
            or self.env_mean is None
            or self.env_scale is None
        ):
            return None
        return X[:, self.env_feature_index] * self.env_scale + self.env_mean

    def _env_multiplier(self, X: np.ndarray) -> Optional[np.ndarray]:
        env_values = self._env_values(X)
        if env_values is None:
            return None
        baseline = self.env_baseline or 1.0
        multiplier = env_values / baseline
        multiplier[~np.isfinite(multiplier)] = 1.0
        multiplier[multiplier <= 0] = 1.0
        return multiplier

    def predict(self, X: np.ndarray) -> np.ndarray:
        buffer_values = self._buffer_values(X)
        if buffer_values is not None:
            mask_small = np.isfinite(buffer_values) & (
                buffer_values <= self.threshold
            )
        else:
            mask_small = None

        env_multiplier = self._env_multiplier(X) if self.scale_ram_by_env else None

        outputs = []
        for target in TARGET_COLUMNS:
            target_model = self.models[target]
            if isinstance(target_model, dict):
                preds_raw = np.empty(X.shape[0])
                preds_raw[:] = np.nan
                if mask_small is not None and mask_small.any():
                    small_idx = mask_small.copy()
                else:
                    small_idx = np.zeros(X.shape[0], dtype=bool)
                large_idx = ~small_idx
                if "small" in target_model and small_idx.any():
                    preds_raw[small_idx] = target_model["small"].predict(
                        X[small_idx]
                    )
                if "large" in target_model and large_idx.any():
                    preds_raw[large_idx] = target_model["large"].predict(
                        X[large_idx]
                    )
                nan_idx = np.isnan(preds_raw)
                if nan_idx.any():
                    fallback = target_model.get("large") or target_model.get(
                        "small"
                    )
                    if fallback is not None:
                        preds_raw[nan_idx] = fallback.predict(X[nan_idx])
                preds = preds_raw
            else:
                preds = target_model.predict(X)

            if target in LOG_TARGETS:
                preds = np.exp(preds)
                if env_multiplier is not None:
                    preds = preds * env_multiplier
            outputs.append(preds)

        return np.column_stack(outputs)


TIMER_GAUGES = {
    "Sorter.Step.mean": "timer_step_mean",
    "Sorter.Environment.EpisodeLength.mean": "timer_episode_length_mean",
    "Sorter.Environment.CumulativeReward.mean": "timer_cumulative_reward_mean",
    "Sorter.Policy.Entropy.mean": "timer_entropy_mean",
    "Sorter.Policy.ExtrinsicReward.mean": "timer_extrinsic_reward_mean",
    "Sorter.Policy.ExtrinsicValueEstimate.mean": "timer_value_estimate_mean",
    "Sorter.Policy.LearningRate.mean": "timer_learning_rate_mean",
    "Sorter.Policy.Beta.mean": "timer_beta_mean",
    "Sorter.Policy.Epsilon.mean": "timer_epsilon_mean",
    "Sorter.Losses.PolicyLoss.mean": "timer_policy_loss_mean",
    "Sorter.Losses.ValueLoss.mean": "timer_value_loss_mean",
}


def extract_timer_metrics(timers_path: Path) -> Dict[str, float]:
    features: Dict[str, float] = {}
    try:
        payload = json.loads(timers_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return features

    gauges = payload.get("gauges", {})
    for raw_name, feature_name in TIMER_GAUGES.items():
        stats = gauges.get(raw_name)
        if not isinstance(stats, dict):
            continue
        value = stats.get("value")
        if isinstance(value, (int, float)):
            features[feature_name] = float(value)
    return features


def extract_training_status_features(status_path: Path) -> Dict[str, float]:
    features: Dict[str, float] = {}
    try:
        payload = json.loads(status_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return features

    for curriculum_key in ("num_tiles", "reward_mode"):
        lesson = payload.get(curriculum_key, {}).get("lesson_num")
        if isinstance(lesson, (int, float)):
            features[f"status_{curriculum_key}_lesson_num"] = float(lesson)

    sorter_block = payload.get("Sorter", {})
    checkpoints = sorter_block.get("checkpoints") or []
    if isinstance(checkpoints, list):
        features["status_checkpoint_count"] = float(len(checkpoints))
        checkpoint_steps = [
            cp.get("steps") for cp in checkpoints if isinstance(cp.get("steps"), (int, float))
        ]
        checkpoint_rewards = [
            cp.get("reward") for cp in checkpoints if isinstance(cp.get("reward"), (int, float))
        ]
        if checkpoint_steps:
            features["status_checkpoint_steps_max"] = float(max(checkpoint_steps))
            features["status_checkpoint_steps_min"] = float(min(checkpoint_steps))
        if checkpoint_rewards:
            features["status_checkpoint_reward_max"] = float(max(checkpoint_rewards))
            features["status_checkpoint_reward_min"] = float(min(checkpoint_rewards))
            features["status_checkpoint_reward_last"] = float(checkpoint_rewards[-1])

    final_checkpoint = sorter_block.get("final_checkpoint", {})
    for field in ("steps", "reward", "creation_time"):
        value = final_checkpoint.get(field)
        if isinstance(value, (int, float)):
            features[f"status_final_{field}"] = float(value)

    return features


def extract_run_log_features(run_dir: Path) -> Dict[str, float]:
    run_logs_dir = run_dir / "run_logs"
    features: Dict[str, float] = {}
    features.update(extract_timer_metrics(run_logs_dir / "timers.json"))
    features.update(extract_training_status_features(run_logs_dir / "training_status.json"))
    return features


def extract_environment_parameters(env_params: Dict) -> Dict[str, float]:
    """Pull basic curriculum stats (start/end values) from environment params."""
    extracted: Dict[str, float] = {}
    for name, payload in env_params.items():
        curriculum = payload.get("curriculum") if isinstance(payload, dict) else None
        if not curriculum:
            continue
        def _get_value(entry):
            val = entry.get("value", {})
            sampler_params = val.get("sampler_parameters", {})
            return sampler_params.get("value")

        first_val = _get_value(curriculum[0])
        last_val = _get_value(curriculum[-1])
        if first_val is not None:
            extracted[f"envparam_{name}_start"] = first_val
        if last_val is not None:
            extracted[f"envparam_{name}_end"] = last_val
        extracted[f"envparam_{name}_lessons"] = len(curriculum)
    return extracted


def flatten_configuration(config_path: Path) -> Dict[str, float]:
    """Extract numeric/categorical features from configuration.yaml."""
    cfg = yaml.safe_load(config_path.read_text())
    features: Dict[str, float] = {}

    behavior = cfg.get("behaviors", {}).get("Sorter", {})
    hyper = behavior.get("hyperparameters", {})
    for key, value in hyper.items():
        features[f"hyper_{key}"] = _maybe_numeric(value)

    net = behavior.get("network_settings", {})
    for key, value in net.items():
        if key == "memory" or value is None:
            continue
        features[f"net_{key}"] = _maybe_numeric(value)

    features["max_steps"] = behavior.get("max_steps")
    features["time_horizon"] = behavior.get("time_horizon")
    features["summary_freq"] = behavior.get("summary_freq")

    env_settings = cfg.get("env_settings", {})
    for key in ("num_envs", "num_areas", "base_port", "seed"):
        if key in env_settings:
            features[f"env_{key}"] = env_settings[key]

    engine_settings = cfg.get("engine_settings", {})
    for key in ("time_scale", "quality_level", "no_graphics", "target_frame_rate"):
        if key in engine_settings:
            features[f"engine_{key}"] = _maybe_numeric(engine_settings[key])

    env_params = cfg.get("environment_parameters", {})
    features.update(extract_environment_parameters(env_params))

    return features


def collect_runs(root: Path) -> pd.DataFrame:
    """Build a DataFrame with features + targets for every run under root."""
    rows: List[Dict[str, float]] = []
    for config_path in root.rglob("configuration.yaml"):
        run_dir = config_path.parent
        details_path = run_dir / "time_RAM_details.txt"
        try:
            PreProcessing.verifyContent(run_dir)
            PreProcessing.verifyRAM(run_dir)
        except Exception as exc:
            print(f"PreProcessing check failed for {run_dir}: {exc}")
        stats = parse_time_ram_file(details_path)
        if not stats:
            continue
        relative_run_dir = str(run_dir.relative_to(root))
        features = flatten_configuration(config_path)
        log_features = extract_run_log_features(run_dir)
        row = {**features, **log_features, **stats}
        row["run_dir"] = relative_run_dir
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No runs with configuration + time_RAM_details found under {root}")

    df = pd.DataFrame(rows)
    df = df.dropna(axis=0, subset=TARGET_COLUMNS)
    if "run_dir" in df.columns:
        df = df.sort_values("run_dir").reset_index(drop=True)
    return df


def clean_dataset(
    df: pd.DataFrame,
    min_wall_clock: Optional[float] = None,
    max_wall_clock: Optional[float] = None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    wall = df.get("wall_clock_seconds")
    if wall is not None:
        if min_wall_clock is not None:
            mask &= wall >= min_wall_clock
        if max_wall_clock is not None:
            mask &= wall <= max_wall_clock
    cleaned = df.loc[mask].copy()
    return cleaned


def filter_by_buffer_size(
    df: pd.DataFrame,
    min_buffer_size: Optional[float] = None,
    max_buffer_size: Optional[float] = None,
) -> pd.DataFrame:
    """Drop runs based on hyper_buffer_size bounds if that column exists."""

    if "hyper_buffer_size" not in df.columns:
        return df

    mask = pd.Series(True, index=df.index)
    if min_buffer_size is not None:
        mask &= df["hyper_buffer_size"] >= min_buffer_size
    if max_buffer_size is not None:
        mask &= df["hyper_buffer_size"] <= max_buffer_size

    return df.loc[mask].copy()


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.divide(denominator)
    return result.replace([np.inf, -np.inf], np.nan)


def _add_if_available(df: pd.DataFrame, col: str, values: pd.Series):
    df[col] = values


def add_run_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    categories = []
    groups = []
    leaves = []
    pairwise = []
    for entry in df["run_dir"].astype(str):
        parts = Path(entry).parts
        category = parts[0] if parts else "root"
        group = parts[1] if len(parts) > 1 else category
        leaf = parts[-1] if parts else entry
        categories.append(category)
        groups.append(group)
        leaves.append(leaf)
        pairwise.append(int("pairwise tests" in category.lower()))
    df["run_category"] = categories
    df["run_group"] = groups
    df["run_leaf"] = leaves
    df["is_pairwise"] = pairwise
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def has_cols(cols: List[str]) -> bool:
        return all(col in df.columns for col in cols)

    if has_cols(["hyper_batch_size"]):
        df["log_hyper_batch_size"] = np.log(df["hyper_batch_size"].replace(0, np.nan))
    if has_cols(["hyper_buffer_size"]):
        df["log_hyper_buffer_size"] = np.log(df["hyper_buffer_size"].replace(0, np.nan))
        df["is_small_buffer"] = (df["hyper_buffer_size"] <= 10240).astype(int)
    if has_cols(["hyper_buffer_size", "hyper_batch_size"]):
        df["ratio_buffer_per_batch"] = _safe_divide(
            df["hyper_buffer_size"], df["hyper_batch_size"]
        )
    if has_cols(["hyper_buffer_size", "time_horizon"]):
        df["ratio_buffer_per_time"] = _safe_divide(
            df["hyper_buffer_size"], df["time_horizon"]
        )
    if has_cols(["hyper_batch_size", "time_horizon"]):
        df["ratio_batch_per_time"] = _safe_divide(
            df["hyper_batch_size"], df["time_horizon"]
        )
    if has_cols(["net_hidden_units", "net_num_layers"]):
        df["network_units_total"] = df["net_hidden_units"] * df["net_num_layers"]
        df["network_units_quadratic"] = (
            (df["net_hidden_units"] ** 2) * df["net_num_layers"]
        )
    if has_cols(["time_horizon", "env_num_envs"]):
        df["time_horizon_per_env"] = _safe_divide(
            df["time_horizon"], df["env_num_envs"]
        )
    if has_cols(["max_steps", "env_num_envs"]):
        df["max_steps_per_env"] = _safe_divide(df["max_steps"], df["env_num_envs"])
    if has_cols(["summary_freq", "env_num_envs"]):
        df["summary_freq_per_env"] = _safe_divide(
            df["summary_freq"], df["env_num_envs"]
        )
    if has_cols(["envparam_num_tiles_end", "envparam_num_tiles_start"]):
        df["envparam_num_tiles_span"] = (
            df["envparam_num_tiles_end"] - df["envparam_num_tiles_start"]
        )
    if has_cols(["envparam_reward_mode_end", "envparam_reward_mode_start"]):
        df["envparam_reward_mode_span"] = (
            df["envparam_reward_mode_end"] - df["envparam_reward_mode_start"]
        )

    return df


def build_prediction_comparisons(
    df: pd.DataFrame, y_true: pd.DataFrame, predictions: np.ndarray
) -> pd.DataFrame:
    """Assemble a small table comparing actual vs predicted targets."""

    comparisons = pd.DataFrame({
        "run_dir": df.loc[y_true.index, "run_dir"].astype(str).values
    })
    for idx, target in enumerate(TARGET_COLUMNS):
        comparisons[f"actual_{target}"] = y_true.iloc[:, idx].values
        comparisons[f"pred_{target}"] = predictions[:, idx]
        comparisons[f"diff_{target}"] = (
            predictions[:, idx] - y_true.iloc[:, idx].values
        )
    return comparisons.reset_index(drop=True)


def train_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_ram_by_env: bool = False,
):
    """Encode features, split train/test, and fit buffer-aware regressors."""
    feature_cols = [col for col in df.columns if col not in TARGET_COLUMNS + ["run_dir"]]
    X = pd.get_dummies(df[feature_cols], drop_first=False)
    y = df[TARGET_COLUMNS]

    env_counts = df.get("env_num_envs")
    if env_counts is None:
        env_counts = pd.Series(1.0, index=df.index)
    env_counts = env_counts.astype(float).fillna(1.0).clip(lower=1.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    env_train = env_counts.loc[X_train.index].fillna(1.0)
    env_test = env_counts.loc[X_test.index].fillna(1.0)
    if scale_ram_by_env:
        for target in LOG_TARGETS:
            if target in y_train.columns:
                y_train.loc[:, target] = y_train[target] / env_train
    else:
        env_train = pd.Series(1.0, index=env_train.index)
        env_test = pd.Series(1.0, index=env_test.index)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    buffer_sizes = df.get("hyper_buffer_size")
    if buffer_sizes is None:
        buffer_sizes = pd.Series(np.nan, index=df.index)
    buffer_sizes = buffer_sizes.astype(float)
    buffer_train = buffer_sizes.loc[X_train.index]
    buffer_test = buffer_sizes.loc[X_test.index]
    small_train_mask = (buffer_train <= SMALL_BUFFER_THRESHOLD).fillna(False)
    small_test_mask = (buffer_test <= SMALL_BUFFER_THRESHOLD).fillna(False)
    small_train_mask = small_train_mask.reindex(X_train.index, fill_value=False)
    small_test_mask = small_test_mask.reindex(X_test.index, fill_value=False)

    base_params = dict(
        n_estimators=600,
        random_state=random_state,
        max_features=0.3,
        n_jobs=-1,
    )

    tuned_models: Dict[str, Dict[str, RandomForestRegressor] | RandomForestRegressor] = {}
    predictions = np.zeros((len(X_test), len(TARGET_COLUMNS)))
    metrics = []

    for idx, target in enumerate(TARGET_COLUMNS):
        y_train_target = y_train[target]
        y_test_target = y_test[target]

        if target in LOG_TARGETS:
            models_per_regime: Dict[str, RandomForestRegressor] = {}
            log_train = np.log(y_train_target.clip(lower=1))
            small_mask_arr = small_train_mask.to_numpy()
            large_mask_arr = ~small_mask_arr
            min_regime_samples = 10

            if small_mask_arr.sum() >= min_regime_samples and large_mask_arr.sum() >= min_regime_samples:
                reg_small = RandomForestRegressor(**base_params)
                reg_small.fit(X_train_scaled[small_mask_arr], log_train[small_mask_arr])
                reg_large = RandomForestRegressor(**base_params)
                reg_large.fit(X_train_scaled[large_mask_arr], log_train[large_mask_arr])
                models_per_regime["small"] = reg_small
                models_per_regime["large"] = reg_large
            else:
                fallback = RandomForestRegressor(**base_params)
                fallback.fit(X_train_scaled, log_train)
                models_per_regime["large"] = fallback

            tuned_models[target] = models_per_regime

            preds_log = np.empty(len(X_test_scaled))
            preds_log[:] = np.nan
            small_test_arr = small_test_mask.to_numpy()
            large_test_arr = ~small_test_arr
            if "small" in models_per_regime and small_test_arr.any():
                preds_log[small_test_arr] = models_per_regime["small"].predict(
                    X_test_scaled[small_test_arr]
                )
            if "large" in models_per_regime and large_test_arr.any():
                preds_log[large_test_arr] = models_per_regime["large"].predict(
                    X_test_scaled[large_test_arr]
                )
            nan_idx = np.isnan(preds_log)
            if nan_idx.any():
                fallback = models_per_regime.get("large") or models_per_regime.get("small")
                preds_log[nan_idx] = fallback.predict(X_test_scaled[nan_idx])

            env_test_arr = env_test.to_numpy()
            preds_linear = np.exp(preds_log)
            preds_linear *= env_test_arr
            predictions[:, idx] = preds_linear

            mae = mean_absolute_error(y_test_target, preds_linear)
            r2 = r2_score(y_test_target, preds_linear)
            log_actual = np.log(y_test_target.clip(lower=1))
            if scale_ram_by_env:
                log_preds = preds_log + np.log(env_test_arr.clip(min=1.0))
            else:
                log_preds = preds_log
            entry = {
                "target": target,
                "MAE": mae,
                "R2": r2,
                "R2_log": r2_score(log_actual, log_preds),
            }
            metrics.append(entry)
        else:
            regressor = RandomForestRegressor(**base_params)
            regressor.fit(X_train_scaled, y_train_target)
            tuned_models[target] = regressor
            preds_linear = regressor.predict(X_test_scaled)
            predictions[:, idx] = preds_linear
            mae = mean_absolute_error(y_test_target, preds_linear)
            r2 = r2_score(y_test_target, preds_linear)
            metrics.append({"target": target, "MAE": mae, "R2": r2})

    report = pd.DataFrame(metrics)
    comparison_table = build_prediction_comparisons(df, y_test, predictions)

    buffer_feature_index = (
        X.columns.get_loc("hyper_buffer_size") if "hyper_buffer_size" in X.columns else None
    )
    buffer_mean = (
        scaler.mean_[buffer_feature_index] if buffer_feature_index is not None else None
    )
    buffer_scale = (
        scaler.scale_[buffer_feature_index] if buffer_feature_index is not None else None
    )

    env_feature_index = (
        X.columns.get_loc("env_num_envs") if "env_num_envs" in X.columns else None
    )
    env_mean = (
        scaler.mean_[env_feature_index] if env_feature_index is not None else None
    )
    env_scale = (
        scaler.scale_[env_feature_index] if env_feature_index is not None else None
    )

    model = PerTargetRegressor(
        tuned_models,
        buffer_feature_index=buffer_feature_index,
        buffer_mean=buffer_mean,
        buffer_scale=buffer_scale,
        env_feature_index=env_feature_index,
        env_mean=env_mean,
        env_scale=env_scale,
        scale_ram_by_env=scale_ram_by_env,
    )
    return model, scaler, X.columns.tolist(), report, comparison_table

def save_model(model, scaler, feature_columns: List[str], path: Path):
    payload = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "targets": TARGET_COLUMNS,
    }
    joblib.dump(payload, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/sorter/YAMLchanged_Louise"),
        help="Directory that contains sorter experiment subfolders",
    )
    parser.add_argument(
        "--export-dataset",
        type=Path,
        default=None,
        help="Optional path to write the assembled dataset CSV",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Optional path to persist the trained model via joblib",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of runs to reserve for evaluation",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split and the regressor",
    )
    parser.add_argument(
        "--min-wall-clock",
        type=float,
        default=30.0,
        help="Drop runs with wall-clock time below this threshold (set negative to disable)",
    )
    parser.add_argument(
        "--max-wall-clock",
        type=float,
        default=None,
        help="Upper bound for wall-clock filtering (optional)",
    )
    parser.add_argument(
        "--comparison-samples",
        type=int,
        default=5,
        help="Number of hold-out runs to print for actual vs predicted comparison",
    )
    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Print MAE/R^2 validation metrics in addition to sample comparisons",
    )
    parser.add_argument(
        "--scale-ram-by-env",
        action="store_true",
        help="Divide RAM targets by env_num_envs during training and multiply predictions by env count",
    )
    parser.add_argument(
        "--min-buffer-size",
        type=float,
        default=None,
        help="Drop runs whose hyper_buffer_size is below this threshold (set negative to disable)",
    )
    parser.add_argument(
        "--max-buffer-size",
        type=float,
        default=None,
        help="Drop runs whose hyper_buffer_size exceeds this threshold",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root
    df_raw = collect_runs(root)
    print(f"Loaded {len(df_raw)} runs from {root}")

    min_wall = args.min_wall_clock if args.min_wall_clock is not None and args.min_wall_clock >= 0 else None
    df_clean = clean_dataset(
        df_raw,
        min_wall_clock=min_wall,
        max_wall_clock=args.max_wall_clock,
    )
    removed = len(df_raw) - len(df_clean)
    if removed:
        print(f"Removed {removed} run(s) based on wall-clock filters")

    min_buffer = (
        args.min_buffer_size
        if args.min_buffer_size is not None and args.min_buffer_size >= 0
        else None
    )
    df_buffer = filter_by_buffer_size(
        df_clean,
        min_buffer_size=min_buffer,
        max_buffer_size=args.max_buffer_size,
    )
    removed_buffer = len(df_clean) - len(df_buffer)
    if removed_buffer:
        print(f"Removed {removed_buffer} run(s) based on buffer-size filters")

    df_with_meta = add_run_metadata(df_buffer)
    df = add_engineered_features(df_with_meta)
    print(f"Dataset after feature engineering: {len(df)} runs, {len(df.columns)} columns")

    if args.export_dataset:
        args.export_dataset.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.export_dataset, index=False)
        print(f"Dataset exported to {args.export_dataset}")

    model, scaler, columns, report, comparisons = train_models(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        scale_ram_by_env=args.scale_ram_by_env,
    )
    sample_count = max(1, args.comparison_samples)
    print("\nSample hold-out predictions vs actuals:")
    print(
        comparisons.head(sample_count).to_string(index=False)
    )

    if args.show_metrics:
        print("\nValidation metrics:")
        print(report.to_string(index=False))

    if args.model_out:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        save_model(model, scaler, columns, args.model_out)
        print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
