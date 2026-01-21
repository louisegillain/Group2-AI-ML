"""Train an XGBoost regressor for wall-clock predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent))

from train_resource_predictor import (  # noqa
    add_engineered_features,
    add_run_metadata,
    clean_dataset,
    collect_runs,
    filter_by_buffer_size,
)
from gbdt_model import GBDTmodel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/sorter/YAMLchanged_Louise"),
        help="Directory with sorter experiment runs",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of runs reserved for validation",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=18,
        help="Random seed for train/test split and the regressor",
    )
    parser.add_argument(
        "--min-wall-clock",
        type=float,
        default=30.0,
        help="Drop runs with wall-clock below this threshold (negative to disable)",
    )
    parser.add_argument(
        "--min-buffer-size",
        type=float,
        default=10000,
        help="Minimum hyper_buffer_size to keep (negative to disable)",
    )
    parser.add_argument(
        "--max-buffer-size",
        type=float,
        default=None,
        help="Maximum hyper_buffer_size to keep",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("wallclock_model.pkl"),
        help="Where to persist the trained model artifact",
    )
    return parser.parse_args()


def load_dataset(args) -> tuple[pd.DataFrame, pd.Series]:
    df_raw = collect_runs(args.root)
    min_wc = args.min_wall_clock if args.min_wall_clock is not None and args.min_wall_clock >= 0 else None
    df_clean = clean_dataset(df_raw, min_wall_clock=min_wc, max_wall_clock=None)
    min_buffer = args.min_buffer_size if args.min_buffer_size is None or args.min_buffer_size >= 0 else None
    df_buffer = filter_by_buffer_size(
        df_clean,
        min_buffer_size=min_buffer,
        max_buffer_size=args.max_buffer_size,
    )
    df = add_engineered_features(add_run_metadata(df_buffer))
    feature_cols = [col for col in df.columns if col not in {"wall_clock_seconds", "run_dir"}]
    X = pd.get_dummies(df[feature_cols], drop_first=False)
    y = df["wall_clock_seconds"]
    return X, y


def main() -> None:
    args = parse_args()
    X, y = load_dataset(args)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GBDTmodel()
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Wall-clock MAE: {mae:.2f} seconds")
    print(f"Wall-clock R^2: {r2:.3f}")

    if args.model_out:
        payload = {
            "model": model,
            "scaler": scaler,
            "feature_columns": X.columns.tolist(),
            "target": "wall_clock_seconds",
            "random_state": args.random_state,
        }
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, args.model_out)
        print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
