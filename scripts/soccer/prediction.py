import pandas as pd
import numpy as np

from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    make_scorer
)

df = pd.read_csv("shared_repo/Group9/dataset.csv")
X = df.drop(columns=["run_id", "env_name", "algorithm", "gamma", "lambda", "mean_group_reward", "group_cumulative_reward", "episode_length", "mean_policy_loss", "mean_value_loss", "mean_entropy", "ELO", "total_duration", "efficiency_score"])

# df = pd.read_csv("results/soccer/group9.csv", sep=";")
# X = df.drop(columns=["strategy", "game_play", "num_layers", "hidden_units", "total_duration"])

y = df["total_duration"]

numerical_features = [col for col in X.columns]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features)
    ]
)

models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42
    )
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    "MedianAE": make_scorer(median_absolute_error, greater_is_better=False),
    "R2": make_scorer(r2_score)
}

print(X.columns)
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring
    )

    print(name)
    print(f"  MAE: {-scores['test_MAE'].mean():.2f}")
    print(f"  Median AE: {-scores['test_MedianAE'].mean():.2f}")
    print(f"  R²: {scores['test_R2'].mean():.3f}")
    print()
