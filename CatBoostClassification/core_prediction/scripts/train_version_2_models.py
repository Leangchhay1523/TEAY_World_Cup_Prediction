"""Train the Version 2 hybrid outcome and goal models.

Active Version 2 architecture:
- CatBoostClassifier predicts match outcome probabilities.
- CatBoostRegressor models predict each team's expected goals.
- Statistical xG is blended with CatBoost xG and calibrated by a tuned goal scale.
- The prediction script defaults to highest Poisson score-probability selection.

The dataset builder remains leakage-aware. It computes pre-match rolling team
features from historical results before each match is added to team history.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error

from v25_feature_engineering import (
    CATEGORICAL_MODEL_FEATURES,
    EXISTING_FIFA_FEATURE_COLUMNS,
    FIFA_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    NEW_FIFA_FEATURE_COLUMNS,
    RollingTeamState,
    build_fifa_feature_pair,
    build_form_feature_pair,
    calibrate_expected_goals,
    clamp_expected_goals,
    estimate_statistical_expected_goals,
    get_tournament_importance_weight,
    get_tournament_type_group,
    load_results,
    outcome_label,
    safe_ratio,
    safe_rate,
    score_winner,
    select_best_score_candidate,
    select_highest_score_probability_candidate,
    standardize_team_name,
    update_h2h_after_match,
    update_states_after_match,
)

try:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
except ImportError as exc:  # pragma: no cover - handled when training starts.
    CatBoostClassifier = None
    CatBoostRegressor = None
    Pool = None
    CATBOOST_IMPORT_ERROR = exc
else:
    CATBOOST_IMPORT_ERROR = None


VERSION_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION_DIR.parent
DATA_ROOT = PROJECT_ROOT.parent / "data"

DEFAULT_RESULTS_PATHS = [
    DATA_ROOT / "raw" / "results.csv",
    PROJECT_ROOT / "version_1_baseline" / "data" / "results.csv",
]
DEFAULT_FIFA_PATHS = [
    DATA_ROOT / "raw" / "fifa_rankings.csv",
]

TRAINING_DATA_PATH = VERSION_DIR / "processed_data" / "training_dataset_v2.csv"
REPORT_PATH = VERSION_DIR / "reports" / "training_report_v2.md"
FEATURE_IMPORTANCE_PATH = VERSION_DIR / "outputs" / "feature_importance_v2.csv"
METRICS_PATH = VERSION_DIR / "outputs" / "training_metrics_v2.json"
OUTCOME_MODEL_PATH = VERSION_DIR / "models" / "catboost_outcome_model.cbm"
GOALS_TEAM_1_MODEL_PATH = VERSION_DIR / "models" / "catboost_goals_team_1.cbm"
GOALS_TEAM_2_MODEL_PATH = VERSION_DIR / "models" / "catboost_goals_team_2.cbm"
GOAL_ENSEMBLE_CONFIG_PATH = VERSION_DIR / "models" / "goal_ensemble_config.json"
GOAL_ENSEMBLE_TUNING_RESULTS_PATH = VERSION_DIR / "outputs" / "goal_ensemble_tuning_results.csv"
SCORE_SELECTION_CONFIG_PATH = VERSION_DIR / "models" / "score_selection_config.json"
SCORE_SELECTION_TUNING_RESULTS_PATH = VERSION_DIR / "outputs" / "score_selection_tuning_results.csv"
ENSEMBLE_WEIGHT_GRID = [round(weight / 10.0, 1) for weight in range(11)]
GOAL_SCALE_GRID = [0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50]

def find_existing_path(candidates: list[Path], label: str, required: bool = True) -> Path | None:
    """Return the first existing path from accepted locations."""

    for path in candidates:
        if path.exists():
            return path
    if required:
        checked = "\n".join(f"- {path}" for path in candidates)
        raise FileNotFoundError(f"Could not find {label}. Checked:\n{checked}")
    return None


def load_latest_fifa(path: Path | None) -> pd.DataFrame:
    """Load latest FIFA rankings for optional experimental features."""

    if path is None:
        return pd.DataFrame()
    fifa = pd.read_csv(path)
    if "team" not in fifa.columns:
        return pd.DataFrame()
    fifa = fifa.copy()
    fifa["team_key"] = fifa["team"].map(standardize_team_name)
    for column in ["rank", "previous_rank", "ranking_move", "points", "previous_points", "rated_matches"]:
        if column in fifa.columns:
            fifa[column] = pd.to_numeric(fifa[column], errors="coerce")
    return fifa


def get_fifa_feature(fifa: pd.DataFrame, team: str, column: str) -> Any:
    """Read a latest FIFA feature. Disabled by default to avoid leakage."""

    if fifa.empty or column not in fifa.columns:
        return np.nan
    match = fifa[fifa["team_key"].eq(team)]
    if match.empty:
        return np.nan
    return match.iloc[0][column]


def get_fifa_row(fifa: pd.DataFrame, team: str) -> pd.Series | None:
    """Return one latest FIFA snapshot row when available."""

    if fifa.empty or "team_key" not in fifa.columns:
        return None
    match = fifa[fifa["team_key"].eq(team)]
    if match.empty:
        return None
    return match.iloc[0]


def build_training_dataset(
    results: pd.DataFrame,
    fifa_table: pd.DataFrame | None = None,
    allow_latest_rating_features: bool = False,
) -> pd.DataFrame:
    """Build one row per historical match with pre-match features."""

    fifa_table = fifa_table if fifa_table is not None else pd.DataFrame()
    states: dict[str, RollingTeamState] = {}
    h2h_history: dict[tuple[str, str], list[Any]] = {}
    rows: list[dict[str, Any]] = []

    for _, matches_on_date in results.groupby("date", sort=True):
        # Build every row for the date before updating states, so same-date
        # results cannot leak into another match's pre-match features.
        pending_updates: list[tuple[RollingTeamState, RollingTeamState, str, str, int, int, bool]] = []

        for match in matches_on_date.itertuples(index=False):
            team_1 = match.home_team
            team_2 = match.away_team
            goals_1 = int(match.home_score)
            goals_2 = int(match.away_score)
            state_1 = states.setdefault(team_1, RollingTeamState())
            state_2 = states.setdefault(team_2, RollingTeamState())

            team_1_goal_rate = safe_rate(state_1.goals_for, state_1.matches)
            team_2_goal_rate = safe_rate(state_2.goals_for, state_2.matches)
            team_1_concede_rate = safe_rate(state_1.goals_against, state_1.matches)
            team_2_concede_rate = safe_rate(state_2.goals_against, state_2.matches)
            form_features = build_form_feature_pair(state_1, state_2, team_1, team_2, h2h_history)
            tournament_type_group = get_tournament_type_group(match.tournament)

            row = {
                "date": match.date.date().isoformat(),
                "year": int(match.date.year),
                "team_1": team_1,
                "team_2": team_2,
                "stage": match.tournament,
                "tournament": match.tournament,
                "neutral": bool(match.neutral),
                "team_1_elo": state_1.elo,
                "team_2_elo": state_2.elo,
                "elo_diff": state_1.elo - state_2.elo,
                "elo_ratio": safe_ratio(state_1.elo, state_2.elo),
                "team_1_recent_form": form_features["team_1_form_points_last_5"],
                "team_2_recent_form": form_features["team_2_form_points_last_5"],
                "recent_form_diff": form_features["form_points_diff_last_5"],
                "team_1_goal_rate": team_1_goal_rate,
                "team_2_goal_rate": team_2_goal_rate,
                "goal_rate_diff": team_1_goal_rate - team_2_goal_rate,
                "team_1_concede_rate": team_1_concede_rate,
                "team_2_concede_rate": team_2_concede_rate,
                "concede_rate_diff": team_1_concede_rate - team_2_concede_rate,
                "tournament_type_group": tournament_type_group,
                "tournament_importance_weight": get_tournament_importance_weight(tournament_type_group),
                "goals_team_1": goals_1,
                "goals_team_2": goals_2,
                "target_outcome": outcome_label(goals_1, goals_2),
            }
            row.update(form_features)

            row.update(
                build_fifa_feature_pair(
                    get_fifa_row(fifa_table, team_1),
                    get_fifa_row(fifa_table, team_2),
                    enabled=allow_latest_rating_features,
                )
            )

            rows.append(row)
            pending_updates.append((state_1, state_2, team_1, team_2, goals_1, goals_2, bool(match.neutral)))

        for state_1, state_2, team_1, team_2, goals_1, goals_2, neutral in pending_updates:
            update_states_after_match(state_1, state_2, goals_1, goals_2, neutral=neutral)
            update_h2h_after_match(h2h_history, team_1, team_2, goals_1, goals_2)

    dataset = pd.DataFrame(rows)
    dataset["neutral"] = dataset["neutral"].astype(str)
    return dataset


def model_features(dataset: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return the shared classifier/regressor features in canonical order."""

    missing = [name for name in MODEL_FEATURE_COLUMNS if name not in dataset.columns]
    if missing:
        raise ValueError(f"Training dataset is missing required Version 2.5 feature columns: {missing}")
    features = MODEL_FEATURE_COLUMNS.copy()
    categorical = [name for name in CATEGORICAL_MODEL_FEATURES if name in features]
    return features, categorical


def time_aware_split(dataset: pd.DataFrame, train_fraction: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on older matches and validate on newer matches."""

    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0.5 and 1.0")
    dataset = dataset.sort_values(["date", "team_1", "team_2"]).reset_index(drop=True)
    split_index = max(1, min(len(dataset) - 1, int(len(dataset) * train_fraction)))
    return dataset.iloc[:split_index].copy(), dataset.iloc[split_index:].copy()


def outcome_probability_maps(probabilities: np.ndarray, classes: list[Any]) -> list[dict[str, float]]:
    """Convert CatBoost probability arrays into named outcome dictionaries."""

    labels = [str(label) for label in classes]
    maps = []
    for row in probabilities:
        probability_map = {label: float(probability) for label, probability in zip(labels, row)}
        if not all(label in probability_map for label in ["team_1_win", "draw", "team_2_win"]):
            probability_map = {
                label: float(probability)
                for label, probability in zip(["team_1_win", "draw", "team_2_win"], row)
            }
        maps.append({label: probability_map[label] for label in ["team_1_win", "draw", "team_2_win"]})
    return maps


def competition_points(
    predicted_goals_1: int,
    predicted_goals_2: int,
    actual_goals_1: int,
    actual_goals_2: int,
) -> int:
    """Score one selected prediction with the active competition rules."""

    predicted_winner = score_winner(predicted_goals_1, predicted_goals_2)
    actual_winner = score_winner(actual_goals_1, actual_goals_2)
    predicted_goal_difference = predicted_goals_1 - predicted_goals_2
    actual_goal_difference = actual_goals_1 - actual_goals_2
    points = 0
    if predicted_winner == actual_winner:
        points += 3
    if predicted_goal_difference == actual_goal_difference:
        points += 2
    if predicted_goals_1 == actual_goals_1 and predicted_goals_2 == actual_goals_2:
        points += 5
    return points


def tune_goal_ensemble(
    valid_df: pd.DataFrame,
    outcome_probabilities: np.ndarray,
    outcome_classes: list[Any],
    catboost_xg_team_1: np.ndarray,
    catboost_xg_team_2: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Backtest the CatBoost/statistical xG blend on the validation split."""

    probability_maps = outcome_probability_maps(outcome_probabilities, outcome_classes)
    statistical_xg = [estimate_statistical_expected_goals(row) for _, row in valid_df.iterrows()]
    statistical_xg_team_1 = np.array([xg[0] for xg in statistical_xg], dtype=float)
    statistical_xg_team_2 = np.array([xg[1] for xg in statistical_xg], dtype=float)
    actual_goals_1 = valid_df["goals_team_1"].astype(int).to_numpy()
    actual_goals_2 = valid_df["goals_team_2"].astype(int).to_numpy()

    rows = []
    for catboost_weight in ENSEMBLE_WEIGHT_GRID:
        statistical_weight = round(1.0 - catboost_weight, 1)
        exact_correct = 0
        goal_difference_correct = 0
        winner_correct = 0
        total_points = 0
        ensemble_xg_1_values = []
        ensemble_xg_2_values = []

        for index in range(len(valid_df)):
            ensemble_xg_1 = clamp_expected_goals(
                catboost_weight * catboost_xg_team_1[index]
                + statistical_weight * statistical_xg_team_1[index]
            )
            ensemble_xg_2 = clamp_expected_goals(
                catboost_weight * catboost_xg_team_2[index]
                + statistical_weight * statistical_xg_team_2[index]
            )
            ensemble_xg_1_values.append(ensemble_xg_1)
            ensemble_xg_2_values.append(ensemble_xg_2)

            best_candidate = select_best_score_candidate(
                ensemble_xg_1,
                ensemble_xg_2,
                probability_maps[index],
            )
            predicted_goals_1 = int(best_candidate["goals_team_1"])
            predicted_goals_2 = int(best_candidate["goals_team_2"])
            actual_1 = int(actual_goals_1[index])
            actual_2 = int(actual_goals_2[index])

            exact_correct += int(predicted_goals_1 == actual_1 and predicted_goals_2 == actual_2)
            goal_difference_correct += int((predicted_goals_1 - predicted_goals_2) == (actual_1 - actual_2))
            winner_correct += int(score_winner(predicted_goals_1, predicted_goals_2) == score_winner(actual_1, actual_2))
            total_points += competition_points(predicted_goals_1, predicted_goals_2, actual_1, actual_2)

        rows.append(
            {
                "catboost_weight": catboost_weight,
                "statistical_weight": statistical_weight,
                "exact_score_accuracy": exact_correct / len(valid_df),
                "goal_difference_accuracy": goal_difference_correct / len(valid_df),
                "winner_accuracy_from_score": winner_correct / len(valid_df),
                "avg_competition_points": total_points / len(valid_df),
                "goals_team_1_mae": float(mean_absolute_error(actual_goals_1, ensemble_xg_1_values)),
                "goals_team_2_mae": float(mean_absolute_error(actual_goals_2, ensemble_xg_2_values)),
            }
        )

    tuning_results = pd.DataFrame(rows)
    ranking = tuning_results.assign(
        mean_goal_mae=(tuning_results["goals_team_1_mae"] + tuning_results["goals_team_2_mae"]) / 2.0
    ).sort_values(
        [
            "avg_competition_points",
            "exact_score_accuracy",
            "goal_difference_accuracy",
            "mean_goal_mae",
        ],
        ascending=[False, False, False, True],
    )
    best = ranking.iloc[0]
    config = {
        "best_catboost_weight": float(best["catboost_weight"]),
        "best_statistical_weight": float(best["statistical_weight"]),
        "selection_metric": "average_competition_points",
        "validation_exact_score_accuracy": float(best["exact_score_accuracy"]),
        "validation_goal_difference_accuracy": float(best["goal_difference_accuracy"]),
        "validation_winner_accuracy_from_score": float(best["winner_accuracy_from_score"]),
        "validation_avg_competition_points": float(best["avg_competition_points"]),
        "validation_goals_team_1_mae": float(best["goals_team_1_mae"]),
        "validation_goals_team_2_mae": float(best["goals_team_2_mae"]),
    }
    return tuning_results, config


def tune_score_selection(
    valid_df: pd.DataFrame,
    catboost_xg_team_1: np.ndarray,
    catboost_xg_team_2: np.ndarray,
    goal_ensemble_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Backtest goal-scale calibration with score-probability selection."""

    catboost_weight = float(goal_ensemble_config["best_catboost_weight"])
    statistical_weight = float(goal_ensemble_config["best_statistical_weight"])
    statistical_xg = [estimate_statistical_expected_goals(row) for _, row in valid_df.iterrows()]
    statistical_xg_team_1 = np.array([xg[0] for xg in statistical_xg], dtype=float)
    statistical_xg_team_2 = np.array([xg[1] for xg in statistical_xg], dtype=float)
    ensemble_xg_team_1 = np.array(
        [
            clamp_expected_goals(catboost_weight * catboost_xg_team_1[index] + statistical_weight * statistical_xg_team_1[index])
            for index in range(len(valid_df))
        ],
        dtype=float,
    )
    ensemble_xg_team_2 = np.array(
        [
            clamp_expected_goals(catboost_weight * catboost_xg_team_2[index] + statistical_weight * statistical_xg_team_2[index])
            for index in range(len(valid_df))
        ],
        dtype=float,
    )
    actual_goals_1 = valid_df["goals_team_1"].astype(int).to_numpy()
    actual_goals_2 = valid_df["goals_team_2"].astype(int).to_numpy()
    actual_avg_total_goals = float(np.mean(actual_goals_1 + actual_goals_2))

    rows = []
    for goal_scale in GOAL_SCALE_GRID:
        exact_correct = 0
        goal_difference_correct = 0
        winner_correct = 0
        total_absolute_goal_error = 0.0
        predicted_total_goals = []
        winner_counts = {"team_1_win": 0, "draw": 0, "team_2_win": 0}

        for index in range(len(valid_df)):
            calibrated_xg_1 = calibrate_expected_goals(ensemble_xg_team_1[index], goal_scale)
            calibrated_xg_2 = calibrate_expected_goals(ensemble_xg_team_2[index], goal_scale)
            best_candidate = select_highest_score_probability_candidate(calibrated_xg_1, calibrated_xg_2)
            predicted_goals_1 = int(best_candidate["goals_team_1"])
            predicted_goals_2 = int(best_candidate["goals_team_2"])
            actual_1 = int(actual_goals_1[index])
            actual_2 = int(actual_goals_2[index])
            predicted_winner = score_winner(predicted_goals_1, predicted_goals_2)
            actual_winner = score_winner(actual_1, actual_2)

            exact_correct += int(predicted_goals_1 == actual_1 and predicted_goals_2 == actual_2)
            goal_difference_correct += int((predicted_goals_1 - predicted_goals_2) == (actual_1 - actual_2))
            winner_correct += int(predicted_winner == actual_winner)
            total_absolute_goal_error += abs(predicted_goals_1 - actual_1) + abs(predicted_goals_2 - actual_2)
            predicted_total_goals.append(predicted_goals_1 + predicted_goals_2)
            winner_counts[predicted_winner] += 1

        rows.append(
            {
                "goal_scale": goal_scale,
                "exact_score_accuracy": exact_correct / len(valid_df),
                "goal_difference_accuracy": goal_difference_correct / len(valid_df),
                "winner_accuracy_from_score": winner_correct / len(valid_df),
                "avg_absolute_goal_error": total_absolute_goal_error / len(valid_df),
                "predicted_avg_total_goals": float(np.mean(predicted_total_goals)),
                "actual_avg_total_goals": actual_avg_total_goals,
                "draw_prediction_rate": winner_counts["draw"] / len(valid_df),
                "home_win_prediction_rate": winner_counts["team_1_win"] / len(valid_df),
                "away_win_prediction_rate": winner_counts["team_2_win"] / len(valid_df),
            }
        )

    tuning_results = pd.DataFrame(rows)
    ranking = tuning_results.assign(
        total_goal_distribution_distance=(
            tuning_results["predicted_avg_total_goals"] - tuning_results["actual_avg_total_goals"]
        ).abs()
    ).sort_values(
        [
            "exact_score_accuracy",
            "goal_difference_accuracy",
            "winner_accuracy_from_score",
            "total_goal_distribution_distance",
        ],
        ascending=[False, False, False, True],
    )
    best = ranking.iloc[0]
    config = {
        "decision_rule": "highest_adjusted_score_probability",
        "goal_scale": float(best["goal_scale"]),
        "selection_metric": "exact_score_accuracy",
        "exact_score_accuracy": float(best["exact_score_accuracy"]),
        "goal_difference_accuracy": float(best["goal_difference_accuracy"]),
        "winner_accuracy_from_score": float(best["winner_accuracy_from_score"]),
        "avg_absolute_goal_error": float(best["avg_absolute_goal_error"]),
        "predicted_avg_total_goals": float(best["predicted_avg_total_goals"]),
        "actual_avg_total_goals": float(best["actual_avg_total_goals"]),
        "draw_prediction_rate": float(best["draw_prediction_rate"]),
        "home_win_prediction_rate": float(best["home_win_prediction_rate"]),
        "away_win_prediction_rate": float(best["away_win_prediction_rate"]),
    }
    return tuning_results, config


def train_models(
    dataset: pd.DataFrame,
    train_fraction: float = 0.8,
    random_seed: int = 42,
    iterations: int = 500,
) -> dict[str, Any]:
    """Train the CatBoost outcome classifier and both goal regressors."""

    if CATBOOST_IMPORT_ERROR is not None:
        raise RuntimeError(
            "CatBoost is required for Version 2 training. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from CATBOOST_IMPORT_ERROR

    features, categorical = model_features(dataset)
    train_df, valid_df = time_aware_split(dataset, train_fraction=train_fraction)
    x_train = train_df[features]
    x_valid = valid_df[features]
    y_train = train_df["target_outcome"]
    y_valid = valid_df["target_outcome"]
    y_goals_1_train = train_df["goals_team_1"]
    y_goals_1_valid = valid_df["goals_team_1"]
    y_goals_2_train = train_df["goals_team_2"]
    y_goals_2_valid = valid_df["goals_team_2"]
    categorical_indices = [features.index(name) for name in categorical]

    train_pool = Pool(x_train, y_train, cat_features=categorical_indices)
    valid_pool = Pool(x_valid, y_valid, cat_features=categorical_indices)
    outcome_model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="Accuracy",
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        random_seed=random_seed,
        train_dir=str(VERSION_DIR / "outputs" / "catboost_info_classifier"),
        verbose=100,
    )
    outcome_model.fit(train_pool, eval_set=valid_pool)

    regressor_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": iterations,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": random_seed,
        "verbose": 100,
    }
    goals_team_1_model = CatBoostRegressor(
        **regressor_params,
        train_dir=str(VERSION_DIR / "outputs" / "catboost_info_goals_team_1"),
    )
    goals_team_2_model = CatBoostRegressor(
        **regressor_params,
        train_dir=str(VERSION_DIR / "outputs" / "catboost_info_goals_team_2"),
    )
    goals_team_1_model.fit(
        Pool(x_train, y_goals_1_train, cat_features=categorical_indices),
        eval_set=Pool(x_valid, y_goals_1_valid, cat_features=categorical_indices),
    )
    goals_team_2_model.fit(
        Pool(x_train, y_goals_2_train, cat_features=categorical_indices),
        eval_set=Pool(x_valid, y_goals_2_valid, cat_features=categorical_indices),
    )

    predictions = outcome_model.predict(x_valid).reshape(-1)
    probabilities = outcome_model.predict_proba(x_valid)
    goals_team_1_predictions = goals_team_1_model.predict(x_valid)
    goals_team_2_predictions = goals_team_2_model.predict(x_valid)
    labels_seen = list(outcome_model.classes_)
    goal_ensemble_tuning_results, goal_ensemble_config = tune_goal_ensemble(
        valid_df,
        probabilities,
        labels_seen,
        np.array([clamp_expected_goals(value) for value in goals_team_1_predictions], dtype=float),
        np.array([clamp_expected_goals(value) for value in goals_team_2_predictions], dtype=float),
    )
    score_selection_tuning_results, score_selection_config = tune_score_selection(
        valid_df,
        np.array([clamp_expected_goals(value) for value in goals_team_1_predictions], dtype=float),
        np.array([clamp_expected_goals(value) for value in goals_team_2_predictions], dtype=float),
        goal_ensemble_config,
    )
    metrics = {
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "train_start": str(train_df["date"].min()),
        "train_end": str(train_df["date"].max()),
        "validation_start": str(valid_df["date"].min()),
        "validation_end": str(valid_df["date"].max()),
        "outcome_accuracy": float(accuracy_score(y_valid, predictions)),
        "outcome_log_loss": float(log_loss(y_valid, probabilities, labels=labels_seen)),
        "goals_team_1_mae": float(mean_absolute_error(y_goals_1_valid, goals_team_1_predictions)),
        "goals_team_2_mae": float(mean_absolute_error(y_goals_2_valid, goals_team_2_predictions)),
        "goals_team_1_rmse": float(np.sqrt(mean_squared_error(y_goals_1_valid, goals_team_1_predictions))),
        "goals_team_2_rmse": float(np.sqrt(mean_squared_error(y_goals_2_valid, goals_team_2_predictions))),
        "goal_ensemble_best_catboost_weight": goal_ensemble_config["best_catboost_weight"],
        "goal_ensemble_best_statistical_weight": goal_ensemble_config["best_statistical_weight"],
        "goal_ensemble_validation_avg_competition_points": goal_ensemble_config[
            "validation_avg_competition_points"
        ],
        "goal_ensemble_validation_exact_score_accuracy": goal_ensemble_config[
            "validation_exact_score_accuracy"
        ],
        "goal_ensemble_validation_goal_difference_accuracy": goal_ensemble_config[
            "validation_goal_difference_accuracy"
        ],
        "score_selection_goal_scale": score_selection_config["goal_scale"],
        "score_selection_exact_score_accuracy": score_selection_config["exact_score_accuracy"],
        "score_selection_goal_difference_accuracy": score_selection_config["goal_difference_accuracy"],
        "score_selection_winner_accuracy_from_score": score_selection_config[
            "winner_accuracy_from_score"
        ],
        "score_selection_draw_prediction_rate": score_selection_config["draw_prediction_rate"],
    }

    importance = pd.DataFrame(
        {
            "feature": features,
            "outcome_importance": outcome_model.get_feature_importance(),
            "goals_team_1_importance": goals_team_1_model.get_feature_importance(),
            "goals_team_2_importance": goals_team_2_model.get_feature_importance(),
        }
    ).sort_values("outcome_importance", ascending=False)

    return {
        "features": features,
        "categorical_features": categorical,
        "metrics": metrics,
        "feature_importance": importance,
        "goal_ensemble_tuning_results": goal_ensemble_tuning_results,
        "goal_ensemble_config": goal_ensemble_config,
        "score_selection_tuning_results": score_selection_tuning_results,
        "score_selection_config": score_selection_config,
        "outcome_model": outcome_model,
        "goals_team_1_model": goals_team_1_model,
        "goals_team_2_model": goals_team_2_model,
    }


def save_training_outputs(dataset: pd.DataFrame, training_result: dict[str, Any], allow_latest: bool) -> None:
    """Save dataset, models, feature importance, metrics, and report."""

    VERSION_DIR.joinpath("models").mkdir(parents=True, exist_ok=True)
    VERSION_DIR.joinpath("processed_data").mkdir(parents=True, exist_ok=True)
    VERSION_DIR.joinpath("outputs").mkdir(parents=True, exist_ok=True)
    VERSION_DIR.joinpath("reports").mkdir(parents=True, exist_ok=True)

    dataset.to_csv(TRAINING_DATA_PATH, index=False)
    training_result["outcome_model"].save_model(OUTCOME_MODEL_PATH)
    training_result["goals_team_1_model"].save_model(GOALS_TEAM_1_MODEL_PATH)
    training_result["goals_team_2_model"].save_model(GOALS_TEAM_2_MODEL_PATH)
    training_result["feature_importance"].to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    training_result["goal_ensemble_tuning_results"].to_csv(GOAL_ENSEMBLE_TUNING_RESULTS_PATH, index=False)
    training_result["score_selection_tuning_results"].to_csv(SCORE_SELECTION_TUNING_RESULTS_PATH, index=False)
    GOAL_ENSEMBLE_CONFIG_PATH.write_text(
        json.dumps(training_result["goal_ensemble_config"], indent=2),
        encoding="utf-8",
    )
    SCORE_SELECTION_CONFIG_PATH.write_text(
        json.dumps(training_result["score_selection_config"], indent=2),
        encoding="utf-8",
    )

    metrics = training_result["metrics"]
    ensemble_config = training_result["goal_ensemble_config"]
    score_selection_config = training_result["score_selection_config"]
    top_features = training_result["feature_importance"].head(15)
    report = [
        "# Version 2 Training Report",
        "",
        "## Summary",
        "",
        "- Active outcome model: CatBoostClassifier for `team_1_win`, `draw`, and `team_2_win`.",
        "- Active goal models: CatBoostRegressor for `goals_team_1` and `goals_team_2`.",
        "- The outcome model predicts winner probabilities only.",
        "- The goal models predict expected goals only.",
        "- Statistical xG provides a stable expected-goals baseline from rates, recent goals, Elo, FIFA points when available, and tournament importance.",
        "- The goal ensemble blends CatBoost xG and statistical xG before Poisson scoring.",
        "- Goal-scale calibration adjusts ensemble xG before building the Poisson score matrix.",
        "- Default prediction selects the highest expected competition points.",
        "- Pure score-probability selection is retained only as optional prediction/debug mode.",
        "- Version 2 upgrade: leakage-safe recent form, goals, opponent-strength, Package A, Package B, Package C, and tournament-type features",
        "- Split strategy: time-aware split, older matches for training and newer matches for validation",
        f"- Latest rating snapshot features used in historical training: `{allow_latest}`",
        "",
        "## Data",
        "",
        f"- Training rows: {metrics['train_rows']}",
        f"- Validation rows: {metrics['validation_rows']}",
        f"- Training period: {metrics['train_start']} to {metrics['train_end']}",
        f"- Validation period: {metrics['validation_start']} to {metrics['validation_end']}",
        f"- Processed dataset: `{TRAINING_DATA_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "## Evaluation",
        "",
        f"- Outcome accuracy: {metrics['outcome_accuracy']:.4f}",
        f"- Outcome log loss: {metrics['outcome_log_loss']:.4f}",
        f"- Team 1 goals MAE: {metrics['goals_team_1_mae']:.4f}",
        f"- Team 2 goals MAE: {metrics['goals_team_2_mae']:.4f}",
        f"- Team 1 goals RMSE: {metrics['goals_team_1_rmse']:.4f}",
        f"- Team 2 goals RMSE: {metrics['goals_team_2_rmse']:.4f}",
        "",
        "## Goal Ensemble Tuning",
        "",
        f"- Best CatBoost xG weight: {ensemble_config['best_catboost_weight']:.1f}",
        f"- Best statistical xG weight: {ensemble_config['best_statistical_weight']:.1f}",
        f"- Validation exact score accuracy: {ensemble_config['validation_exact_score_accuracy']:.4f}",
        f"- Validation goal difference accuracy: {ensemble_config['validation_goal_difference_accuracy']:.4f}",
        f"- Validation winner accuracy from selected score: {ensemble_config['validation_winner_accuracy_from_score']:.4f}",
        f"- Validation average competition points: {ensemble_config['validation_avg_competition_points']:.4f}",
        f"- Tuning results: `{GOAL_ENSEMBLE_TUNING_RESULTS_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Saved ensemble config: `{GOAL_ENSEMBLE_CONFIG_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "## Score Selection Calibration",
        "",
        "- Default final score selection is highest expected competition points.",
        "- Pure score-probability selection is retained only as optional prediction/debug mode.",
        f"- Best goal scale: {score_selection_config['goal_scale']:.2f}",
        f"- Exact score accuracy: {score_selection_config['exact_score_accuracy']:.4f}",
        f"- Goal difference accuracy: {score_selection_config['goal_difference_accuracy']:.4f}",
        f"- Winner accuracy from selected score: {score_selection_config['winner_accuracy_from_score']:.4f}",
        f"- Average absolute goal error: {score_selection_config['avg_absolute_goal_error']:.4f}",
        f"- Predicted average total goals: {score_selection_config['predicted_avg_total_goals']:.4f}",
        f"- Actual average total goals: {score_selection_config['actual_avg_total_goals']:.4f}",
        f"- Draw prediction rate: {score_selection_config['draw_prediction_rate']:.4f}",
        f"- Tuning results: `{SCORE_SELECTION_TUNING_RESULTS_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Saved score-selection config: `{SCORE_SELECTION_CONFIG_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "## Saved Models",
        "",
        f"- `{OUTCOME_MODEL_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{GOALS_TEAM_1_MODEL_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{GOALS_TEAM_2_MODEL_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "## Most Important Features",
        "",
        "```text",
        top_features.to_csv(index=False, lineterminator="\n").replace(",", " | ").strip(),
        "```",
        "",
        "## Version 2 Feature Families",
        "",
        "- Recent result form: wins, draws, and losses in each team's last 5 and last 10 matches.",
        "- Recent points form: football points over last 5 and last 10, plus team-difference columns.",
        "- Recent attacking/defensive form: average goals scored and conceded over last 5 and last 10.",
        "- Opponent strength: average pre-match rolling Elo of previous opponents over last 5 and last 10.",
        "- Package A: Elo ratio, draw rate, clean-sheet rate, failed-to-score rate, and average goal-difference features.",
        "- Package B: total-goals threshold rates and goal-scoring/conceding volatility features.",
        "- Package C: head-to-head, neutral-ground, home/away style, and attack-vs-defense interaction features.",
        "- Tournament context: `tournament_type_group` and `tournament_importance_weight`.",
        "",
        "## Leakage Controls",
        "",
        "- Rolling Elo, form, goal rate, concede rate, and opponent-strength features are computed before each match is added to team history.",
        "- The default training mode does not use latest FIFA snapshots as historical match features.",
        "- Use `--allow-latest-rating-features` only for experiments where you accept that leakage risk.",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    metadata = {
        "metrics": metrics,
        "features": training_result["features"],
        "categorical_features": training_result["categorical_features"],
        "model_types": {
            "outcome": "CatBoostClassifier",
            "goals_team_1": "CatBoostRegressor",
            "goals_team_2": "CatBoostRegressor",
        },
        "targets": {
            "outcome": "target_outcome",
            "goals_team_1": "goals_team_1",
            "goals_team_2": "goals_team_2",
        },
        "version_2_5_feature_note": (
            "Rolling result counts, points, goals, opponent Elo, Package A rates, Package B exact-score features, Package C matchup features, and tournament type are computed "
            "from prior matches only."
        ),
        "allow_latest_rating_features": allow_latest,
        "active_model_paths": {
            "outcome": str(OUTCOME_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "goals_team_1": str(GOALS_TEAM_1_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "goals_team_2": str(GOALS_TEAM_2_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "goal_ensemble_config": str(GOAL_ENSEMBLE_CONFIG_PATH.relative_to(PROJECT_ROOT)),
            "score_selection_config": str(SCORE_SELECTION_CONFIG_PATH.relative_to(PROJECT_ROOT)),
        },
        "goal_ensemble_config": training_result["goal_ensemble_config"],
        "score_selection_config": training_result["score_selection_config"],
        "prediction_architecture": (
            "CatBoostClassifier predicts winner probabilities; CatBoostRegressor goal models and the "
            "statistical xG baseline are blended by the saved ensemble weight; calibrated xG is "
            "scaled by the saved goal scale; Poisson produces 0-0 to 6-6 score probabilities; "
            "default prediction selects the highest expected competition points."
        ),
    }
    METRICS_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Train Version 2 CatBoost outcome and goal models.")
    parser.add_argument("--results-path", type=Path, default=None, help="Historical results CSV.")
    parser.add_argument("--fifa-path", type=Path, default=None, help="Latest FIFA ranking CSV.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Chronological train fraction.")
    parser.add_argument("--iterations", type=int, default=500, help="CatBoost iterations.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--allow-latest-rating-features",
        action="store_true",
        help="Use current FIFA snapshot features in historical rows. This may leak future information.",
    )
    return parser.parse_args()


def main() -> None:
    """Run training end to end."""

    args = parse_args()
    results_path = args.results_path or find_existing_path(DEFAULT_RESULTS_PATHS, "historical results")
    fifa_path = args.fifa_path or find_existing_path(DEFAULT_FIFA_PATHS, "latest FIFA rankings", required=False)
    results = load_results(results_path)
    latest_fifa = load_latest_fifa(fifa_path)

    print(f"Loaded historical results: {results_path}")
    print(f"Rows: {len(results):,}")
    if args.allow_latest_rating_features:
        print("WARNING: latest FIFA snapshot features are being used in historical rows.")
    else:
        print("Leakage-safe mode: latest FIFA snapshot features are loaded but not used for historical rows.")
    print(f"Number of FIFA features before expansion: {len(EXISTING_FIFA_FEATURE_COLUMNS)}")
    print(f"Number of FIFA features after expansion: {len(FIFA_FEATURE_COLUMNS)}")
    print("Newly added FIFA features:")
    for feature in NEW_FIFA_FEATURE_COLUMNS:
        print(f"- {feature}")
    removed_fifa_features = [
        feature for feature in EXISTING_FIFA_FEATURE_COLUMNS if feature not in MODEL_FEATURE_COLUMNS
    ]
    if removed_fifa_features:
        raise ValueError(f"Existing FIFA features were removed from the model contract: {removed_fifa_features}")
    print("Confirmed no existing FIFA feature has been removed.")

    dataset = build_training_dataset(
        results,
        fifa_table=latest_fifa,
        allow_latest_rating_features=args.allow_latest_rating_features,
    )
    print(f"Built training dataset: {dataset.shape[0]:,} rows x {dataset.shape[1]:,} columns")

    training_result = train_models(
        dataset,
        train_fraction=args.train_fraction,
        random_seed=args.random_seed,
        iterations=args.iterations,
    )
    save_training_outputs(dataset, training_result, allow_latest=args.allow_latest_rating_features)

    print(f"Saved training dataset: {TRAINING_DATA_PATH}")
    print(f"Saved outcome model: {OUTCOME_MODEL_PATH}")
    print(f"Saved team 1 goal model: {GOALS_TEAM_1_MODEL_PATH}")
    print(f"Saved team 2 goal model: {GOALS_TEAM_2_MODEL_PATH}")
    print(f"Saved goal ensemble config: {GOAL_ENSEMBLE_CONFIG_PATH}")
    print(f"Saved goal ensemble tuning results: {GOAL_ENSEMBLE_TUNING_RESULTS_PATH}")
    print(f"Saved score selection config: {SCORE_SELECTION_CONFIG_PATH}")
    print(f"Saved score selection tuning results: {SCORE_SELECTION_TUNING_RESULTS_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
