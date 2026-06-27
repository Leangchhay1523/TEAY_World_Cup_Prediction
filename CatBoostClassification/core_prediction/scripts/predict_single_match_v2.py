"""Predict one FIFA World Cup 2026 match with Version 2.

Active architecture:
- CatBoostClassifier predicts outcome probabilities.
- CatBoostRegressor goal models estimate expected goals.
- Statistical xG and goal-scale calibration stabilize Poisson lambdas.
- Poisson matrix generates exact-score and goal-difference probabilities.
- Default decision mode selects the highest expected competition points.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v25_feature_engineering import (
    MODEL_FEATURE_COLUMNS,
    VERSION_2_5_FEATURE_COLUMNS,
    RollingTeamState,
    build_fifa_feature_pair,
    build_form_feature_pair,
    build_historical_context_before_date,
    calibrate_expected_goals,
    clamp_expected_goals,
    estimate_statistical_expected_goals,
    generate_candidates as shared_generate_candidates,
    generate_score_matrix as shared_generate_score_matrix,
    get_tournament_importance_weight,
    get_tournament_type_group,
    load_results,
    normalize_team_name as shared_normalize_team_name,
    poisson_pmf as shared_poisson_pmf,
    safe_ratio,
    safe_rate,
    score_winner as shared_score_winner,
    standardize_team_name as shared_standardize_team_name,
)


VERSION_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION_DIR.parent
DATA_ROOT = PROJECT_ROOT.parent / "data"

FIXTURE_PATH = DATA_ROOT / "worldcup_2026_fixtures" / "worldcup_2026_fixtures_cleaned.csv"
FUTURE_FIXTURE_PATH = DATA_ROOT / "worldcup_2026_fixtures" / "future_match.csv"
RESULTS_PATH = DATA_ROOT / "raw" / "results.csv"
ELO_PATH = DATA_ROOT / "raw" / "elo_ratings.csv"
FIFA_PATH = DATA_ROOT / "raw" / "fifa_rankings.csv"
OUTCOME_MODEL_PATH = VERSION_DIR / "models" / "catboost_outcome_model.cbm"
GOALS_TEAM_1_MODEL_PATH = VERSION_DIR / "models" / "catboost_goals_team_1.cbm"
GOALS_TEAM_2_MODEL_PATH = VERSION_DIR / "models" / "catboost_goals_team_2.cbm"
GOAL_ENSEMBLE_CONFIG_PATH = VERSION_DIR / "models" / "goal_ensemble_config.json"
SCORE_SELECTION_CONFIG_PATH = VERSION_DIR / "models" / "score_selection_config.json"
PREDICTION_OUTPUT_PATH = VERSION_DIR / "outputs" / "version_2_predictions.csv"
TRAINING_METRICS_PATH = VERSION_DIR / "outputs" / "training_metrics_v2.json"

PREDICTION_METHOD = "CatBoostClassifier + CatBoostRegressor/Statistical xG Ensemble + Expected-Points Score Selection"
DECISION_MODES = ["score_probability", "expected_points"]
OUTCOME_LABELS = ["team_1_win", "draw", "team_2_win"]
FIXTURE_COLUMNS = [
    "match_id",
    "date",
    "team_1",
    "team_2",
    "stage",
    "group",
    "host_city",
    "host_country",
    "stadium",
    "venue",
    "kickoff_time",
]

TEAM_ALIASES = {
    "USA": "United States",
    "US": "United States",
    "United States of America": "United States",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "South Korea": "South Korea",
    "Cote d'Ivoire": "Cote d'Ivoire",
    "Cote d Ivoire": "Cote d'Ivoire",
    "Czech Republic": "Czechia",
    "Turkiye": "Turkey",
}

NORMALIZED_TEAM_ALIASES = {
    "ivory coast": "Cote d'Ivoire",
    "cote d'ivoire": "Cote d'Ivoire",
    "cote divoire": "Cote d'Ivoire",
    "côte d'ivoire": "Cote d'Ivoire",
    "cote d ivoire": "Cote d'Ivoire",
    "usa": "United States",
    "us": "United States",
    "united states of america": "United States",
    "south korea": "South Korea",
    "korea republic": "South Korea",
    "republic of korea": "South Korea",
    "cabo verde": "Cabo Verde",
    "cape verde": "Cabo Verde",
}

DISPLAY_TEAM_NAMES = {
    "Cote d'Ivoire": "Ivory Coast",
}

PLACEHOLDER_PATTERNS = [
    r"^TBD$",
    r"^[123][A-L]+$",
    r"^[WL]\d+$",
    r"^Winner\b",
    r"^Runner[- ]up\b",
    r"^Runner up\b",
    r".*\bPlayoff\b.*",
    r".*\bGroup [A-L]\b.*",
]

_TRAINING_METADATA_CACHE: dict[str, Any] | None = None


@dataclass(frozen=True)
class FixtureMatch:
    """Validated fixture row."""

    match_date: str
    team_1: str
    team_2: str
    stage: str
    host_country: str
    fixture_team_1: str
    fixture_team_2: str


@dataclass(frozen=True)
class PredictionResult:
    """Final prediction record saved to CSV."""

    match_date: str
    team_1: str
    team_2: str
    stage: str
    predicted_winner: str
    predicted_score: str
    predicted_goal_difference: int
    confidence: float
    outcome_confidence: float
    exact_score_confidence: float
    confidence_type: str
    team_1_win_probability: float
    draw_probability: float
    team_2_win_probability: float
    expected_goals_team_1: float
    expected_goals_team_2: float
    catboost_xg_team_1: float
    catboost_xg_team_2: float
    statistical_xg_team_1: float
    statistical_xg_team_2: float
    ensemble_xg_team_1: float
    ensemble_xg_team_2: float
    goal_ensemble_weight_catboost: float
    goal_ensemble_weight_statistical: float
    decision_mode: str
    goal_scale: float
    calibrated_xg_team_1: float
    calibrated_xg_team_2: float
    selected_score_probability: float
    predicted_total_goals: int
    expected_competition_points: float
    prediction_method: str
    explanation: str


@dataclass(frozen=True)
class ExpectedGoalsBreakdown:
    """CatBoost, statistical, and final ensemble expected goals."""

    catboost_xg_team_1: float
    catboost_xg_team_2: float
    statistical_xg_team_1: float
    statistical_xg_team_2: float
    ensemble_xg_team_1: float
    ensemble_xg_team_2: float
    calibrated_xg_team_1: float
    calibrated_xg_team_2: float
    catboost_weight: float
    statistical_weight: float
    goal_scale: float


def import_catboost_models() -> tuple[Any, Any]:
    """Import CatBoost classes only when model inference is needed."""

    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError(
            "CatBoost is required for Version 2 predictions. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc
    return CatBoostClassifier, CatBoostRegressor


def normalize_team_name(value: Any) -> str:
    """Canonical team name used for all fixture/rating comparisons.

    The matching key deliberately removes accents and normalizes known aliases,
    so user input like "Ivory Coast" can match fixture text like
    "Côte d'Ivoire" and rating text like "Ivory Coast".
    """

    return shared_normalize_team_name(value)


def standardize_team_name(value: Any) -> str:
    """Backward-compatible wrapper for existing feature code."""

    return shared_standardize_team_name(value)


def display_team_name(team_name: str) -> str:
    """Convert internal canonical names into user-facing display names."""

    return DISPLAY_TEAM_NAMES.get(team_name, team_name)


def load_training_metadata() -> dict[str, Any]:
    """Load the saved training feature contract when it is available."""

    global _TRAINING_METADATA_CACHE
    if _TRAINING_METADATA_CACHE is not None:
        return _TRAINING_METADATA_CACHE
    if not TRAINING_METRICS_PATH.exists():
        _TRAINING_METADATA_CACHE = {}
        return _TRAINING_METADATA_CACHE
    with TRAINING_METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
        _TRAINING_METADATA_CACHE = json.load(metrics_file)
    return _TRAINING_METADATA_CACHE


def training_feature_columns() -> list[str]:
    """Return the exact feature names and order saved by training."""

    metadata = load_training_metadata()
    features = metadata.get("features")
    if isinstance(features, list) and features:
        return [str(feature) for feature in features]
    return MODEL_FEATURE_COLUMNS.copy()


def training_uses_latest_rating_features() -> bool:
    """Return whether the saved model was trained with current FIFA snapshots."""

    metadata = load_training_metadata()
    return bool(metadata.get("allow_latest_rating_features", False))


def load_goal_ensemble_config() -> dict[str, float]:
    """Load the saved CatBoost/statistical xG ensemble weights."""

    if GOAL_ENSEMBLE_CONFIG_PATH.exists():
        with GOAL_ENSEMBLE_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    else:
        # Keeps the CLI usable before the next training run; training writes the tuned config.
        config = {
            "best_catboost_weight": 1.0,
            "best_statistical_weight": 0.0,
            "selection_metric": "fallback_catboost_only",
        }

    catboost_weight = float(config.get("best_catboost_weight", 1.0))
    catboost_weight = max(0.0, min(1.0, catboost_weight))
    statistical_weight = float(config.get("best_statistical_weight", 1.0 - catboost_weight))
    statistical_weight = max(0.0, min(1.0, statistical_weight))
    if abs((catboost_weight + statistical_weight) - 1.0) > 1e-6:
        statistical_weight = 1.0 - catboost_weight
    return {
        **config,
        "best_catboost_weight": catboost_weight,
        "best_statistical_weight": statistical_weight,
    }


def load_score_selection_config() -> dict[str, float | str]:
    """Load the saved score-probability goal-scale calibration."""

    if SCORE_SELECTION_CONFIG_PATH.exists():
        with SCORE_SELECTION_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    else:
        config = {
            "decision_rule": "highest_adjusted_score_probability",
            "goal_scale": 1.0,
            "selection_metric": "fallback_unscaled_score_probability",
        }
    goal_scale = max(0.1, min(2.0, float(config.get("goal_scale", 1.0))))
    return {
        **config,
        "goal_scale": goal_scale,
    }


def is_unresolved_placeholder(team_name: str) -> bool:
    """Return True when a fixture entry is not yet a real team."""

    text = standardize_team_name(team_name)
    return any(re.match(pattern, text, flags=re.IGNORECASE) for pattern in PLACEHOLDER_PATTERNS)


def load_fixtures(path: Path | None = None) -> pd.DataFrame:
    """Load World Cup 2026 fixtures from the existing fixture CSVs."""

    fixture_paths = [path] if path is not None else [FIXTURE_PATH, FUTURE_FIXTURE_PATH]
    frames = []
    for fixture_path in fixture_paths:
        if fixture_path is None or not fixture_path.exists():
            continue
        frame = pd.read_csv(fixture_path)
        if {"date", "team_1", "team_2", "stage", "host_country"}.difference(frame.columns):
            frame = pd.read_csv(fixture_path, header=None, names=FIXTURE_COLUMNS)
        frame["fixture_source"] = fixture_path.name
        frames.append(frame)

    if not frames:
        checked = "\n".join(f"- {fixture_path}" for fixture_path in fixture_paths)
        raise FileNotFoundError(f"No fixture files found. Checked:\n{checked}")

    fixtures = pd.concat(frames, ignore_index=True)
    required = {"date", "team_1", "team_2", "stage", "host_country"}
    missing = required.difference(fixtures.columns)
    if missing:
        raise ValueError(f"Fixture file missing columns: {sorted(missing)}")

    fixtures = fixtures.copy()
    if "match_id" in fixtures.columns:
        fixtures = fixtures.drop_duplicates(subset=["match_id"], keep="last")
    else:
        fixtures = fixtures.drop_duplicates(subset=["date", "team_1", "team_2", "stage"], keep="last")
    # Keep fixture dates as date objects so comparison is not affected by string formatting.
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce").dt.date
    fixtures["team_1_norm"] = fixtures["team_1"].map(normalize_team_name)
    fixtures["team_2_norm"] = fixtures["team_2"].map(normalize_team_name)
    fixtures["team_1_std"] = fixtures["team_1_norm"]
    fixtures["team_2_std"] = fixtures["team_2_norm"]
    fixtures["stage_std"] = fixtures["stage"].astype(str).str.strip().str.lower()
    return fixtures


def load_elo(path: Path = ELO_PATH) -> pd.DataFrame:
    """Load latest Elo/team-stat CSV."""

    if not path.exists():
        raise FileNotFoundError(f"Elo file not found: {path}")
    elo = pd.read_csv(path)
    required = {"team_name", "elo", "matches", "goals_for", "goals_against", "recent_form"}
    missing = required.difference(elo.columns)
    if missing:
        raise ValueError(f"Elo file missing columns: {sorted(missing)}")

    elo = elo.copy()
    elo["team_key"] = elo["team_name"].map(standardize_team_name)
    for column in ["elo", "matches", "goals_for", "goals_against", "recent_form"]:
        elo[column] = pd.to_numeric(elo[column], errors="coerce")
    elo["goal_rate"] = np.where(elo["matches"] > 0, elo["goals_for"] / elo["matches"], np.nan)
    elo["concede_rate"] = np.where(elo["matches"] > 0, elo["goals_against"] / elo["matches"], np.nan)
    return elo


def load_fifa(path: Path = FIFA_PATH) -> pd.DataFrame:
    """Load latest FIFA ranking CSV."""

    if not path.exists():
        raise FileNotFoundError(f"FIFA ranking file not found: {path}")
    fifa = pd.read_csv(path)
    required = {"team", "rank", "points", "confederation"}
    missing = required.difference(fifa.columns)
    if missing:
        raise ValueError(f"FIFA file missing columns: {sorted(missing)}")

    fifa = fifa.copy()
    fifa["team_key"] = fifa["team"].map(standardize_team_name)
    for column in ["rank", "previous_rank", "ranking_move", "points", "previous_points", "rated_matches"]:
        if column not in fifa.columns:
            fifa[column] = np.nan
        fifa[column] = pd.to_numeric(fifa[column], errors="coerce")
    return fifa


def get_single_team_row(table: pd.DataFrame, team: str, label: str) -> pd.Series:
    """Return one team row from a rating table."""

    matches = table[table["team_key"].eq(team)]
    if matches.empty:
        raise ValueError(f"{team} was not found in the latest {label} CSV.")
    return matches.iloc[0]


def closest_fixture_team_names(team_name: str, fixtures_on_date: pd.DataFrame) -> list[str]:
    """Return closest fixture team names for an improved no-match error."""

    if fixtures_on_date.empty:
        return []
    teams = sorted(set(fixtures_on_date["team_1_norm"]).union(set(fixtures_on_date["team_2_norm"])))
    return difflib.get_close_matches(team_name, teams, n=5, cutoff=0.35)


def find_fixture(
    team_1: str,
    team_2: str,
    match_date: str,
    fixture_df: pd.DataFrame,
    stage: str | None = None,
) -> pd.Series | None:
    """Find a fixture by normalized teams and date, independent of team order."""

    team_1_norm = normalize_team_name(team_1)
    team_2_norm = normalize_team_name(team_2)
    input_date = pd.to_datetime(match_date, errors="raise").date()

    print("DEBUG INPUT:", team_1, team_2, match_date)
    print("DEBUG NORMALIZED:", team_1_norm, team_2_norm)

    fixtures_on_date = fixture_df[fixture_df["date"].eq(input_date)].copy()
    print("DEBUG FIXTURES ON DATE:")
    if fixtures_on_date.empty:
        print("(none)")
    else:
        print(fixtures_on_date[["team_1", "team_2", "date"]].to_string(index=False))

    if stage is not None:
        stage_std = str(stage).strip().lower()
        fixtures_on_date = fixtures_on_date[fixtures_on_date["stage_std"].eq(stage_std)].copy()

    # Match either user order or fixture-file order.
    match = fixtures_on_date[
        (
            fixtures_on_date["team_1_norm"].eq(team_1_norm)
            & fixtures_on_date["team_2_norm"].eq(team_2_norm)
        )
        | (
            fixtures_on_date["team_1_norm"].eq(team_2_norm)
            & fixtures_on_date["team_2_norm"].eq(team_1_norm)
        )
    ]

    if match.empty:
        print("No exact fixture found. Available matches on this date:")
        if fixtures_on_date.empty:
            print("(none)")
        else:
            print(fixtures_on_date[["team_1", "team_2", "date"]].to_string(index=False))
        print("Closest matches by team name:")
        print(f"- {team_1_norm}: {closest_fixture_team_names(team_1_norm, fixtures_on_date)}")
        print(f"- {team_2_norm}: {closest_fixture_team_names(team_2_norm, fixtures_on_date)}")
        return None

    return match.iloc[0]


def validate_fixture(
    team_1: str,
    team_2: str,
    stage: str,
    match_date: str,
    fixtures: pd.DataFrame,
    elo: pd.DataFrame,
    fifa: pd.DataFrame,
) -> FixtureMatch:
    """Validate fixture existence, placeholders, and rating coverage."""

    team_1_std = normalize_team_name(team_1)
    team_2_std = normalize_team_name(team_2)
    input_date = pd.to_datetime(match_date, errors="raise").date()

    if is_unresolved_placeholder(team_1_std) or is_unresolved_placeholder(team_2_std):
        raise ValueError("Prediction rejected: both teams must be real resolved teams, not placeholders.")

    fixture = find_fixture(team_1, team_2, match_date, fixtures, stage=stage)
    if fixture is None:
        raise ValueError(
            f"No fixture found for {team_1_std} vs {team_2_std} on {input_date.isoformat()}."
        )

    for fixture_team in [fixture["team_1_norm"], fixture["team_2_norm"]]:
        if is_unresolved_placeholder(fixture_team):
            raise ValueError(f"Prediction rejected: fixture contains unresolved placeholder `{fixture_team}`.")

    get_single_team_row(elo, team_1_std, "Elo")
    get_single_team_row(elo, team_2_std, "Elo")
    get_single_team_row(fifa, team_1_std, "FIFA ranking")
    get_single_team_row(fifa, team_2_std, "FIFA ranking")

    return FixtureMatch(
        match_date=input_date.isoformat(),
        team_1=team_1_std,
        team_2=team_2_std,
        stage=str(stage).strip(),
        host_country=normalize_team_name(fixture["host_country"]),
        fixture_team_1=fixture["team_1_norm"],
        fixture_team_2=fixture["team_2_norm"],
    )


def infer_neutral_flag(team_1: str, team_2: str, host_country: str) -> str:
    """Infer neutral venue from host country."""

    return str(host_country not in {team_1, team_2})


def _fallback_rate(primary: float, fallback: float) -> float:
    """Use a fallback value when a rolling rate is unavailable."""

    return float(fallback) if pd.isna(primary) else float(primary)


def build_feature_row(
    fixture: FixtureMatch,
    elo: pd.DataFrame,
    fifa: pd.DataFrame,
    results: pd.DataFrame | None = None,
    allow_latest_rating_features: bool | None = None,
) -> pd.DataFrame:
    """Build the shared model feature row from the saved training contract."""

    elo_1 = get_single_team_row(elo, fixture.team_1, "Elo")
    elo_2 = get_single_team_row(elo, fixture.team_2, "Elo")
    fifa_1 = get_single_team_row(fifa, fixture.team_1, "FIFA ranking")
    fifa_2 = get_single_team_row(fifa, fixture.team_2, "FIFA ranking")
    use_latest_ratings = (
        training_uses_latest_rating_features()
        if allow_latest_rating_features is None
        else allow_latest_rating_features
    )
    historical_results = results if results is not None else load_results(RESULTS_PATH)
    states, h2h_history = build_historical_context_before_date(historical_results, fixture.match_date)
    state_1 = states.get(fixture.team_1, RollingTeamState())
    state_2 = states.get(fixture.team_2, RollingTeamState())
    form_features = build_form_feature_pair(state_1, state_2, fixture.team_1, fixture.team_2, h2h_history)

    team_1_goal_rate = _fallback_rate(safe_rate(state_1.goals_for, state_1.matches), elo_1["goal_rate"])
    team_2_goal_rate = _fallback_rate(safe_rate(state_2.goals_for, state_2.matches), elo_2["goal_rate"])
    team_1_concede_rate = _fallback_rate(safe_rate(state_1.goals_against, state_1.matches), elo_1["concede_rate"])
    team_2_concede_rate = _fallback_rate(safe_rate(state_2.goals_against, state_2.matches), elo_2["concede_rate"])
    tournament = "FIFA World Cup"
    tournament_type_group = get_tournament_type_group(tournament)

    row = {
        "team_1_elo": float(elo_1["elo"]),
        "team_2_elo": float(elo_2["elo"]),
        "elo_diff": float(elo_1["elo"] - elo_2["elo"]),
        "elo_ratio": safe_ratio(elo_1["elo"], elo_2["elo"]),
        "team_1_recent_form": form_features["team_1_form_points_last_5"],
        "team_2_recent_form": form_features["team_2_form_points_last_5"],
        "recent_form_diff": form_features["form_points_diff_last_5"],
        "team_1_goal_rate": team_1_goal_rate,
        "team_2_goal_rate": team_2_goal_rate,
        "goal_rate_diff": team_1_goal_rate - team_2_goal_rate,
        "team_1_concede_rate": team_1_concede_rate,
        "team_2_concede_rate": team_2_concede_rate,
        "concede_rate_diff": team_1_concede_rate - team_2_concede_rate,
        "tournament": tournament,
        "tournament_type_group": tournament_type_group,
        "tournament_importance_weight": get_tournament_importance_weight(tournament_type_group),
        "neutral": infer_neutral_flag(fixture.team_1, fixture.team_2, fixture.host_country),
        "year": int(pd.to_datetime(fixture.match_date).year),
    }
    row.update(build_fifa_feature_pair(fifa_1, fifa_2, enabled=use_latest_ratings))
    row.update(form_features)
    return pd.DataFrame([row])


def model_feature_columns() -> list[str]:
    """Feature order used by the outcome classifier and goal regressors."""

    return training_feature_columns()


def aligned_model_input(feature_row: pd.DataFrame) -> pd.DataFrame:
    """Validate and order features exactly as they were used during training."""

    features = model_feature_columns()
    missing = [feature for feature in features if feature not in feature_row.columns]
    if missing:
        raise ValueError(f"Feature row is missing trained model columns: {missing}")
    return feature_row[features]


def load_prediction_models() -> tuple[Any, Any, Any]:
    """Load the outcome classifier and both goal regressors."""

    if not OUTCOME_MODEL_PATH.exists():
        raise FileNotFoundError(f"Required outcome model not found: {OUTCOME_MODEL_PATH}")
    if not GOALS_TEAM_1_MODEL_PATH.exists():
        raise FileNotFoundError(f"Required team 1 goal model not found: {GOALS_TEAM_1_MODEL_PATH}")
    if not GOALS_TEAM_2_MODEL_PATH.exists():
        raise FileNotFoundError(f"Required team 2 goal model not found: {GOALS_TEAM_2_MODEL_PATH}")

    CatBoostClassifier, CatBoostRegressor = import_catboost_models()
    outcome_model = CatBoostClassifier()
    goals_team_1_model = CatBoostRegressor()
    goals_team_2_model = CatBoostRegressor()
    outcome_model.load_model(OUTCOME_MODEL_PATH)
    goals_team_1_model.load_model(GOALS_TEAM_1_MODEL_PATH)
    goals_team_2_model.load_model(GOALS_TEAM_2_MODEL_PATH)
    validate_model_feature_contract(outcome_model, goals_team_1_model, goals_team_2_model)
    return outcome_model, goals_team_1_model, goals_team_2_model


def validate_model_feature_contract(*models: Any) -> None:
    """Ensure all loaded CatBoost models share the saved training feature order."""

    expected_features = model_feature_columns()
    for model in models:
        model_features = list(getattr(model, "feature_names_", []) or [])
        if model_features and model_features != expected_features:
            raise ValueError(
                "Loaded model feature names do not match the saved Version 2 training contract."
            )


def predict_outcome_probabilities(model: Any, feature_row: pd.DataFrame) -> dict[str, float]:
    """Return P(team_1_win), P(draw), and P(team_2_win)."""

    probabilities = model.predict_proba(aligned_model_input(feature_row))[0]
    classes = getattr(model, "classes_", OUTCOME_LABELS)
    labels = [str(label) for label in classes]
    probability_map = {label: float(prob) for label, prob in zip(labels, probabilities)}
    if not all(label in probability_map for label in OUTCOME_LABELS):
        probability_map = {label: float(prob) for label, prob in zip(OUTCOME_LABELS, probabilities)}
    return {label: probability_map[label] for label in OUTCOME_LABELS}


def clamp(value: float, lower: float = 0.05, upper: float = 5.0) -> float:
    """Keep expected goals football-realistic."""

    return clamp_expected_goals(value, lower=lower, upper=upper)


def predict_catboost_expected_goals(
    goals_team_1_model: Any,
    goals_team_2_model: Any,
    feature_row: pd.DataFrame,
) -> tuple[float, float]:
    """Predict and clamp CatBoost goal-model expected goals."""

    model_input = aligned_model_input(feature_row)
    expected_goals_1 = clamp_expected_goals(float(goals_team_1_model.predict(model_input)[0]))
    expected_goals_2 = clamp_expected_goals(float(goals_team_2_model.predict(model_input)[0]))
    return round(expected_goals_1, 4), round(expected_goals_2, 4)


def predict_expected_goals(
    goals_team_1_model: Any,
    goals_team_2_model: Any,
    feature_row: pd.DataFrame,
) -> tuple[float, float]:
    """Backward-compatible wrapper for CatBoost goal-model predictions."""

    return predict_catboost_expected_goals(goals_team_1_model, goals_team_2_model, feature_row)


def predict_ensemble_expected_goals(
    goals_team_1_model: Any,
    goals_team_2_model: Any,
    feature_row: pd.DataFrame,
    ensemble_config: dict[str, float],
    goal_scale: float = 1.0,
) -> ExpectedGoalsBreakdown:
    """Combine CatBoost xG with statistical xG, then apply goal-scale calibration."""

    catboost_xg_1, catboost_xg_2 = predict_catboost_expected_goals(
        goals_team_1_model,
        goals_team_2_model,
        feature_row,
    )
    statistical_xg_1, statistical_xg_2 = estimate_statistical_expected_goals(feature_row.iloc[0])
    catboost_weight = float(ensemble_config["best_catboost_weight"])
    statistical_weight = float(ensemble_config["best_statistical_weight"])
    ensemble_xg_1 = clamp_expected_goals(
        catboost_weight * catboost_xg_1 + statistical_weight * statistical_xg_1
    )
    ensemble_xg_2 = clamp_expected_goals(
        catboost_weight * catboost_xg_2 + statistical_weight * statistical_xg_2
    )
    calibrated_xg_1 = calibrate_expected_goals(ensemble_xg_1, goal_scale)
    calibrated_xg_2 = calibrate_expected_goals(ensemble_xg_2, goal_scale)
    return ExpectedGoalsBreakdown(
        catboost_xg_team_1=round(catboost_xg_1, 4),
        catboost_xg_team_2=round(catboost_xg_2, 4),
        statistical_xg_team_1=round(statistical_xg_1, 4),
        statistical_xg_team_2=round(statistical_xg_2, 4),
        ensemble_xg_team_1=round(ensemble_xg_1, 4),
        ensemble_xg_team_2=round(ensemble_xg_2, 4),
        calibrated_xg_team_1=round(calibrated_xg_1, 4),
        calibrated_xg_team_2=round(calibrated_xg_2, 4),
        catboost_weight=round(catboost_weight, 4),
        statistical_weight=round(statistical_weight, 4),
        goal_scale=round(float(goal_scale), 4),
    )


def poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function."""

    return shared_poisson_pmf(k, lam)


def score_winner(goals_1: int, goals_2: int) -> str:
    """Map a scoreline to an outcome label."""

    return shared_score_winner(goals_1, goals_2)


def generate_score_matrix(expected_goals_1: float, expected_goals_2: float, max_goals: int = 6) -> pd.DataFrame:
    """Generate normalized score probabilities from 0-0 to 6-6."""

    return shared_generate_score_matrix(expected_goals_1, expected_goals_2, max_goals=max_goals)


def generate_candidates(score_matrix: pd.DataFrame, outcome_probabilities: dict[str, float]) -> pd.DataFrame:
    """Score every candidate by expected competition points."""

    return shared_generate_candidates(score_matrix, outcome_probabilities)


def select_final_candidate(candidates: pd.DataFrame, decision_mode: str) -> pd.Series:
    """Select the final scoreline with the requested decision rule."""

    if decision_mode == "score_probability":
        return candidates.sort_values("poisson_score_probability", ascending=False).iloc[0]
    if decision_mode == "expected_points":
        return candidates.sort_values(
            ["expected_competition_points", "poisson_score_probability"],
            ascending=[False, False],
        ).iloc[0]
    raise ValueError(f"Unsupported decision_mode `{decision_mode}`. Choose from {DECISION_MODES}.")


def print_top_candidates(candidates: pd.DataFrame, limit: int = 5) -> None:
    """Print the expected-points ranking for debug comparison."""

    print("\nTop 5 scorelines by expected points:")
    columns = [
        "predicted_score",
        "winner_from_score",
        "goal_difference",
        "poisson_score_probability",
        "outcome_probability",
        "goal_difference_probability",
        "expected_competition_points",
    ]
    print(candidates[columns].head(limit).to_string(index=False))


def print_top_score_probabilities(candidates: pd.DataFrame, limit: int = 5) -> None:
    """Print the default score-probability ranking."""

    print("\nTop 5 scorelines by score probability:")
    columns = [
        "predicted_score",
        "winner_from_score",
        "goal_difference",
        "poisson_score_probability",
    ]
    ranked = candidates.sort_values("poisson_score_probability", ascending=False).reset_index(drop=True)
    print(ranked[columns].head(limit).to_string(index=False))


def print_form_features(fixture: FixtureMatch, feature_row: pd.DataFrame) -> None:
    """Print Version 2 rolling form features for both teams."""

    row = feature_row.iloc[0]
    print("\nVersion 2 form features:")
    for prefix, team_name in [
        ("team_1", display_team_name(fixture.team_1)),
        ("team_2", display_team_name(fixture.team_2)),
    ]:
        team_columns = [
            column for column in VERSION_2_5_FEATURE_COLUMNS if column.startswith(prefix)
        ]
        print(f"\n{team_name}:")
        for column in team_columns:
            value = row[column]
            if pd.isna(value):
                print(f"- {column}: NaN")
            else:
                print(f"- {column}: {float(value):.4f}")

    diff_columns = [
        column for column in VERSION_2_5_FEATURE_COLUMNS if not column.startswith(("team_1", "team_2"))
    ]
    print("\nTeam 1 minus Team 2 differences:")
    for column in diff_columns:
        value = row[column]
        if pd.isna(value):
            print(f"- {column}: NaN")
        else:
            print(f"- {column}: {float(value):.4f}")


def print_final_selection(prediction: PredictionResult) -> None:
    """Print the final selected prediction for debugging."""

    print("\nFinal selected prediction:")
    print(f"- Decision mode: {prediction.decision_mode}")
    print(f"- Winner: {prediction.predicted_winner}")
    print(f"- Score: {prediction.predicted_score}")
    print(f"- Goal difference: {prediction.predicted_goal_difference}")
    print(f"- Predicted winner: {prediction.predicted_winner}")
    print(f"- Confidence source: {prediction.confidence_type}")
    print(f"- Outcome confidence: {prediction.outcome_confidence:.4f}")
    print(f"- Exact score confidence: {prediction.exact_score_confidence:.4f}")
    print(f"- Confidence: {prediction.confidence:.2f}/100")
    print(f"- Selected score probability: {prediction.selected_score_probability:.2%}")
    print(f"- Expected competition points: {prediction.expected_competition_points:.4f}")


def print_outcome_probabilities(probabilities: dict[str, float], team_1: str, team_2: str) -> None:
    """Print classifier probabilities in a readable debug block."""

    print("\nOutcome probabilities from CatBoostClassifier:")
    print(f"- {team_1} win: {probabilities['team_1_win']:.4f}")
    print(f"- Draw: {probabilities['draw']:.4f}")
    print(f"- {team_2} win: {probabilities['team_2_win']:.4f}")


def print_expected_goals(
    expected_goals_1: float,
    expected_goals_2: float,
    team_1: str,
    team_2: str,
) -> None:
    """Print CatBoost-only expected goals for backward-compatible callers."""

    print("\nExpected goals from CatBoostRegressor goal models:")
    print(f"- {team_1}: {expected_goals_1:.4f}")
    print(f"- {team_2}: {expected_goals_2:.4f}")


def print_goal_ensemble(breakdown: ExpectedGoalsBreakdown, team_1: str, team_2: str) -> None:
    """Print the full xG ensemble debug block."""

    print("\nExpected goals ensemble:")
    print("CatBoostRegressor xG:")
    print(f"- {team_1}: {breakdown.catboost_xg_team_1:.4f}")
    print(f"- {team_2}: {breakdown.catboost_xg_team_2:.4f}")
    print("Statistical xG:")
    print(f"- {team_1}: {breakdown.statistical_xg_team_1:.4f}")
    print(f"- {team_2}: {breakdown.statistical_xg_team_2:.4f}")
    print("Ensemble weights:")
    print(f"- CatBoost: {breakdown.catboost_weight:.4f}")
    print(f"- Statistical: {breakdown.statistical_weight:.4f}")
    print("Final ensemble xG before goal scale:")
    print(f"- {team_1}: {breakdown.ensemble_xg_team_1:.4f}")
    print(f"- {team_2}: {breakdown.ensemble_xg_team_2:.4f}")
    print("Goal-scale calibration:")
    print(f"- Goal scale: {breakdown.goal_scale:.4f}")
    print(f"- Calibrated {team_1} xG: {breakdown.calibrated_xg_team_1:.4f}")
    print(f"- Calibrated {team_2} xG: {breakdown.calibrated_xg_team_2:.4f}")


def build_explanation(
    fixture: FixtureMatch,
    feature_row: pd.DataFrame,
    probabilities: dict[str, float],
    xg_breakdown: ExpectedGoalsBreakdown,
    best_candidate: pd.Series,
    decision_mode: str,
) -> str:
    """Create a concise explanation for the final row."""

    row = feature_row.iloc[0]
    display_team_1 = display_team_name(fixture.team_1)
    display_team_2 = display_team_name(fixture.team_2)
    winner_label = best_candidate["winner_from_score"]
    winner_text = {
        "team_1_win": display_team_1,
        "draw": "Draw",
        "team_2_win": display_team_2,
    }[winner_label]
    fifa_text = (
        "FIFA snapshot features were disabled by the training contract"
        if pd.isna(row["fifa_points_diff"])
        else f"FIFA points diff {row['fifa_points_diff']:.1f}"
    )
    decision_text = (
        "highest calibrated Poisson score probability"
        if decision_mode == "score_probability"
        else "highest expected competition points"
    )
    return (
        f"{winner_text} selected because its candidate scoreline had the {decision_text}. "
        f"CatBoost outcome probabilities were {display_team_1} {probabilities['team_1_win']:.1%}, "
        f"draw {probabilities['draw']:.1%}, {display_team_2} {probabilities['team_2_win']:.1%}. "
        f"CatBoost xG was {display_team_1} {xg_breakdown.catboost_xg_team_1:.2f} and "
        f"{display_team_2} {xg_breakdown.catboost_xg_team_2:.2f}; statistical xG was "
        f"{display_team_1} {xg_breakdown.statistical_xg_team_1:.2f} and "
        f"{display_team_2} {xg_breakdown.statistical_xg_team_2:.2f}. "
        f"Final ensemble xG was {display_team_1} {xg_breakdown.ensemble_xg_team_1:.2f} and "
        f"{display_team_2} {xg_breakdown.ensemble_xg_team_2:.2f} "
        f"using CatBoost weight {xg_breakdown.catboost_weight:.2f}; calibrated xG was "
        f"{display_team_1} {xg_breakdown.calibrated_xg_team_1:.2f} and "
        f"{display_team_2} {xg_breakdown.calibrated_xg_team_2:.2f} "
        f"with goal scale {xg_breakdown.goal_scale:.2f}. "
        f"Confidence represents predicted winner/outcome confidence: "
        f"{winner_text} {probabilities[winner_label]:.1%}. "
        f"Selected score probability represents exact-score confidence: "
        f"{float(best_candidate['poisson_score_probability']):.1%}. "
        f"Elo diff {row['elo_diff']:.1f}, {fifa_text}, "
        f"last-5 form points diff {row['form_points_diff_last_5']:.1f}, "
        f"last-10 opponent Elo diff {row['avg_opponent_elo_diff_last_10']:.1f}."
    )


def predict_single_match(
    team_1: str,
    team_2: str,
    stage: str,
    match_date: str,
    decision_mode: str = "expected_points",
) -> PredictionResult:
    """Run the complete one-match prediction workflow."""

    if decision_mode not in DECISION_MODES:
        raise ValueError(f"Unsupported decision_mode `{decision_mode}`. Choose from {DECISION_MODES}.")

    fixtures = load_fixtures()
    elo = load_elo()
    fifa = load_fifa()
    results = load_results(RESULTS_PATH)
    fixture = validate_fixture(team_1, team_2, stage, match_date, fixtures, elo, fifa)
    feature_row = build_feature_row(fixture, elo, fifa, results)
    display_team_1 = display_team_name(fixture.team_1)
    display_team_2 = display_team_name(fixture.team_2)

    print("\nNormalized team names:")
    print(f"- team_1: {fixture.team_1}")
    print(f"- team_2: {fixture.team_2}")
    print(
        "Fixture validation result: "
        f"valid -> {display_team_1} vs {display_team_2}, {fixture.stage}, {fixture.match_date}"
    )
    print(
        "Training feature contract: "
        f"{len(model_feature_columns())} features, "
        f"latest FIFA snapshot features enabled = {training_uses_latest_rating_features()}"
    )
    print("Built feature row:")
    print(aligned_model_input(feature_row).to_string(index=False))
    print_form_features(fixture, feature_row)

    outcome_model, goals_team_1_model, goals_team_2_model = load_prediction_models()
    probabilities = predict_outcome_probabilities(outcome_model, feature_row)
    print_outcome_probabilities(probabilities, display_team_1, display_team_2)
    ensemble_config = load_goal_ensemble_config()
    score_selection_config = load_score_selection_config()
    goal_scale = float(score_selection_config["goal_scale"])
    xg_breakdown = predict_ensemble_expected_goals(
        goals_team_1_model,
        goals_team_2_model,
        feature_row,
        ensemble_config,
        goal_scale=goal_scale,
    )
    print_goal_ensemble(xg_breakdown, display_team_1, display_team_2)
    score_matrix = generate_score_matrix(xg_breakdown.calibrated_xg_team_1, xg_breakdown.calibrated_xg_team_2)
    candidates = generate_candidates(score_matrix, probabilities)
    print_top_score_probabilities(candidates, limit=5)
    print_top_candidates(candidates, limit=5)
    print(f"\nFinal decision mode: {decision_mode}")

    best_candidate = select_final_candidate(candidates, decision_mode)
    winner_label = best_candidate["winner_from_score"]
    predicted_winner = {
        "team_1_win": display_team_1,
        "draw": "Draw",
        "team_2_win": display_team_2,
    }[winner_label]
    selected_score_probability = float(best_candidate["poisson_score_probability"])
    outcome_confidence = float(probabilities[winner_label])
    exact_score_confidence = selected_score_probability
    confidence_type = "outcome_probability"
    confidence = round(max(0.0, min(100.0, outcome_confidence * 100.0)), 2)
    explanation = build_explanation(
        fixture,
        feature_row,
        probabilities,
        xg_breakdown,
        best_candidate,
        decision_mode,
    )

    prediction = PredictionResult(
        match_date=fixture.match_date,
        team_1=display_team_1,
        team_2=display_team_2,
        stage=fixture.stage,
        predicted_winner=predicted_winner,
        predicted_score=str(best_candidate["predicted_score"]),
        predicted_goal_difference=int(best_candidate["goal_difference"]),
        confidence=confidence,
        outcome_confidence=round(outcome_confidence, 6),
        exact_score_confidence=round(exact_score_confidence, 6),
        confidence_type=confidence_type,
        team_1_win_probability=round(probabilities["team_1_win"], 4),
        draw_probability=round(probabilities["draw"], 4),
        team_2_win_probability=round(probabilities["team_2_win"], 4),
        expected_goals_team_1=xg_breakdown.calibrated_xg_team_1,
        expected_goals_team_2=xg_breakdown.calibrated_xg_team_2,
        catboost_xg_team_1=xg_breakdown.catboost_xg_team_1,
        catboost_xg_team_2=xg_breakdown.catboost_xg_team_2,
        statistical_xg_team_1=xg_breakdown.statistical_xg_team_1,
        statistical_xg_team_2=xg_breakdown.statistical_xg_team_2,
        ensemble_xg_team_1=xg_breakdown.ensemble_xg_team_1,
        ensemble_xg_team_2=xg_breakdown.ensemble_xg_team_2,
        goal_ensemble_weight_catboost=xg_breakdown.catboost_weight,
        goal_ensemble_weight_statistical=xg_breakdown.statistical_weight,
        decision_mode=decision_mode,
        goal_scale=xg_breakdown.goal_scale,
        calibrated_xg_team_1=xg_breakdown.calibrated_xg_team_1,
        calibrated_xg_team_2=xg_breakdown.calibrated_xg_team_2,
        selected_score_probability=round(selected_score_probability, 6),
        predicted_total_goals=int(best_candidate["goals_team_1"] + best_candidate["goals_team_2"]),
        expected_competition_points=round(float(best_candidate["expected_competition_points"]), 4),
        prediction_method=PREDICTION_METHOD,
        explanation=explanation,
    )
    print_final_selection(prediction)
    return prediction


def save_prediction(prediction: PredictionResult, path: Path = PREDICTION_OUTPUT_PATH) -> None:
    """Append one prediction row to the Version 2 output CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    row = prediction.__dict__
    fieldnames = list(row.keys())
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as existing_file:
            reader = csv.DictReader(existing_file)
            existing_header = reader.fieldnames or []
            existing_rows = list(reader)
        if existing_header and existing_header != fieldnames:
            if not set(existing_header).issubset(set(fieldnames)):
                raise ValueError(
                    "Existing prediction CSV schema has unknown columns that cannot be migrated. "
                    f"Expected {fieldnames}, found {existing_header}."
                )
            with path.open("w", newline="", encoding="utf-8") as migrated_file:
                writer = csv.DictWriter(migrated_file, fieldnames=fieldnames)
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow({field: existing_row.get(field, "") for field in fieldnames})
            print(f"Migrated prediction CSV to the current prediction schema: {path}")

    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    """Parse command-line input."""

    parser = argparse.ArgumentParser(description="Predict one World Cup 2026 match with Version 2.")
    parser.add_argument(
        "--team_1",
        "--term_1",
        dest="team_1",
        required=True,
        help="First team from the user perspective.",
    )
    parser.add_argument(
        "--team_2",
        "--term_2",
        dest="team_2",
        required=True,
        help="Second team from the user perspective.",
    )
    parser.add_argument("--stage", required=True, help="World Cup stage, for example 'Group Stage'.")
    parser.add_argument("--match_date", required=True, help="Match date in YYYY-MM-DD format.")
    parser.add_argument(
        "--decision_mode",
        choices=DECISION_MODES,
        default="expected_points",
        help=(
            "Final score selector. Default uses expected competition points, combining "
            "outcome, goal-difference, and exact-score probabilities."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    try:
        prediction = predict_single_match(
            args.team_1,
            args.team_2,
            args.stage,
            args.match_date,
            decision_mode=args.decision_mode,
        )
        save_prediction(prediction)
    except Exception as exc:
        print(f"Prediction failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\nPrediction saved.")
    print(f"{prediction.team_1} vs {prediction.team_2} on {prediction.match_date}")
    print(f"Winner: {prediction.predicted_winner}")
    print(f"Score: {prediction.predicted_score}")
    print(f"Goal difference: {prediction.predicted_goal_difference}")
    print(f"Decision mode: {prediction.decision_mode}")
    print(f"Confidence: {prediction.confidence:.2f}/100")
    print(f"Selected score probability: {prediction.selected_score_probability:.2%}")
    print(f"Expected competition points: {prediction.expected_competition_points:.4f}")
    print(prediction.explanation)


if __name__ == "__main__":
    main()
