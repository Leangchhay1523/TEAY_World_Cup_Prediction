"""Predict one FIFA World Cup 2026 match with simplified Version 2.

Active architecture:
- CatBoostClassifier predicts outcome probabilities.
- Rating/statistical logic estimates expected goals.
- Poisson matrix generates exact-score and goal-difference probabilities.
- Candidate optimizer selects the scoreline with best expected competition points.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSION_DIR = PROJECT_ROOT / "version_2_ml"

FIXTURE_PATH = PROJECT_ROOT / "data" / "processed" / "worldcup_2026_fixtures_cleaned.csv"
ELO_PATH = PROJECT_ROOT / "data" / "live_updates" / "elo_ratings.csv"
FIFA_PATH = PROJECT_ROOT / "data" / "live_updates" / "fifa_rankings.csv"
OUTCOME_MODEL_PATH = VERSION_DIR / "models" / "catboost_outcome_model.cbm"
PREDICTION_OUTPUT_PATH = VERSION_DIR / "outputs" / "version_2_predictions.csv"

PREDICTION_METHOD = (
    "CatBoost Outcome Model + Rating-Based Poisson Goal Model + "
    "Candidate Expected-Points Optimizer"
)
OUTCOME_LABELS = ["team_1_win", "draw", "team_2_win"]

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
    team_1_win_probability: float
    draw_probability: float
    team_2_win_probability: float
    expected_goals_team_1: float
    expected_goals_team_2: float
    expected_competition_points: float
    prediction_method: str
    explanation: str


def import_catboost_classifier() -> Any:
    """Import CatBoost only when model inference is needed."""

    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError(
            "CatBoost is required for Version 2 predictions. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc
    return CatBoostClassifier


def normalize_team_name(value: Any) -> str:
    """Canonical team name used for all fixture/rating comparisons.

    The matching key deliberately removes accents and normalizes known aliases,
    so user input like "Ivory Coast" can match fixture text like
    "Côte d'Ivoire" and rating text like "Ivory Coast".
    """

    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = unicodedata.normalize("NFKC", text).replace("-", " ")
    text = " ".join(text.split())
    direct_alias = TEAM_ALIASES.get(text)
    if direct_alias:
        text = direct_alias

    ascii_key = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    ascii_key = " ".join(ascii_key.replace("-", " ").split())
    return NORMALIZED_TEAM_ALIASES.get(ascii_key, text)


def standardize_team_name(value: Any) -> str:
    """Backward-compatible wrapper for existing feature code."""

    return normalize_team_name(value)


def display_team_name(team_name: str) -> str:
    """Convert internal canonical names into user-facing display names."""

    return DISPLAY_TEAM_NAMES.get(team_name, team_name)


def is_unresolved_placeholder(team_name: str) -> bool:
    """Return True when a fixture entry is not yet a real team."""

    text = standardize_team_name(team_name)
    return any(re.match(pattern, text, flags=re.IGNORECASE) for pattern in PLACEHOLDER_PATTERNS)


def load_fixtures(path: Path = FIXTURE_PATH) -> pd.DataFrame:
    """Load World Cup 2026 fixtures."""

    if not path.exists():
        raise FileNotFoundError(f"Fixture file not found: {path}")
    fixtures = pd.read_csv(path)
    required = {"date", "team_1", "team_2", "stage", "host_country"}
    missing = required.difference(fixtures.columns)
    if missing:
        raise ValueError(f"Fixture file missing columns: {sorted(missing)}")

    fixtures = fixtures.copy()
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
    for column in ["rank", "points"]:
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


def build_feature_row(fixture: FixtureMatch, elo: pd.DataFrame, fifa: pd.DataFrame) -> pd.DataFrame:
    """Build the classifier feature row from latest Elo/FIFA data."""

    elo_1 = get_single_team_row(elo, fixture.team_1, "Elo")
    elo_2 = get_single_team_row(elo, fixture.team_2, "Elo")
    fifa_1 = get_single_team_row(fifa, fixture.team_1, "FIFA ranking")
    fifa_2 = get_single_team_row(fifa, fixture.team_2, "FIFA ranking")

    row = {
        "team_1_elo": float(elo_1["elo"]),
        "team_2_elo": float(elo_2["elo"]),
        "elo_diff": float(elo_1["elo"] - elo_2["elo"]),
        "team_1_fifa_rank": float(fifa_1["rank"]),
        "team_2_fifa_rank": float(fifa_2["rank"]),
        "fifa_rank_diff": float(fifa_1["rank"] - fifa_2["rank"]),
        "team_1_fifa_points": float(fifa_1["points"]),
        "team_2_fifa_points": float(fifa_2["points"]),
        "fifa_points_diff": float(fifa_1["points"] - fifa_2["points"]),
        "team_1_recent_form": float(elo_1["recent_form"]),
        "team_2_recent_form": float(elo_2["recent_form"]),
        "recent_form_diff": float(elo_1["recent_form"] - elo_2["recent_form"]),
        "team_1_goal_rate": float(elo_1["goal_rate"]),
        "team_2_goal_rate": float(elo_2["goal_rate"]),
        "goal_rate_diff": float(elo_1["goal_rate"] - elo_2["goal_rate"]),
        "team_1_concede_rate": float(elo_1["concede_rate"]),
        "team_2_concede_rate": float(elo_2["concede_rate"]),
        "concede_rate_diff": float(elo_1["concede_rate"] - elo_2["concede_rate"]),
        "tournament": fixture.stage,
        "neutral": infer_neutral_flag(fixture.team_1, fixture.team_2, fixture.host_country),
        "year": int(pd.to_datetime(fixture.match_date).year),
        "team_1_confederation": str(fifa_1["confederation"]),
        "team_2_confederation": str(fifa_2["confederation"]),
    }
    return pd.DataFrame([row])


def model_feature_columns() -> list[str]:
    """Feature order used by the outcome classifier."""

    return [
        "team_1_elo",
        "team_2_elo",
        "elo_diff",
        "team_1_fifa_rank",
        "team_2_fifa_rank",
        "fifa_rank_diff",
        "team_1_fifa_points",
        "team_2_fifa_points",
        "fifa_points_diff",
        "team_1_recent_form",
        "team_2_recent_form",
        "recent_form_diff",
        "team_1_goal_rate",
        "team_2_goal_rate",
        "goal_rate_diff",
        "team_1_concede_rate",
        "team_2_concede_rate",
        "concede_rate_diff",
        "tournament",
        "neutral",
        "year",
        "team_1_confederation",
        "team_2_confederation",
    ]


def load_outcome_model() -> Any:
    """Load only the active CatBoost outcome classifier."""

    if not OUTCOME_MODEL_PATH.exists():
        raise FileNotFoundError(f"Required outcome model not found: {OUTCOME_MODEL_PATH}")
    CatBoostClassifier = import_catboost_classifier()
    model = CatBoostClassifier()
    model.load_model(OUTCOME_MODEL_PATH)
    return model


def predict_outcome_probabilities(model: Any, feature_row: pd.DataFrame) -> dict[str, float]:
    """Return P(team_1_win), P(draw), and P(team_2_win)."""

    probabilities = model.predict_proba(feature_row[model_feature_columns()])[0]
    classes = getattr(model, "classes_", OUTCOME_LABELS)
    labels = [str(label) for label in classes]
    probability_map = {label: float(prob) for label, prob in zip(labels, probabilities)}
    if not all(label in probability_map for label in OUTCOME_LABELS):
        probability_map = {label: float(prob) for label, prob in zip(OUTCOME_LABELS, probabilities)}
    return {label: probability_map[label] for label in OUTCOME_LABELS}


def clamp(value: float, lower: float = 0.2, upper: float = 4.5) -> float:
    """Keep expected goals football-realistic."""

    return max(lower, min(upper, value))


def estimate_expected_goals(feature_row: pd.DataFrame) -> tuple[float, float]:
    """Estimate expected goals without CatBoostRegressor.

    The logic blends team attacking rate, opponent concede rate, Elo strength,
    FIFA points strength, and recent form. Both teams retain a minimum scoring
    expectation so weaker teams do not collapse to zero.
    """

    row = feature_row.iloc[0]
    base_1 = np.nanmean([row["team_1_goal_rate"], row["team_2_concede_rate"]])
    base_2 = np.nanmean([row["team_2_goal_rate"], row["team_1_concede_rate"]])

    if np.isnan(base_1):
        base_1 = 1.2
    if np.isnan(base_2):
        base_2 = 1.2

    elo_component = row["elo_diff"] / 400.0
    fifa_component = row["fifa_points_diff"] / 350.0
    form_component = row["recent_form_diff"] / 300.0
    strength_adjustment = 0.22 * elo_component + 0.12 * fifa_component + 0.08 * form_component

    neutral_modifier = 0.0 if str(row["neutral"]).lower() == "true" else 0.08
    expected_goals_1 = base_1 + strength_adjustment + neutral_modifier
    expected_goals_2 = base_2 - strength_adjustment

    # Very strong mismatches should tilt goals, but stay realistic.
    if row["elo_diff"] > 200:
        expected_goals_1 += 0.15
        expected_goals_2 -= 0.08
    elif row["elo_diff"] < -200:
        expected_goals_1 -= 0.08
        expected_goals_2 += 0.15

    return float(round(clamp(expected_goals_1), 4)), float(round(clamp(expected_goals_2), 4))


def poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function."""

    return math.exp(-lam) * math.pow(lam, k) / math.factorial(k)


def score_winner(goals_1: int, goals_2: int) -> str:
    """Map a scoreline to an outcome label."""

    if goals_1 > goals_2:
        return "team_1_win"
    if goals_1 < goals_2:
        return "team_2_win"
    return "draw"


def generate_score_matrix(expected_goals_1: float, expected_goals_2: float, max_goals: int = 6) -> pd.DataFrame:
    """Generate normalized score probabilities from 0-0 to 6-6."""

    lambda_1 = clamp(expected_goals_1)
    lambda_2 = clamp(expected_goals_2)
    rows = []
    for goals_1 in range(max_goals + 1):
        for goals_2 in range(max_goals + 1):
            probability = poisson_pmf(goals_1, lambda_1) * poisson_pmf(goals_2, lambda_2)
            rows.append(
                {
                    "goals_team_1": goals_1,
                    "goals_team_2": goals_2,
                    "predicted_score": f"{goals_1}-{goals_2}",
                    "poisson_score_probability": probability,
                    "goal_difference": goals_1 - goals_2,
                    "winner_from_score": score_winner(goals_1, goals_2),
                }
            )
    matrix = pd.DataFrame(rows)
    matrix["poisson_score_probability"] = (
        matrix["poisson_score_probability"] / matrix["poisson_score_probability"].sum()
    )
    return matrix


def generate_candidates(score_matrix: pd.DataFrame, outcome_probabilities: dict[str, float]) -> pd.DataFrame:
    """Score every candidate by expected competition points."""

    goal_diff_probabilities = (
        score_matrix.groupby("goal_difference")["poisson_score_probability"].sum().to_dict()
    )
    candidates = score_matrix.copy()
    candidates["outcome_probability"] = candidates["winner_from_score"].map(outcome_probabilities)
    candidates["goal_difference_probability"] = candidates["goal_difference"].map(goal_diff_probabilities)
    candidates["expected_competition_points"] = (
        3.0 * candidates["outcome_probability"]
        + 2.0 * candidates["goal_difference_probability"]
        + 5.0 * candidates["poisson_score_probability"]
    )
    return candidates.sort_values(
        ["expected_competition_points", "poisson_score_probability"],
        ascending=[False, False],
    ).reset_index(drop=True)


def print_top_candidates(candidates: pd.DataFrame, limit: int = 10) -> None:
    """Print useful debugging information before final selection."""

    print("\nTop candidate predictions:")
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


def build_explanation(
    fixture: FixtureMatch,
    feature_row: pd.DataFrame,
    probabilities: dict[str, float],
    expected_goals_1: float,
    expected_goals_2: float,
    best_candidate: pd.Series,
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
    return (
        f"{winner_text} selected because its candidate scoreline maximized expected competition points. "
        f"CatBoost outcome probabilities were {display_team_1} {probabilities['team_1_win']:.1%}, "
        f"draw {probabilities['draw']:.1%}, {display_team_2} {probabilities['team_2_win']:.1%}. "
        f"Poisson expected goals were {display_team_1} {expected_goals_1:.2f} and "
        f"{display_team_2} {expected_goals_2:.2f}. "
        f"Elo diff {row['elo_diff']:.1f}, FIFA points diff {row['fifa_points_diff']:.1f}."
    )


def predict_single_match(team_1: str, team_2: str, stage: str, match_date: str) -> PredictionResult:
    """Run the complete one-match prediction workflow."""

    fixtures = load_fixtures()
    elo = load_elo()
    fifa = load_fifa()
    fixture = validate_fixture(team_1, team_2, stage, match_date, fixtures, elo, fifa)
    feature_row = build_feature_row(fixture, elo, fifa)
    display_team_1 = display_team_name(fixture.team_1)
    display_team_2 = display_team_name(fixture.team_2)

    print(f"Validated fixture: {display_team_1} vs {display_team_2}, {fixture.stage}, {fixture.match_date}")
    print("Built feature row:")
    print(feature_row[model_feature_columns()].to_string(index=False))

    outcome_model = load_outcome_model()
    probabilities = predict_outcome_probabilities(outcome_model, feature_row)
    expected_goals_1, expected_goals_2 = estimate_expected_goals(feature_row)
    score_matrix = generate_score_matrix(expected_goals_1, expected_goals_2)
    candidates = generate_candidates(score_matrix, probabilities)
    print_top_candidates(candidates)

    best_candidate = candidates.iloc[0]
    winner_label = best_candidate["winner_from_score"]
    predicted_winner = {
        "team_1_win": display_team_1,
        "draw": "Draw",
        "team_2_win": display_team_2,
    }[winner_label]
    selected_winner_probability = probabilities[winner_label]
    confidence = round(max(0.0, min(100.0, selected_winner_probability * 100.0)), 2)
    explanation = build_explanation(
        fixture,
        feature_row,
        probabilities,
        expected_goals_1,
        expected_goals_2,
        best_candidate,
    )

    return PredictionResult(
        match_date=fixture.match_date,
        team_1=display_team_1,
        team_2=display_team_2,
        stage=fixture.stage,
        predicted_winner=predicted_winner,
        predicted_score=str(best_candidate["predicted_score"]),
        predicted_goal_difference=int(best_candidate["goal_difference"]),
        confidence=confidence,
        team_1_win_probability=round(probabilities["team_1_win"], 4),
        draw_probability=round(probabilities["draw"], 4),
        team_2_win_probability=round(probabilities["team_2_win"], 4),
        expected_goals_team_1=expected_goals_1,
        expected_goals_team_2=expected_goals_2,
        expected_competition_points=round(float(best_candidate["expected_competition_points"]), 4),
        prediction_method=PREDICTION_METHOD,
        explanation=explanation,
    )


def save_prediction(prediction: PredictionResult, path: Path = PREDICTION_OUTPUT_PATH) -> None:
    """Append one prediction row to the Version 2 output CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    row = prediction.__dict__
    fieldnames = list(row.keys())
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as existing_file:
            reader = csv.reader(existing_file)
            existing_header = next(reader, [])
        if existing_header and existing_header != fieldnames:
            archive_dir = VERSION_DIR / "_archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived_path = archive_dir / f"version_2_predictions_legacy_{timestamp}.csv"
            path.replace(archived_path)
            print(f"Archived old prediction schema to: {archived_path}")

    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    """Parse command-line input."""

    parser = argparse.ArgumentParser(description="Predict one World Cup 2026 match with Version 2.")
    parser.add_argument("--team_1", required=True, help="First team from the user perspective.")
    parser.add_argument("--team_2", required=True, help="Second team from the user perspective.")
    parser.add_argument("--stage", required=True, help="World Cup stage, for example 'Group Stage'.")
    parser.add_argument("--match_date", required=True, help="Match date in YYYY-MM-DD format.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    try:
        prediction = predict_single_match(args.team_1, args.team_2, args.stage, args.match_date)
        save_prediction(prediction)
    except Exception as exc:
        print(f"Prediction failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\nPrediction saved.")
    print(f"{prediction.team_1} vs {prediction.team_2} on {prediction.match_date}")
    print(f"Winner: {prediction.predicted_winner}")
    print(f"Score: {prediction.predicted_score}")
    print(f"Goal difference: {prediction.predicted_goal_difference}")
    print(f"Confidence: {prediction.confidence:.2f}/100")
    print(f"Expected competition points: {prediction.expected_competition_points:.4f}")
    print(prediction.explanation)


if __name__ == "__main__":
    main()
