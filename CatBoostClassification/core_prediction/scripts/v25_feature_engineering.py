"""Shared Version 2 football feature engineering helpers.

The functions in this module are intentionally side-effect free. Training and
single-match prediction both use them so rolling form, tournament grouping, and
opponent-strength features are computed consistently.
"""

from __future__ import annotations

import math
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTCOME_LABELS = ["team_1_win", "draw", "team_2_win"]

TEAM_ALIASES = {
    "USA": "United States",
    "US": "United States",
    "United States of America": "United States",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Czech Republic": "Czechia",
    "Turkiye": "Turkey",
    "Cote d'Ivoire": "Ivory Coast",
    "Cote d Ivoire": "Ivory Coast",
    "Ivory Coast": "Ivory Coast",
    "Cape Verde": "Cabo Verde",
}

NORMALIZED_TEAM_ALIASES = {
    "usa": "United States",
    "us": "United States",
    "united states of america": "United States",
    "south korea": "South Korea",
    "korea republic": "South Korea",
    "republic of korea": "South Korea",
    "czech republic": "Czechia",
    "turkiye": "Turkey",
    "cote d'ivoire": "Ivory Coast",
    "cote divoire": "Ivory Coast",
    "cote d ivoire": "Ivory Coast",
    "ivory coast": "Ivory Coast",
    "cabo verde": "Cabo Verde",
    "cape verde": "Cabo Verde",
    "curacao": "Curacao",
}

TOURNAMENT_IMPORTANCE_WEIGHTS = {
    "FIFA World Cup": 1.00,
    "Continental Championship": 0.90,
    "World Cup Qualification": 0.80,
    "Continental Qualification": 0.70,
    "Nations League": 0.60,
    "Friendly": 0.30,
    "Other": 0.50,
}

CONTINENTAL_CHAMPIONSHIP_KEYWORDS = [
    "afc asian cup",
    "africa cup of nations",
    "african cup of nations",
    "copa america",
    "copa américa",
    "gold cup",
    "ofc nations cup",
    "oceania nations cup",
    "uefa euro",
]

CONTINENTAL_QUALIFICATION_KEYWORDS = [
    "afc asian cup qualification",
    "africa cup of nations qualification",
    "african cup of nations qualification",
    "copa america qualification",
    "copa américa qualification",
    "gold cup qualification",
    "ofc nations cup qualification",
    "oceania nations cup qualification",
    "uefa euro qualification",
]

FORM_WINDOWS = [5, 10]

RESULT_FORM_FEATURE_COLUMNS = [
    f"{team}_{result}_last_{window}"
    for team in ["team_1", "team_2"]
    for window in FORM_WINDOWS
    for result in ["wins", "draws", "losses"]
]

FORM_POINTS_FEATURE_COLUMNS = [
    "team_1_form_points_last_5",
    "team_1_form_points_last_10",
    "team_2_form_points_last_5",
    "team_2_form_points_last_10",
    "form_points_diff_last_5",
    "form_points_diff_last_10",
]

GOAL_FORM_FEATURE_COLUMNS = [
    "team_1_avg_goals_scored_last_5",
    "team_1_avg_goals_scored_last_10",
    "team_1_avg_goals_conceded_last_5",
    "team_1_avg_goals_conceded_last_10",
    "team_2_avg_goals_scored_last_5",
    "team_2_avg_goals_scored_last_10",
    "team_2_avg_goals_conceded_last_5",
    "team_2_avg_goals_conceded_last_10",
    "avg_goals_scored_diff_last_5",
    "avg_goals_scored_diff_last_10",
    "avg_goals_conceded_diff_last_5",
    "avg_goals_conceded_diff_last_10",
]

OPPONENT_ELO_FEATURE_COLUMNS = [
    "team_1_avg_opponent_elo_last_5",
    "team_1_avg_opponent_elo_last_10",
    "team_2_avg_opponent_elo_last_5",
    "team_2_avg_opponent_elo_last_10",
    "avg_opponent_elo_diff_last_5",
    "avg_opponent_elo_diff_last_10",
]

VERSION_2_5_FEATURE_COLUMNS = (
    RESULT_FORM_FEATURE_COLUMNS
    + FORM_POINTS_FEATURE_COLUMNS
    + GOAL_FORM_FEATURE_COLUMNS
    + OPPONENT_ELO_FEATURE_COLUMNS
)

MODEL_FEATURE_COLUMNS = [
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
    *VERSION_2_5_FEATURE_COLUMNS,
    "tournament",
    "tournament_type_group",
    "tournament_importance_weight",
    "neutral",
    "year",
    "team_1_confederation",
    "team_2_confederation",
]

CATEGORICAL_MODEL_FEATURES = [
    "tournament",
    "tournament_type_group",
    "neutral",
    "team_1_confederation",
    "team_2_confederation",
]


@dataclass
class RollingTeamState:
    """Pre-match state built only from previous matches."""

    elo: float = 1500.0
    matches: int = 0
    goals_for: int = 0
    goals_against: int = 0
    recent_results: deque[str] = field(default_factory=lambda: deque(maxlen=10))
    recent_points: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_goals_for: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_goals_against: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_opponent_elos: deque[float] = field(default_factory=lambda: deque(maxlen=10))


def normalize_team_name(value: Any) -> str:
    """Normalize common team-name spelling, accent, and alias differences."""

    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.replace("-", " ").split())
    text = TEAM_ALIASES.get(text, text)

    ascii_key = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    ascii_key = " ".join(ascii_key.replace("-", " ").split())
    return NORMALIZED_TEAM_ALIASES.get(ascii_key, text)


def standardize_team_name(value: Any) -> str:
    """Backward-compatible name for normalization used by existing scripts."""

    return normalize_team_name(value)


def load_results(path: Path) -> pd.DataFrame:
    """Load historical match results in canonical, chronological form."""

    results = pd.read_csv(path)
    required = {"date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Historical results missing columns: {sorted(missing)}")

    results = results.copy()
    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    results["home_score"] = pd.to_numeric(results["home_score"], errors="coerce")
    results["away_score"] = pd.to_numeric(results["away_score"], errors="coerce")
    results["home_team"] = results["home_team"].map(normalize_team_name)
    results["away_team"] = results["away_team"].map(normalize_team_name)
    results["neutral"] = results["neutral"].astype(str).str.lower().isin(["true", "1", "yes"])

    results = results.dropna(subset=["date", "home_score", "away_score"])
    results = results[results["home_team"].ne("") & results["away_team"].ne("")]
    return results.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def safe_rate(numerator: float, denominator: float) -> float:
    """Return a rate, using NaN when no previous matches exist."""

    if denominator <= 0:
        return np.nan
    return numerator / denominator


def expected_score(rating_a: float, rating_b: float) -> float:
    """Elo expected result for team A."""

    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def update_elo(rating_a: float, rating_b: float, score_a: float, k_factor: float = 20.0) -> tuple[float, float]:
    """Update two rolling Elo ratings."""

    expected_a = expected_score(rating_a, rating_b)
    change = k_factor * (score_a - expected_a)
    return rating_a + change, rating_b - change


def outcome_label(goals_1: int, goals_2: int) -> str:
    """Convert a scoreline into a classifier target."""

    if goals_1 > goals_2:
        return "team_1_win"
    if goals_1 < goals_2:
        return "team_2_win"
    return "draw"


def result_points(goals_for: int, goals_against: int) -> tuple[str, float, float]:
    """Return result label, football points, and Elo result score."""

    if goals_for > goals_against:
        return "win", 3.0, 1.0
    if goals_for < goals_against:
        return "loss", 0.0, 0.0
    return "draw", 1.0, 0.5


def _last_values(values: deque[Any], window: int) -> list[Any]:
    """Return the last N values from a rolling deque."""

    return list(values)[-window:]


def _sum_last(values: deque[float], window: int) -> float:
    """Sum the last N rolling values."""

    return float(sum(_last_values(values, window)))


def _average_last(values: deque[float], window: int) -> float:
    """Average the last N rolling values, returning NaN when unavailable."""

    selected = _last_values(values, window)
    if not selected:
        return np.nan
    return float(np.mean(selected))


def get_tournament_type_group(tournament: Any) -> str:
    """Map detailed tournament names to Version 2 tournament groups."""

    if pd.isna(tournament):
        return "Other"

    text = str(tournament).strip()
    text_lower = text.lower()
    text_ascii = (
        unicodedata.normalize("NFKD", text_lower)
        .encode("ascii", "ignore")
        .decode("ascii")
    )

    if "friendly" in text_ascii:
        return "Friendly"
    if "nations league" in text_ascii:
        return "Nations League"
    if "qualification" in text_ascii and "world cup" in text_ascii:
        return "World Cup Qualification"
    if any(keyword in text_lower for keyword in CONTINENTAL_QUALIFICATION_KEYWORDS):
        return "Continental Qualification"
    if text_lower in {"fifa world cup", "world cup"}:
        return "FIFA World Cup"
    if any(keyword in text_lower for keyword in CONTINENTAL_CHAMPIONSHIP_KEYWORDS):
        return "Continental Championship"
    return "Other"


def get_tournament_importance_weight(tournament_or_group: Any) -> float:
    """Return numeric importance weight for a tournament group."""

    group = str(tournament_or_group)
    if group not in TOURNAMENT_IMPORTANCE_WEIGHTS:
        group = get_tournament_type_group(tournament_or_group)
    return float(TOURNAMENT_IMPORTANCE_WEIGHTS.get(group, TOURNAMENT_IMPORTANCE_WEIGHTS["Other"]))


def build_team_form_features(prefix: str, state: RollingTeamState) -> dict[str, float]:
    """Build rolling Version 2 form features for one team."""

    features: dict[str, float] = {}
    for window in FORM_WINDOWS:
        recent_results = _last_values(state.recent_results, window)
        features[f"{prefix}_wins_last_{window}"] = float(recent_results.count("win"))
        features[f"{prefix}_draws_last_{window}"] = float(recent_results.count("draw"))
        features[f"{prefix}_losses_last_{window}"] = float(recent_results.count("loss"))
        features[f"{prefix}_form_points_last_{window}"] = _sum_last(state.recent_points, window)
        features[f"{prefix}_avg_goals_scored_last_{window}"] = _average_last(state.recent_goals_for, window)
        features[f"{prefix}_avg_goals_conceded_last_{window}"] = _average_last(state.recent_goals_against, window)
        features[f"{prefix}_avg_opponent_elo_last_{window}"] = _average_last(state.recent_opponent_elos, window)
    return features


def build_form_feature_pair(state_1: RollingTeamState, state_2: RollingTeamState) -> dict[str, float]:
    """Build all Version 2 rolling features for both teams and differences."""

    features = {
        **build_team_form_features("team_1", state_1),
        **build_team_form_features("team_2", state_2),
    }
    for window in FORM_WINDOWS:
        features[f"form_points_diff_last_{window}"] = (
            features[f"team_1_form_points_last_{window}"]
            - features[f"team_2_form_points_last_{window}"]
        )
        features[f"avg_goals_scored_diff_last_{window}"] = (
            features[f"team_1_avg_goals_scored_last_{window}"]
            - features[f"team_2_avg_goals_scored_last_{window}"]
        )
        features[f"avg_goals_conceded_diff_last_{window}"] = (
            features[f"team_1_avg_goals_conceded_last_{window}"]
            - features[f"team_2_avg_goals_conceded_last_{window}"]
        )
        features[f"avg_opponent_elo_diff_last_{window}"] = (
            features[f"team_1_avg_opponent_elo_last_{window}"]
            - features[f"team_2_avg_opponent_elo_last_{window}"]
        )
    return features


def update_states_after_match(
    state_1: RollingTeamState,
    state_2: RollingTeamState,
    goals_1: int,
    goals_2: int,
    k_factor: float = 20.0,
) -> None:
    """Update rolling states after a match has contributed its pre-match row."""

    pre_match_elo_1 = state_1.elo
    pre_match_elo_2 = state_2.elo
    result_1, points_1, elo_score_1 = result_points(goals_1, goals_2)
    result_2, points_2, _ = result_points(goals_2, goals_1)

    state_1.elo, state_2.elo = update_elo(pre_match_elo_1, pre_match_elo_2, elo_score_1, k_factor)
    state_1.matches += 1
    state_2.matches += 1
    state_1.goals_for += goals_1
    state_1.goals_against += goals_2
    state_2.goals_for += goals_2
    state_2.goals_against += goals_1
    state_1.recent_results.append(result_1)
    state_2.recent_results.append(result_2)
    state_1.recent_points.append(points_1)
    state_2.recent_points.append(points_2)
    state_1.recent_goals_for.append(float(goals_1))
    state_1.recent_goals_against.append(float(goals_2))
    state_2.recent_goals_for.append(float(goals_2))
    state_2.recent_goals_against.append(float(goals_1))
    state_1.recent_opponent_elos.append(float(pre_match_elo_2))
    state_2.recent_opponent_elos.append(float(pre_match_elo_1))


def build_states_before_date(results: pd.DataFrame, cutoff_date: Any) -> dict[str, RollingTeamState]:
    """Build rolling team states from matches strictly before cutoff_date."""

    cutoff = pd.to_datetime(cutoff_date, errors="raise")
    states: dict[str, RollingTeamState] = {}

    for match in results.sort_values(["date", "home_team", "away_team"]).itertuples(index=False):
        if pd.Timestamp(match.date) >= cutoff:
            break
        team_1 = match.home_team
        team_2 = match.away_team
        state_1 = states.setdefault(team_1, RollingTeamState())
        state_2 = states.setdefault(team_2, RollingTeamState())
        update_states_after_match(
            state_1,
            state_2,
            int(match.home_score),
            int(match.away_score),
        )

    return states

