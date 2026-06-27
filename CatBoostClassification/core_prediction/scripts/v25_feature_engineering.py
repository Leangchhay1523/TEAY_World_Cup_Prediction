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
EXPECTED_GOALS_MIN = 0.05
EXPECTED_GOALS_MAX = 7.0
CALIBRATED_EXPECTED_GOALS_MAX = 6.0

TEAM_ALIASES = {
    "USA": "United States",
    "US": "United States",
    "United States of America": "United States",
    "IR Iran": "Iran",
    "Islamic Republic of Iran": "Iran",
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
    "ir iran": "Iran",
    "iran": "Iran",
    "islamic republic of iran": "Iran",
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

PACKAGE_A_FEATURE_COLUMNS = [
    "team_1_draw_rate_last_5",
    "team_2_draw_rate_last_5",
    "team_1_draw_rate_last_10",
    "team_2_draw_rate_last_10",
    "draw_rate_diff_last_5",
    "draw_rate_diff_last_10",
    "team_1_clean_sheet_rate_last_5",
    "team_2_clean_sheet_rate_last_5",
    "team_1_clean_sheet_rate_last_10",
    "team_2_clean_sheet_rate_last_10",
    "clean_sheet_rate_diff_last_5",
    "clean_sheet_rate_diff_last_10",
    "team_1_failed_to_score_rate_last_5",
    "team_2_failed_to_score_rate_last_5",
    "team_1_failed_to_score_rate_last_10",
    "team_2_failed_to_score_rate_last_10",
    "failed_to_score_rate_diff_last_5",
    "failed_to_score_rate_diff_last_10",
    "team_1_avg_goal_difference_last_5",
    "team_2_avg_goal_difference_last_5",
    "team_1_avg_goal_difference_last_10",
    "team_2_avg_goal_difference_last_10",
    "avg_goal_difference_diff_last_5",
    "avg_goal_difference_diff_last_10",
]

PACKAGE_B_FEATURE_COLUMNS = [
    "team_1_over_2_5_rate_last_5",
    "team_2_over_2_5_rate_last_5",
    "team_1_over_2_5_rate_last_10",
    "team_2_over_2_5_rate_last_10",
    "over_2_5_rate_diff_last_5",
    "over_2_5_rate_diff_last_10",
    "team_1_over_3_5_rate_last_5",
    "team_2_over_3_5_rate_last_5",
    "team_1_over_3_5_rate_last_10",
    "team_2_over_3_5_rate_last_10",
    "over_3_5_rate_diff_last_5",
    "over_3_5_rate_diff_last_10",
    "team_1_under_2_5_rate_last_5",
    "team_2_under_2_5_rate_last_5",
    "team_1_under_2_5_rate_last_10",
    "team_2_under_2_5_rate_last_10",
    "under_2_5_rate_diff_last_5",
    "under_2_5_rate_diff_last_10",
    "team_1_goals_scored_std_last_5",
    "team_2_goals_scored_std_last_5",
    "team_1_goals_scored_std_last_10",
    "team_2_goals_scored_std_last_10",
    "goals_scored_std_diff_last_5",
    "goals_scored_std_diff_last_10",
    "team_1_goals_conceded_std_last_5",
    "team_2_goals_conceded_std_last_5",
    "team_1_goals_conceded_std_last_10",
    "team_2_goals_conceded_std_last_10",
    "goals_conceded_std_diff_last_5",
    "goals_conceded_std_diff_last_10",
]

PACKAGE_C_FEATURE_COLUMNS = [
    "h2h_matches_count",
    "h2h_team_1_win_rate",
    "h2h_team_2_win_rate",
    "h2h_draw_rate",
    "h2h_avg_total_goals",
    "h2h_avg_goal_difference_team_1",
    "h2h_last_match_goal_difference_team_1",
    "team_1_neutral_win_rate_last_10",
    "team_2_neutral_win_rate_last_10",
    "team_1_neutral_goal_rate_last_10",
    "team_2_neutral_goal_rate_last_10",
    "team_1_neutral_concede_rate_last_10",
    "team_2_neutral_concede_rate_last_10",
    "neutral_win_rate_diff_last_10",
    "neutral_goal_rate_diff_last_10",
    "neutral_concede_rate_diff_last_10",
    "team_1_home_goal_rate_last_10",
    "team_1_away_goal_rate_last_10",
    "team_2_home_goal_rate_last_10",
    "team_2_away_goal_rate_last_10",
    "team_1_home_win_rate_last_10",
    "team_1_away_win_rate_last_10",
    "team_2_home_win_rate_last_10",
    "team_2_away_win_rate_last_10",
    "team_1_attack_vs_team_2_defense_ratio_last_10",
    "team_2_attack_vs_team_1_defense_ratio_last_10",
    "attack_defense_ratio_diff_last_10",
]

EXISTING_FIFA_FEATURE_COLUMNS = [
    "team_1_fifa_rank",
    "team_2_fifa_rank",
    "fifa_rank_diff",
    "team_1_fifa_points",
    "team_2_fifa_points",
    "fifa_points_diff",
    "team_1_confederation",
    "team_2_confederation",
]

NEW_FIFA_FEATURE_COLUMNS = [
    "team_1_previous_fifa_rank",
    "team_2_previous_fifa_rank",
    "team_1_ranking_move",
    "team_2_ranking_move",
    "ranking_move_diff",
    "team_1_previous_fifa_points",
    "team_2_previous_fifa_points",
    "team_1_points_change",
    "team_2_points_change",
    "points_change_diff",
    "team_1_rated_matches",
    "team_2_rated_matches",
    "rated_matches_diff",
    "fifa_rank_ratio",
    "fifa_points_ratio",
    "team_1_top10",
    "team_2_top10",
    "team_1_top20",
    "team_2_top20",
    "team_1_top30",
    "team_2_top30",
    "team_1_top50",
    "team_2_top50",
]

FIFA_FEATURE_COLUMNS = EXISTING_FIFA_FEATURE_COLUMNS + NEW_FIFA_FEATURE_COLUMNS

VERSION_2_5_FEATURE_COLUMNS = (
    RESULT_FORM_FEATURE_COLUMNS
    + FORM_POINTS_FEATURE_COLUMNS
    + GOAL_FORM_FEATURE_COLUMNS
    + OPPONENT_ELO_FEATURE_COLUMNS
    + PACKAGE_A_FEATURE_COLUMNS
    + PACKAGE_B_FEATURE_COLUMNS
    + PACKAGE_C_FEATURE_COLUMNS
)

MODEL_FEATURE_COLUMNS = [
    "team_1_elo",
    "team_2_elo",
    "elo_diff",
    "elo_ratio",
    "team_1_fifa_rank",
    "team_2_fifa_rank",
    "fifa_rank_diff",
    "team_1_fifa_points",
    "team_2_fifa_points",
    "fifa_points_diff",
    "team_1_previous_fifa_rank",
    "team_2_previous_fifa_rank",
    "team_1_ranking_move",
    "team_2_ranking_move",
    "ranking_move_diff",
    "team_1_previous_fifa_points",
    "team_2_previous_fifa_points",
    "team_1_points_change",
    "team_2_points_change",
    "points_change_diff",
    "team_1_rated_matches",
    "team_2_rated_matches",
    "rated_matches_diff",
    "fifa_rank_ratio",
    "fifa_points_ratio",
    "team_1_top10",
    "team_2_top10",
    "team_1_top20",
    "team_2_top20",
    "team_1_top30",
    "team_2_top30",
    "team_1_top50",
    "team_2_top50",
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
    recent_goal_differences: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_total_goals: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_opponent_elos: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_neutral_wins: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_neutral_goals_for: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_neutral_goals_against: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_home_wins: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_home_goals_for: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_away_wins: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_away_goals_for: deque[float] = field(default_factory=lambda: deque(maxlen=10))


@dataclass(frozen=True)
class HeadToHeadResult:
    """One previous match between two teams in canonical team order."""

    team_1: str
    team_2: str
    goals_1: int
    goals_2: int


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


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a ratio, using NaN when the denominator is unavailable."""

    if pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0.0:
        return np.nan
    return float(numerator) / float(denominator)


def _fifa_value(row: pd.Series | dict[str, Any] | None, column: str) -> Any:
    """Read a FIFA snapshot value from a row-like object."""

    if row is None:
        return np.nan
    try:
        return row[column]
    except (KeyError, TypeError):
        return np.nan


def _fifa_number(row: pd.Series | dict[str, Any] | None, column: str) -> float:
    """Read one numeric FIFA snapshot value."""

    value = _fifa_value(row, column)
    if pd.isna(value):
        return np.nan
    return float(value)


def _fifa_denominator(value: float) -> float:
    """Use the requested max(value, 1) denominator rule for FIFA ratios."""

    if pd.isna(value):
        return np.nan
    return max(float(value), 1.0)


def _fifa_tier_flag(rank: float, threshold: int) -> float:
    """Return 1/0 for available FIFA ranks and NaN when ranking is disabled."""

    if pd.isna(rank):
        return np.nan
    return float(int(float(rank) <= threshold))


def empty_fifa_features() -> dict[str, Any]:
    """Return the complete FIFA feature family with leakage-safe empty values."""

    return {
        "team_1_fifa_rank": np.nan,
        "team_2_fifa_rank": np.nan,
        "fifa_rank_diff": np.nan,
        "team_1_fifa_points": np.nan,
        "team_2_fifa_points": np.nan,
        "fifa_points_diff": np.nan,
        "team_1_previous_fifa_rank": np.nan,
        "team_2_previous_fifa_rank": np.nan,
        "team_1_ranking_move": np.nan,
        "team_2_ranking_move": np.nan,
        "ranking_move_diff": np.nan,
        "team_1_previous_fifa_points": np.nan,
        "team_2_previous_fifa_points": np.nan,
        "team_1_points_change": np.nan,
        "team_2_points_change": np.nan,
        "points_change_diff": np.nan,
        "team_1_rated_matches": np.nan,
        "team_2_rated_matches": np.nan,
        "rated_matches_diff": np.nan,
        "fifa_rank_ratio": np.nan,
        "fifa_points_ratio": np.nan,
        "team_1_top10": np.nan,
        "team_2_top10": np.nan,
        "team_1_top20": np.nan,
        "team_2_top20": np.nan,
        "team_1_top30": np.nan,
        "team_2_top30": np.nan,
        "team_1_top50": np.nan,
        "team_2_top50": np.nan,
        "team_1_confederation": "unknown",
        "team_2_confederation": "unknown",
    }


def build_fifa_feature_pair(
    team_1_fifa: pd.Series | dict[str, Any] | None,
    team_2_fifa: pd.Series | dict[str, Any] | None,
    enabled: bool,
) -> dict[str, Any]:
    """Build current, momentum, experience, ratio, tier, and confederation FIFA features."""

    if not enabled:
        return empty_fifa_features()

    rank_1 = _fifa_number(team_1_fifa, "rank")
    rank_2 = _fifa_number(team_2_fifa, "rank")
    previous_rank_1 = _fifa_number(team_1_fifa, "previous_rank")
    previous_rank_2 = _fifa_number(team_2_fifa, "previous_rank")
    ranking_move_1 = _fifa_number(team_1_fifa, "ranking_move")
    ranking_move_2 = _fifa_number(team_2_fifa, "ranking_move")
    points_1 = _fifa_number(team_1_fifa, "points")
    points_2 = _fifa_number(team_2_fifa, "points")
    previous_points_1 = _fifa_number(team_1_fifa, "previous_points")
    previous_points_2 = _fifa_number(team_2_fifa, "previous_points")
    points_change_1 = points_1 - previous_points_1
    points_change_2 = points_2 - previous_points_2
    rated_matches_1 = _fifa_number(team_1_fifa, "rated_matches")
    rated_matches_2 = _fifa_number(team_2_fifa, "rated_matches")

    return {
        "team_1_fifa_rank": rank_1,
        "team_2_fifa_rank": rank_2,
        "fifa_rank_diff": rank_1 - rank_2,
        "team_1_fifa_points": points_1,
        "team_2_fifa_points": points_2,
        "fifa_points_diff": points_1 - points_2,
        "team_1_previous_fifa_rank": previous_rank_1,
        "team_2_previous_fifa_rank": previous_rank_2,
        "team_1_ranking_move": ranking_move_1,
        "team_2_ranking_move": ranking_move_2,
        "ranking_move_diff": ranking_move_1 - ranking_move_2,
        "team_1_previous_fifa_points": previous_points_1,
        "team_2_previous_fifa_points": previous_points_2,
        "team_1_points_change": points_change_1,
        "team_2_points_change": points_change_2,
        "points_change_diff": points_change_1 - points_change_2,
        "team_1_rated_matches": rated_matches_1,
        "team_2_rated_matches": rated_matches_2,
        "rated_matches_diff": rated_matches_1 - rated_matches_2,
        "fifa_rank_ratio": safe_ratio(rank_1, _fifa_denominator(rank_2)),
        "fifa_points_ratio": safe_ratio(points_1, _fifa_denominator(points_2)),
        "team_1_top10": _fifa_tier_flag(rank_1, 10),
        "team_2_top10": _fifa_tier_flag(rank_2, 10),
        "team_1_top20": _fifa_tier_flag(rank_1, 20),
        "team_2_top20": _fifa_tier_flag(rank_2, 20),
        "team_1_top30": _fifa_tier_flag(rank_1, 30),
        "team_2_top30": _fifa_tier_flag(rank_2, 30),
        "team_1_top50": _fifa_tier_flag(rank_1, 50),
        "team_2_top50": _fifa_tier_flag(rank_2, 50),
        "team_1_confederation": _fifa_value(team_1_fifa, "confederation")
        if not pd.isna(_fifa_value(team_1_fifa, "confederation"))
        else "unknown",
        "team_2_confederation": _fifa_value(team_2_fifa, "confederation")
        if not pd.isna(_fifa_value(team_2_fifa, "confederation"))
        else "unknown",
    }


def clamp_expected_goals(
    value: float,
    lower: float = EXPECTED_GOALS_MIN,
    upper: float = EXPECTED_GOALS_MAX,
) -> float:
    """Clamp an expected-goals estimate to a realistic football range."""

    if pd.isna(value):
        return lower
    return max(lower, min(upper, float(value)))


def calibrate_expected_goals(value: float, goal_scale: float) -> float:
    """Apply the tuned goal scale and clamp the final Poisson lambda."""

    return clamp_expected_goals(
        float(value) * float(goal_scale),
        lower=EXPECTED_GOALS_MIN,
        upper=CALIBRATED_EXPECTED_GOALS_MAX,
    )


def _row_value(row: pd.Series | dict[str, Any], column: str, default: float = np.nan) -> float:
    """Read one numeric feature from a row-like object."""

    try:
        value = row[column]
    except (KeyError, TypeError):
        return default
    if pd.isna(value):
        return default
    return float(value)


def _weighted_mean(values: list[tuple[float, float]], default: float = 1.25) -> float:
    """Average available values while ignoring missing inputs."""

    total = 0.0
    weight_total = 0.0
    for value, weight in values:
        if pd.isna(value):
            continue
        total += float(value) * weight
        weight_total += weight
    if weight_total == 0.0:
        return default
    return total / weight_total


def estimate_statistical_expected_goals(row: pd.Series | dict[str, Any]) -> tuple[float, float]:
    """Estimate stable expected goals from leakage-safe model features.

    This baseline uses pre-match scoring rates, opponent concede rates, recent
    goal form, Elo difference, optional FIFA points, and tournament importance.
    It deliberately avoids any current-match result information.
    """

    team_1_attack = _weighted_mean(
        [
            (_row_value(row, "team_1_goal_rate"), 0.35),
            (_row_value(row, "team_1_avg_goals_scored_last_5"), 0.35),
            (_row_value(row, "team_1_avg_goals_scored_last_10"), 0.20),
        ]
    )
    team_2_attack = _weighted_mean(
        [
            (_row_value(row, "team_2_goal_rate"), 0.35),
            (_row_value(row, "team_2_avg_goals_scored_last_5"), 0.35),
            (_row_value(row, "team_2_avg_goals_scored_last_10"), 0.20),
        ]
    )
    team_1_opponent_allows = _weighted_mean(
        [
            (_row_value(row, "team_2_concede_rate"), 0.35),
            (_row_value(row, "team_2_avg_goals_conceded_last_5"), 0.35),
            (_row_value(row, "team_2_avg_goals_conceded_last_10"), 0.20),
        ]
    )
    team_2_opponent_allows = _weighted_mean(
        [
            (_row_value(row, "team_1_concede_rate"), 0.35),
            (_row_value(row, "team_1_avg_goals_conceded_last_5"), 0.35),
            (_row_value(row, "team_1_avg_goals_conceded_last_10"), 0.20),
        ]
    )

    team_1_xg = 0.58 * team_1_attack + 0.42 * team_1_opponent_allows
    team_2_xg = 0.58 * team_2_attack + 0.42 * team_2_opponent_allows

    elo_diff = _row_value(row, "elo_diff", 0.0)
    elo_adjustment = float(np.clip(elo_diff / 1000.0, -0.35, 0.35))
    team_1_multiplier = 1.0 + elo_adjustment
    team_2_multiplier = 1.0 - elo_adjustment

    fifa_points_diff = _row_value(row, "fifa_points_diff", 0.0)
    fifa_adjustment = 0.0 if pd.isna(fifa_points_diff) else float(np.clip(fifa_points_diff / 2500.0, -0.18, 0.18))
    team_1_multiplier += fifa_adjustment
    team_2_multiplier -= fifa_adjustment

    importance = _row_value(row, "tournament_importance_weight", 0.5)
    tournament_multiplier = 0.92 + 0.13 * float(np.clip(importance, 0.0, 1.0))

    team_1_xg *= team_1_multiplier * tournament_multiplier
    team_2_xg *= team_2_multiplier * tournament_multiplier
    return clamp_expected_goals(team_1_xg), clamp_expected_goals(team_2_xg)


def poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function."""

    lam = clamp_expected_goals(lam)
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

    lambda_1 = clamp_expected_goals(expected_goals_1)
    lambda_2 = clamp_expected_goals(expected_goals_2)
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


def select_best_score_candidate(
    expected_goals_1: float,
    expected_goals_2: float,
    outcome_probabilities: dict[str, float],
    max_goals: int = 6,
) -> dict[str, Any]:
    """Select the best scoreline without constructing pandas objects per match."""

    matrix_total = 0.0
    score_rows: list[dict[str, Any]] = []
    diff_probabilities: dict[int, float] = {}
    for goals_1 in range(max_goals + 1):
        for goals_2 in range(max_goals + 1):
            probability = poisson_pmf(goals_1, expected_goals_1) * poisson_pmf(goals_2, expected_goals_2)
            goal_difference = goals_1 - goals_2
            winner = score_winner(goals_1, goals_2)
            score_rows.append(
                {
                    "goals_team_1": goals_1,
                    "goals_team_2": goals_2,
                    "predicted_score": f"{goals_1}-{goals_2}",
                    "poisson_score_probability": probability,
                    "goal_difference": goal_difference,
                    "winner_from_score": winner,
                }
            )
            matrix_total += probability

    for row in score_rows:
        row["poisson_score_probability"] = row["poisson_score_probability"] / matrix_total
        diff = int(row["goal_difference"])
        diff_probabilities[diff] = diff_probabilities.get(diff, 0.0) + row["poisson_score_probability"]

    for row in score_rows:
        row["outcome_probability"] = float(outcome_probabilities[row["winner_from_score"]])
        row["goal_difference_probability"] = diff_probabilities[int(row["goal_difference"])]
        row["expected_competition_points"] = (
            3.0 * row["outcome_probability"]
            + 2.0 * row["goal_difference_probability"]
            + 5.0 * row["poisson_score_probability"]
        )

    return max(
        score_rows,
        key=lambda row: (row["expected_competition_points"], row["poisson_score_probability"]),
    )


def select_highest_score_probability_candidate(
    expected_goals_1: float,
    expected_goals_2: float,
    max_goals: int = 6,
) -> dict[str, Any]:
    """Select the most likely exact score from the Poisson matrix."""

    best_candidate: dict[str, Any] | None = None
    matrix_total = 0.0
    score_rows: list[dict[str, Any]] = []
    for goals_1 in range(max_goals + 1):
        for goals_2 in range(max_goals + 1):
            probability = poisson_pmf(goals_1, expected_goals_1) * poisson_pmf(goals_2, expected_goals_2)
            row = {
                "goals_team_1": goals_1,
                "goals_team_2": goals_2,
                "predicted_score": f"{goals_1}-{goals_2}",
                "poisson_score_probability": probability,
                "goal_difference": goals_1 - goals_2,
                "winner_from_score": score_winner(goals_1, goals_2),
            }
            score_rows.append(row)
            matrix_total += probability

    for row in score_rows:
        row["poisson_score_probability"] = row["poisson_score_probability"] / matrix_total
        if best_candidate is None or (
            row["poisson_score_probability"],
            -abs(row["goal_difference"]),
            -(row["goals_team_1"] + row["goals_team_2"]),
        ) > (
            best_candidate["poisson_score_probability"],
            -abs(best_candidate["goal_difference"]),
            -(best_candidate["goals_team_1"] + best_candidate["goals_team_2"]),
        ):
            best_candidate = row

    if best_candidate is None:
        raise ValueError("Could not select a score probability candidate.")
    return best_candidate


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


def _rate_last(values: deque[Any], window: int, target: Any) -> float:
    """Rate of target values in the previous N matches."""

    selected = _last_values(values, window)
    if not selected:
        return np.nan
    return float(selected.count(target) / len(selected))


def _zero_rate_last(values: deque[float], window: int) -> float:
    """Rate of previous N matches where the tracked goal value was zero."""

    selected = _last_values(values, window)
    if not selected:
        return np.nan
    return float(sum(1 for value in selected if float(value) == 0.0) / len(selected))


def _threshold_rate_last(values: deque[float], window: int, threshold: float, operator: str) -> float:
    """Rate of previous N matches above or below a numeric threshold."""

    selected = _last_values(values, window)
    if not selected:
        return np.nan
    if operator == "gt":
        count = sum(1 for value in selected if float(value) > threshold)
    elif operator == "lt":
        count = sum(1 for value in selected if float(value) < threshold)
    else:
        raise ValueError(f"Unsupported threshold operator: {operator}")
    return float(count / len(selected))


def _std_last(values: deque[float], window: int) -> float:
    """Population standard deviation over prior matches, or 0 when history is too short."""

    selected = _last_values(values, window)
    if len(selected) < 2:
        return 0.0
    return float(np.std(selected, ddof=0))


def _average_last_or_default(values: deque[float], window: int, default: float = 0.0) -> float:
    """Average the last N rolling values, using a safe default when unavailable."""

    value = _average_last(values, window)
    return default if pd.isna(value) else value


def _h2h_key(team_1: str, team_2: str) -> tuple[str, str]:
    """Return an order-independent key for a head-to-head matchup."""

    return tuple(sorted((team_1, team_2)))


def build_h2h_features(
    team_1: str,
    team_2: str,
    h2h_history: dict[tuple[str, str], list[HeadToHeadResult]],
) -> dict[str, float]:
    """Build Package C head-to-head features from prior matches only."""

    previous_matches = h2h_history.get(_h2h_key(team_1, team_2), [])
    if not previous_matches:
        return {
            "h2h_matches_count": 0.0,
            "h2h_team_1_win_rate": 0.0,
            "h2h_team_2_win_rate": 0.0,
            "h2h_draw_rate": 0.0,
            "h2h_avg_total_goals": 0.0,
            "h2h_avg_goal_difference_team_1": 0.0,
            "h2h_last_match_goal_difference_team_1": 0.0,
        }

    goal_differences = []
    total_goals = []
    team_1_wins = 0
    team_2_wins = 0
    draws = 0

    for match in previous_matches:
        if match.team_1 == team_1:
            goals_for = match.goals_1
            goals_against = match.goals_2
        else:
            goals_for = match.goals_2
            goals_against = match.goals_1

        goal_difference = goals_for - goals_against
        goal_differences.append(float(goal_difference))
        total_goals.append(float(goals_for + goals_against))
        if goal_difference > 0:
            team_1_wins += 1
        elif goal_difference < 0:
            team_2_wins += 1
        else:
            draws += 1

    match_count = len(previous_matches)
    return {
        "h2h_matches_count": float(match_count),
        "h2h_team_1_win_rate": float(team_1_wins / match_count),
        "h2h_team_2_win_rate": float(team_2_wins / match_count),
        "h2h_draw_rate": float(draws / match_count),
        "h2h_avg_total_goals": float(np.mean(total_goals)),
        "h2h_avg_goal_difference_team_1": float(np.mean(goal_differences)),
        "h2h_last_match_goal_difference_team_1": float(goal_differences[-1]),
    }


def update_h2h_after_match(
    h2h_history: dict[tuple[str, str], list[HeadToHeadResult]],
    team_1: str,
    team_2: str,
    goals_1: int,
    goals_2: int,
) -> None:
    """Add one completed match to the head-to-head history."""

    h2h_history.setdefault(_h2h_key(team_1, team_2), []).append(
        HeadToHeadResult(team_1=team_1, team_2=team_2, goals_1=goals_1, goals_2=goals_2)
    )


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
        # Package A: share of prior matches in the window that ended level.
        features[f"{prefix}_draw_rate_last_{window}"] = _rate_last(state.recent_results, window, "draw")
        # Package A: share of prior matches in the window with zero goals conceded.
        features[f"{prefix}_clean_sheet_rate_last_{window}"] = _zero_rate_last(state.recent_goals_against, window)
        # Package A: share of prior matches in the window with zero goals scored.
        features[f"{prefix}_failed_to_score_rate_last_{window}"] = _zero_rate_last(state.recent_goals_for, window)
        # Package A: average goals-for minus goals-against across prior matches in the window.
        features[f"{prefix}_avg_goal_difference_last_{window}"] = _average_last(
            state.recent_goal_differences,
            window,
        )
        # Package B: exact-score context from total-goal thresholds in prior matches.
        features[f"{prefix}_over_2_5_rate_last_{window}"] = _threshold_rate_last(
            state.recent_total_goals,
            window,
            2.5,
            "gt",
        )
        features[f"{prefix}_over_3_5_rate_last_{window}"] = _threshold_rate_last(
            state.recent_total_goals,
            window,
            3.5,
            "gt",
        )
        features[f"{prefix}_under_2_5_rate_last_{window}"] = _threshold_rate_last(
            state.recent_total_goals,
            window,
            2.5,
            "lt",
        )
        # Package B: scoring volatility from previous goals scored and conceded.
        features[f"{prefix}_goals_scored_std_last_{window}"] = _std_last(state.recent_goals_for, window)
        features[f"{prefix}_goals_conceded_std_last_{window}"] = _std_last(state.recent_goals_against, window)
    # Package C: neutral-ground and home/away style over prior context.
    features[f"{prefix}_neutral_win_rate_last_10"] = _average_last_or_default(state.recent_neutral_wins, 10)
    features[f"{prefix}_neutral_goal_rate_last_10"] = _average_last_or_default(state.recent_neutral_goals_for, 10)
    features[f"{prefix}_neutral_concede_rate_last_10"] = _average_last_or_default(
        state.recent_neutral_goals_against,
        10,
    )
    features[f"{prefix}_home_goal_rate_last_10"] = _average_last_or_default(state.recent_home_goals_for, 10)
    features[f"{prefix}_away_goal_rate_last_10"] = _average_last_or_default(state.recent_away_goals_for, 10)
    features[f"{prefix}_home_win_rate_last_10"] = _average_last_or_default(state.recent_home_wins, 10)
    features[f"{prefix}_away_win_rate_last_10"] = _average_last_or_default(state.recent_away_wins, 10)
    return features


def build_form_feature_pair(
    state_1: RollingTeamState,
    state_2: RollingTeamState,
    team_1: str = "",
    team_2: str = "",
    h2h_history: dict[tuple[str, str], list[HeadToHeadResult]] | None = None,
) -> dict[str, float]:
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
        # Package A difference features are always Team 1 minus Team 2.
        features[f"draw_rate_diff_last_{window}"] = (
            features[f"team_1_draw_rate_last_{window}"]
            - features[f"team_2_draw_rate_last_{window}"]
        )
        features[f"clean_sheet_rate_diff_last_{window}"] = (
            features[f"team_1_clean_sheet_rate_last_{window}"]
            - features[f"team_2_clean_sheet_rate_last_{window}"]
        )
        features[f"failed_to_score_rate_diff_last_{window}"] = (
            features[f"team_1_failed_to_score_rate_last_{window}"]
            - features[f"team_2_failed_to_score_rate_last_{window}"]
        )
        features[f"avg_goal_difference_diff_last_{window}"] = (
            features[f"team_1_avg_goal_difference_last_{window}"]
            - features[f"team_2_avg_goal_difference_last_{window}"]
        )
        # Package B difference features are always Team 1 minus Team 2.
        features[f"over_2_5_rate_diff_last_{window}"] = (
            features[f"team_1_over_2_5_rate_last_{window}"]
            - features[f"team_2_over_2_5_rate_last_{window}"]
        )
        features[f"over_3_5_rate_diff_last_{window}"] = (
            features[f"team_1_over_3_5_rate_last_{window}"]
            - features[f"team_2_over_3_5_rate_last_{window}"]
        )
        features[f"under_2_5_rate_diff_last_{window}"] = (
            features[f"team_1_under_2_5_rate_last_{window}"]
            - features[f"team_2_under_2_5_rate_last_{window}"]
        )
        features[f"goals_scored_std_diff_last_{window}"] = (
            features[f"team_1_goals_scored_std_last_{window}"]
            - features[f"team_2_goals_scored_std_last_{window}"]
        )
        features[f"goals_conceded_std_diff_last_{window}"] = (
            features[f"team_1_goals_conceded_std_last_{window}"]
            - features[f"team_2_goals_conceded_std_last_{window}"]
        )
    features.update(build_h2h_features(team_1, team_2, h2h_history or {}))
    features["neutral_win_rate_diff_last_10"] = (
        features["team_1_neutral_win_rate_last_10"]
        - features["team_2_neutral_win_rate_last_10"]
    )
    features["neutral_goal_rate_diff_last_10"] = (
        features["team_1_neutral_goal_rate_last_10"]
        - features["team_2_neutral_goal_rate_last_10"]
    )
    features["neutral_concede_rate_diff_last_10"] = (
        features["team_1_neutral_concede_rate_last_10"]
        - features["team_2_neutral_concede_rate_last_10"]
    )
    team_1_attack = _row_value(features, "team_1_avg_goals_scored_last_10", 0.0)
    team_2_attack = _row_value(features, "team_2_avg_goals_scored_last_10", 0.0)
    team_1_defense_allows = max(_row_value(features, "team_1_avg_goals_conceded_last_10", 0.0), 0.1)
    team_2_defense_allows = max(_row_value(features, "team_2_avg_goals_conceded_last_10", 0.0), 0.1)
    features["team_1_attack_vs_team_2_defense_ratio_last_10"] = team_1_attack / team_2_defense_allows
    features["team_2_attack_vs_team_1_defense_ratio_last_10"] = team_2_attack / team_1_defense_allows
    features["attack_defense_ratio_diff_last_10"] = (
        features["team_1_attack_vs_team_2_defense_ratio_last_10"]
        - features["team_2_attack_vs_team_1_defense_ratio_last_10"]
    )
    return features


def update_states_after_match(
    state_1: RollingTeamState,
    state_2: RollingTeamState,
    goals_1: int,
    goals_2: int,
    neutral: bool = False,
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
    state_1.recent_goal_differences.append(float(goals_1 - goals_2))
    state_2.recent_goal_differences.append(float(goals_2 - goals_1))
    state_1.recent_total_goals.append(float(goals_1 + goals_2))
    state_2.recent_total_goals.append(float(goals_1 + goals_2))
    state_1.recent_opponent_elos.append(float(pre_match_elo_2))
    state_2.recent_opponent_elos.append(float(pre_match_elo_1))
    state_1.recent_home_wins.append(1.0 if result_1 == "win" else 0.0)
    state_1.recent_home_goals_for.append(float(goals_1))
    state_2.recent_away_wins.append(1.0 if result_2 == "win" else 0.0)
    state_2.recent_away_goals_for.append(float(goals_2))
    if neutral:
        state_1.recent_neutral_wins.append(1.0 if result_1 == "win" else 0.0)
        state_2.recent_neutral_wins.append(1.0 if result_2 == "win" else 0.0)
        state_1.recent_neutral_goals_for.append(float(goals_1))
        state_1.recent_neutral_goals_against.append(float(goals_2))
        state_2.recent_neutral_goals_for.append(float(goals_2))
        state_2.recent_neutral_goals_against.append(float(goals_1))


def build_historical_context_before_date(
    results: pd.DataFrame,
    cutoff_date: Any,
) -> tuple[dict[str, RollingTeamState], dict[tuple[str, str], list[HeadToHeadResult]]]:
    """Build team and head-to-head context from matches strictly before cutoff_date."""

    cutoff = pd.to_datetime(cutoff_date, errors="raise")
    states: dict[str, RollingTeamState] = {}
    h2h_history: dict[tuple[str, str], list[HeadToHeadResult]] = {}

    for match in results.sort_values(["date", "home_team", "away_team"]).itertuples(index=False):
        if pd.Timestamp(match.date) >= cutoff:
            break
        team_1 = match.home_team
        team_2 = match.away_team
        goals_1 = int(match.home_score)
        goals_2 = int(match.away_score)
        state_1 = states.setdefault(team_1, RollingTeamState())
        state_2 = states.setdefault(team_2, RollingTeamState())
        update_states_after_match(
            state_1,
            state_2,
            goals_1,
            goals_2,
            neutral=bool(match.neutral),
        )
        update_h2h_after_match(h2h_history, team_1, team_2, goals_1, goals_2)

    return states, h2h_history


def build_states_before_date(results: pd.DataFrame, cutoff_date: Any) -> dict[str, RollingTeamState]:
    """Build rolling team states from matches strictly before cutoff_date."""

    states, _ = build_historical_context_before_date(results, cutoff_date)
    return states
