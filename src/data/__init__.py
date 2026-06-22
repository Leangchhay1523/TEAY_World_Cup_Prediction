import pandas as pd

from src.data.elo_ratings import EloRatings, load_elo_csv
from src.data.fifa_rankings import FifaRankings, load_fifa_csv
from src.data.fixtures import load_fixtures, is_unresolved_placeholder
from src.data.results import load_results
from src.data.shared import (
    TEAM_ALIASES,
    NORMALIZED_TEAM_ALIASES,
    DISPLAY_TEAM_NAMES,
    standardize_team_name,
    normalize_team_name,
    display_team_name,
)


__all__ = [
    "EloRatings",
    "FifaRankings",
    "full_data",
    "merged_data",
    "load_elo_csv",
    "load_fifa_csv",
    "load_results",
    "load_fixtures",
    "is_unresolved_placeholder",
    "TEAM_ALIASES",
    "NORMALIZED_TEAM_ALIASES",
    "DISPLAY_TEAM_NAMES",
    "standardize_team_name",
    "normalize_team_name",
    "display_team_name",
]


def full_data() -> dict[str, pd.DataFrame]:
    """Fetch, clean, and return all datasets as a dict of DataFrames."""
    elo = EloRatings()
    fifa = FifaRankings()
    return {
        "elo_ratings": elo.run(),
        "fifa_rankings": fifa.run(),
    }


def merged_data(how: str = "inner") -> pd.DataFrame:
    """Return elo and FIFA data merged on ``country_code``."""
    data = full_data()
    elo = data["elo_ratings"]
    fifa = data["fifa_rankings"]
    return elo.merge(fifa, on="country_code", how=how)
