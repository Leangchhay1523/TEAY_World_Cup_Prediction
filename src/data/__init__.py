import pandas as pd

from src.data.elo_ratings import EloRatings
from src.data.fifa_rankings import FifaRankings


__all__ = ["EloRatings", "FifaRankings", "full_data", "merged_data"]


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
