from pathlib import Path

import pandas as pd
import requests

from src.data.base import BaseDataPipeline
from src.data.shared import standardize_team_name


def load_fifa_csv(path: Path) -> pd.DataFrame:
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


class FifaRankings(BaseDataPipeline):
    """Fetch, clean, and export FIFA rankings data from the official API."""

    API_URL = (
        "https://api.fifa.com/api/v3/fifarankings/rankings/live"
        "?gender=1&sportType=0&language=en"
    )
    HEADERS = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://inside.fifa.com/",
        "Origin": "https://inside.fifa.com",
    }

    def fetch(self) -> list[dict]:
        resp = requests.get(self.API_URL, headers=self.HEADERS)
        resp.raise_for_status()
        return resp.json()

    def clean(self, raw: dict) -> pd.DataFrame:
        rows = []
        for team in raw["Results"]:
            rows.append(
                {
                    "id_team": team["IdTeam"],
                    "country_code": team["IdCountry"],
                    "team": team["TeamName"][0]["Description"],
                    "gender": "Men" if team["Gender"] == 1 else "Women",
                    "rank": team["Rank"],
                    "previous_rank": team["PrevRank"],
                    "ranking_move": team["RankingMovement"],
                    "points": team["TotalPoints"],
                    "previous_points": team["PrevPoints"],
                    "rated_matches": team["RatedMatches"],
                    "confederation": team["ConfederationName"],
                    "movement": team["RankingMovement"],
                }
            )
        return pd.DataFrame(rows)
