import requests
import pandas as pd

from src.data.base import BaseDataPipeline


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
