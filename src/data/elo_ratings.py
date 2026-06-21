import numpy as np
import pandas as pd

from src.data.base import BaseDataPipeline


class EloRatings(BaseDataPipeline):
    """Fetch, clean, and export Elo ratings data from eloratings.net."""

    ELO_URL = "https://www.eloratings.net/World.tsv"
    TEAMS_URL = "https://www.eloratings.net/en.teams.tsv?_=1781977931498"

    ELO_COLUMNS = [
        "rank",
        "previous_rank",
        "country_code",
        "elo",
        "elo_rank_change",
        "max_elo",
        "max_rank",
        "min_elo_since",
        "min_rank_since",
        "lowest_elo",
        "change1",
        "value1",
        "change2",
        "value2",
        "change3",
        "value3",
        "change4",
        "value4",
        "change5",
        "value5",
        "change6",
        "value6",
        "matches",
        "wins",
        "draws",
        "losses",
        "home_matches",
        "home_wins",
        "home_draws",
        "goals_for",
        "goals_against",
    ]

    DROP_COLS = [
        "rank",
        "elo_rank_change",
        "max_rank",
        "min_elo_since",
        "min_rank_since",
        "lowest_elo",
        "change1",
        "change2",
        "change3",
        "change4",
        "change5",
        "change6",
    ]

    FORM_COLS = ["value1", "value2", "value3", "value4", "value5", "value6"]

    def fetch(self) -> pd.DataFrame:
        df = pd.read_csv(self.ELO_URL, sep="\t", names=self.ELO_COLUMNS)
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=self.DROP_COLS)

        for col in self.FORM_COLS:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("−", "-", regex=False)
                .replace("-", np.nan)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["recent_form"] = df[self.FORM_COLS].sum(axis=1)
        df = df.drop(columns=self.FORM_COLS)

        teams = pd.read_csv(
            self.TEAMS_URL,
            sep="\t",
            header=None,
            usecols=[0, 1],
            names=["country_code", "team_name"],
            engine="python",
        )

        df = df.merge(teams, on="country_code", how="left")
        cols = ["team_name"] + [c for c in df.columns if c != "team_name"]
        df = df[cols]

        return df
