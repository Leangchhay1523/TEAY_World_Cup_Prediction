from pathlib import Path

import pandas as pd

from src.data.shared import standardize_team_name


def load_results(path: Path) -> pd.DataFrame:
    results = pd.read_csv(path)
    required = {"date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Historical results missing columns: {sorted(missing)}")

    results = results.copy()
    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    results["home_score"] = pd.to_numeric(results["home_score"], errors="coerce")
    results["away_score"] = pd.to_numeric(results["away_score"], errors="coerce")
    results["home_team"] = results["home_team"].map(standardize_team_name)
    results["away_team"] = results["away_team"].map(standardize_team_name)
    results["neutral"] = results["neutral"].astype(str).str.lower().isin(["true", "1", "yes"])

    results = results.dropna(subset=["date", "home_score", "away_score"])
    results = results[results["home_team"].ne("") & results["away_team"].ne("")]
    return results.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
