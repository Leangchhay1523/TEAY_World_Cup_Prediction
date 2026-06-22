import re
from pathlib import Path

import pandas as pd

from src.data.shared import normalize_team_name


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


def load_fixtures(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fixture file not found: {path}")
    fixtures = pd.read_csv(path)
    required = {"date", "team_1", "team_2", "stage", "host_country"}
    missing = required.difference(fixtures.columns)
    if missing:
        raise ValueError(f"Fixture file missing columns: {sorted(missing)}")

    fixtures = fixtures.copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce").dt.date
    fixtures["team_1_norm"] = fixtures["team_1"].map(normalize_team_name)
    fixtures["team_2_norm"] = fixtures["team_2"].map(normalize_team_name)
    fixtures["stage_std"] = fixtures["stage"].astype(str).str.strip().str.lower()
    return fixtures


def is_unresolved_placeholder(team_name: str) -> bool:
    return any(re.match(pattern, team_name, flags=re.IGNORECASE) for pattern in PLACEHOLDER_PATTERNS)
