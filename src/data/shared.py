import unicodedata
from typing import Any

import pandas as pd


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
    "cote d'ivoire": "Cote d'Ivoire",
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


def standardize_team_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.replace("-", " ").split())
    return TEAM_ALIASES.get(text, text)


def normalize_team_name(value: Any) -> str:
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


def display_team_name(team_name: str) -> str:
    return DISPLAY_TEAM_NAMES.get(team_name, team_name)
