"""
teams.py — the app's team list and team colors.

    teams.csv   one row per team: its name as used everywhere in the app (filters, tables, Add a Game, Schedule)
                and its primary color (Maroon, Purple, Red, ...). Colors are shown on the Rankings page and can be
                changed there; COLOR_HEX turns each color name into the shade the app draws.
"""
from __future__ import annotations

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HERE, "teams.csv")
TEAM = "Dowling Catholic"

COLOR_HEX = {
    "Maroon": "#8C1D2C",
    "Red": "#C8102E",
    "Orange": "#E87722",
    "Yellow": "#FFC72C",
    "Green": "#1B6B3A",
    "Blue": "#1F4E9E",
    "Light Blue": "#6CACE4",
    "Purple": "#4B2E83",
    "Silver": "#A7A8AA",
    "Black": "#111111",
    "Gold": "#B3A369",
    "Navy": "#13294B",
}

# Abbreviations spelled out in the dropdowns' search text, so "cedar r" finds CR Kennedy.
ABBREVIATIONS = {"CR": "Cedar Rapids", "IC": "Iowa City", "DM": "Des Moines", "SC": "Sioux City",
                 "CB": "Council Bluffs", "SEP": "Southeast Polk", "DCG": "Dallas Center-Grimes",
                 "NW": "Northwest"}


def load(path: str = FILE) -> pd.DataFrame:
    """teams.csv: team, color."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["team", "color"])
    t = pd.read_csv(path, dtype=str).fillna("")
    t = pd.DataFrame({"team": t.get("team", pd.Series(dtype=str)).astype(str).str.strip(),
                      "color": t.get("color", pd.Series(dtype=str)).astype(str).str.strip()})
    return t[t["team"] != ""].reset_index(drop=True)


def names(path: str = FILE) -> list[str]:
    return load(path)["team"].tolist()


def colors(path: str = FILE) -> dict[str, str]:
    """{team: '#RRGGBB'} for teams whose color is one of COLOR_HEX's names."""
    t = load(path)
    lookup = {k.casefold(): v for k, v in COLOR_HEX.items()}
    return {team: lookup[c.casefold()] for team, c in zip(t["team"], t["color"]) if c.casefold() in lookup}


def to_csv(t: pd.DataFrame) -> bytes:
    t = t[["team", "color"]].drop_duplicates("team", keep="last")
    return t.sort_values("team", key=lambda s: s.str.casefold()).to_csv(index=False).encode()


def team_options(*extra) -> list[str]:
    """Every team for the team dropdowns: teams.csv plus any names passed in (e.g. teams already in games.csv),
    sorted, Dowling first."""
    found = set(names())
    for group in extra:
        found |= {str(n).strip() for n in group if str(n).strip()}
    return [TEAM] + sorted((n for n in found if n != TEAM), key=str.casefold)


def option_label(team: str) -> str:
    """'CR Kennedy (Cedar Rapids Kennedy)' for a dropdown, so either spelling finds it; other names as they are."""
    words = team.split()
    spelled = [ABBREVIATIONS.get(w, w) for w in words]
    return f"{team} ({' '.join(spelled)})" if spelled != words else team
