"""Checks for rankings.py: the per-team table, ranks, and the 'number (rank)' text."""
import pandas as pd
import pytest

import insights as ins
import rankings as rk
import visuals as v


@pytest.fixture(scope="module")
def table():
    df = ins.add_game_state(v.load_data())
    return rk.team_table(df[df["SEASON"] == 2026]).reset_index(drop=True)


def test_every_team_once(table):
    assert table["Team"].is_unique
    assert {"Dowling Catholic", "Cedar Falls", "IC High", "Waukee NW"} <= set(table["Team"])
    assert "Iowa City High" not in set(table["Team"])


def test_dowling_counts(table):
    d = table.set_index("Team").loc["Dowling Catholic"]
    assert d["Games"] == 9
    assert d["_record"] == "8-1"
    assert 0 <= d["Off Success"] <= 1 and 0 <= d["Off Pass Rate"] <= 1
    assert d["Net EPA"] == pytest.approx(d["Off EPA"] - d["Def EPA"])


def test_rank_direction(table):
    r = rk.ranks(table)
    best_off = table.loc[table["Off EPA"].idxmax(), "Team"]
    best_def = table.loc[table["Def EPA"].idxmin(), "Team"]   # lowest EPA allowed is best
    assert r.loc[table["Team"] == best_off, "Off EPA"].item() == 1
    assert r.loc[table["Team"] == best_def, "Def EPA"].item() == 1
    assert "Games" not in r   # counts aren't ranked


def test_display_text():
    t = pd.DataFrame([{"Team": "A", "Off Pass EPA": 0.2, "FG %": 0.75, "_fg": "3/4", "_pat": "0/0"},
                      {"Team": "B", "Off Pass EPA": 0.31, "FG %": float("nan"), "_fg": "0/0", "_pat": "0/0"}])
    for c in rk.COLUMNS:
        if c.label not in t:
            t[c.label] = float("nan")
    t["_record"] = "0-0"
    text = rk.display_text(t, rk.ranks(t))
    assert text.loc[0, "Off Pass EPA"] == "+0.20 (2)"
    assert text.loc[1, "Off Pass EPA"] == "+0.31 (1)"
    assert text.loc[0, "FG %"] == "75% (3/4) (1)"
    assert text.loc[1, "FG %"] == "–"


def test_text_color_contrast():
    assert rk._text_color("#8C1D2C") == "#FFFFFF"
    assert rk._text_color("#FFD100") == "#111111"
