"""Checks for schedule.py: results from film, and the next game in schedule.csv."""
from datetime import date

import pandas as pd
import pytest

import schedule as sch
import visuals as v


@pytest.fixture(scope="module")
def df():
    return v.load_data()


def test_dowling_results(df):
    r = sch.results(df, "Dowling Catholic")
    r26 = r[r["Season"] == 2026]
    assert len(r26) == 9
    assert (r26["Result"] == "W").sum() == 8
    first = r26.iloc[0]
    assert (first["Week"], first["Date"], first["Opponent"]) == (1, date(2026, 8, 28), "Valley")
    assert r26["Score"].str.fullmatch(r"\d+-\d+").all()


def test_scout_team_results(df):
    r = sch.results(df, "Cedar Falls")
    assert "Dowling Catholic" not in set(r["Opponent"])
    assert len(r) >= 7


def _next(df, text, today, monkeypatch):
    monkeypatch.setattr(sch, "read_text", lambda path=sch.FILE: text)
    return sch.next_game(df, today)


def test_next_game(df, monkeypatch):
    text = "season,date,opponent\n2026,2026-11-06,Cedar Falls\n"
    g = _next(df, text, date(2026, 11, 3), monkeypatch)
    assert g["opponent"] == "Cedar Falls" and g["week"] == 11 and sch.when(g) == "Week 11, Fri 11/6"
    assert _next(df, text, date(2026, 11, 7), monkeypatch) is None            # date passed


def test_played_game_is_not_next(df, monkeypatch):
    # Dowling vs Marshalltown 10/23/26 is on film
    text = "season,date,opponent\n2026,2026-10-23,Marshalltown\n2026,2026-11-06,Cedar Falls\n"
    assert _next(df, text, date(2026, 10, 20), monkeypatch)["opponent"] == "Cedar Falls"


def test_add_next_game_replaces_unplayed(df):
    before = "season,date,opponent\n2026,2026-10-23,Marshalltown\n2026,2026-11-06,Ankeny\n"
    out = pd.read_csv(pd.io.common.BytesIO(sch.with_next_game(before, "Cedar Falls", date(2026, 11, 6), df,
                                                               today=date(2026, 11, 3))), dtype=str)
    assert out.values.tolist() == [["2026", "2026-10-23", "Marshalltown"], ["2026", "2026-11-06", "Cedar Falls"]]
    assert sch.with_next_game("", "Cedar Falls", date(2026, 11, 6), df).decode().splitlines() == \
        ["season,date,opponent", "2026,2026-11-06,Cedar Falls"]
