"""Checks for schedule.py: results from film, and the next game in schedule.csv."""
from datetime import date

import pandas as pd
import pytest

import schedule as sch
import visuals as v


@pytest.fixture(scope="module")
def df():
    return v.load_data()


def test_add_next_game_replaces_unplayed(df):
    before = "season,date,opponent\n2026,2026-10-23,Marshalltown\n2026,2026-11-06,Ankeny\n"
    out = pd.read_csv(pd.io.common.BytesIO(sch.with_next_game(before, "Cedar Falls", date(2026, 11, 6), df,
                                                               today=date(2026, 11, 3))), dtype=str)
    assert out.values.tolist() == [["2026", "2026-10-23", "Marshalltown"], ["2026", "2026-11-06", "Cedar Falls"]]
    assert sch.with_next_game("", "Cedar Falls", date(2026, 11, 6), df).decode().splitlines() == \
        ["season,date,opponent", "2026,2026-11-06,Cedar Falls"]
