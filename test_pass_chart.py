"""Checks for pass_chart.py on the Dowling vs Johnston game."""
import pandas as pd
import pytest

import build_data
import pass_chart as pc

GAME = "Dowling Catholic_Johnston_2026_09_04"


@pytest.fixture(scope="module")
def game():
    d = build_data.load_games()[0]
    return d[d["game_id"] == GAME]


def test_stat_lines(game):
    assert {k: pc.stat_line(pc.throws(game, "Dowling Catholic"))[k] for k in ("comp", "att", "td", "int")} == \
        {"comp": 7, "att": 11, "td": 3, "int": 0}
    assert {k: pc.stat_line(pc.throws(game, "Johnston"))[k] for k in ("comp", "att", "td", "int")} == \
        {"comp": 13, "att": 29, "td": 4, "int": 1}


def test_throws_land_in_their_zone_and_never_behind_the_line(game):
    t = pc.throws(game, "Johnston")
    third = (t["PASS ZONE"] - 1) % 3
    assert ((t["x"] >= third * pc.FIELD_W / 3) & (t["x"] <= (third + 1) * pc.FIELD_W / 3)).all()
    assert ((t["y"] - t["AIR YARDS"]).abs() <= 0.6).all()
    assert (t["y"] >= 0).all()
    again = pc.throws(game, "Johnston")
    assert (again["x"] == t["x"]).all()           # jitter is the same every time


def test_html_draws(game):
    import base64, re
    out = pc.pass_chart_html(game, "Dowling Catholic")
    svg = base64.b64decode(re.search(r"base64,([^\"]+)", out).group(1)).decode()
    assert svg.count("<circle") == 11 and "7/11" in out and "3-0" in out
    assert pc.pass_chart_html(game, "Nobody") == ""
    # Defense view: Johnston's throws are the ones against Dowling's defense.
    assert len(pc.throws(game, "Dowling Catholic", side="defense")) == 29
