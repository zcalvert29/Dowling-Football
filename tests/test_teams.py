"""Checks for teams.py: the team list and colors in teams.csv."""
import pandas as pd

import teams as tm
import visuals as v


def test_every_team_in_the_data_has_a_color():
    df = v.load_data()
    in_data = set(df["offense"].dropna()) | set(df["defense"].dropna())
    colors = tm.colors()
    assert in_data <= set(colors), sorted(in_data - set(colors))


def test_colors_are_known_names():
    t = tm.load()
    assert t["team"].is_unique
    assert set(t["color"]) <= set(tm.COLOR_HEX)
    assert tm.colors()["Dowling Catholic"] == tm.COLOR_HEX["Maroon"]


def test_good_teams_use_team_names():
    good = set(pd.read_csv("good_teams.csv")["team"])
    assert good <= set(tm.names())


def test_options_and_labels():
    opts = tm.team_options(["Cedar Falls", "Zz New Team"])
    assert opts[0] == "Dowling Catholic" and "Zz New Team" in opts and len(opts) == len(set(opts))
    assert tm.option_label("CR Kennedy") == "CR Kennedy (Cedar Rapids Kennedy)"
    assert tm.option_label("SC East") == "SC East (Sioux City East)"
    assert tm.option_label("Cedar Falls") == "Cedar Falls"
