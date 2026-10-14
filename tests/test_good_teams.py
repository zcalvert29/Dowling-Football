"""Checks for good_teams.py: which games the Good Teams filter keeps."""
import pytest

import good_teams as gt
import visuals as v

DOWLING = "Dowling Catholic"


@pytest.fixture(scope="module")
def df():
    return v.load_data()


def test_default_list_is_saved():
    assert set(gt.load()) == set(gt.DEFAULT) and len(gt.DEFAULT) == 16


def test_dowling_games_split_by_opponent(df):
    good = set(gt.keep_games(df, gt.DEFAULT, gt.GOOD, [DOWLING]))
    other = set(gt.keep_games(df, gt.DEFAULT, gt.OTHER, [DOWLING]))
    assert "Dowling Catholic_Waukee_2026_10_09" in good and "Dowling Catholic_Waukee_2026_10_09" not in other
    assert "Dowling Catholic_Waterloo West_2026_09_25" in other and "Dowling Catholic_Waterloo West_2026_09_25" not in good
    assert good | other == set(df["game_id"]) and not good & other     # every game in exactly one bucket


def test_scouted_team_judged_by_its_opponent(df):
    # CR Kennedy isn't a good team, but its game against Waukee is a game against a good team.
    good = gt.keep_games(df, gt.DEFAULT, gt.GOOD, [DOWLING, "CR Kennedy"])
    assert "CR Kennedy_Waukee_2026_09_11" in good and "CR Kennedy_Marshalltown_2026_09_25" not in good


def test_all_opponents_and_names_ignore_case(df):
    assert len(gt.apply(df, gt.ALL, [DOWLING], good=[])) == len(df)
    lower = gt.keep_games(df, [t.lower() for t in gt.DEFAULT], gt.GOOD, [DOWLING])
    assert lower == gt.keep_games(df, gt.DEFAULT, gt.GOOD, [DOWLING])
