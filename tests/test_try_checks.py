"""The PAT / 2-point tagging check reads tries from the filming team's side, in scout film as well as Dowling film."""

import pytest

import qol
import visuals as v


@pytest.fixture(scope="module")
def df():
    return v.load_data()


def _try_issues(g):
    i = qol.validate_plays(g)
    return i[i["Issue"].str.contains(" try")]


def test_scout_game_tries_are_not_flagged(df):
    g = df[df["game_id"].str.startswith("CR Kennedy_Linn-Mar_2026")]
    assert (g["PLAY TYPE"] == "Extra Pt.").sum() >= 3                    # CR Kennedy's own PATs
    assert _try_issues(g).empty


def test_scout_game_mistagged_try_is_flagged(df):
    g = df[df["game_id"].str.startswith("CR Kennedy_Linn-Mar_2026")].copy()
    n = g.loc[g["PLAY TYPE"] == "Extra Pt.", "PLAY #"].iat[0]
    g.loc[g["PLAY #"] == n, "PLAY TYPE"] = "Extra Pt. Block"            # CR Kennedy's PAT tagged as Linn-Mar's
    issues = _try_issues(g)
    assert list(issues["PLAY #"]) == [n]
    assert issues["Issue"].iat[0] == ("Extra Pt. Block after a CR Kennedy touchdown: "
                                      "the tag says it's the opponent's try")


def test_dowling_game_mistagged_try_is_still_flagged(df):
    g = df[df["game_id"] == "Dowling Catholic_Waterloo West_2026_09_25"].copy()
    assert _try_issues(g).empty
    n = g.loc[g["PLAY TYPE"] == "Extra Pt.", "PLAY #"].iat[0]
    g.loc[g["PLAY #"] == n, "PLAY TYPE"] = "Extra Pt. Block"
    assert "after a Dowling Catholic touchdown" in _try_issues(g)["Issue"].iat[0]
