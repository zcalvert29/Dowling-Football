"""Game Recap: scores on the drive chart, scores that aren't drives, the grouped boxes, and Head to head."""

import pytest

import insights as ins
import visuals as v

TEAM = "Dowling Catholic"
WATERLOO = "Dowling Catholic_Waterloo West_2026_09_25"   # 55-0: two pick-sixes and a fumble-return TD
VALLEY = "Dowling Catholic_Valley_2025_08_29"            # 19-20: a Dowling safety
SEP = "Dowling Catholic_SEP_2025_09_19"                  # 45-14: a Dowling punt return TD
CEDAR_FALLS = "Dowling Catholic_Cedar Falls_2026_11_06"  # 42-14: Cedar Falls' opening kickoff return TD


@pytest.fixture(scope="module")
def df():
    return ins.add_game_state(v.load_data())


def _game(df, gid):
    return df[df["game_id"] == gid]


def _boxes(g, team):
    drives, rows = ins.drive_rows(g, team)
    return {k: (val, note) for _, items in ins.drive_chart_stats(drives, rows) for k, val, note in items}


@pytest.mark.parametrize("gid", [WATERLOO, VALLEY, SEP, CEDAR_FALLS])
def test_points_add_up_to_the_final_score(df, gid):
    g = _game(df, gid)
    final = ins.score_for(g.sort_values("PLAY #").iloc[-1], TEAM)
    opp = ins.other_team(g, TEAM)
    assert (_boxes(g, TEAM)["Points"][0], _boxes(g, opp)["Points"][0]) == final
    _, rows = ins.drive_rows(g, TEAM)
    assert rows[-1]["score"] == final                                 # the last row shows the final


def test_scores_that_arent_drives(df):
    events = lambda gid, team: [(e["label"], e["unit"]) for e in ins.scoring_events(_game(df, gid), team)]
    assert events(WATERLOO, TEAM) == [("Pick-six", "defense")] * 2 + [("Fumble return TD", "defense")]
    assert events(VALLEY, TEAM) == [("Safety", "defense")]
    assert events(SEP, TEAM) == [("Punt return TD", "special teams")]
    assert events(CEDAR_FALLS, "Cedar Falls") == [("Kick return TD", "special teams")]
    assert events(CEDAR_FALLS, TEAM) == []


def test_rows_show_points_and_the_score(df):
    _, rows = ins.drive_rows(_game(df, SEP), TEAM)
    ret = next(r for r in rows if r["kind"] == "event")
    assert (ret["points"], ret["score"]) == (7, (21, 0))
    assert all(r["points"] >= 0 for r in rows)
    b = _boxes(_game(df, WATERLOO), TEAM)
    assert b["Points"] == (55, "34 off · 21 def") and b["Touchdowns"] == (8, "5 off · 3 def")
    assert _boxes(_game(df, WATERLOO), "Waterloo West")["Turnovers"] == (4, "3 returned for a TD")


def test_second_half_rows_start_from_the_halftime_score(df):
    g = _game(df, SEP)
    _, rows = ins.drive_rows(g[g["QTR"] >= 3], TEAM, full=g)
    assert sum(r["points"] for r in rows) == 45 - ins.score_for(g[g["QTR"] <= 2].sort_values("PLAY #").iloc[-1],
                                                                TEAM)[0]


def test_head_to_head_groups_and_edges(df):
    g = _game(df, VALLEY)
    groups = dict(ins.head_to_head(g, TEAM, "Valley"))
    assert list(groups) == ["Overall", "Running", "Passing", "Situational", "Turnovers & penalties"]
    rows = {k: (a, b, e) for lines in groups.values() for k, a, b, e in lines}
    assert {"Rush EPA / play", "Rush success", "Pass EPA / play", "Pass success"} <= set(rows)
    assert rows["Rushing"] == ("11 for 39", "44 for 169", "Valley")
    assert rows["Success"][2] == ""                                   # 40% vs 43%: too close to call
    assert rows["Turnover margin"] == ("-1", "+1", "Valley")
    assert rows["Plays"][2] == "" and rows["QB scrambles"][2] == ""   # no better direction


def test_points_off_turnovers(df):
    assert ins._points_off_turnovers(_game(df, WATERLOO), TEAM) == (28, 0)   # 3 defensive TDs + a TD drive
    assert ins._points_off_turnovers(_game(df, CEDAR_FALLS), TEAM) == (0, 0)  # two missed FGs and the end of the game
