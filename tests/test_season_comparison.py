"""Checks for season_comparison.py: the summary tables and their colors."""
import pytest

import insights as ins
import season_comparison as sc
import visuals as v

TEAM = "Dowling Catholic"


@pytest.fixture(scope="module")
def df():
    return ins.add_game_state(v.load_data())


def test_seasons_and_colors(df):
    assert sc.team_seasons(df, TEAM)[-2:] == [2025, 2026]
    assert sc.team_seasons(df, "Waukee") == [2026]                       # nothing to compare yet
    assert sc.team_color(TEAM) == "#8C1D2C" and sc.team_color("Waukee") == "#185FA5"


def test_down_conversions_match_the_plays(df):
    g = sc.season_games(df, TEAM, 2026)
    s = sc.unit_stats(g, TEAM, "offense")
    rate, made, att = s["3rd Down Conv. Rate"]
    snaps = v.run_pass(g[g["offense"] == TEAM])
    third = snaps[(snaps["DN"] == 3) & (snaps["RESULT"] != "Penalty")]
    assert att == len(third) and made == int(third["THIRD_DOWN_CONVERTED"].sum()) and rate == pytest.approx(made / att)
    assert s["Avg 3rd Down Distance"] == pytest.approx(third["DIST"].mean())
    assert s["4th Down Conv. Rate"][2] > 0


def test_defense_rows_count_lower_as_better(df):
    a = sc.unit_stats(sc.season_games(df, TEAM, 2025), TEAM, "defense")
    b = dict(a, **{"EPA/Play": a["EPA/Play"] - 0.2, "Avg 3rd Down Distance": a["Avg 3rd Down Distance"] + 3})
    t, tones = sc.unit_table(a, b, 2025, 2026, "defense")
    tone = dict(zip(t["Stat"], tones))
    assert tone["EPA/Play"] == 1                                         # opponents' EPA went down: good
    assert tone["Avg 3rd Down Distance"] == 1                            # opponents' 3rd downs got longer: good
    assert "Opp. Starting Field Pos." in tone
    t, tones = sc.unit_table(a, b, 2025, 2026, "offense")
    tone = dict(zip(t["Stat"], tones))
    assert tone["EPA/Play"] == -1 and tone["Avg 3rd Down Distance"] == -1


def test_special_teams_has_epa_for_each_phase(df):
    s = sc.st_stats(sc.season_games(df, TEAM, 2026), TEAM)
    for key in [("Field goals", "EPA per kick"), ("Kickoffs", "EPA per kick"), ("Kick returns", "EPA per return"),
                ("Punting", "EPA per punt"), ("Punt returns", "EPA per return")]:
        assert key in s and s[key][0] != "–"
    t, tones = sc.st_table(s, s, 2026, 2026)
    assert set(tones) == {0}                                             # same season both sides: nothing colored


def test_grouped_table_shows_each_group_once(df):
    a = sc.unit_stats(sc.season_games(df, TEAM, 2025), TEAM, "offense")
    t, tones = sc.unit_table(a, a, 2025, 2026, "offense")
    out = sc.grouped_table_html(t, tones, "Section")
    for group in t["Section"].unique():
        assert out.count(f'<tr class="sc-band"><td colspan="4">{group}</td></tr>') == 1
    assert "<th>Section</th>" not in out and out.count("<tr>") == len(t) + 1   # header row + one row per stat
