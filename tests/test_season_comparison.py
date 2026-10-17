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
    one_season = [t for t in df["offense"].dropna().unique() if len(sc.team_seasons(df, t)) == 1]
    assert one_season                                                    # e.g. a scout team with only 2026 film
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


def test_run_comparison_matches_run_scheme_detail(df):
    dc = ins.dowling_only(df)
    ta, tb, schemes = sc.run_scheme_seasons(dc, 2025, 2026)
    assert ta.equals(v.run_scheme_table(dc[dc["SEASON"] == 2025]))         # same numbers as DCHS O Run Game
    assert set(schemes) == set(ta.index) | set(tb.index)
    out = sc.run_comparison_html(ta, tb, schemes, 2025, 2026)
    assert out.count("<tr") == len(schemes) + 2                             # two header rows + one per scheme
    only_b = [s for s in schemes if s not in ta.index]
    if only_b:
        assert f"<b>{only_b[0]}</b> <span class=\"tag\">new in 2026</span>" in out


def test_run_comparison_colors_only_real_samples():
    import pandas as pd
    cols = ["Avg Yards Gained", "Success Rate", "EPA per Play", "Explosive Rate", "Plays"]
    ta = pd.DataFrame([[4.0, 0.50, 0.00, 0.10, 40], [4.0, 0.50, 0.00, 0.10, 5]], index=["BIG", "TINY"], columns=cols)
    tb = pd.DataFrame([[6.0, 0.60, 0.30, 0.20, 40], [9.0, 0.90, 0.90, 0.50, 5]], index=["BIG", "TINY"], columns=cols)
    out = sc.run_comparison_html(ta, tb, ["BIG", "TINY"], 2025, 2026)
    big, tiny = out.split("<b>BIG</b>")[1].split("</tr>")[0], out.split("<b>TINY</b>")[1].split("</tr>")[0]
    assert big.count(sc.GOOD) == 4 and sc.GOOD not in tiny                  # small sample: shown, not colored
    assert '<tr class="small">' in out
