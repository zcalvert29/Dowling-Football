"""Checks for teams.py: GoBound names -> app names, and reading a schedule / standings page.

The HTML here is written for these tests (gobound.com couldn't be reached when they were written), so they check
the parsing rules, not GoBound's exact markup. Run `python teams.py` to see what it reads from the live pages."""
from datetime import date

import teams as tm


def test_app_names():
    assert tm.app_name("Cedar Rapids Kennedy") == "CR Kennedy"
    assert tm.app_name("Cedar Rapids Prairie") == "CR Prairie"
    assert tm.app_name("Dallas Center-Grimes") == "DCG"
    assert tm.app_name("Dallas Center Grimes") == "DCG"
    assert tm.app_name("Waukee Northwest") == "Waukee NW"
    assert tm.app_name("Iowa City High") == "IC High"
    assert tm.app_name("Iowa City Liberty") == "IC Liberty"
    assert tm.app_name("Iowa City West") == "IC West"
    assert tm.app_name("Southeast Polk") == "SEP"
    assert tm.app_name("West Des Moines Valley") == "Valley"
    assert tm.app_name("  Cedar   Falls ") == "Cedar Falls"
    # App names come back unchanged
    for n in ("CR Kennedy", "DCG", "Waukee NW", "IC High", "SEP", "Valley", "Dowling Catholic", "Linn-Mar"):
        assert tm.app_name(n) == n


def test_season_slug_and_urls():
    assert tm.season_slug(date(2026, 11, 3)) == "2026-27"
    assert tm.season_slug(date(2027, 2, 1)) == "2026-27"
    assert tm.schedule_url(date(2026, 9, 1)).endswith("/ia/ihsaa/football/2026-27/dowling/v/schedule")
    assert tm.standings_url(date(2026, 9, 1)).endswith("/ia/ihsaa/football/2026-27/standings")


SCHEDULE_TABLE = """
<html><body><script>var x = "<a href='/ia/ihsaa/football/2026-27/fake/v'>Fake</a>";</script>
<a href="/ia/ihsaa/football/2026-27/dowling/v">Dowling Catholic</a>
<table>
<tr><td>Fri 10/23</td><td>vs <a href="/ia/ihsaa/football/2026-27/marshalltown/v">Marshalltown</a></td><td>W 49-0</td></tr>
<tr><td>Fri 10/30</td><td>@ <a href="/ia/ihsaa/football/2026-27/cr-kennedy/v">Cedar Rapids Kennedy</a></td><td>L 14-17</td></tr>
<tr><td>Fri 11/6</td><td>vs <a href="/ia/ihsaa/football/2026-27/cedar-falls/v">Cedar Falls</a></td><td>7:00 PM</td></tr>
<tr><td>Fri 11/13</td><td>vs <a href="/ia/ihsaa/football/2026-27/dcg/v">Dallas Center-Grimes</a></td><td>TBD</td></tr>
</table></body></html>
"""

SCHEDULE_CARDS = """
<div class="game"><a href="/ia/ihsaa/football/2026-27/sep/v/schedule"><img src="logo.png"></a>
  <a href="/ia/ihsaa/football/2026-27/sep/v/schedule">Southeast Polk</a><span>Sep 18</span><span>Final</span>
  <span>Dowling 21</span><span>Southeast Polk 28</span></div>
<div class="game"><a href="/ia/ihsaa/football/2026-27/waukee-northwest/v">Waukee Northwest</a>
  <span>Oct 2, 2026</span><span>7:00 PM</span></div>
"""


def test_schedule_from_a_table():
    games = tm.parse_schedule(SCHEDULE_TABLE, date(2026, 11, 3))
    assert [(g.day, g.opponent, g.played) for g in games] == [
        (date(2026, 10, 23), "Marshalltown", True),
        (date(2026, 10, 30), "CR Kennedy", True),
        (date(2026, 11, 6), "Cedar Falls", False),
        (date(2026, 11, 13), "DCG", False),
    ]
    assert tm.next_game(games, date(2026, 11, 3)).opponent == "Cedar Falls"
    # After Cedar Falls is played (scored), the next game moves on
    played = SCHEDULE_TABLE.replace("<td>7:00 PM</td>", "<td>W 28-21</td>")
    assert tm.next_game(tm.parse_schedule(played, date(2026, 11, 7)), date(2026, 11, 7)).opponent == "DCG"


def test_schedule_from_cards_with_date_after_name():
    games = tm.parse_schedule(SCHEDULE_CARDS, date(2026, 9, 1))
    assert [(g.day, g.opponent, g.played) for g in games] == [
        (date(2026, 9, 18), "SEP", True),        # logo link + name link count once
        (date(2026, 10, 2), "Waukee NW", False),
    ]


def test_a_past_unscored_game_is_skipped():
    games = tm.parse_schedule(SCHEDULE_TABLE.replace("<td>W 49-0</td>", "<td></td>"), date(2026, 11, 3))
    assert tm.next_game(games, date(2026, 11, 3)).opponent == "Cedar Falls"


STANDINGS = """
<h2>Class 5A - District 2</h2>
<table><tr><td><a href="/ia/ihsaa/football/2026-27/dowling/v">West Des Moines Dowling Catholic</a></td><td>8-1</td></tr>
<tr><td><a href="/ia/ihsaa/football/2026-27/cedar-falls/v">Cedar Falls (7-2)</a></td></tr>
<tr><td><a href="/ia/ihsaa/football/2026-27/iowa-city-high/v">Iowa City High</a></td></tr>
<tr><td><a href="/ia/ihsaa/football/2026-27/cedar-falls/v">Cedar Falls</a></td></tr>
<tr><td><a href="/ia/ihsaa/football/2026-27/standings?class=4A">4A</a></td></tr>
<tr><td><a href="/news/123">Week 10 preview</a></td></tr></table>
"""


def test_standings_teams():
    names = tm.parse_standings(STANDINGS)
    assert names == ["West Des Moines Dowling Catholic", "Cedar Falls", "Iowa City High"]
    assert [tm.app_name(n) for n in names] == ["Dowling Catholic", "Cedar Falls", "IC High"]


def test_team_options_and_merge():
    opts = tm.team_options(["Cedar Falls", "Zz New Team"], gobound_names=["Cedar Rapids Washington", "Iowa City High"])
    assert opts[0] == "Dowling Catholic"
    assert {"CR Washington", "IC High", "Zz New Team", "Waukee NW"} <= set(opts)
    assert len(opts) == len(set(opts))
    t = tm.merge(tm.load(), ["Cedar Rapids Washington", "Dallas Center-Grimes"])
    assert t.loc[t["team"] == "CR Washington", "gobound"].item() == "Cedar Rapids Washington"
    assert (t["team"] == "DCG").sum() == 1
    assert tm.colors()["Dowling Catholic"] == "#8C1D2C"


def test_teams_csv_uses_app_names():
    for name in tm.load()["team"]:
        assert tm.app_name(name) == name
