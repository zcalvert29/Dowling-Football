"""Checks for build_data.py: the plain-English notes about drives that come out as 'Error'."""
import pandas as pd
import pytest

import build_data as b

LINN_MAR = b.Game(file="Scout/crk-vs-linn-mar-091826.xlsx", team="CR Kennedy", opponent="Linn-Mar", season=2026)


def test_missing_punt_names_the_plays():
    _, notes = b.curate_game(LINN_MAR)
    assert len(notes) == 1
    assert "from CR Kennedy to Linn-Mar after #48 (Q2, 3rd & 13, Pass, Incomplete)" in notes[0]
    assert "Next play: #49" in notes[0]


def test_dropped_s_row_is_pointed_out():
    _, notes = b.curate_game(b.Game(file="Scout/waukee-vs-valley-091826.xlsx", team="Waukee", opponent="Valley",
                                    season=2026))
    assert any("#97 is a blank S row" in n for n in notes)


def test_film_that_stops_early(tmp_path):
    raw = pd.read_excel(b._path(LINN_MAR.file))
    # Cut the film off in the middle of a 3rd-quarter drive.
    q3 = raw[(raw["QTR"] == 3) & (raw["ODK"].isin(["O", "D"])) & (raw["DN"] == 2)]
    cut = raw[raw["PLAY #"] <= q3["PLAY #"].iat[0]]
    f = tmp_path / "crk-vs-linn-mar-091826.xlsx"
    cut.to_excel(f, index=False)
    _, notes = b.curate_game(b.Game(file=str(f), team="CR Kennedy", opponent="Linn-Mar", season=2026))
    assert any("the film ends after" in n and "in Q3" in n for n in notes)


def test_clean_game_has_no_notes():
    _, notes = b.curate_game(b.Game(file="Scout/waukee-vs-dcg-092526.xlsx", team="Waukee", opponent="DCG", season=2026))
    assert notes == []


def test_week_from_date():
    from datetime import date
    assert b.week_of(date(2026, 8, 28), 2026) == 1
    assert b.week_of(date(2026, 9, 4), 2026) == 2
    assert b.week_of(date(2026, 9, 11), 2026) == 3
    assert b.week_of(date(2026, 11, 20), 2026) == 13
    assert b.week_of(date(2026, 9, 3), 2026) == 2          # Thursday game: that week's Friday
    assert b.week_of(date(2026, 8, 29), 2026) == 1          # Saturday game
    for outside in (date(2026, 8, 21), date(2026, 11, 27)):
        with pytest.raises(ValueError, match="isn't in the 2026 season"):
            b.week_of(outside, 2026)
    assert b.week1(2027) == date(2027, 8, 27)               # unlisted season: last Friday of August


def test_past_seasons():
    from datetime import date
    for season, first in {2025: date(2025, 8, 29), 2024: date(2024, 8, 30), 2023: date(2023, 8, 26),
                          2022: date(2022, 8, 26), 2021: date(2021, 8, 27)}.items():
        assert b.week_of(first, season) == 1
        assert b.week_of(date.fromordinal(first.toordinal() + 7), season) == 2
        assert b.week_of(date.fromordinal(first.toordinal() + 84), season) == 13
    assert b.week_of(date(2023, 8, 25), 2023) == 1          # Friday before 2023's Saturday Week 1 date
    assert b.week_of(date(2023, 9, 1), 2023) == 2
    assert b.week_of(date(2025, 11, 21), 2025) == 13
    with pytest.raises(ValueError, match="isn't in the 2024 season"):
        b.week_of(date(2024, 8, 23), 2024)


def test_built_games_have_season_and_week():
    df, _ = b.build()
    per_game = df.groupby("game_id")[["SEASON", "WEEK"]].nunique()
    assert (per_game == 1).all().all()
    seasons = df.groupby("game_id")["SEASON"].first()
    assert all(int(gid.split("_")[2]) == season for gid, season in seasons.items())   # season = the date's year
    assert df.loc[df["game_id"] == "Dowling Catholic_Waukee_2026_10_09", "WEEK"].iat[0] == 7
