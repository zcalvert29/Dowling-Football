"""Offense films: named with the game's date, matched to the game play by play, and ignored if they don't match."""
import shutil

import pandas as pd

import build_data as b
import curate_pbp as cp


def _game(f, opp, season):
    return b.Game(file=f, team="Dowling Catholic", opponent=opp, season=season)


def test_each_season_uses_its_own_offense_film():
    assert b.offense_name("Dowling/dowling-vs-valley-082826.xlsx") == "Dowling/dowling-vs-valley-082826-offense.xlsx"
    assert _game("Dowling/dowling-vs-valley-082826.xlsx", "Valley", 2026).offense_file == \
        "Dowling/dowling-vs-valley-082826-offense.xlsx"
    # No 2025 offense film: the 2025 game uses its own tags, never the 2026 offense film.
    g25 = _game("Dowling/dowling-vs-valley-082925.xlsx", "Valley", 2025)
    assert g25.offense_file == g25.file
    df, _ = b.curate_game(g25)
    raw = pd.read_excel(b._path(g25.file)).set_index("PLAY #")
    got = df.set_index("PLAY #")["OFF PLAY"]
    assert ((got == raw.loc[got.index, "OFF PLAY"]) | got.isna() & raw.loc[got.index, "OFF PLAY"].isna()).all()
    assert df.loc[df["PLAY #"] == 2, "OFF PLAY"].iat[0] == "CHARGE ODD EGYPT"


def test_offense_film_matched_by_content_not_play_number():
    # Waukee 2026: the offense film is one play behind from #24 on. Game #28 (Run, 1st & 10) is offense film #27.
    g = _game("Dowling/dowling-vs-waukee-100926.xlsx", "Waukee", 2026)
    off = pd.read_excel(b._path(g.offense_file)).set_index("PLAY #")
    df, _ = b.curate_game(g)
    assert df.loc[df["PLAY #"] == 28, "OFF PLAY"].iat[0] == off.loc[27, "OFF PLAY"]


def test_offense_film_from_another_game_is_ignored(tmp_path):
    # The 2026 Valley offense film saved as if it were the 2025 game's: it doesn't match, so it's left out.
    shutil.copy(b._path("Dowling/dowling-vs-valley-082925.xlsx"), tmp_path / "dowling-vs-valley-082925.xlsx")
    shutil.copy(b._path("Dowling/dowling-vs-valley-082826-offense.xlsx"),
                tmp_path / "dowling-vs-valley-082925-offense.xlsx")
    g = _game(str(tmp_path / "dowling-vs-valley-082925.xlsx"), "Valley", 2025)
    df, notes = b.curate_game(g)
    assert any("doesn't look like the same game" in n for n in notes)
    assert df.loc[df["PLAY #"] == 2, "OFF PLAY"].iat[0] == "CHARGE ODD EGYPT"     # the game file's own tag


def test_well_matched_offense_film_matches_every_snap():
    for f, opp in [("Dowling/dowling-vs-johnston-090426.xlsx", "Johnston"), ("Dowling/dowling-vs-sep-091826.xlsx", "SEP")]:
        g = _game(f, opp, 2026)
        P = cp.curate_play_by_play_data(b._read_film(g.file, ()), g.team, g.opponent, g.date, g.week)
        O = cp.curate_play_by_play_data(b._read_film(g.offense_file, ()), g.team, g.opponent, g.date, g.week)
        assert cp.offense_film_match(P, O) == 1.0
