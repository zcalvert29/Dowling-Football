"""Checks for build_data.py: the plain-English notes about drives that come out as 'Error'."""
import pandas as pd

import build_data as b

LINN_MAR = b.Game(file="Scout/crk-vs-linn-mar-091826.xlsx", team="CR Kennedy", opponent="Linn-Mar", week=4)


def test_missing_punt_names_the_plays():
    _, notes = b.curate_game(LINN_MAR)
    assert len(notes) == 1
    assert "from CR Kennedy to Linn-Mar after #48 (Q2, 3rd & 13, Pass, Incomplete)" in notes[0]
    assert "Next play: #49" in notes[0]


def test_dropped_s_row_is_pointed_out():
    _, notes = b.curate_game(b.Game(file="Scout/waukee-vs-valley-091826.xlsx", team="Waukee", opponent="Valley",
                                    week=4))
    assert any("#97 is a blank S row" in n for n in notes)


def test_film_that_stops_early(tmp_path):
    raw = pd.read_excel(b._path(LINN_MAR.file))
    # Cut the film off in the middle of a 3rd-quarter drive.
    q3 = raw[(raw["QTR"] == 3) & (raw["ODK"].isin(["O", "D"])) & (raw["DN"] == 2)]
    cut = raw[raw["PLAY #"] <= q3["PLAY #"].iat[0]]
    f = tmp_path / "crk-vs-linn-mar-091826.xlsx"
    cut.to_excel(f, index=False)
    _, notes = b.curate_game(b.Game(file=str(f), team="CR Kennedy", opponent="Linn-Mar", week=4))
    assert any("the film ends after" in n and "in Q3" in n for n in notes)


def test_clean_game_has_no_notes():
    _, notes = b.curate_game(b.Game(file="Scout/waukee-vs-dcg-092526.xlsx", team="Waukee", opponent="DCG", week=5))
    assert notes == []
