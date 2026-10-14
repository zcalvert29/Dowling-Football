"""Checks for the Add a Game page's file naming and games.csv update."""
import io
from datetime import date

import pandas as pd

import insights as ins


def test_reupload_keeps_existing_file_names():
    # Typed opponent differs from the file name's "crk", and in case.
    assert ins.new_game_files("Dowling Catholic", "cr kennedy", date(2026, 10, 2)) == \
        ("Dowling/dowling-vs-crk-100226.xlsx", "Dowling/dowling-vs-crk-offense.xlsx")
    assert ins.new_game_files("Waukee", "Ankeny", date(2026, 9, 4)) == ("Scout/waukee-vs-ankeny-090426.xlsx", None)


def test_new_game_names():
    assert ins.new_game_files("Dowling Catholic", "Ankeny", date(2026, 10, 16)) == \
        ("Dowling/dowling-vs-ankeny-101626.xlsx", "Dowling/dowling-vs-ankeny-offense.xlsx")
    assert ins.new_game_files("Ankeny Centennial", "Ankeny", date(2026, 10, 16)) == \
        ("Scout/ankeny-centennial-vs-ankeny-101626.xlsx", None)


def _manifest(*args) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(ins.updated_manifest(*args)), dtype=str).fillna("")


def test_manifest_update_replaces_in_place():
    before = pd.read_csv("games.csv", dtype=str)
    m = _manifest("Dowling/dowling-vs-valley-082826.xlsx", "Dowling Catholic", "Valley", 1)
    assert len(m) == len(before)
    assert m.iloc[0].tolist() == ["Dowling/dowling-vs-valley-082826.xlsx", "Dowling Catholic", "Valley", "1", "82", "57"]


def test_manifest_adds_new_game():
    before = pd.read_csv("games.csv", dtype=str)
    m = _manifest("Dowling/dowling-vs-ankeny-101626.xlsx", "Dowling Catholic", "Ankeny", 8)
    assert len(m) == len(before) + 1
    assert m.iloc[-1].tolist() == ["Dowling/dowling-vs-ankeny-101626.xlsx", "Dowling Catholic", "Ankeny", "8", "", ""]
