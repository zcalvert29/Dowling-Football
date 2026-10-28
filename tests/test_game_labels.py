"""Game Review labels: season and week together ('2026W1'), so games from different seasons can't be confused."""

import pytest

import insights as ins
import visuals as v


@pytest.fixture(scope="module")
def df():
    return v.load_data()


def test_dowling_labels_show_the_season(df):
    assert ins.game_label(df, "Dowling Catholic_Valley_2026_08_28", with_season=True) == "2026W1 · Valley"
    assert ins.game_label(df, "Dowling Catholic_Valley_2025_08_29", with_season=True) == "2025W1 · Valley"
    assert ins.game_label(df, "Dowling Catholic_Valley_2026_08_28") == "W1 · Valley"     # other pages unchanged


def test_scout_labels_show_the_season(df):
    gid = next(g for g in df["game_id"].unique() if g.startswith("Johnston_Waukee_2026"))
    assert ins.scout_game_label(df, gid, with_season=True) == "2026W1 Johnston vs Waukee"
    assert ins.scout_game_label(df, gid) == "W1 Johnston vs Waukee"


def test_fourth_down_review_labels_show_the_season(df):
    t = ins.fourth_down_review(ins.add_game_state(df[df["game_id"].str.startswith("Dowling Catholic_SEP_")]))
    assert set(t["Game"]) == {"2025W4 · SEP", "2026W4 · SEP"}
