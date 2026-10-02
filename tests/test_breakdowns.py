"""Checks for breakdowns.py: tendency tree, field/strength, sequencing, drives, aggressiveness."""
import numpy as np
import pandas as pd
import pytest

import breakdowns as bd
import insights as ins
import visuals as v


@pytest.fixture(scope="module")
def df():
    return v.load_data("curated-pbp.xlsx")


def test_tree_levels_add_up(df):
    d = bd._team_plays(df, v.TEAM)
    t = bd.tendency_tree_frame(d, depth=3)
    for lvl in range(3):
        assert t.loc[t["level"] == lvl, "n"].sum() == len(d)          # every play appears once per level
    kids = t[t["level"] == 1]
    for _, f in t[t["level"] == 0].iterrows():                          # children stack inside their parent
        c = kids[kids["path"].str.startswith(f["path"] + " › ")]
        assert c["n"].sum() == f["n"] and c["y0"].min() == f["y0"] and c["y1"].max() == f["y1"]


def test_small_formations_fold_into_other(df):
    d = bd._team_plays(df, v.TEAM)
    t = bd.tendency_tree_frame(d, depth=1)
    fold = max(4, int(np.ceil(0.03 * len(d))))
    named = t[~t["label"].isin(["Other", "Untagged"])]
    assert (named["n"] >= fold).all()


def test_field_side_logic():
    d = pd.DataFrame({"HASH": ["L", "L", "R", "M"], "PLAY DIR": ["R", "L", "L", "R"],
                      "OFF STR": ["R", "R", "BAL", "L"]})
    f = bd._field_strength_flags(d)
    # Left hash: field is to the right. Right hash: field is to the left. Middle: no field side.
    assert f["to_field"].tolist()[:3] == [1.0, 0.0, 1.0] and np.isnan(f["to_field"].iat[3])
    assert f["to_strength"].tolist()[:2] == [1.0, 0.0] and np.isnan(f["to_strength"].iat[2])


def test_sequence_previous_play_stays_in_drive(df):
    seq = bd.sequence_frame(df, v.TEAM)
    firsts = seq[seq["first_of_drive"]]
    assert firsts["prev_type"].isna().all()                            # nothing carries over between drives
    assert seq.loc[~seq["first_of_drive"], "prev_type"].isin(["Run", "Pass"]).all()


def test_long_touchdowns_are_not_red_zone_trips(df):
    d = bd.season_drives(df, v.TEAM, "offense")
    long_td = d[(d["result"] == "Touchdown") & (d["snap_deepest"] < 80)]
    assert (long_td["deepest"] == 100).all()                           # funnel: crossed every line
    trips = d[d["snap_deepest"] >= 80]
    assert not trips.index.isin(long_td.index).any()                   # red zone: not a trip


def test_drive_points(df):
    d = bd.season_drives(df, v.TEAM, "offense")
    assert set(d.loc[d["points"] > 0, "result"]) <= {"Touchdown", "FG Made"}
    assert (d.loc[d["result"] == "Touchdown", "points"] == 7).all()


def test_aggressiveness_matches_ledger(df):
    t = ins.fourth_down_review(df)
    a = bd.aggressiveness_frame(t, ins.games(df))
    ledger = t.loc[t["category"] == "Costly", "WP left (pts)"].sum()
    assert a["wp_left"].sum() == pytest.approx(ledger)
    assert (a["went_on_go"] <= a["model_go"]).all()
