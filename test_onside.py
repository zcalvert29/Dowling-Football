"""Onside kicks: "Onside Kick" = we kick, "Onside Kick Rec" = we receive; RESULT "Recovered" = kicking team got it
back, "Return" = receiving team. Checked on the Johnston game with our kickoff at PLAY # 25 relabeled."""
import contextlib
import io

import numpy as np
import pandas as pd
import pytest

import curate_pbp as cp

RAW = "Dowling/dowling-vs-johnston-090426.xlsx"
US, THEM = "Dowling Catholic", "Johnston"


def curate(play_type, result, we_keep_ball=False):
    raw = pd.read_excel(RAW)
    i = raw.index[raw["PLAY #"] == 25][0]
    raw.loc[i, ["PLAY TYPE", "RESULT"]] = [play_type, result]
    if we_keep_ball:  # the next series is ours
        raw.loc[raw["PLAY #"].between(26, 28), "ODK"] = "O"
    with contextlib.redirect_stdout(io.StringIO()) as out:
        df = cp.curate_play_by_play_data(raw, US, THEM, "2026-09-04", 2)
    assert "unrecognized" not in out.getvalue()
    return df.set_index("PLAY #", drop=False)


def test_we_onside_and_recover():
    df = curate("Onside Kick", "Recovered", we_keep_ball=True)
    kick = df.loc[25]
    assert kick["offense"] == US and kick["defense"] == THEM
    assert kick["KICKOFF"] == 1 and kick["ONSIDE"] == 1 and kick["ONSIDE_RECOVERED"] == 1
    assert np.isnan(kick["SERIES"])
    assert kick["ep"] < 0 and kick["epa"] > 0               # getting the ball back is a big gain for the kicking team
    assert df.loc[24, "drive_result"] == "Touchdown"        # the scoring drive still ends in a TD
    assert df.loc[25, "drive"] == df.loc[26, "drive"] != df.loc[24, "drive"]  # new possession starts at the kick


def test_we_onside_they_return():
    df = curate("Onside Kick", "Return")
    kick = df.loc[25]
    assert kick["offense"] == THEM and kick["ONSIDE_RECOVERED"] == 0 and kick["KICKOFF"] == 1
    assert df.loc[24, "drive_result"] == "Touchdown"
    assert df.loc[25, "drive"] == df.loc[26, "drive"] != df.loc[24, "drive"]


def test_they_onside_and_recover():
    raw = pd.read_excel(RAW)
    # Their kickoff to us at PLAY # 1 (tagged "KO Rec"), relabeled as their recovered onside kick.
    i = raw.index[raw["PLAY #"] == 1][0]
    raw.loc[i, ["PLAY TYPE", "RESULT"]] = ["Onside Kick Rec", "Recovered"]
    with contextlib.redirect_stdout(io.StringIO()):
        df = cp.curate_play_by_play_data(raw, US, THEM, "2026-09-04", 2)
    assert df.iloc[0]["offense"] == THEM and df.iloc[0]["ONSIDE_RECOVERED"] == 1


def test_existing_kickoffs_unchanged():
    raw = pd.read_excel(RAW)
    with contextlib.redirect_stdout(io.StringIO()):
        df = cp.curate_play_by_play_data(raw, US, THEM, "2026-09-04", 2)
    assert df["ONSIDE"].sum() == 0
    ko = df[df["PLAY TYPE"] == "KO"]
    assert (ko["offense"] == THEM).all()


def test_special_teams_counts_onside_separately():
    import special_teams as stp
    df = curate("Onside Kick", "Recovered", we_keep_ball=True)
    df["game_id"], df["WEEK"] = "Dowling Catholic_Johnston_2026_09_04", 2
    ev = getattr(stp.kick_events, "__wrapped__", stp.kick_events)(df.reset_index(drop=True))
    on = ev[ev["kind"] == "Onside"]
    assert len(on) == 1 and on.iloc[0]["kicker"] == US and on.iloc[0]["result"] == "Recovered"
    assert 25 not in ev[ev["kind"] == "Kickoff"].index  # not mixed into kickoff touchback/start numbers
