"""
special_projects.py — one-off studies that don't belong on a regular page.

First study: does the first play of a drive predict how many points the drive scores? Dowling Catholic offensive
drives are split by the yards gained on their first snap (4+ yards vs. under 4 yards). Four yards is the success
line on 1st & 10 (40% of the distance), so the buckets are "on schedule" and "behind schedule".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import breakdowns as bd
import insights as ins
import visuals as v

FIRST_PLAY_YARDS = 4  # yards on the drive's first snap that puts a drive "on schedule"
ON_SCHEDULE = f"On schedule ({FIRST_PLAY_YARDS}+ yds on 1st play)"
BEHIND = f"Behind schedule (under {FIRST_PLAY_YARDS} yds on 1st play)"


def _drive_points(g: pd.DataFrame, team: str) -> pd.Series:
    """
    Points `team` actually scored on each of its drives in one game (touchdown + the try's real result, or a
    field goal), indexed by drive number. A drive's rows include its try, and team_score/opponent_score are the
    score after each play, so a drive's points = the team's score after its last row minus the score after the
    row before the drive started.
    """
    g = g.sort_values("PLAY #", kind="stable")
    col = "team_score" if str(g["game_id"].iat[0]).startswith(f"{team}_") else "opponent_score"
    end = g.groupby("drive", sort=True)[col].last()
    start = end.shift(1, fill_value=0)
    pts = (end - start).clip(lower=0)  # a defensive score by Dowling happens on the other team's drive
    offense = g.groupby("drive", sort=True)["offense"].agg(lambda s: s.dropna().iat[0] if s.notna().any() else None)
    return pts[offense == team]


def first_play_drives(df_games: pd.DataFrame, team: str = v.TEAM) -> pd.DataFrame:
    """
    One row per `team` offensive drive (same drives as the Season Drives page) with first_play_yards (yards on
    the drive's first run or pass; penalties before the first snap are skipped), bucket, and points (actual).
    Drives with no run or pass snap get bucket = NaN.
    """
    drives = bd.season_drives(df_games, team)
    if drives.empty:
        return drives
    rows = []
    for gid, d in drives.groupby("game_id", sort=False):
        g = df_games[df_games["game_id"] == gid]
        snaps = v.run_pass(g[g["offense"] == team]).sort_values("PLAY #", kind="stable")
        first = snaps.groupby("drive")["GN/LS"].first()
        pts = _drive_points(g, team)
        rows.append(d.assign(first_play_yards=d["drive"].map(first), points=d["drive"].map(pts).fillna(0)))
    out = pd.concat(rows, ignore_index=True)
    has_snap = out["first_play_yards"].notna()
    out["bucket"] = np.where(~has_snap, None,
                             np.where(out["first_play_yards"] >= FIRST_PLAY_YARDS, ON_SCHEDULE, BEHIND))
    return out


def first_play_table(drives: pd.DataFrame) -> pd.DataFrame:
    d = drives[drives["bucket"].notna()]
    rows = [(name, d[d["bucket"] == name]) for name in (ON_SCHEDULE, BEHIND)] + [("All drives", d)]
    t = pd.DataFrame([{"First play of the drive": name, "Avg Points Scored per Drive": x["points"].mean()
                       if len(x) else np.nan, "Drives": len(x)} for name, x in rows])
    return t.set_index("First play of the drive")


def render_first_play_study(df_all: pd.DataFrame) -> None:
    st.subheader("Does the first play set the tone?")
    st.caption(f"Dowling Catholic offensive drives, split by the yards gained on the drive's first play. "
               f"{FIRST_PLAY_YARDS} yards is the success line on 1st & 10, so a drive that gets it is on schedule.")
    weeks = sorted(df_all["WEEK"].dropna().unique())
    sel = st.multiselect("Weeks", weeks, default=weeks, key="sp_weeks", format_func=lambda w: f"Week {int(w)}")
    games = df_all[df_all["WEEK"].isin(sel)]
    drives = first_play_drives(ins.dowling_only(games))
    if drives.empty:
        st.info("No Dowling Catholic drives in the selected weeks.")
        return
    t = first_play_table(drives)
    st.dataframe(t.style.format({"Avg Points Scored per Drive": "{:.2f}", "Drives": "{:,.0f}"}, na_rep="–"),
                 width="stretch")
    skipped = int(drives["bucket"].isna().sum())
    note = ("Points are what the drive actually scored: a touchdown counts 6 plus the try's real result (so a "
            "missed PAT is 6, a 2-pointer is 8), a made field goal 3. Every possession counts, including ones that "
            "ran out the clock at the end of a half or game.")
    if skipped:
        note += f" {skipped} drive{'s' if skipped != 1 else ''} with no run or pass snap {'are' if skipped != 1 else 'is'} left out."
    st.caption(note)
    with st.expander("Every drive"):
        shown = drives[drives["bucket"].notna()][["Game", "QTR", "start", "first_play_yards", "bucket", "plays",
                                                   "result", "points"]].copy()
        shown["start"] = shown["start"].map(ins._yard_label)
        st.dataframe(shown.rename(columns={"start": "Start", "first_play_yards": "1st play yds", "bucket": "Bucket",
                                           "plays": "Plays", "result": "Result", "points": "Points"}),
                     hide_index=True, width="stretch")
