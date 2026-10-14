"""
good_teams.py — which teams count as "good" for the sidebar's Good Teams filter, so coaches can compare film
against good opponents only (or only the rest).

The list lives in good_teams.csv (one team per row) and is edited on Data > Good Teams Filter. Edits apply to the
coach's own session right away; Save commits the file to GitHub (github_sync.py) so it's the list for everyone.

The filter looks at the opponent of the team being studied: Dowling on the DCHS pages, plus the scouted team on
the Scouting pages. So scouting CR Kennedy with "Good teams only" keeps CR Kennedy vs Waukee and drops CR Kennedy
vs Marshalltown, whether or not CR Kennedy itself is on the list.
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

import insights as ins

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HERE, "good_teams.csv")
DEFAULT = ["SEP", "Dowling Catholic", "Johnston", "Ankeny", "Waukee", "Waukee Northwest", "Sioux City East",
           "Indianola", "Valley", "DCG", "Iowa City Liberty", "Iowa City High", "Cedar Falls", "Bettendorf",
           "Pleasant Valley", "Urbandale"]
ALL, GOOD, OTHER = "All opponents", "Good teams only", "Other teams only"
MODES = [ALL, GOOD, OTHER]
SESSION_KEY = "good_teams"  # this session's edited list, until it's saved
SAVED_KEY = "good_teams_saved"  # what this session saved to GitHub (the file catches up when the app redeploys)


def load(path: str = FILE) -> list[str]:
    """The saved list (DEFAULT if good_teams.csv is missing)."""
    if not os.path.exists(path):
        return list(DEFAULT)
    t = pd.read_csv(path, dtype=str)["team"].dropna().str.strip()
    return [x for x in t if x]


def to_csv(teams) -> bytes:
    return pd.DataFrame({"team": sorted(set(teams), key=str.casefold)}).to_csv(index=False).encode()


def current() -> list[str]:
    """This session's list: the coach's unsaved edits if there are any, else the saved list."""
    return st.session_state.get(SESSION_KEY) or load()


def opponents_by_game(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """{game_id: {team: the team it played}} from who had the ball in each game."""
    out = {}
    for gid, g in df.groupby("game_id", sort=False):
        teams = ins.teams_in_game(g)
        if len(teams) == 2:
            out[gid] = {teams[0]: teams[1], teams[1]: teams[0]}
    return out


def keep_games(df: pd.DataFrame, good, mode: str, focus) -> list[str]:
    """game_ids to keep. For each game, look at the opponents of the focus teams playing in it (or, if no focus
    team played, of the filming team, the first name in the game_id): "Good teams only" keeps the game if one of
    those opponents is on the list, "Other teams only" if one isn't. When Dowling plays the scouted team, the game
    can count both ways (Dowling's opponent and the scouted team's opponent are different teams)."""
    good = {t.casefold() for t in good}
    focus = set(focus)
    keep = []
    for gid, opp in opponents_by_game(df).items():
        filming = str(gid).split("_")[0]
        studied = [t for t in opp if t in focus] or [t for t in opp if t == filming] or list(opp)
        vs_good = [opp[t].casefold() in good for t in studied]
        if (mode == GOOD and any(vs_good)) or (mode == OTHER and not all(vs_good)):
            keep.append(gid)
    return keep


def apply(df: pd.DataFrame, mode: str, focus, good=None) -> pd.DataFrame:
    if mode not in (GOOD, OTHER):
        return df
    return df[df["game_id"].isin(keep_games(df, current() if good is None else good, mode, focus))]


def teams_with_data(df: pd.DataFrame) -> pd.Series:
    """Every team in the data, with how many games it played."""
    games = [t for _, g in df.groupby("game_id", sort=False) for t in ins.teams_in_game(g)]
    return pd.Series(games, dtype=str).value_counts()


def render_page(df_all: pd.DataFrame) -> None:
    import github_sync

    st.markdown("Pick the teams that count as **good**. The **Good Teams** filter in the sidebar (on the DCHS, "
                "Special Teams, and Scouting pages) then shows only games against good teams, or only the rest. "
                "Teams without film yet can stay on the list; they'll count as soon as their games are added.")
    saved = st.session_state.get(SAVED_KEY) or load()
    chosen = current()
    counts = teams_with_data(df_all)
    teams = sorted(set(counts.index) | set(chosen) | set(saved), key=str.casefold)
    table = pd.DataFrame({"Good team": [t in chosen for t in teams], "Team": teams,
                          "Games in app": [int(counts.get(t, 0)) for t in teams]})
    edited = st.data_editor(
        table, hide_index=True, num_rows="dynamic", width="stretch", key="good_teams_editor",
        height=35 * (len(table) + 2) + 3,  # every team visible, plus a row for adding one
        column_config={
            "Good team": st.column_config.CheckboxColumn(default=True, width="small"),
            "Team": st.column_config.TextColumn(help="Spelled the way it appears in the app. Add a row for a team "
                                                     "that has no film yet."),
            "Games in app": st.column_config.NumberColumn(disabled=True, width="small"),
        })
    picked = sorted({str(t).strip() for t, on in zip(edited["Team"], edited["Good team"])
                     if pd.notna(t) and str(t).strip() and on is True}, key=str.casefold)
    if set(picked) != set(chosen):
        st.session_state[SESSION_KEY] = picked
        st.rerun()

    st.caption(f"{len(picked)} good teams, {sum(1 for t in picked if counts.get(t, 0))} with games in the app.")
    if set(picked) == set(saved):
        st.success("This is the saved list everyone sees.")
        return
    st.info("You've changed the list. The filter already uses your changes, but only for you until you save.")
    cfg = github_sync.config()
    if cfg is None:
        st.caption("Saving for everyone isn't set up (it needs a `[github]` section in the app's secrets). "
                   "Download the list and replace `good_teams.csv` in the repo instead.")
        st.download_button("Download good_teams.csv", to_csv(picked), file_name="good_teams.csv", mime="text/csv",
                           on_click="ignore")
        return
    if not ins._upload_unlocked():
        return
    if not st.button("Save for everyone", type="primary"):
        return
    added = sorted(set(picked) - set(saved), key=str.casefold)
    removed = sorted(set(saved) - set(picked), key=str.casefold)
    detail = "\n".join([f"Added: {', '.join(added)}" if added else "", f"Removed: {', '.join(removed)}" if removed
                        else ""]).strip()
    try:
        with st.spinner("Saving to GitHub…"):
            url = github_sync.commit_files(cfg, lambda read: {"good_teams.csv": to_csv(picked)},
                                           f"Update good teams list\n\n{detail}\n\nSaved from the app's Good Teams "
                                           "Filter page.")
    except github_sync.SaveError as e:
        st.error(f"Couldn't save: {e}")
        return
    st.session_state[SAVED_KEY] = picked
    st.success("Saved. Everyone gets this list when the app redeploys in a minute or two."
               + (f" [See the commit]({url})." if url else ""))
