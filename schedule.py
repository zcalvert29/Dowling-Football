"""
schedule.py — the Schedule page and the app's "next game".

    Results     each team's games come straight from the film in the app: week, date, opponent, and final score.
    Next game   schedule.csv holds Dowling's next game (season, date, opponent), set with Add Next Game on the
                Schedule page and saved to GitHub the same way Add a Game saves film. The Home and Scouting pages
                open on that opponent. Once the game's film is in the app (a Dowling game against that opponent
                within a few days of the date) or the date has passed, it's no longer "next".

schedule.csv columns:
    season      e.g. 2026
    date        YYYY-MM-DD
    opponent    spelled the way it appears in the app (teams.csv)
"""
from __future__ import annotations

import io
import os
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

import insights as ins
import teams as tm

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HERE, "schedule.csv")
TEAM = ins.TEAM
COLUMNS = ["season", "date", "opponent"]
SAVED_KEY = "schedule_saved"   # schedule.csv text this session saved (the file catches up when the app redeploys)
SAME_GAME_DAYS = 6             # film within this many days of a scheduled date is that game


# ---------------------------------------------------------------------------
# schedule.csv
# ---------------------------------------------------------------------------
def parse(text: str) -> pd.DataFrame:
    if not text or not text.strip():
        return pd.DataFrame(columns=COLUMNS)
    s = pd.read_csv(io.StringIO(text), dtype=str).fillna("")
    s = s.reindex(columns=COLUMNS, fill_value="")
    s = s[s["opponent"].str.strip() != ""].copy()
    s["opponent"] = s["opponent"].str.strip()
    s["date"] = pd.to_datetime(s["date"], errors="coerce").dt.date
    s["season"] = pd.to_numeric(s["season"], errors="coerce")
    s = s.dropna(subset=["date"])
    s["season"] = s["season"].fillna(pd.Series([d.year for d in s["date"]], index=s.index)).astype(int)
    return s.sort_values("date").reset_index(drop=True)


def read_text(path: str = FILE) -> str:
    """This session's saved schedule if it just saved one, else schedule.csv."""
    try:
        saved = st.session_state.get(SAVED_KEY)
    except Exception:  # noqa: BLE001 — no Streamlit session (tests, command line)
        saved = None
    if saved is not None:
        return saved
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def load(path: str = FILE) -> pd.DataFrame:
    return parse(read_text(path))


def to_csv(s: pd.DataFrame) -> bytes:
    out = s.reindex(columns=COLUMNS).copy()
    out["date"] = [d.isoformat() for d in out["date"]]
    return out.sort_values("date").to_csv(index=False).encode()


def with_next_game(current_text: str, opponent: str, day: date, df: pd.DataFrame, today: date | None = None) -> bytes:
    """schedule.csv with this game as the next one: any other game still to be played is replaced (there's only one
    "next"), games already played or past are kept."""
    today = today or datetime.now().date()
    s = parse(current_text)
    keep = [i for i, r in s.iterrows() if is_played(r["opponent"], r["date"], df) or r["date"] < today]
    s = s.loc[keep]
    s = pd.concat([s, pd.DataFrame([{"season": day.year, "date": day, "opponent": opponent}])], ignore_index=True)
    return to_csv(s)


# ---------------------------------------------------------------------------
# Games on film and the next game
# ---------------------------------------------------------------------------
def game_date(game_id: str) -> date | None:
    try:
        return datetime.strptime("_".join(str(game_id).split("_")[-3:]), "%Y_%m_%d").date()
    except ValueError:
        return None


def is_played(opponent: str, day: date, df: pd.DataFrame) -> bool:
    """True if the app has film of Dowling vs this opponent within a few days of day."""
    g = df[((df["offense"] == TEAM) & (df["defense"] == opponent)) | ((df["offense"] == opponent)
                                                                       & (df["defense"] == TEAM))]
    return any(abs((d - day).days) <= SAME_GAME_DAYS for d in map(game_date, g["game_id"].unique()) if d)


def next_game(df: pd.DataFrame, today: date | None = None) -> dict | None:
    """{"opponent", "date", "season", "week"} for the next game in schedule.csv (today or later, no film yet), or None."""
    import build_data

    today = today or datetime.now().date()
    for _, r in load().iterrows():
        if r["date"] >= today and not is_played(r["opponent"], r["date"], df):
            try:
                week = build_data.week_of(r["date"], int(r["season"]))
            except ValueError:
                week = None
            return {"opponent": r["opponent"], "date": r["date"], "season": int(r["season"]), "week": week}
    return None


def when(g: dict) -> str:
    """'Week 11, Fri 11/6'."""
    day = f"{g['date']:%a} {g['date'].month}/{g['date'].day}"
    return f"Week {g['week']}, {day}" if g.get("week") else day


def results(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """One row per game of film with this team: Week, Date, Opponent, Result, Score (this team's points first)."""
    import rankings

    finals = rankings.final_scores(df)
    rows = []
    for gid, g in df.groupby("game_id", sort=False):
        pair = ins.teams_in_game(g)
        if team not in pair or len(pair) != 2:
            continue
        opp = pair[1] if pair[0] == team else pair[0]
        f = finals[(finals["game"] == gid) & (finals["team"] == team)]
        pf, pa = (int(f["pf"].iat[0]), int(f["pa"].iat[0])) if len(f) else (None, None)
        result = "" if pf is None else "W" if pf > pa else "L" if pf < pa else "T"
        rows.append({"Season": int(g["SEASON"].iat[0]), "Week": int(g["WEEK"].iat[0]), "Date": game_date(gid),
                     "Opponent": opp, "Result": result, "Score": "" if pf is None else f"{pf}-{pa}"})
    out = pd.DataFrame(rows, columns=["Season", "Week", "Date", "Opponent", "Result", "Score"])
    return out.sort_values(["Date", "Week"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def _next_friday(after: date) -> date:
    return after + timedelta(days=(4 - after.weekday()) % 7 or 7)


def render_page(df_all: pd.DataFrame) -> None:
    team_names = sorted({t for gid, g in df_all.groupby("game_id") for t in ins.teams_in_game(g)}, key=str.casefold)
    seasons = sorted(int(s) for s in df_all["SEASON"].dropna().unique())
    c1, c2 = st.columns([2, 1])
    team = c1.selectbox("Team", team_names, index=team_names.index(TEAM) if TEAM in team_names else 0,
                        key="sch_team", format_func=tm.option_label)
    season = c2.selectbox("Season", seasons[::-1], key="sch_season")

    games = results(df_all, team)
    games = games[games["Season"] == season].drop(columns="Season")
    nxt = next_game(df_all) if team == TEAM else None
    shown = games.copy()
    if nxt is not None and nxt["season"] == season:
        shown = pd.concat([shown, pd.DataFrame([{"Week": nxt["week"], "Date": nxt["date"], "Opponent": nxt["opponent"],
                                                 "Result": "Up next", "Score": ""}])], ignore_index=True)
    if shown.empty:
        st.info(f"No {season} games of film with {team} yet.")
    else:
        wins, losses = int((games["Result"] == "W").sum()), int((games["Result"] == "L").sum())
        st.markdown(f"**{team} {season}: {wins}-{losses}** in {len(games)} game{'s' if len(games) != 1 else ''} "
                    "of film")
        colors = tm.colors()

        def opp_css(name):
            import rankings

            c = colors.get(name)
            return f"background-color: {c}; color: {rankings._text_color(c)}; font-weight: 600" if c else ""

        def result_css(r):
            return {"W": "color: #1D7A4F; font-weight: 600", "L": "color: #B42318; font-weight: 600",
                    "Up next": "font-weight: 600; font-style: italic"}.get(r, "")

        shown["Week"] = shown["Week"].astype("Int64")
        sty = (shown.style.map(opp_css, subset=["Opponent"]).map(result_css, subset=["Result"])
               .format({"Date": lambda d: f"{d:%a} {d.month}/{d.day}" if pd.notna(d) else ""}))
        st.dataframe(sty, hide_index=True, width="content",
                     column_config={"Week": st.column_config.NumberColumn(width="small"),
                                    "Result": st.column_config.TextColumn(width="small"),
                                    "Score": st.column_config.TextColumn(help=f"{team}'s points first.")})
        st.caption("Scores come from the film in the app. A game shows up here once its film is added on Add a Game.")

    if team == TEAM:
        _render_next_game(df_all, games, nxt)


def _render_next_game(df_all: pd.DataFrame, games: pd.DataFrame, nxt: dict | None) -> None:
    import build_data
    import github_sync

    st.subheader("Next game")
    if nxt is not None:
        st.markdown(f"Up next: **{nxt['opponent']}** ({when(nxt)}). The Home and Scouting pages open on them.")
    else:
        st.markdown("No next game set. The Home and Scouting pages open on next_opponent from the app's secrets "
                    "(or SEP).")
    with st.expander("Add Next Game", expanded=nxt is None):
        st.caption("Pick who Dowling plays next. Saving makes them the default opponent for everyone and replaces "
                   "any other game still to be played on the schedule.")
        last = max(games["Date"]) if len(games) else None
        today = datetime.now().date()
        default_day = _next_friday(max(last, today - timedelta(days=1)) if last else today - timedelta(days=1))
        options = ins.add_game_team_options()
        c1, c2 = st.columns(2)
        opponent = c1.selectbox("Opponent", [o for o in options if o != TEAM], index=None, key="sch_opp",
                                format_func=tm.option_label, accept_new_options=True, placeholder="Type to search")
        day = c2.date_input("Game date", default_day, key="sch_date")
        try:
            c2.caption(f"Week {build_data.week_of(day, day.year)} of {day.year}")
        except ValueError as e:
            st.warning(str(e).replace("Check the date and season.", "Check the date."))
            return
        if not opponent:
            return
        opponent = opponent.strip()
        n_film = df_all.loc[(df_all["offense"] == opponent) | (df_all["defense"] == opponent), "game_id"].nunique()
        if not n_film:
            st.caption(f"No film on {opponent} in the app yet, so the Scouting pages can't open on them until "
                       "some is added.")
        data = with_next_game(read_text(), opponent, day, df_all)
        cfg = github_sync.config()
        if cfg is None:
            st.caption("Saving straight to the app isn't set up (it needs a `[github]` section in the app's secrets). "
                       "Download schedule.csv and replace it in the repo instead.")
            st.download_button("Download schedule.csv", data, file_name="schedule.csv", mime="text/csv",
                               on_click="ignore")
            return
        if not ins._upload_unlocked():
            return
        if not st.button("Save next game", type="primary"):
            return
        message = (f"Set next game: Dowling Catholic vs {opponent} ({day:%m/%d/%Y})\n\n"
                   "Saved from the app's Schedule page.")

        def build_files(read):
            return {"schedule.csv": with_next_game(read("schedule.csv") or "", opponent, day, df_all)}

        try:
            with st.spinner("Saving to GitHub…"):
                url = github_sync.commit_files(cfg, build_files, message)
        except github_sync.SaveError as e:
            st.error(f"Couldn't save: {e}")
            return
        st.session_state[SAVED_KEY] = data.decode()
        st.session_state["f_opp"] = opponent   # this session's default opponent too (reset if they have no film)
        st.success(f"Saved. {opponent} is up next for everyone once the app redeploys in a minute or two."
                   + (f" [See the commit]({url})." if url else ""))
