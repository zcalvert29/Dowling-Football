"""
insights.py — the newer, coach-facing pages' building blocks:

  Game review    drive charts, game recap, explosive-play logs, win
                 probability, 4th-down decision review
  Self-scout     Dowling's own tendencies an opponent could exploit
  Scouting       opponent's best plays, matchup comparison, one-page report
  Data checks    tagging coverage by week, "add a game" processing

Everything works off the same curated play-by-play frame the scouting pages
use (visuals.load_data), so the sidebar filters apply where it makes sense.
"""
from __future__ import annotations

import io
from datetime import date as _date

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import visuals as v

TEAM = v.TEAM
KICKOFF_TYPES = {"KO", "KO Rec"}
TRY_TYPES = {"Extra Pt.", "Extra Pt. Block", "2 Pt.", "2 Pt. Block", "2 Pt. Defend"}
PUNT_TYPES = {"Punt", "Punt Rec"}
FG_TYPES = {"FG", "FG Block"}

GREEN, RED, AMBER, GRAY = "#1D9E75", "#E24B4A", "#EF9F27", "#888780"


# ---------------------------------------------------------------------------
# Situational filters (used by the sidebar "Situation" control)
# ---------------------------------------------------------------------------
SITUATIONS = {
    "All plays": None,
    "Red zone (inside the 20)": lambda d: d["YARDLINE_100"] <= 20,
    "Goal to go": lambda d: d["DIST"] >= d["YARDLINE_100"],
    "Backed up (own 10 and in)": lambda d: d["YARDLINE_100"] >= 90,
    "3rd & short (1-3 yds)": lambda d: (d["DN"] == 3) & (d["DIST"] <= 3),
    "Short yardage (3rd/4th & 1-2)": lambda d: d["DN"].isin([3, 4]) & (d["DIST"] <= 2),
    "Long yardage (2nd/3rd & 8+)": lambda d: d["DN"].isin([2, 3]) & (d["DIST"] >= 8),
    # Needs the "neutral" column from add_game_state() (app.py adds it when the data loads).
    "Neutral": lambda d: d["neutral"] if "neutral" in d else pd.Series(True, index=d.index),
}
NEUTRAL_MARGIN = 7          # score within this many points at the snap...
NEUTRAL_LATE_SECONDS = 120  # ...and not inside the last 2 minutes of the 2nd or 4th quarter (estimated clock)


def add_game_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds, for every scrimmage snap: pre_snap_margin (offense minus defense, score at the snap), est_q_seconds
    (estimated high school seconds left in the quarter, the same estimate the win probability and 4th-down
    pages use), and neutral (True when the margin is within NEUTRAL_MARGIN and it isn't the last 2 minutes of
    a half or overtime). Kicks, tries, and snaps that can't be placed get neutral = False.
    """
    out = df.copy()
    out["pre_snap_margin"] = np.nan
    out["est_q_seconds"] = np.nan
    out["neutral"] = False
    d = _scrimmage(df[df["game_id"].notna() & df["QTR"].notna()])
    if d.empty:
        return out
    d = _estimate_clock(d)
    margin = d["pre_off_score"] - d["pre_def_score"]
    q = d["QTR"].astype(int)
    late = q.isin([2, 4]) & (d["q_seconds"] <= NEUTRAL_LATE_SECONDS)
    out.loc[d.index, "pre_snap_margin"] = margin
    out.loc[d.index, "est_q_seconds"] = d["q_seconds"]
    out.loc[d.index, "neutral"] = (margin.abs() <= NEUTRAL_MARGIN) & q.between(1, 4) & ~late
    out["neutral"] = out["neutral"].astype(bool)
    return out


def apply_situation(df: pd.DataFrame, name: str) -> pd.DataFrame:
    rule = SITUATIONS.get(name)
    return df if rule is None else df[rule(df).fillna(False).astype(bool)]


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------
def opponent_of(game_id: str) -> str:
    parts = str(game_id).split("_")
    return parts[1] if len(parts) > 1 else str(game_id)


def game_date_of(game_id: str) -> str:
    parts = str(game_id).split("_")
    if len(parts) >= 5:
        try:
            return _date(int(parts[2]), int(parts[3]), int(parts[4])).strftime("%b %d").replace(" 0", " ")
        except ValueError:
            pass
    return ""


def games(df: pd.DataFrame) -> list[str]:
    """game_ids in week order (game date breaks ties when two games share a week)."""
    g = df.dropna(subset=["game_id"]).groupby("game_id")["WEEK"].min().reset_index()
    g["_date"] = ["_".join(str(x).split("_")[2:5]) for x in g["game_id"]]
    return g.sort_values(["WEEK", "_date"], kind="stable")["game_id"].tolist()


def teams_in_game(g: pd.DataFrame) -> list[str]:
    """The two teams in one game, from who had the ball (game_id order isn't reliable for scout film)."""
    return sorted(set(g["offense"].dropna()) | set(g["defense"].dropna()))


def other_team(g: pd.DataFrame, team: str) -> str | None:
    rest = [t for t in teams_in_game(g) if t != team]
    return rest[0] if rest else None


def dowling_game_ids(df: pd.DataFrame) -> set:
    """Games Dowling actually played in (everything else is scout film of other teams)."""
    return set(df.loc[(df["offense"] == TEAM) | (df["defense"] == TEAM), "game_id"].dropna())


def dowling_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["game_id"].isin(dowling_game_ids(df))]


def scout_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["game_id"].isin(dowling_game_ids(df))]


def game_teams(game_id: str) -> tuple[str, str]:
    """(first, second) team from the game_id. The first is the side team_score / opponent_score were kept for."""
    parts = str(game_id).split("_")
    return parts[0], (parts[1] if len(parts) > 1 else "")


def scout_game_label(df: pd.DataFrame, game_id: str) -> str:
    """'W1 Johnston vs Waukee'."""
    week = df.loc[df["game_id"] == game_id, "WEEK"].min()
    a, b = game_teams(game_id)
    return (f"W{int(week)} " if pd.notna(week) else "") + f"{a} vs {b}"


def score_for(row: pd.Series, team: str) -> tuple[int, int]:
    """(team's score, other team's score) from a row's team_score / opponent_score."""
    first, _ = game_teams(row["game_id"])
    a, b = int(row["team_score"]), int(row["opponent_score"])
    return (a, b) if team == first else (b, a)


def game_label(df: pd.DataFrame, game_id: str) -> str:
    week = df.loc[df["game_id"] == game_id, "WEEK"].min()
    wk = f"W{int(week)} · " if pd.notna(week) else ""
    return f"{wk}{opponent_of(game_id)}"


def last_updated_text(df: pd.DataFrame) -> str:
    """'Data through Week 6 vs CR Kennedy (Oct 2)': Dowling's most recent game, not the latest scout film."""
    d = dowling_only(df)
    gs = games(d)
    if not gs:
        return "No Dowling games loaded"
    last = gs[-1]
    g = d[d["game_id"] == last]
    week = g["WEEK"].min()
    opp = other_team(g, TEAM) or opponent_of(last)
    when = game_date_of(last)
    wk = f"Week {int(week)} vs " if pd.notna(week) else "vs "
    return f"Data through {wk}{opp}" + (f" ({when})" if when else "")


# ---------------------------------------------------------------------------
# Drive chart
# ---------------------------------------------------------------------------
YELLOW = "#E8B923"
DRIVE_COLORS = {
    "Touchdown": GREEN, "FG Made": YELLOW, "FG Missed": RED, "Punt": GRAY, "Turnover": RED, "Turnover, Def TD": RED,
    "Turnover on Downs": RED, "Safety": RED, "End of Half": GRAY, "End of Game": GRAY, "Error": GRAY,
}
DRIVE_LEGEND = [("Touchdown", GREEN), ("FG made", YELLOW), ("Turnover / downs / missed FG / safety", RED),
                ("Punt / end of half", GRAY)]


def drive_summary(g: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    One row per possession for `team` in a single game. start/end are yards
    from the team's own goal line (0-100). end is the spot of the drive's
    last snap, or 100 for a touchdown.
    """
    d = g[(g["offense"] == team) & ~g["PLAY TYPE"].isin(KICKOFF_TYPES | TRY_TYPES)]
    d = d[d["YARDLINE_100"].notna()]
    rows = []
    for drive, p in d.groupby("drive", sort=True):
        result = p["drive_result"].dropna()
        result = result.iat[0] if len(result) else "Error"
        if result == "FG Attempt":
            fg = p[p["PLAY TYPE"].isin(FG_TYPES)]
            made = len(fg) and str(fg["RESULT"].iat[-1]) == "Good"
            result = "FG Made" if made else "FG Missed"
        snaps = p[p["PLAY TYPE"].isin(["Run", "Pass"])]
        start = 100 - p["YARDLINE_100"].iat[0]
        end = 100 if result == "Touchdown" else 100 - p["YARDLINE_100"].iat[-1]
        rows.append({
            "drive": drive, "QTR": int(p["QTR"].iat[0]), "start": float(start), "end": float(end),
            "plays": len(snaps), "yards": float(p["GN/LS"].fillna(0)[p["PLAY TYPE"].isin(["Run", "Pass"])].sum()),
            "result": result, "three_and_out": result == "Punt" and len(snaps) <= 3,
        })
    return pd.DataFrame(rows)


def _yard_label(x: float) -> str:
    """Yards from own goal -> 'Own 25' / 'Opp 40' / '50'."""
    x = int(round(x))
    if x == 50:
        return "50"
    return f"Own {x}" if x < 50 else f"Opp {100 - x}"


def drive_chart_html(drives: pd.DataFrame, ink: str = "inherit") -> str:
    if drives.empty:
        return ""
    muted = "opacity:.65"
    ticks = "".join(
        f'<span style="position:absolute;left:{x}%;transform:translateX(-50%)">'
        f'{"G" if x in (0, 100) else (x if x <= 50 else 100 - x)}</span>'
        for x in range(0, 101, 10)
    )
    yard_lines = "".join(
        f'<span style="position:absolute;top:0;bottom:0;left:{x}%;border-left:'
        f'{"1px solid rgba(128,128,128,.55)" if x == 50 else "0.5px solid rgba(128,128,128,.3)"}"></span>'
        for x in range(10, 100, 10)
    )
    rows = []
    for _, r in drives.iterrows():
        color = DRIVE_COLORS.get(r["result"], GRAY)
        left, width = min(r["start"], r["end"]), max(abs(r["end"] - r["start"]), 1.2)
        tip = f'{_yard_label(r["start"])} to {"end zone" if r["end"] >= 100 else _yard_label(r["end"])}'
        result = "3-and-out" if r["three_and_out"] else r["result"]
        rows.append(
            f'<div style="font-size:12px;{muted}">Q{r["QTR"]}</div>'
            f'<div style="position:relative;height:22px;background:rgba(128,128,128,.08);border-radius:4px">'
            f'{yard_lines}<span title="{tip}" style="position:absolute;top:4px;height:14px;border-radius:3px;'
            f'left:{left}%;width:{width}%;background:{color}"></span></div>'
            f'<div style="font-size:12px">{r["plays"]} play{"" if r["plays"] == 1 else "s"} · {result}</div>'
        )
    legend = "".join(
        f'<span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:10px;'
        f'border-radius:2px;background:{c}"></span>{lab}</span>' for lab, c in DRIVE_LEGEND
    )
    return (
        f'<div style="max-width:860px;color:{ink};font-family:inherit">'
        f'<div style="display:flex;flex-wrap:wrap;gap:14px;font-size:12px;margin-bottom:8px;{muted}">{legend}'
        f'<span style="margin-left:auto">Driving left to right</span></div>'
        f'<div style="display:grid;grid-template-columns:36px 1fr 150px;gap:6px 10px;align-items:center">'
        f'<div></div><div style="position:relative;height:14px;font-size:11px;{muted}">{ticks}</div><div></div>'
        f'{"".join(rows)}</div></div>'
    )


def render_drive_chart(g: pd.DataFrame, team: str, title: str) -> pd.DataFrame:
    st.markdown(f"**{title}**")
    drives = drive_summary(g, team)
    if drives.empty:
        st.info("No drives for this team in the selected game.")
        return drives
    stats = [
        ("Drives", len(drives)),
        ("Touchdowns", int((drives["result"] == "Touchdown").sum())),
        ("Avg start", _yard_label(drives["start"].mean())),
        ("3-and-outs", int(drives["three_and_out"].sum())),
        ("Turnovers", int(drives["result"].isin(["Turnover", "Turnover, Def TD", "Turnover on Downs"]).sum())),
    ]
    stats_html = (
        '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:10px">'
        + "".join(f'<div style="background:rgba(128,128,128,.08);border-radius:8px;padding:6px 12px">'
                  f'<div style="font-size:12px;opacity:.65">{k}</div>'
                  f'<div style="font-size:18px;font-weight:600">{val}</div></div>' for k, val in stats)
        + "</div>"
    )
    st.html(drive_chart_html(drives) + stats_html)
    return drives


# ---------------------------------------------------------------------------
# Game recap
# ---------------------------------------------------------------------------
# How each head-to-head row is compared: (better direction, value used, display)
def _unit_stats(d: pd.DataFrame) -> dict:
    """Head-to-head numbers for one offense: {row: (sort value, display text)}."""
    rp = v.run_pass(d)
    third, fourth = rp[rp["DN"] == 3], rp[rp["DN"] == 4]
    expl = int(pd.to_numeric(rp["explosive_play"], errors="coerce").sum())

    def conv(rows, col):
        if not len(rows):
            return np.nan, "–"
        made = int(rows[col].sum())
        return made / len(rows), f"{made}/{len(rows)} ({made / len(rows):.0%})"

    epa, sr = rp["epa"].mean(), rp["success"].mean()
    avg3 = third["DIST"].mean() if len(third) else np.nan
    return {
        "Plays": (len(rp), f"{len(rp)}"),
        "EPA / play": (epa, "–" if pd.isna(epa) else f"{epa:+.2f}"),
        "Success": (sr, "–" if pd.isna(sr) else f"{sr:.0%}"),
        "Explosive plays": (expl, f"{expl}"),
        "3rd down": conv(third, "THIRD_DOWN_CONVERTED"),
        "Avg 3rd down distance": (avg3, "–" if pd.isna(avg3) else f"{avg3:.1f} yds"),
        "4th down": conv(fourth, "FOURTH_DOWN_CONVERTED"),
        "Turnovers": (int(rp["turnover"].sum()), f"{int(rp['turnover'].sum())}"),
    }


# Rows where a higher number is better for that offense (True) or lower (False).
HEAD_TO_HEAD_BETTER = {"EPA / play": True, "Success": True, "Explosive plays": True, "3rd down": True,
                       "Avg 3rd down distance": False, "4th down": True, "Turnovers": False}


PLAY_COLS = ["WEEK", "QTR", "DN", "DIST", "YARD LN", "PLAY TYPE", "OFF FORM", "OFF PLAY", "RESULT", "GN/LS", "epa"]


def _ordinal(n) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(int(n), f"{int(n)}th") if pd.notna(n) else ""


def _field_spot(ytg) -> str:
    """Yards to goal -> 'Own 33' / 'Opp 29' / '50', from the offense's view."""
    if pd.isna(ytg):
        return ""
    y = int(round(ytg))
    return "50" if y == 50 else f"Own {100 - y}" if y > 50 else f"Opp {y}"


def plays_table(d: pd.DataFrame) -> pd.DataFrame:
    """
    One row per play, outcome first: EPA, yards, and result lead so they stay
    visible when tables sit side by side, followed by a readable situation
    ("Q1 · 3rd & 5 · Own 33") instead of separate numeric columns.
    """
    out = pd.DataFrame(index=d.index)
    if "epa" in d:
        out["EPA"] = d["epa"]
    if "GN/LS" in d:
        # Text so a missing value shows blank instead of Streamlit's "None".
        out["Yards"] = [str(int(x)) if pd.notna(x) else "" for x in d["GN/LS"]]
    if "RESULT" in d:
        out["Result"] = d["RESULT"].fillna("")
    spot = d["YARDLINE_100"].map(_field_spot) if "YARDLINE_100" in d else ""
    down = [f"{_ordinal(n)} & {int(x)}" if pd.notna(n) and pd.notna(x) else "" for n, x in zip(d["DN"], d["DIST"])]
    out["Situation"] = [" · ".join(p for p in (f"Q{int(q)}" if pd.notna(q) else "", dd, sp) if p)
                        for q, dd, sp in zip(d["QTR"], down, spot if len(spot) else [""] * len(d))]
    for src, name in (("PLAY TYPE", "Type"), ("OFF FORM", "Formation"), ("OFF PLAY", "Play")):
        if src in d:
            out[name] = d[src].astype("string").fillna("")
    if "WEEK" in d and d["WEEK"].nunique() > 1:
        out["Week"] = d["WEEK"].astype("Int64")
    return out


def show_plays(title: str, d: pd.DataFrame, good_high: bool = True) -> None:
    st.markdown(f"**{title}**")
    if d.empty:
        st.info("No plays.")
        return
    t = plays_table(d)

    def color(col):
        return [("background-color:#E1F5EE;color:#085041" if (x >= 0) == good_high else
                 "background-color:#FCEBEB;color:#791F1F") if pd.notna(x) else "" for x in col]

    sty = t.style.format(na_rep="").format({"EPA": "{:+.2f}"}, na_rep="").apply(color, subset=["EPA"])
    st.dataframe(sty, width="stretch", hide_index=True)


def render_game_recap(df: pd.DataFrame, game_id: str, half: str, team: str = TEAM) -> None:
    """
    Full recap for one game (or season-wide logs when game_id is None). `team` is the side the recap is told
    from: Dowling for Dowling games, the first team in the game_id for scout film.
    """
    TEAM = team  # noqa: N806 — everything below reads from this team's side
    g = df if game_id is None else df[df["game_id"] == game_id]
    if half == "First half":
        g = g[g["QTR"] <= 2]
    elif half == "Second half":
        g = g[g["QTR"] >= 3]
    if g.empty:
        st.info("No plays for this selection.")
        return
    opp = None if game_id is None else (other_team(df[df["game_id"] == game_id], team) or opponent_of(game_id))
    them = opp or "Opponents"
    scout = team != v.TEAM

    if game_id is not None:
        last = g.iloc[-1]
        us, they = score_for(last, team)
        when = {"First half": "at halftime", "Second half": "final"}.get(half, "final")
        st.subheader(f"{TEAM} {us}, {opp} {they} ({when})")

    ours = _unit_stats(g[g["offense"] == TEAM])
    theirs = _unit_stats(g[(g["offense"] != TEAM) & g["offense"].notna()])
    st.markdown("**Head to head**")
    rows = []
    for k in ours:
        (a, a_txt), (b, b_txt) = ours[k], theirs[k]
        better_high = HEAD_TO_HEAD_BETTER.get(k)
        edge = ""
        if better_high is not None and pd.notna(a) and pd.notna(b) and a != b:
            edge = TEAM if (a > b) == better_high else them
        rows.append((k, a_txt, b_txt, edge))
    table = pd.DataFrame(rows, columns=["", f"{TEAM} offense", f"{them} offense", "Edge"]).set_index("")

    def color(col):
        return ["background-color:#E1F5EE;color:#085041" if x == TEAM else
                "background-color:#FCEBEB;color:#791F1F" if x else "" for x in col]

    st.dataframe(table.style.apply(color, subset=["Edge"]), width="stretch")
    st.caption("3rd and 4th down count runs and passes only (punts and field goals aren't conversion tries). "
               "A shorter average 3rd down distance is the edge.")

    if game_id is not None:
        c1, c2 = st.columns(2)
        with c1:
            render_drive_chart(g, TEAM, f"{TEAM} drives")
        with c2:
            render_drive_chart(g, opp, f"{opp} drives")

        by_q = g[g["PLAY TYPE"].isin(["Run", "Pass"])].assign(
            side=lambda d: np.where(d["offense"] == TEAM, f"{TEAM} offense", f"{opp} offense"))
        q = by_q.pivot_table(index="QTR", columns="side", values="epa", aggfunc="mean")
        q.index = [f"Q{int(i)}" for i in q.index]
        st.markdown("**EPA per play by quarter**")
        st.dataframe(q.style.format("{:+.2f}", na_rep="–"), width="stretch")

    rp = v.run_pass(g)
    ours_rp, theirs_rp = rp[rp["offense"] == TEAM], rp[rp["offense"] != TEAM]
    c1, c2 = st.columns(2)
    with c1:
        show_plays(f"{TEAM} best plays", ours_rp.nlargest(5, "epa"))
        show_plays(f"{TEAM} worst plays", ours_rp.nsmallest(5, "epa"))
    with c2:
        show_plays(f"{them} best plays" + ("" if scout else " (our worst on D)"), theirs_rp.nlargest(5, "epa"),
                   good_high=scout)
        show_plays(f"{them} worst plays" + ("" if scout else " (our best on D)"), theirs_rp.nsmallest(5, "epa"),
                   good_high=scout)

    expl = pd.to_numeric(rp["explosive_play"], errors="coerce") == 1
    show_plays(f"Explosive plays for {TEAM}", rp[expl & (rp["offense"] == TEAM)])
    show_plays(f"Explosive plays for {them}" if scout else "Explosive plays allowed",
               rp[expl & (rp["offense"] != TEAM)], good_high=scout)


# ---------------------------------------------------------------------------
# Self-scout
# ---------------------------------------------------------------------------
def _tell(share: float) -> str:
    return "Strong" if share >= 0.8 else "Lean" if share >= 0.7 else "Mixed"


def _tell_style(col):
    return [{"Strong": "background-color:#FCEBEB;color:#791F1F",
             "Lean": "background-color:#FAEEDA;color:#633806"}.get(x, "") for x in col]


def _show_tells(title: str, t: pd.DataFrame, pct_cols: list[str], caption: str) -> None:
    st.markdown(f"**{title}**")
    if t.empty:
        st.info("Not enough tagged plays yet.")
        return
    sty = t.style.format({c: "{:.0%}" for c in pct_cols}, na_rep="–").apply(_tell_style, subset=["Tell"])
    st.dataframe(sty, width="stretch")
    st.caption(caption)


def render_self_scout(df: pd.DataFrame, df_any_down: pd.DataFrame, min_plays: int = 10) -> None:
    v.render_dd_tendencies(df_any_down, TEAM, "DCHS O down & distance tendencies")
    rp = v.run_pass(df[df["offense"] == TEAM])
    tell_caption = "Tell: Strong = 80%+ one way, Lean = 70-79%, Mixed = under 70%."

    f = rp.dropna(subset=["OFF FORM"]).groupby("OFF FORM").agg(Plays=("PASS", "size"), Pass=("PASS", "mean"))
    f = f[f["Plays"] >= min_plays].sort_values("Plays", ascending=False)
    f["Run"] = 1 - f["Pass"]
    f["Tell"] = [_tell(max(p, 1 - p)) for p in f["Pass"]]
    _show_tells(f"Run/pass by formation ({min_plays}+ plays)", f[["Plays", "Run", "Pass", "Tell"]],
                ["Run", "Pass"], tell_caption)

    # Hash: run/pass, and whether the play goes to the field or boundary.
    h = rp.dropna(subset=["HASH"]).copy()
    dir_ = h["PLAY DIR"].astype("string").str.upper()
    field_dir = h["HASH"].map({"L": "R", "R": "L"})
    h["to_field"] = np.where(dir_.isin(["L", "R"]) & field_dir.notna(), (dir_ == field_dir).astype(float), np.nan)
    ht = h.groupby("HASH").agg(Plays=("PASS", "size"), Pass=("PASS", "mean"), **{"To field": ("to_field", "mean")})
    ht["Run"] = 1 - ht["Pass"]
    ht["Tell"] = [_tell(max(p, 1 - p, *([tf, 1 - tf] if pd.notna(tf) else [])))
                  for p, tf in zip(ht["Pass"], ht["To field"])]
    ht.index = [{"L": "Left hash", "M": "Middle", "R": "Right hash"}.get(h, h) for h in ht.index]
    _show_tells("By hash", ht[["Plays", "Run", "Pass", "To field", "Tell"]], ["Run", "Pass", "To field"],
                "To field = share of plays with a direction that went to the wide side of the field. " + tell_caption)

    # Strength and motion: does the play go toward them?
    out = []
    s = rp[rp["OFF STR"].isin(["L", "R"]) & rp["PLAY DIR"].isin(["L", "R"])]
    if len(s):
        share = (s["OFF STR"] == s["PLAY DIR"]).mean()
        out.append(("Toward formation strength", len(s), share))
    m = rp[rp["MOTION DIR"].isin(["L", "R"]) & rp["PLAY DIR"].isin(["L", "R"])]
    if len(m):
        share = (m["MOTION DIR"] == m["PLAY DIR"]).mean()
        out.append(("Toward the motion", len(m), share))
    t = pd.DataFrame(out, columns=["Check", "Plays", "Share"]).set_index("Check")
    if not t.empty:
        t["Tell"] = [_tell(max(x, 1 - x)) if n >= min_plays else "Low n" for x, n in zip(t["Share"], t["Plays"])]
    _show_tells("Play direction vs strength and motion", t, ["Share"],
                f"Share = how often the play went that way. Under {min_plays} plays shows as low n. " + tell_caption)


# ---------------------------------------------------------------------------
# Opponent scouting: best plays, matchup, coverage by formation
# ---------------------------------------------------------------------------
def render_best_plays(df: pd.DataFrame, team: str, n: int = 10) -> None:
    rp = v.run_pass(df[df["offense"] == team])
    show_plays(f"{team} best {n} plays (most likely to come back)", rp.nlargest(n, "epa"), good_high=False)


CONVERTED_COL = {3: "THIRD_DOWN_CONVERTED", 4: "FOURTH_DOWN_CONVERTED"}


def render_down_calls(df: pd.DataFrame, team: str, down: int) -> None:
    """
    Every run/pass `team` ran on 3rd or 4th down, in game order, laid out like the best-plays table plus a
    Converted column. Converted comes from THIRD_DOWN_CONVERTED / FOURTH_DOWN_CONVERTED (1 = gained the line
    to gain on that snap; a first down by penalty isn't counted).
    """
    label = {3: "3rd", 4: "4th"}[down]
    title = f"{team} {label} down calls and results"
    d = v.run_pass(df[(df["offense"] == team) & (df["DN"] == down)])
    st.markdown(f"**{title}**")
    if d.empty:
        st.info(f"No {label} down plays match the current filters.")
        return
    d = d.sort_values(["WEEK", "game_id", "PLAY #"], kind="stable")
    t = plays_table(d)
    conv = pd.to_numeric(d[CONVERTED_COL[down]], errors="coerce")
    t.insert(0, "Converted", ["Yes" if x == 1 else "No" if x == 0 else "" for x in conv])
    sty = t.style.format(na_rep="").format({"EPA": "{:+.2f}"}, na_rep="")   # no colors: just the log
    made = int((conv == 1).sum())
    st.dataframe(sty, width="stretch", hide_index=True)
    st.caption(f"Converted {made} of {int(conv.notna().sum())}. Every {label} down run or pass in the selected games, "
               "oldest first.")


def _side_metrics(d: pd.DataFrame) -> dict:
    """Offensive numbers for a set of plays (an offense's plays, or the plays a defense faced)."""
    rp = v.run_pass(d)
    third = rp[rp["DN"] == 3]
    return {
        "Plays": len(rp),
        "Pass rate": rp["PASS"].mean(),
        "Rush EPA / play": rp.loc[rp["PLAY TYPE"] == "Run", "epa"].mean(),
        "Pass EPA / play": rp.loc[rp["PLAY TYPE"] == "Pass", "epa"].mean(),
        "Success rate": rp["success"].mean(),
        "Explosive rate": pd.to_numeric(rp["explosive_play"], errors="coerce").mean(),
        "3rd down conversion": third["THIRD_DOWN_CONVERTED"].mean() if len(third) else np.nan,
        # Sample size behind each number, used to shrink small samples in the projection.
        "_n": {"Rush EPA / play": int((rp["PLAY TYPE"] == "Run").sum()),
               "Pass EPA / play": int((rp["PLAY TYPE"] == "Pass").sum()),
               "Success rate": len(rp), "Explosive rate": len(rp), "3rd down conversion": len(third)},
    }


# How far the projected matchup has to be from the baseline to call an edge.
EDGE_THRESHOLDS = {"Rush EPA / play": 0.10, "Pass EPA / play": 0.10, "Success rate": 0.04,
                   "Explosive rate": 0.03, "3rd down conversion": 0.06}


# Shrinkage: before projecting, each number is pulled toward average as if it
# came with this many extra plays of exactly-average football. One game of
# film (~30 passes) is mostly noise for EPA, so it gets pulled about halfway;
# a full season of Dowling plays barely moves. Rough stabilization points;
# tune them as the data grows.
SHRINK_PLAYS = {"Rush EPA / play": 40, "Pass EPA / play": 40, "Success rate": 40,
                "Explosive rate": 80, "3rd down conversion": 15}


def _shrink(x: float, n: int, base: float, k: int) -> float:
    return base + (x - base) * n / (n + k) if pd.notna(x) else x


def _baselines(df: pd.DataFrame) -> dict:
    """What an average offense does: EPA = 0 by definition; rates = every offense in the data combined."""
    allp = _side_metrics(df)
    return {"Rush EPA / play": 0.0, "Pass EPA / play": 0.0, "Success rate": allp["Success rate"],
            "Explosive rate": allp["Explosive rate"], "3rd down conversion": allp["3rd down conversion"]}


def _matchup_table(off: dict, deff: dict, base: dict, off_name: str, def_name: str) -> pd.DataFrame:
    """
    Both columns are offensive numbers: what the offense produces, and what
    the defense gives up. Each is judged against an average offense
    (baseline), so a defense that ALLOWS positive EPA is a weakness, not a
    match for a good offense. Projected = offense + defense - baseline: how
    far above/below average this offense should be against this defense.
    """
    rows = []
    for k in off:
        if k.startswith("_"):
            continue
        a, b, b0 = off[k], deff[k], base.get(k, np.nan)
        proj, edge = np.nan, ""
        thr = EDGE_THRESHOLDS.get(k)
        if thr and pd.notna(a) and pd.notna(b) and pd.notna(b0):
            a_s = _shrink(a, off["_n"][k], b0, SHRINK_PLAYS[k])
            b_s = _shrink(b, deff["_n"][k], b0, SHRINK_PLAYS[k])
            proj = a_s + b_s - b0
            edge = (off_name if proj - b0 >= thr else def_name.replace(" allows", "") if b0 - proj >= thr
                    else "Even")
        rows.append((k, a, b, b0, proj, edge))
    return pd.DataFrame(rows, columns=["Metric", off_name, def_name, "Average offense", "Projected", "Edge"]
                        ).set_index("Metric")


# Columns still computed (they drive the colors) but not shown on the Matchup page.
MATCHUP_HIDDEN = ["Average offense", "Projected", "Edge"]


def _fmt_metric(k, x):
    if pd.isna(x):
        return "–"
    if k == "Plays":
        return f"{int(x)}"
    if "EPA" in k:
        return f"{x:+.2f}"
    return f"{x:.0%}"


GOOD_CSS, BAD_CSS = "background-color:#E1F5EE;color:#085041", "background-color:#FCEBEB;color:#791F1F"


def _show_matchup(title: str, t: pd.DataFrame, dowling_on_offense: bool) -> None:
    """
    Shows the offense and defense columns only. Colors are from Dowling's point of view (green = good for
    Dowling) and need to be clearly away from an average offense, except EPA in the opposing offense's column,
    which is simply green when positive and red when negative.
    """
    st.markdown(f"**{title}**")
    off_col, def_col = t.columns[0], t.columns[1]
    shown = t[[off_col, def_col]].copy()
    for col in shown.columns:
        shown[col] = [_fmt_metric(k, x) for k, x in zip(t.index, t[col])]

    def good_for_dowling(offense_above_avg: bool) -> bool:
        return offense_above_avg == dowling_on_offense

    def style_row(row):
        k = row.name
        thr = EDGE_THRESHOLDS.get(k)
        styles = {c: "" for c in shown.columns}
        if thr:
            b0 = t.loc[k, "Average offense"]
            for col in (off_col, def_col):
                x = t.loc[k, col]
                if pd.isna(x):
                    continue
                if "EPA" in k and col == off_col and not dowling_on_offense:
                    styles[col] = GOOD_CSS if x > 0 else BAD_CSS if x < 0 else ""
                elif pd.notna(b0) and abs(x - b0) >= thr:
                    styles[col] = GOOD_CSS if good_for_dowling(x > b0) else BAD_CSS
        return [styles[c] for c in shown.columns]

    st.dataframe(shown.style.apply(style_row, axis=1), width="stretch")


def render_matchup(df: pd.DataFrame, opponent: str) -> None:
    base = _baselines(df)
    # Dowling's side leaves out the head-to-head games, so a game already
    # played against this opponent isn't counted on both sides of the table.
    h2h = set(df.loc[((df["offense"] == opponent) | (df["defense"] == opponent)) & ((df["offense"] == TEAM)
                     | (df["defense"] == TEAM)), "game_id"])
    others = df[~df["game_id"].isin(h2h)]
    dchs = others if not others.empty else df
    opp_games = df.loc[(df["offense"] == opponent) | (df["defense"] == opponent), "game_id"].unique()
    only_h2h = len(opp_games) > 0 and set(opp_games) <= h2h

    if only_h2h:
        st.warning(f"All of {opponent}'s film is from their game against Dowling, so their numbers below are really "
                   f"\"how they did against us.\" Add film of {opponent} against someone else for an independent read.",
                   icon="⚠️")

    opp_o = _side_metrics(df[df["offense"] == opponent])
    dchs_d = _side_metrics(dchs[dchs["defense"] == TEAM])
    dchs_o = _side_metrics(dchs[dchs["offense"] == TEAM])
    opp_d = _side_metrics(df[df["defense"] == opponent])
    _show_matchup(f"{opponent} offense vs DCHS defense",
                  _matchup_table(opp_o, dchs_d, base, f"{opponent} offense", "DCHS defense allows"),
                  dowling_on_offense=False)
    _show_matchup(f"DCHS offense vs {opponent} defense",
                  _matchup_table(dchs_o, opp_d, base, "DCHS offense", f"{opponent} defense allows"),
                  dowling_on_offense=True)
    h2h_note = (f" Dowling's columns leave out the {len(h2h)} game(s) against {opponent} so that game isn't counted "
                f"twice." if h2h and not others.empty else "")
    st.caption(
        "Every number is from the offense's point of view: what the offense gets, and what the defense gives up. "
        f"EPA in the {opponent} offense column is green when positive and red when negative. Every other colored "
        "cell is clearly above or below an average offense (EPA 0; rates = every offense in the data combined), "
        "with green = good for Dowling and red = bad. "
        f"{opponent}'s numbers come from the {len(opp_games)} game(s) in the data that include them.{h2h_note}"
    )


def render_coverage_by_formation(df: pd.DataFrame, top_n: int = 10) -> None:
    st.markdown("**DCHS D coverage by opponent formation**")
    d = v.run_pass(df[df["defense"] == TEAM]).dropna(subset=["OFF FORM", "COVERAGE"])
    if d.empty:
        st.info("No plays with both a formation and a coverage tag match the current filters.")
        return
    forms = d["OFF FORM"].value_counts().head(top_n).index
    covs = d["COVERAGE"].value_counts().head(8).index
    d = d[d["OFF FORM"].isin(forms) & d["COVERAGE"].isin(covs)]
    t = pd.crosstab(d["OFF FORM"], d["COVERAGE"], normalize="index").reindex(index=forms, columns=covs)
    t.insert(0, "EPA per Play", d.groupby("OFF FORM")["epa"].mean().reindex(forms))
    t["Plays"] = d["OFF FORM"].value_counts().reindex(forms)

    def epa_color(col):  # the opponent's EPA: positive is bad for Dowling
        return ["" if pd.isna(x) else BAD_CSS if x >= 0.10 else GOOD_CSS if x <= -0.10 else "" for x in col]

    # One format call: a second .format() call resets every column it isn't given back to the raw number,
    # which is what showed coverage shares as 0.257143 instead of 25.7%.
    sty = (t.style.format({**{c: "{:.1%}" for c in covs}, "EPA per Play": "{:+.2f}", "Plays": "{:,.0f}"},
                          na_rep="–")
           .background_gradient(cmap="Blues", subset=list(covs), vmin=0, vmax=1)
           .apply(epa_color, subset=["EPA per Play"]))
    st.dataframe(sty, width="stretch")
    st.caption("Each row is one opponent formation: the share of those snaps Dowling played each coverage. "
               "A dark cell is a coverage we lean on against that look. EPA per Play is what the opponent gained "
               "from that formation (green = Dowling held it down, red = it hurt us).")


# ---------------------------------------------------------------------------
# Scouting report (one page + downloadable HTML for printing)
# ---------------------------------------------------------------------------
REPORT_CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#31333F;margin:24px;max-width:900px}
h1{font-size:24px;margin:0 0 4px} h2{font-size:17px;margin:28px 0 8px;border-bottom:1px solid #ddd;padding-bottom:4px}
p.sub{color:#666;margin:0 0 12px} table{border-collapse:collapse;font-size:13px;width:100%}
th,td{padding:4px 8px;border-bottom:1px solid #eee;text-align:right} th:first-child,td:first-child{text-align:left}
@media print{h2{break-after:avoid} .block{break-inside:avoid}}
"""


def _table_html(t: pd.DataFrame, fmt: dict, index: bool = True) -> str:
    sty = t.style.format(na_rep="–").format(fmt, na_rep="–")
    return (sty if index else sty.hide(axis="index")).to_html()


def build_report_html(df: pd.DataFrame, df_any_down: pd.DataFrame, opponent: str, filters_text: str,
                      notes: str = "", df_games: pd.DataFrame | None = None) -> str:
    import html as _html

    import profiles

    ink = "#31333F"
    sections = [f"<h1>{opponent} offense scouting report</h1>",
                f'<p class="sub">{TEAM} · {filters_text} · generated {_date.today():%b %d, %Y}</p>']
    if df_games is not None:
        prof = profiles.profiles_report_html(df, df_games, opponent)
        if prof:
            sections.append(f'<div><h2>Team profiles</h2>{prof}</div>')
    if notes:
        sections.append(f'<div class="block"><h2>Staff notes</h2><p style="white-space:pre-wrap">'
                        f'{_html.escape(notes)}</p></div>')
    dd = v.dd_tendency_html(df_any_down, opponent, ink)
    if dd:
        sections.append(f'<div class="block"><h2>Down and distance tendencies</h2>{dd}</div>')
    gaps = v.run_gaps_html(df, "offense", opponent, good_high=False, ink=ink)
    if gaps:
        sections.append(f'<div class="block"><h2>Run game by gap</h2>{gaps}</div>')
    zones = v.pass_zones_html(df, "offense", opponent, "Share of throws", good_high=False, ink=ink)
    if zones:
        sections.append(f'<div class="block"><h2>Where they throw</h2>{zones}</div>')
    forms = v.opp_formations_table(df, opponent)
    if not forms.empty:
        fmt = {c: v.FORMATS[c] for c in forms.columns if c in v.FORMATS}
        sections.append(f'<div class="block"><h2>Formations</h2>{_table_html(forms, fmt)}</div>')
    for kind in ("Run", "Pass"):
        calls = v.opp_play_call_table(df, opponent, kind)
        if not calls.empty:
            fmt = {c: v.FORMATS[c] for c in calls.columns if c in v.FORMATS}
            sections.append(f'<div class="block"><h2>{kind} plays</h2>{_table_html(calls, fmt)}</div>')
    best = v.run_pass(df[df["offense"] == opponent]).nlargest(10, "epa")
    if not best.empty:
        sections.append(f'<div class="block"><h2>Their best plays</h2>'
                        f'{_table_html(plays_table(best), {"EPA": "{:+.2f}"}, index=False)}</div>')
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{opponent} scouting report</title>"
            f"<style>{REPORT_CSS}</style></head><body>{''.join(sections)}</body></html>")


# ---------------------------------------------------------------------------
# Tagging coverage
# ---------------------------------------------------------------------------
TAG_CHECKS = [
    # column, label, which plays it should be on
    ("OFF FORM", "Off form", "All run/pass plays"),
    ("PLAY DIR", "Play dir", "All run/pass plays"),
    ("OFF PLAY", "Off play", "All run/pass plays"),
    ("MEN IN BOX", "Men in box", "All run/pass plays"),
    ("GAP", "Gap", "Runs"),
    ("PASS ZONE", "Pass zone", "Passes"),
    ("AIR YARDS", "Air yards", "Passes"),
    ("COVERAGE", "Coverage", "Plays vs DCHS defense"),
]


def _tag_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    rp = v.run_pass(df)
    if scope == "Runs":
        return rp[rp["PLAY TYPE"] == "Run"]
    if scope == "Passes":
        return rp[rp["PLAY TYPE"] == "Pass"]
    if scope == "Plays vs DCHS defense":
        return rp[rp["defense"] == TEAM]
    return rp


@st.cache_data(show_spinner=False)
def tagging_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for gid in games(df):
        g = df[df["game_id"] == gid]
        rows[game_label(df, gid)] = {
            label: (_tag_scope(g, scope)[col].notna().mean() if len(_tag_scope(g, scope)) else np.nan)
            for col, label, scope in TAG_CHECKS
        }
    return pd.DataFrame(rows).T


def render_tagging_coverage(df: pd.DataFrame) -> None:
    st.markdown("**Share of plays tagged, by game**")
    t = tagging_coverage(df)
    if t.empty:
        st.info("No games loaded.")
        return

    def color(col):
        return ["" if pd.isna(x) else
                "background-color:#E1F5EE;color:#085041" if x >= 0.9 else
                "background-color:#FAEEDA;color:#633806" if x >= 0.5 else
                "background-color:#FCEBEB;color:#791F1F" for x in col]

    st.dataframe(t.style.format("{:.0%}", na_rep="–").apply(color, axis=0), width="stretch")
    scopes = "; ".join(f"{label}: {scope.lower()}" for _, label, scope in TAG_CHECKS)
    st.caption(f"Green 90%+, amber 50-89%, red under 50%. Each column is measured against the plays it applies "
               f"to ({scopes}).")

    st.markdown("**Find the untagged plays**")
    c1, c2 = st.columns(2)
    gid = c1.selectbox("Game", games(df)[::-1], format_func=lambda g: game_label(df, g), key="tag_game")
    label = c2.selectbox("Tag", [lab for _, lab, _ in TAG_CHECKS], key="tag_col")
    col, _, scope = next(x for x in TAG_CHECKS if x[1] == label)
    missing = _tag_scope(df[df["game_id"] == gid], scope)
    missing = missing[missing[col].isna()]
    if missing.empty:
        st.success(f"Every play that needs a {label.lower()} tag has one.")
    else:
        st.caption(f"{len(missing)} plays are missing a {label.lower()} tag. Use PLAY # to find them in Hudl.")
        cols = ["PLAY #", "QTR", "DN", "DIST", "YARD LN", "PLAY TYPE", "offense", "OFF FORM", "RESULT"]
        st.dataframe(missing[[c for c in cols if c in missing]], width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Win probability and 4th-down review
#
# The data has no game clock, so the clock is estimated by spreading each
# quarter's scrimmage plays evenly across the 12-minute quarter. That high
# school clock is then scaled by 15/12 for the model (trained on 15-minute
# college quarters), exactly like the 4th Down Bot page. Home/away isn't in
# the data, so every evaluation is neutral-site (home and away averaged).
# ---------------------------------------------------------------------------
HS_QUARTER = 12 * 60


def _with_pre_snap_score(g: pd.DataFrame) -> pd.DataFrame:
    """
    offense_score/defense_score are the score AFTER each play, so a play's
    situation has to use the previous row's score. Adds pre_off_score and
    pre_def_score (score at the snap, from that play's offense's view).
    """
    g = g.sort_values(["game_id", "PLAY #"]).copy()
    pre_team = g.groupby("game_id")["team_score"].shift(1, fill_value=0)
    pre_opp = g.groupby("game_id")["opponent_score"].shift(1, fill_value=0)
    # team_score belongs to the first team in the game_id (Dowling in Dowling games).
    ours = g["offense"] == g["game_id"].astype(str).str.split("_").str[0]
    g["pre_off_score"] = np.where(ours, pre_team, pre_opp)
    g["pre_def_score"] = np.where(ours, pre_opp, pre_team)
    return g


def _scrimmage(g: pd.DataFrame) -> pd.DataFrame:
    g = _with_pre_snap_score(g)
    d = g[g["DN"].isin([1, 2, 3, 4]) & g["YARDLINE_100"].notna() & g["DIST"].notna()
          & g["offense"].notna() & ~g["PLAY TYPE"].isin(KICKOFF_TYPES | TRY_TYPES)]
    return d.copy()


def _estimate_clock(d: pd.DataFrame) -> pd.DataFrame:
    import fourth_down_core as fd

    d = d.copy()
    d["_i"] = d.groupby(["game_id", "QTR"]).cumcount()
    d["_n"] = d.groupby(["game_id", "QTR"])["QTR"].transform("size")
    secs_in_q = HS_QUARTER * (1 - (d["_i"] + 0.5) / d["_n"])
    d["seconds_remaining"] = [fd.model_seconds_remaining(q, s) for q, s in zip(d["QTR"], secs_in_q)]
    d["half_seconds"] = [fd.half_seconds_from_game(s) for s in d["seconds_remaining"]]
    d["clock"] = [f"Q{int(qq)} {int(s // 60)}:{int(s % 60):02d}" for qq, s in zip(d["QTR"], secs_in_q)]
    d["q_seconds"] = secs_in_q  # high school clock: estimated seconds left in the quarter
    return d.drop(columns=["_i", "_n"])


def _predict_wp(d: pd.DataFrame) -> np.ndarray:
    """Offense win probability for each row, neutral site (vectorized predict_wp_chained)."""
    import model_utils as mu

    ytg = d["YARDLINE_100"].astype(float).to_numpy()
    dist = d["DIST"].astype(float).to_numpy()
    down = d["DN"].astype(int).to_numpy()
    sd = (d["pre_off_score"] - d["pre_def_score"]).astype(float).to_numpy()
    half = d["half_seconds"].to_numpy(dtype=float)
    game = d["seconds_remaining"].to_numpy(dtype=float)
    dummies = np.stack([(down == k).astype(float) for k in (1, 2, 3, 4)], axis=1)
    to = np.full(len(d), 3.0)

    if getattr(mu, "_ep_booster", None) is not None and getattr(mu, "_wp_booster", None) is not None:
        import xgboost as xgb
        ep_x = np.column_stack([ytg, dummies, dist, half, sd, to, to, (half <= 120).astype(float),
                                (dist >= ytg).astype(float)])
        ep = mu._ep_booster.predict(xgb.DMatrix(ep_x, feature_names=mu.EP_FEATURE_ORDER))
        ratio = sd / (game / 60 + 1)
        out = []
        for home in (0.0, 1.0):
            wp_x = np.column_stack([ep, sd, ytg, dummies, dist, half, game, to, to, np.full(len(d), home), ratio])
            out.append(mu._wp_booster.predict(xgb.DMatrix(wp_x, feature_names=mu.WP_FEATURE_ORDER)))
        return (out[0] + out[1]) / 2
    ep = np.array([mu._heuristic_ep(y) for y in ytg])
    return np.array([mu._heuristic_wp(s, gs, e) for s, gs, e in zip(sd, game, ep)])


@st.cache_data(show_spinner=False)
def win_probability(g: pd.DataFrame, team: str = TEAM) -> pd.DataFrame:
    """`team`'s win probability before each scrimmage play of one game (column dchs_wp)."""
    d = _estimate_clock(_scrimmage(g))
    if d.empty:
        return d
    off_wp = _predict_wp(d)
    d["dchs_wp"] = np.where(d["offense"] == team, off_wp, 1 - off_wp)
    d["play_n"] = np.arange(1, len(d) + 1)
    return d


def render_win_probability(g: pd.DataFrame, opponent: str, team: str = TEAM) -> None:
    import model_utils as mu

    st.markdown("**Win probability**")
    d = win_probability(g, team)
    if d.empty:
        st.info("No scrimmage plays in this game.")
        return
    us, them = score_for(g.sort_values("PLAY #").iloc[-1], team)
    final = 1.0 if us > them else 0.0 if us < them else 0.5
    line = pd.concat([d[["play_n", "dchs_wp", "clock"]],
                      pd.DataFrame({"play_n": [len(d) + 1], "dchs_wp": [final], "clock": ["Final"]})])
    q_marks = d.groupby("QTR")["play_n"].min().iloc[1:].reset_index()
    chart = alt.layer(
        alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(color=GRAY, strokeDash=[2, 4]).encode(y="y:Q"),
        alt.Chart(q_marks).mark_rule(color=GRAY, opacity=0.4).encode(x="play_n:Q"),
        alt.Chart(q_marks.assign(lab=lambda x: "Q" + x["QTR"].astype(int).astype(str))).mark_text(
            align="left", dx=4, color=GRAY).encode(x="play_n:Q", y=alt.value(10), text="lab:N"),
        alt.Chart(line).mark_line(color="#185FA5", strokeWidth=2.5, interpolate="step-after").encode(
            x=alt.X("play_n:Q", title="Play"),
            y=alt.Y("dchs_wp:Q", title=f"{team} win probability", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".0%")),
            tooltip=[alt.Tooltip("clock:N", title="Est. clock"), alt.Tooltip("dchs_wp:Q", format=".0%", title="WP")]),
    ).properties(height=320)
    st.altair_chart(chart, width="stretch")
    mode = "trained models" if mu.USING_REAL_MODELS else "fallback heuristic (model files not found)"
    st.caption(f"Uses the 4th Down Bot's {mode}, neutral site. There's no game clock in the data, so time is "
               f"estimated by spreading each quarter's plays evenly. Treat it as the shape of the game, not exact "
               f"numbers.")

    nxt = np.append(d["dchs_wp"].to_numpy()[1:], final)
    d = d.assign(swing=nxt - d["dchs_wp"].to_numpy())
    top = d.reindex(d["swing"].abs().sort_values(ascending=False).index).head(5)
    t = plays_table(top).assign(**{"Est. clock": top["clock"].to_numpy(), "Offense": top["offense"].to_numpy(),
                                   "WP swing": top["swing"].to_numpy()})
    t = t[["WP swing", "Offense", "Est. clock"] + [c for c in t.columns if c not in ("Est. clock", "Offense", "WP swing")]]
    st.markdown("**Biggest swing plays**")

    def color(col):
        return ["background-color:#E1F5EE;color:#085041" if x > 0 else "background-color:#FCEBEB;color:#791F1F"
                for x in col]

    st.dataframe(t.style.format(na_rep="").format({"WP swing": "{:+.0%}", "EPA": "{:+.2f}"}, na_rep="")
                 .apply(color, subset=["WP swing"]), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# 4th-down decision review: ledger, map, and one card per decision
# ---------------------------------------------------------------------------
TOSS_UP_PTS = 1.0   # under 1 point of win probability = either call is fine (nfl4th's cutoff)
# Outside this win-probability band the game is effectively decided, so the
# call barely matters and shouldn't count for or against the staff (the same
# kind of filter rbsdm.com uses).
DECIDED_WP = (0.05, 0.95)
OPTION_SHORT = {"Go for it": "Go", "Field goal": "FG", "Punt": "Punt"}


def _spot_text(ytg: float, opp: str) -> str:
    ytg = int(round(ytg))
    if ytg == 50:
        return "the 50"
    return f"own {100 - ytg}" if ytg > 50 else f"{opp} {ytg}"


@st.cache_data(show_spinner="Checking every 4th down...")
def fourth_down_review(df: pd.DataFrame, team: str = TEAM) -> pd.DataFrame:
    """Every 4th down by `team`: what they did vs what the 4th Down Bot recommends (neutral site)."""
    import fourth_down_core as fd
    import model_utils as mu

    parts = []
    for gid in games(df):
        d = _estimate_clock(_scrimmage(df[df["game_id"] == gid]))
        parts.append(d[(d["offense"] == team) & (d["DN"] == 4)])
    d = pd.concat(parts) if parts else pd.DataFrame()
    if d.empty:
        return pd.DataFrame()

    # Every 4th down priced in one batched model call.
    sd = (d["pre_off_score"] - d["pre_def_score"]).to_numpy(dtype=float)
    res = fd.evaluate_many(d["YARDLINE_100"].to_numpy(dtype=float), np.maximum(d["DIST"].to_numpy(dtype=float), 1),
                           sd, d["seconds_remaining"].to_numpy(dtype=float), is_home_pos=0.5)
    calls = fd.best_calls(res, TOSS_UP_PTS)
    be = fd.break_even_conversion(res)

    rows = []
    for i, (_, r) in enumerate(d.iterrows()):
        gid = r["game_id"]
        opp = other_team(df[df["game_id"] == gid], team) or opponent_of(gid)
        one = fd._to_options(res, i)
        wp = one["wp"]
        pt = r["PLAY TYPE"]
        did = "Punt" if pt in PUNT_TYPES else "Field goal" if pt in FG_TYPES else "Go for it"
        best, best_wp = str(calls["best"][i]), float(calls["best_wp"][i])
        chosen = wp.get(did, np.nan)
        left = (best_wp - chosen) * 100 if pd.notna(chosen) else np.nan
        if not DECIDED_WP[0] < best_wp < DECIDED_WP[1]:
            category = "Decided"
        elif pd.isna(chosen):
            category = "Not modeled"
        elif did == best or left < TOSS_UP_PTS:
            category = "Agreed" if did == best else "Toss-up"
        else:
            category = "Costly"
        us, them = int(r["pre_off_score"]), int(r["pre_def_score"])
        score = f"up {us}-{them}" if us > them else f"down {us}-{them}" if us < them else f"tied {us}-{them}"
        rows.append({
            "game": gid, "Game": game_label(df, gid) if team == TEAM else scout_game_label(df, gid), "opp": opp, "Est. clock": r["clock"], "score_text": score,
            "dist": int(r["DIST"]), "ytg": float(r["YARDLINE_100"]),
            "Situation": f"4th & {int(r['DIST'])} at {_spot_text(r['YARDLINE_100'], opp)}",
            "We chose": did, "Model": best, "Strength": mu.strength_tier(float(calls["margin"][i])),
            "WP if we": chosen, "WP if model": best_wp, "WP left (pts)": left, "category": category,
            "options": one["options"], "Result": r["RESULT"], "break_even": float(be[i]),
            "team": team, "who": "We" if team == TEAM else team,
        })
    return pd.DataFrame(rows)


# --- Ledger ------------------------------------------------------------------
def ledger_html(t: pd.DataFrame) -> str:
    who = t["who"].iat[0] if "who" in t and len(t) else "We"
    call_head = "Our call vs model" if who == "We" else f"{who}'s call vs model"
    decided = int((t["category"] == "Decided").sum())
    t = t[t["category"] != "Decided"]
    n = len(t)
    agreed = int((t["category"] == "Agreed").sum())
    costly = t[t["category"] == "Costly"]
    toss = int((t["category"] == "Toss-up").sum())
    left = costly["WP left (pts)"].sum()
    kpi = "".join(
        f'<div style="background:rgba(128,128,128,.08);border-radius:8px;padding:8px 12px">'
        f'<div style="font-size:12px;opacity:.65">{k}</div><div style="font-size:20px;font-weight:600">{v}</div></div>'
        for k, v in [("4th downs" + (f" (+{decided} decided)" if decided else ""), n), ("Agreed with the model", agreed), ("Costly disagreements", len(costly)),
                     ("Win prob. left on the table", f"{left:.1f} pts")])
    seg = lambda c, k: f'<span style="width:{k / n * 100:.1f}%;background:{c}"></span>' if n and k else ""
    bar = (f'<div style="display:flex;height:14px;border-radius:4px;overflow:hidden;margin:12px 0 6px">'
           f'{seg(GREEN, agreed)}{seg(RED, len(costly))}{seg("#B4B2A9", toss)}</div>')
    legend = (f'<div style="display:flex;flex-wrap:wrap;gap:14px;font-size:12px;opacity:.75;margin-bottom:14px">'
              f'<span>■ <span style="color:{GREEN}">Agreed ({agreed})</span></span>'
              f'<span>■ <span style="color:{RED}">Disagreed, cost 1+ point ({len(costly)})</span></span>'
              f'<span>■ <span style="opacity:.8">Disagreed, toss-up under 1 point ({toss})</span></span>'
              + (f'<span style="opacity:.8">Not counted: {decided} with the game already decided '
                 f'(win probability under {DECIDED_WP[0]:.0%} or over {DECIDED_WP[1]:.0%})</span>' if decided else "")
              + '</div>')
    dis = t[t["category"].isin(["Costly", "Toss-up"])].sort_values("WP left (pts)", ascending=False)
    rows = []
    for _, r in dis.iterrows():
        o = r["options"].get(r["Model"], {})
        if r["Model"] == "Go for it":
            odds = f'{o.get("success_prob", 0):.0%} to convert'
        elif r["Model"] == "Field goal":
            odds = f'{o.get("success_prob", 0):.0%} to make'
        else:
            odds = "punt"
        pts = r["WP left (pts)"]
        faded = "" if r["category"] == "Costly" else "opacity:.55;"
        rows.append(
            f'<div style="display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1.1fr) 150px;gap:12px;'
            f'align-items:center;padding:9px 0;border-top:0.5px solid rgba(128,128,128,.3);font-size:13px;{faded}">'
            f'<div><div style="font-weight:600">{r["Situation"]}</div>'
            f'<div style="font-size:12px;opacity:.7">{r["Game"]} · {r["Est. clock"]} · {r["score_text"]}</div></div>'
            f'<div style="font-size:12px"><span style="padding:1px 8px;border-radius:999px;background:rgba(128,128,128,.15)">'
            f'{r.get("who", "We")}: {OPTION_SHORT[r["We chose"]]}</span> → <span style="padding:1px 8px;border-radius:999px;'
            f'background:#E1F5EE;color:#085041">Model: {OPTION_SHORT[r["Model"]]}</span>'
            f'<div style="opacity:.7;margin-top:3px">{odds}</div></div>'
            f'<div><div style="font-weight:600;margin-bottom:3px">{"&lt; 0.1" if pts < 0.05 else f"+{pts:.1f}"} pts</div>'
            f'<div style="height:12px;background:rgba(128,128,128,.12);border-radius:3px">'
            f'<div style="height:12px;border-radius:3px;width:{max(min(pts / 8, 1) * 100, 2):.0f}%;'
            f'background:{RED if r["category"] == "Costly" else "#B4B2A9"}"></div></div></div></div>')
    body = "".join(rows) if rows else '<div style="font-size:13px;opacity:.7">Every decision agreed with the model.</div>'
    return (f'<div style="font-family:inherit"><div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));'
            f'gap:8px">{kpi}</div>{bar}{legend}'
            f'<div style="font-size:12px;opacity:.6;display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1.1fr) '
            f'150px;gap:12px;padding-bottom:4px"><span>Disagreements</span><span>{call_head}</span>'
            f'<span>Win prob. at stake</span></div>{body}</div>')


# --- Map ---------------------------------------------------------------------
MAP_YTG = list(range(1, 100))  # 1-yard resolution; the batched engine prices it in one call
MAP_STEP = MAP_YTG[1] - MAP_YTG[0]
MAP_DIST = list(range(1, 16))
MAP_COLORS = {"G": "#9FE1CB", "F": "#FAC775", "P": "#D3D1C7", "T": "rgba(128,128,128,.10)"}


@st.cache_data(show_spinner=False)
def decision_map_grid(score_diff: int = 0) -> list[list[str]]:
    """Model's call for each (yards to goal, distance): tie game, start of Q3, neutral site."""
    import fourth_down_core as fd

    secs = fd.model_seconds_remaining(3, HS_QUARTER)
    Y, D = np.meshgrid(MAP_YTG, MAP_DIST, indexing="ij")
    calls = fd.best_calls(fd.evaluate_many(Y.ravel(), D.ravel(), score_diff, secs, is_home_pos=0.5), TOSS_UP_PTS)
    code = np.where(calls["toss_up"], "T", [b[0] for b in calls["best"]])  # G / F / P / T
    code = np.where(D.ravel() > Y.ravel(), "", code).reshape(Y.shape)
    return code.tolist()


def decision_map_html(t: pd.DataFrame, grid: list[list[str]]) -> str:
    who = t["who"].iat[0] if "who" in t and len(t) else "We"
    dots_head = "Our decisions (letter = what we did)" if who == "We" else f"{who} decisions (letter = what they did)"
    max_d = len(MAP_DIST)
    cells = []
    for i, ytg in enumerate(MAP_YTG):
        for j, c in enumerate(grid[i]):
            if not c:
                continue
            cells.append(f'<span style="position:absolute;left:{100 - ytg - MAP_STEP / 2}%;width:{MAP_STEP + 0.05}%;bottom:{j / max_d * 100}%;'
                         f'height:{100 / max_d}%;background:{MAP_COLORS[c]}"></span>')
    lines = "".join(f'<span style="position:absolute;top:0;bottom:0;left:{x}%;border-left:{"1.5px" if x == 50 else "0.5px"} '
                    f'solid rgba(255,255,255,.7)"></span>' for x in range(10, 100, 10))
    seen, dots = {}, []
    for _, r in t.iterrows():
        key = (round(r["ytg"]), min(r["dist"], 15))
        k = seen[key] = seen.get(key, 0) + 1
        col = GREEN if r["category"] == "Agreed" else RED if r["category"] == "Costly" else GRAY
        tip = (f'{r["Game"]} {r["Est. clock"]} · {r["score_text"]} · {r["Situation"]}&#10;{r.get("who", "We")}: {r["We chose"]} · '
               f'Model: {r["Model"]}' + ("" if r["category"] == "Agreed" else f' ({r["WP left (pts)"]:.1f} pts)'))
        x = 100 - r["ytg"] + (k - 1) * 2.2
        y = (min(r["dist"], 15) - 0.5) / max_d * 100
        dots.append(f'<span title="{tip}" style="position:absolute;left:{x}%;bottom:{y}%;width:22px;height:22px;'
                    f'margin:0 0 -11px -11px;border-radius:50%;display:flex;align-items:center;justify-content:center;'
                    f'font-size:10px;font-weight:600;color:#fff;background:{col};border:2px solid #fff;cursor:default">'
                    f'{OPTION_SHORT[r["We chose"]][0]}</span>')
    yticks = "".join(f'<span style="position:absolute;right:4px;bottom:{(d - 0.5) / max_d * 100}%;transform:translateY(50%)">'
                     f'{d}{"+" if d == 15 else ""}</span>' for d in (1, 5, 10, 15))
    xticks = "".join(f'<span style="position:absolute;left:{x}%;transform:translateX(-50%)">'
                     f'{"50" if x == 50 else ("Own " + str(x)) if x < 50 else ("Opp " + str(100 - x))}</span>'
                     for x in range(10, 100, 10))
    sw = lambda c, lab: (f'<span style="display:inline-flex;align-items:center;gap:4px"><span style="width:12px;height:12px;'
                         f'border-radius:2px;background:{c};border:0.5px solid rgba(128,128,128,.4)"></span>{lab}</span>')
    dot = lambda c, lab: (f'<span style="display:inline-flex;align-items:center;gap:4px"><span style="width:12px;'
                          f'height:12px;border-radius:50%;background:{c}"></span>{lab}</span>')
    return (
        f'<div style="font-family:inherit;font-size:12px">'
        f'<div style="display:flex;flex-wrap:wrap;gap:14px;margin-bottom:6px;opacity:.8"><span>Model\'s call (tie game, '
        f'start of Q3, neutral site):</span>{sw(MAP_COLORS["G"], "Go")}{sw(MAP_COLORS["F"], "Field goal")}'
        f'{sw(MAP_COLORS["P"], "Punt")}{sw(MAP_COLORS["T"], "Toss-up")}</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:14px;margin-bottom:8px;opacity:.8"><span>{dots_head}:'
        f'</span>{dot(GREEN, "Agreed")}{dot(RED, "Disagreed, cost 1+ point")}'
        f'{dot(GRAY, "Disagreed, toss-up")}</div>'
        f'<div style="display:grid;grid-template-columns:34px 1fr;gap:6px">'
        f'<div style="position:relative;height:330px;opacity:.6">{yticks}</div>'
        f'<div><div style="position:relative;height:330px;border-radius:6px;overflow:hidden">{"".join(cells)}{lines}'
        f'{"".join(dots)}</div><div style="position:relative;height:16px;opacity:.6">{xticks}</div></div></div>'
        f'<div style="opacity:.6;margin-top:6px">Up = longer to go; right = closer to the opponent\'s goal. A dot can '
        f'disagree with the background because the real score and clock change the call. Hover a dot for details.</div>'
        f'</div>')


# --- Decision card -----------------------------------------------------------
def decision_card_html(r: pd.Series) -> str:
    ytg, dist = r["ytg"], r["dist"]
    pos = lambda x: 6 + x * 0.88
    ball, line = pos(100 - ytg), pos(min(100 - ytg + dist, 100))
    field = ('<div style="position:relative;height:28px;border-radius:6px;background:rgba(128,128,128,.10);overflow:hidden">'
             '<span style="position:absolute;top:0;bottom:0;left:0;width:6%;background:rgba(128,128,128,.18)"></span>'
             '<span style="position:absolute;top:0;bottom:0;right:0;width:6%;background:rgba(128,128,128,.18)"></span>'
             + "".join(f'<span style="position:absolute;top:0;bottom:0;left:{pos(x)}%;border-left:'
                       f'{"1px solid rgba(128,128,128,.6)" if x == 50 else "0.5px solid rgba(128,128,128,.35)"}"></span>'
                       for x in range(10, 100, 10))
             + f'<span style="position:absolute;top:5px;height:18px;left:{ball}%;width:{line - ball}%;'
               f'background:rgba(239,159,39,.22)"></span>'
             f'<span style="position:absolute;top:0;bottom:0;left:{ball}%;border-left:2px solid #185FA5"></span>'
             f'<span style="position:absolute;top:0;bottom:0;left:{line}%;border-left:2px solid #EF9F27"></span></div>')
    did, best, cat = r["We chose"], r["Model"], r["category"]
    verb = {"Go for it": "went for it", "Field goal": "kicked the field goal", "Punt": "punted"}[did]
    who = r.get("who", "We") if isinstance(r.get("who", "We"), str) else "We"
    team_name = r.get("team", TEAM) if isinstance(r.get("team", TEAM), str) else TEAM
    short = "Dowling" if team_name == TEAM else team_name
    our_call = "Our call" if who == "We" else f"{who}'s call"
    if cat == "Decided":
        box, text = ("rgba(128,128,128,.12)", "inherit"), (
            f'{who} {verb}. The game was already decided here (win probability {r["WP if model"]:.0%} for the best '
            f'option), so this one doesn\'t count in the ledger.')
    elif cat == "Agreed":
        box, text = ("#E1F5EE", "#085041"), f'{who} {verb}, and the model agrees ({r["Strength"].lower()}).'
    elif cat == "Toss-up":
        box, text = ("rgba(128,128,128,.12)", "inherit"), (f'{who} {verb}; the model slightly prefers '
                                                            f'{best.lower()}, but it\'s a toss-up (under 1 point).')
    elif cat == "Costly":
        box, text = ("#FAEEDA", "#633806"), (f'{who} {verb}. The model says <b>{best.lower()}</b>, worth '
                                             f'+{r["WP left (pts)"]:.1f} win probability points.')
    else:
        box, text = ("rgba(128,128,128,.12)", "inherit"), f"{who} {verb}; that option isn't modeled from this spot."
    opts = []
    for name in ("Go for it", "Field goal", "Punt"):
        o = r["options"].get(name)
        chips = ""
        if name == best:
            chips += '<span style="font-size:11px;padding:1px 8px;border-radius:999px;background:#E1F5EE;color:#085041;margin-right:4px">Model\'s call</span>'
        if name == did:
            chips += '<span style="font-size:11px;padding:1px 8px;border-radius:999px;background:rgba(128,128,128,.18)">{our_call}</span>'
        if o is None:
            why = "Not an option inside the 35" if name == "Punt" else "Out of field goal range"
            opts.append(f'<div style="display:grid;grid-template-columns:120px 1fr 60px;gap:12px;align-items:center;'
                        f'padding:10px 0;border-top:0.5px solid rgba(128,128,128,.3);opacity:.55">'
                        f'<div style="font-weight:600">{name}</div><div style="font-size:13px">{why}</div><div></div></div>')
            continue
        if o["success_prob"] is None:
            bars = (f'<div style="display:flex;height:24px;border-radius:5px;overflow:hidden;font-size:12px">'
                    f'<span style="width:100%;background:rgba(128,128,128,.18);display:flex;align-items:center;'
                    f'padding:0 8px">Punt → {o["wp"]:.0%} WP</span></div>')
        else:
            ok, bad = ("Convert", "Stopped") if name == "Go for it" else ("Make", "Miss")
            p = o["success_prob"]
            bars = (f'<div style="display:flex;height:24px;border-radius:5px;overflow:hidden;font-size:12px">'
                    f'<span style="width:{p * 100:.0f}%;background:#9FE1CB;color:#04342C;display:flex;align-items:center;'
                    f'padding:0 8px;white-space:nowrap;overflow:hidden">{ok} {p:.0%} → {o["wp_success"]:.0%}</span>'
                    f'<span style="width:{(1 - p) * 100:.0f}%;background:#F7C1C1;color:#501313;display:flex;'
                    f'align-items:center;padding:0 8px;white-space:nowrap;overflow:hidden">{bad} → {o["wp_fail"]:.0%}</span></div>')
        opts.append(f'<div style="display:grid;grid-template-columns:120px 1fr 60px;gap:12px;align-items:center;'
                    f'padding:10px 0;border-top:0.5px solid rgba(128,128,128,.3)">'
                    f'<div><div style="font-weight:600">{name}</div>{chips}</div>{bars}'
                    f'<div style="font-size:20px;font-weight:600;text-align:right">{o["wp"]:.0%}</div></div>')
    return (f'<div style="font-family:inherit;max-width:760px">'
            f'<div style="display:flex;flex-wrap:wrap;gap:6px 16px;align-items:baseline;margin-bottom:6px">'
            f'<span style="font-size:20px;font-weight:600">{r["Situation"]}</span>'
            f'<span style="font-size:13px;opacity:.7">{r["Game"]} · {r["Est. clock"]} (est.) · {short} {r["score_text"]}'
            f' · result: {r["Result"]}</span></div>{field}'
            f'<div style="background:{box[0]};color:{box[1]};border-radius:8px;padding:10px 14px;margin:12px 0;'
            f'font-size:14px">{text}</div>{"".join(opts)}'
            f'<div style="font-size:12px;opacity:.6;margin-top:8px">Bar width = chance of each outcome; arrows show '
            f'win probability after it. The number on the right is the option\'s overall win probability. Blue line = '
            f'ball, orange = line to gain.</div></div>')


def render_fourth_down_review(df: pd.DataFrame, team: str = TEAM) -> None:
    t = fourth_down_review(df, team)
    if t.empty:
        st.info(f"No {'Dowling' if team == TEAM else team} 4th downs in the selected games.")
        return
    st.markdown("**Season ledger**" if df["game_id"].nunique() > 1 else "**Decision ledger**")
    st.html(ledger_html(t))
    if df["game_id"].nunique() > 1:
        import breakdowns as bd  # imported here: breakdowns imports this module
        bd.render_aggressiveness(t, games(df))
    st.markdown("**Decision map**")
    st.html(decision_map_html(t, decision_map_grid()))
    st.markdown("**Decision card**")
    order = t.assign(_k=t["category"].map({"Costly": 0, "Toss-up": 1, "Agreed": 2, "Not modeled": 3, "Decided": 4}))
    order = order.sort_values(["_k", "WP left (pts)"], ascending=[True, False])
    labels = {i: f'{r["Game"]} · {r["Est. clock"]} · {r["Situation"]} ({r["category"].lower()})'
              for i, r in order.iterrows()}
    scope = "all" if df["game_id"].nunique() > 1 else str(df["game_id"].iat[0])
    pick = st.selectbox("Pick a decision", list(labels), format_func=labels.get, key=f"g_fd_pick_{scope}_{team}")
    st.html(decision_card_html(t.loc[pick]))
    st.caption("Same math as the 4th Down Bot: score at the snap, estimated clock scaled to the model's 15-minute "
               "quarters, neutral site, 3 timeouts each, no weather. Under 1 point of win probability is a toss-up.")


# ---------------------------------------------------------------------------
# Add a game: run the curation pipeline on a Hudl export and hand back the
# updated curated-pbp.xlsx to commit
# ---------------------------------------------------------------------------
def _read_upload(f) -> pd.DataFrame:
    name = f.name.lower()
    return pd.read_csv(f) if name.endswith(".csv") else pd.read_excel(f)


def process_new_game(existing_path: str, film, offense_film, opponent: str, game_day: _date, week: int):
    """Returns (combined_df, new_game_df, game_id)."""
    import curate_pbp as cp

    date_str = game_day.strftime("%Y_%m_%d")
    plays = _read_upload(film)
    new = cp.curate_play_by_play_data(plays=plays, team=TEAM, opponent=opponent, date=date_str, week=week)
    if offense_film is not None:
        off = cp.curate_play_by_play_data(plays=_read_upload(offense_film), team=TEAM, opponent=opponent,
                                          date=date_str, week=week)
        new = cp.add_dowling_offense_plays(new, off)
    gid = new["game_id"].iat[0]
    existing = pd.read_excel(existing_path)
    existing = existing[existing["game_id"] != gid]  # re-uploading a game replaces it
    combined = pd.concat([existing, new], ignore_index=True, sort=False)
    return combined, new, gid


def to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Sheet1", index=False)
    return buf.getvalue()


def render_add_game(existing_path: str) -> None:
    st.markdown(
        "Upload a game's Hudl export to run it through the same curation code (`curate_pbp.py`) and get back an "
        "updated `curated-pbp.xlsx`. Nothing on the live app changes until you commit that file to GitHub and "
        "reboot the app. Re-uploading a game that's already in the data replaces it."
    )
    c1, c2, c3 = st.columns(3)
    opponent = c1.text_input("Opponent (as it should appear in the app)", key="add_opp")
    game_day = c2.date_input("Game date", key="add_date")
    week = c3.number_input("Week", 1, 20, 1, key="add_week")
    film = st.file_uploader("Game film export (.xlsx or .csv)", type=["xlsx", "xls", "csv"], key="add_film")
    offense_film = st.file_uploader("Dowling offense film export (optional, fills OFF FORM / OFF PLAY for our "
                                    "offense)", type=["xlsx", "xls", "csv"], key="add_off_film")
    if not st.button("Process game", type="primary", disabled=not (film and opponent.strip())):
        if not opponent.strip() or not film:
            st.caption("Add an opponent name and the film export to continue.")
        return
    try:
        combined, new, gid = process_new_game(existing_path, film, offense_film, opponent.strip(), game_day,
                                              int(week))
    except Exception as e:  # noqa: BLE001 — show the coach what went wrong instead of a stack trace
        st.error(f"Couldn't process that file: {e}")
        return
    errors = int((new["drive_result"] == "Error").sum() + (new["SERIES_RESULT"] == "Error").sum())
    last = new.iloc[-1]
    st.success(f"Processed {gid}: {len(new)} plays, final {TEAM} {int(last['team_score'])}, "
               f"{opponent} {int(last['opponent_score'])}.")
    if errors:
        st.warning(f"{errors} rows came out as 'Error' in drive_result/SERIES_RESULT. Check the tagging before "
                   "committing.")
    st.markdown("**Tagging check for this game**")
    st.dataframe(tagging_coverage(new).style.format("{:.0%}", na_rep="–"), width="stretch")
    import qol  # local import: qol imports this module

    qol.render_validation(new, "Possible tagging mistakes in this game")
    st.download_button("Download updated curated-pbp.xlsx", to_xlsx_bytes(combined), file_name="curated-pbp.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    st.caption("Next: replace curated-pbp.xlsx in the GitHub repo with this file, push, then reboot the app.")
