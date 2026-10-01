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
}


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
    """game_ids in week order."""
    g = df.dropna(subset=["game_id"]).groupby("game_id")["WEEK"].min().sort_values()
    return g.index.tolist()


def game_label(df: pd.DataFrame, game_id: str) -> str:
    week = df.loc[df["game_id"] == game_id, "WEEK"].min()
    wk = f"W{int(week)} · " if pd.notna(week) else ""
    return f"{wk}{opponent_of(game_id)}"


def last_updated_text(df: pd.DataFrame) -> str:
    gs = games(df)
    if not gs:
        return "No games loaded"
    last = gs[-1]
    when = game_date_of(last)
    return f"Data through {game_label(df, last)}" + (f" ({when})" if when else "")


# ---------------------------------------------------------------------------
# Drive chart
# ---------------------------------------------------------------------------
YELLOW = "#E8B923"
DRIVE_COLORS = {
    "Touchdown": GREEN, "FG Made": YELLOW, "FG Missed": RED, "Punt": GRAY, "Turnover": RED,
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
        ("Turnovers", int(drives["result"].isin(["Turnover", "Turnover on Downs"]).sum())),
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


def plays_table(d: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in PLAY_COLS if c in d.columns]
    out = d[cols].rename(columns={"epa": "EPA", "GN/LS": "Yards", "YARD LN": "Yard ln"}).copy()
    for c in ("DN", "DIST", "QTR", "WEEK", "Yard ln", "Yards"):
        if c in out:
            out[c] = out[c].astype("Int64")
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


def render_game_recap(df: pd.DataFrame, game_id: str, half: str) -> None:
    """Full recap for one game (or season-wide logs when game_id is None)."""
    g = df if game_id is None else df[df["game_id"] == game_id]
    if half == "First half":
        g = g[g["QTR"] <= 2]
    elif half == "Second half":
        g = g[g["QTR"] >= 3]
    if g.empty:
        st.info("No plays for this selection.")
        return
    opp = None if game_id is None else opponent_of(game_id)
    them = opp or "Opponents"

    if game_id is not None:
        last = g.iloc[-1]
        us, they = int(last["team_score"]), int(last["opponent_score"])
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
        show_plays(f"{them} best plays (our worst on D)", theirs_rp.nlargest(5, "epa"), good_high=False)
        show_plays(f"{them} worst plays (our best on D)", theirs_rp.nsmallest(5, "epa"), good_high=False)

    expl = pd.to_numeric(rp["explosive_play"], errors="coerce") == 1
    show_plays(f"Explosive plays for {TEAM}", rp[expl & (rp["offense"] == TEAM)])
    show_plays(f"Explosive plays allowed", rp[expl & (rp["offense"] != TEAM)], good_high=False)


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
    }


# How far the projected matchup has to be from the baseline to call an edge.
EDGE_THRESHOLDS = {"Rush EPA / play": 0.10, "Pass EPA / play": 0.10, "Success rate": 0.04,
                   "Explosive rate": 0.03, "3rd down conversion": 0.06}


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
        a, b, b0 = off[k], deff[k], base.get(k, np.nan)
        proj, edge = np.nan, ""
        thr = EDGE_THRESHOLDS.get(k)
        if thr and pd.notna(a) and pd.notna(b) and pd.notna(b0):
            proj = a + b - b0
            edge = off_name if proj - b0 >= thr else def_name if b0 - proj >= thr else "Even"
        rows.append((k, a, b, b0, proj, edge))
    return pd.DataFrame(rows, columns=["Metric", off_name, def_name, "Average offense", "Projected", "Edge"]
                        ).set_index("Metric")


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
    """Colors are from Dowling's point of view: green = good for Dowling, red = bad."""
    st.markdown(f"**{title}**")
    off_col, def_col = t.columns[0], t.columns[1]
    shown = t.copy()
    for col in [off_col, def_col, "Average offense", "Projected"]:
        shown[col] = [_fmt_metric(k, x) for k, x in zip(t.index, t[col])]

    def good_for_dowling(offense_above_avg: bool) -> bool:
        return offense_above_avg == dowling_on_offense

    def style_row(row):
        k = row.name
        thr = EDGE_THRESHOLDS.get(k)
        styles = {c: "" for c in shown.columns}
        if thr:
            b0 = t.loc[k, "Average offense"]
            for col in (off_col, def_col, "Projected"):
                x = t.loc[k, col]
                if pd.notna(x) and pd.notna(b0) and abs(x - b0) >= thr:
                    styles[col] = GOOD_CSS if good_for_dowling(x > b0) else BAD_CSS
            edge = t.loc[k, "Edge"]
            if edge and edge != "Even":
                dowling_edge = (edge == off_col) == dowling_on_offense
                styles["Edge"] = GOOD_CSS if dowling_edge else BAD_CSS
        return [styles[c] for c in shown.columns]

    st.dataframe(shown.style.apply(style_row, axis=1), width="stretch")


def render_matchup(df: pd.DataFrame, opponent: str) -> None:
    base = _baselines(df)
    opp_o = _side_metrics(df[df["offense"] == opponent])
    dchs_d = _side_metrics(df[df["defense"] == TEAM])
    dchs_o = _side_metrics(df[df["offense"] == TEAM])
    opp_d = _side_metrics(df[df["defense"] == opponent])
    _show_matchup(f"{opponent} offense vs DCHS defense",
                  _matchup_table(opp_o, dchs_d, base, f"{opponent} offense", "DCHS defense allows"),
                  dowling_on_offense=False)
    _show_matchup(f"DCHS offense vs {opponent} defense",
                  _matchup_table(dchs_o, opp_d, base, "DCHS offense", f"{opponent} defense allows"),
                  dowling_on_offense=True)
    n_games = df.loc[(df["offense"] == opponent) | (df["defense"] == opponent), "game_id"].nunique()
    st.caption(
        "Every number is from the offense's point of view: what the offense gets, and what the defense gives up. "
        "Both are compared to an average offense (EPA 0; rates = every offense in the data combined), so a "
        "defense that allows positive EPA is a weakness. Projected = offense + defense − average: where this "
        "offense should land against this defense. The edge goes to the offense if the projection is clearly "
        "above average, to the defense if clearly below, otherwise Even. Green = good for Dowling, red = bad. "
        f"{opponent}'s numbers come from the {n_games} game(s) in the data that include them."
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
    t.insert(0, "Plays", d["OFF FORM"].value_counts().reindex(forms))
    sty = t.style.format({c: "{:.0%}" for c in covs}, na_rep="–").background_gradient(
        cmap="Blues", subset=list(covs), vmin=0, vmax=1)
    st.dataframe(sty, width="stretch")
    st.caption("Each row is one opponent formation: the share of those snaps Dowling played each coverage. "
               "A dark cell is a coverage we lean on against that look.")


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


def build_report_html(df: pd.DataFrame, df_any_down: pd.DataFrame, opponent: str, filters_text: str) -> str:
    ink = "#31333F"
    sections = [f"<h1>{opponent} offense scouting report</h1>",
                f'<p class="sub">{TEAM} · {filters_text} · generated {_date.today():%b %d, %Y}</p>']
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
# quarter's plays evenly across the quarter. Time is converted the same way
# the 4th Down Bot does it (12-minute current quarter, 15-minute quarters
# after it), and home/away is unknown, so is_home_pos = 0 throughout.
# ---------------------------------------------------------------------------
HS_QUARTER = 12 * 60
MODEL_QUARTER = 15 * 60


def _with_pre_snap_score(g: pd.DataFrame) -> pd.DataFrame:
    """
    offense_score/defense_score are the score AFTER each play, so a play's
    situation has to use the previous row's score. Adds pre_off_score and
    pre_def_score (score at the snap, from that play's offense's view).
    """
    g = g.sort_values(["game_id", "PLAY #"]).copy()
    pre_team = g.groupby("game_id")["team_score"].shift(1, fill_value=0)
    pre_opp = g.groupby("game_id")["opponent_score"].shift(1, fill_value=0)
    ours = g["offense"] == TEAM
    g["pre_off_score"] = np.where(ours, pre_team, pre_opp)
    g["pre_def_score"] = np.where(ours, pre_opp, pre_team)
    return g


def _scrimmage(g: pd.DataFrame) -> pd.DataFrame:
    g = _with_pre_snap_score(g)
    d = g[g["DN"].isin([1, 2, 3, 4]) & g["YARDLINE_100"].notna() & g["DIST"].notna()
          & g["offense"].notna() & ~g["PLAY TYPE"].isin(KICKOFF_TYPES | TRY_TYPES)]
    return d.copy()


def _estimate_clock(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["_i"] = d.groupby("QTR").cumcount()
    d["_n"] = d.groupby("QTR")["QTR"].transform("size")
    secs_in_q = HS_QUARTER * (1 - (d["_i"] + 0.5) / d["_n"])
    q = d["QTR"].clip(upper=4)
    d["seconds_remaining"] = (4 - q) * MODEL_QUARTER + secs_in_q
    d["half_seconds"] = d["seconds_remaining"].clip(upper=1800)
    d["clock"] = [f"Q{int(qq)} {int(s // 60)}:{int(s % 60):02d}" for qq, s in zip(d["QTR"], secs_in_q)]
    return d.drop(columns=["_i", "_n"])


def _predict_wp(d: pd.DataFrame) -> np.ndarray:
    """Offense win probability for each row (vectorized model_utils.predict_wp_chained)."""
    import model_utils as mu

    ytg = d["YARDLINE_100"].astype(float).to_numpy()
    dist = d["DIST"].astype(float).to_numpy()
    down = d["DN"].astype(int).to_numpy()
    sd = (d["pre_off_score"] - d["pre_def_score"]).astype(float).to_numpy()
    half = d["half_seconds"].to_numpy()
    game = d["seconds_remaining"].to_numpy()
    dummies = np.stack([(down == k).astype(float) for k in (1, 2, 3, 4)], axis=1)
    to = np.full(len(d), 3.0)

    if getattr(mu, "_ep_booster", None) is not None and getattr(mu, "_wp_booster", None) is not None:
        import xgboost as xgb
        ep_x = np.column_stack([ytg, dummies, dist, half, sd, to, to, (half <= 120).astype(float),
                                (dist >= ytg).astype(float)])
        ep = mu._ep_booster.predict(xgb.DMatrix(ep_x, feature_names=mu.EP_FEATURE_ORDER))
        ratio = sd / (game / 60 + 1)
        wp_x = np.column_stack([ep, sd, ytg, dummies, dist, half, game, to, to, np.zeros(len(d)), ratio])
        return mu._wp_booster.predict(xgb.DMatrix(wp_x, feature_names=mu.WP_FEATURE_ORDER))
    ep = np.array([mu._heuristic_ep(y) for y in ytg])
    return np.array([mu._heuristic_wp(s, gs, e) for s, gs, e in zip(sd, game, ep)])


def win_probability(g: pd.DataFrame) -> pd.DataFrame:
    """Dowling's win probability before each scrimmage play of one game."""
    d = _estimate_clock(_scrimmage(g))
    if d.empty:
        return d
    off_wp = _predict_wp(d)
    d["dchs_wp"] = np.where(d["offense"] == TEAM, off_wp, 1 - off_wp)
    d["play_n"] = np.arange(1, len(d) + 1)
    return d


def render_win_probability(g: pd.DataFrame, opponent: str) -> None:
    import model_utils as mu

    st.markdown("**Win probability**")
    d = win_probability(g)
    if d.empty:
        st.info("No scrimmage plays in this game.")
        return
    last = g.iloc[-1]
    final = 1.0 if last["team_score"] > last["opponent_score"] else 0.0 if last["team_score"] < last["opponent_score"] else 0.5
    line = pd.concat([d[["play_n", "dchs_wp", "clock"]],
                      pd.DataFrame({"play_n": [len(d) + 1], "dchs_wp": [final], "clock": ["Final"]})])
    q_marks = d.groupby("QTR")["play_n"].min().iloc[1:].reset_index()
    chart = alt.layer(
        alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(color=GRAY, strokeDash=[2, 4]).encode(y="y:Q"),
        alt.Chart(q_marks).mark_rule(color=GRAY, opacity=0.4).encode(x="play_n:Q"),
        alt.Chart(q_marks).mark_text(align="left", dx=4, dy=-120, color=GRAY).encode(
            x="play_n:Q", text=alt.Text("QTR:Q", format="d")),
        alt.Chart(line).mark_line(color="#185FA5", strokeWidth=2.5, interpolate="step-after").encode(
            x=alt.X("play_n:Q", title="Play"),
            y=alt.Y("dchs_wp:Q", title=f"{TEAM} win probability", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".0%")),
            tooltip=[alt.Tooltip("clock:N", title="Est. clock"), alt.Tooltip("dchs_wp:Q", format=".0%", title="WP")]),
    ).properties(height=320)
    st.altair_chart(chart, width="stretch")
    mode = "trained models" if mu.USING_REAL_MODELS else "fallback heuristic (model files not found)"
    st.caption(f"Uses the 4th Down Bot's {mode}. There's no game clock in the data, so time is estimated by "
               f"spreading each quarter's plays evenly. Treat it as the shape of the game, not exact numbers.")

    nxt = np.append(d["dchs_wp"].to_numpy()[1:], final)
    d["swing"] = nxt - d["dchs_wp"].to_numpy()
    top = d.reindex(d["swing"].abs().sort_values(ascending=False).index).head(5)
    t = plays_table(top).assign(**{"Est. clock": top["clock"].to_numpy(), "Offense": top["offense"].to_numpy(),
                                   "WP swing": top["swing"].to_numpy()})
    t = t[["Est. clock", "Offense"] + [c for c in t.columns if c not in ("Est. clock", "Offense", "WP swing")]
          + ["WP swing"]]
    st.markdown("**Biggest swing plays**")

    def color(col):
        return ["background-color:#E1F5EE;color:#085041" if x > 0 else "background-color:#FCEBEB;color:#791F1F"
                for x in col]

    st.dataframe(t.style.format(na_rep="").format({"WP swing": "{:+.0%}", "EPA": "{:+.2f}"}, na_rep="")
                 .apply(color, subset=["WP swing"]), width="stretch", hide_index=True)


def fourth_down_review(df: pd.DataFrame) -> pd.DataFrame:
    """Every Dowling 4th down: what we did vs what the 4th Down Bot recommends."""
    import fourth_down_core as fd

    rows = []
    for gid in games(df):
        g = df[df["game_id"] == gid]
        d = _estimate_clock(_scrimmage(g))
        d = d[(d["offense"] == TEAM) & (d["DN"] == 4)]
        for _, r in d.iterrows():
            pt = r["PLAY TYPE"]
            did = "Punt" if pt in PUNT_TYPES else "Field goal" if pt in FG_TYPES else "Go for it"
            res = fd.evaluate_options(
                yards_to_goal=float(r["YARDLINE_100"]), distance=float(max(r["DIST"], 1)),
                score_diff=float(r["pre_off_score"] - r["pre_def_score"]),
                seconds_remaining=float(r["seconds_remaining"]), is_home_pos=0,
            )
            wp = res["wp"]
            best = max(wp, key=wp.get)
            chosen = wp.get(did, np.nan)
            rows.append({
                "Game": game_label(df, gid), "Est. clock": r["clock"], "Score": f'{int(r["pre_off_score"])}-'
                f'{int(r["pre_def_score"])}', "Down & dist": f'4th & {int(r["DIST"])}',
                "Spot": _yard_label(100 - r["YARDLINE_100"]), "We chose": did, "Bot says": best,
                "WP if we": chosen, "WP if bot": wp[best], "WP left": wp[best] - chosen if pd.notna(chosen) else np.nan,
                "Result": r["RESULT"],
            })
    return pd.DataFrame(rows)


def render_fourth_down_review(df: pd.DataFrame) -> None:
    st.markdown("**4th-down decisions vs the 4th Down Bot**")
    t = fourth_down_review(df)
    if t.empty:
        st.info("No Dowling 4th downs in the selected games.")
        return
    agree = (t["We chose"] == t["Bot says"]).sum()
    c = st.columns(3)
    c[0].metric("4th downs", len(t))
    c[1].metric("Agreed with the bot", f"{agree} of {len(t)}")
    c[2].metric("WP left on the table", f'{t["WP left"].sum() * 100:.1f} pts')

    def color(row):
        ok = row["We chose"] == row["Bot says"]
        style = "background-color:#E1F5EE;color:#085041" if ok else "background-color:#FCEBEB;color:#791F1F"
        return [style if c in ("We chose", "Bot says") else "" for c in row.index]

    st.dataframe(t.style.format({"WP if we": "{:.0%}", "WP if bot": "{:.0%}", "WP left": "{:+.1%}"}, na_rep="–")
                 .apply(color, axis=1), width="stretch", hide_index=True)
    st.caption("Same math as the 4th Down Bot page, using the score at the snap and the estimated clock "
               "(no game clock in the data), "
               "3 timeouts each, and no weather. WP left = how much win probability the bot's choice was worth "
               "over ours; small numbers mean it was close to a toss-up.")


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
    st.download_button("Download updated curated-pbp.xlsx", to_xlsx_bytes(combined), file_name="curated-pbp.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    st.caption("Next: replace curated-pbp.xlsx in the GitHub repo with this file, push, then reboot the app.")
