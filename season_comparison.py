"""
season_comparison.py — one team, two seasons, side by side: the Team Profiles radars for each season (offense,
defense, special teams) and summary tables for offense, defense, and special teams with the change between them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import insights as ins
import profiles
import special_projects as sp
import special_teams as stm
import visuals as v

GOOD = "background-color:#E1F5EE;color:#085041"
BAD = "background-color:#FCEBEB;color:#791F1F"


def team_color(team: str) -> str:
    """Dowling maroon for Dowling; every other team gets the comparison blue (per-team colors can come later)."""
    return profiles.TEAM_COLORS[0] if team == v.TEAM else profiles.TEAM_COLORS[1]


def season_games(df: pd.DataFrame, team: str, season: int) -> pd.DataFrame:
    """Every play of the games `team` played in `season`."""
    d = df[df["SEASON"] == season]
    ids = d.loc[(d["offense"] == team) | (d["defense"] == team), "game_id"].unique()
    return d[d["game_id"].isin(ids)]


def team_seasons(df: pd.DataFrame, team: str) -> list[int]:
    return sorted(int(s) for s in df.loc[(df["offense"] == team) | (df["defense"] == team), "SEASON"].dropna().unique())


# ---------------------------------------------------------------------------
# Offense / defense table
# ---------------------------------------------------------------------------
def _rate(flags: pd.Series) -> tuple[float, int, int]:
    """(rate, made, attempts)."""
    n = len(flags)
    made = int(flags.sum())
    return (made / n if n else np.nan), made, n


def unit_stats(g: pd.DataFrame, team: str, side: str) -> dict:
    """
    One season's numbers for `team`'s offense (side="offense") or for the offenses its defense faced
    (side="defense"). Snaps are runs and passes on 1st-4th down, penalties left out (as on Team Profiles).
    """
    col = "offense" if side == "offense" else "defense"
    snaps = v.run_pass(g[g[col] == team])
    snaps = snaps[snaps["DN"].isin([1, 2, 3, 4]) & (snaps["RESULT"] != "Penalty")]
    early = snaps[snaps["DN"].isin([1, 2])]
    pas, rush = snaps[snaps["PLAY TYPE"] == "Pass"], snaps[snaps["PLAY TYPE"] == "Run"]
    third, fourth = snaps[snaps["DN"] == 3], snaps[snaps["DN"] == 4]
    drives = sp.first_play_drives(g, team, side=side)          # points = what each drive actually scored
    n_drives = len(drives)
    explosive = lambda d: pd.to_numeric(d["explosive_play"], errors="coerce").mean()
    return {
        "EPA/Play": snaps["epa"].mean(),
        "Success Rate": snaps["success"].mean(),
        "EPA/Drive": snaps["epa"].sum() / n_drives if n_drives else np.nan,
        "Points/Drive": drives["points"].mean() if n_drives else np.nan,
        "Early Down EPA": early["epa"].mean(),
        "Early Down Success": early["success"].mean(),
        "Pass EPA/Play": pas["epa"].mean(),
        "Pass Success Rate": pas["success"].mean(),
        "Explosive Pass %": explosive(pas),
        "Rush EPA/Play": rush["epa"].mean(),
        "Rush Success Rate": rush["success"].mean(),
        "Explosive Rush %": explosive(rush),
        "3rd Down Conv. Rate": _rate(third["THIRD_DOWN_CONVERTED"] == 1),
        "Avg 3rd Down Distance": third["DIST"].mean(),
        "4th Down Conv. Rate": _rate(fourth["FOURTH_DOWN_CONVERTED"] == 1),
        "Starting Field Pos.": drives["start"].mean() if n_drives else np.nan,
        "_snaps": len(snaps), "_drives": n_drives,
    }


# (row, section, kind, threshold for a "clear" change, higher is better on offense)
ROWS = [
    ("EPA/Play", "Efficiency", "epa", 0.05, True),
    ("Success Rate", "Efficiency", "pct", 0.02, True),
    ("EPA/Drive", "Efficiency", "epa2", 0.25, True),
    ("Points/Drive", "Efficiency", "num2", 0.25, True),
    ("Early Down EPA", "Efficiency", "epa", 0.05, True),
    ("Early Down Success", "Efficiency", "pct", 0.02, True),
    ("Pass EPA/Play", "Passing", "epa", 0.05, True),
    ("Pass Success Rate", "Passing", "pct", 0.02, True),
    ("Explosive Pass %", "Passing", "pct", 0.02, True),
    ("Rush EPA/Play", "Rushing", "epa", 0.05, True),
    ("Rush Success Rate", "Rushing", "pct", 0.02, True),
    ("Explosive Rush %", "Rushing", "pct", 0.02, True),
    ("3rd Down Conv. Rate", "Situational", "rate", 0.05, True),
    ("Avg 3rd Down Distance", "Situational", "yds", 1.0, False),
    ("4th Down Conv. Rate", "Situational", "rate", 0.10, True),
    ("Starting Field Pos.", "Field position", "spot", 2.0, True),
]


def _value(x):
    return x[0] if isinstance(x, tuple) else x


def _fmt(kind: str, x) -> str:
    val = _value(x)
    if pd.isna(val):
        return "–"
    if kind == "rate":
        return f"{val:.0%} ({x[1]}/{x[2]})"
    return {"epa": f"{val:+.3f}", "epa2": f"{val:+.2f}", "num2": f"{val:.2f}", "pct": f"{val:.1%}",
            "yds": f"{val:.1f} yds", "spot": ins._yard_label(val)}[kind]


def _change(kind: str, a, b) -> str:
    a, b = _value(a), _value(b)
    if pd.isna(a) or pd.isna(b):
        return "–"
    d = b - a
    if kind in ("pct", "rate"):
        return f"{d * 100:+.1f} pts"
    if kind in ("yds", "spot"):
        return f"{d:+.1f} yds"
    return f"{d:+.3f}" if kind == "epa" else f"{d:+.2f}"


def _tone(a, b, threshold: float, higher_better: bool) -> int:
    """+1 clearly better, -1 clearly worse, 0 about the same (or missing)."""
    a, b = _value(a), _value(b)
    if pd.isna(a) or pd.isna(b) or abs(b - a) < threshold:
        return 0
    return 1 if (b > a) == higher_better else -1


def unit_table(a: dict, b: dict, sa: int, sb: int, side: str) -> tuple[pd.DataFrame, list[int]]:
    """(table, tone per row). Defense rows are the offenses it faced, so lower is better there."""
    defense = side == "defense"
    rows, tones = [], []
    for name, section, kind, thr, higher in ROWS:
        label = f"Opp. {name}" if defense and name == "Starting Field Pos." else name
        rows.append({"Section": section, "Stat": label, str(sa): _fmt(kind, a[name]), str(sb): _fmt(kind, b[name]),
                     "Change": _change(kind, a[name], b[name])})
        tones.append(_tone(a[name], b[name], thr, higher != defense))
    return pd.DataFrame(rows), tones


# ---------------------------------------------------------------------------
# Special teams table
# ---------------------------------------------------------------------------
def st_stats(g: pd.DataFrame, team: str) -> dict:
    """{(unit, stat): (text, value or NaN, kind, threshold)}; every value is signed so higher = better for `team`."""
    ev = stm.kick_events(g)
    ko_k = ev[(ev["kind"] == "Kickoff") & (ev["kicker"] == team)]
    ko_r = ev[(ev["kind"] == "Kickoff") & (ev["receiver"] == team)]
    pu_k = ev[(ev["kind"] == "Punt") & (ev["kicker"] == team)]
    pu_r = ev[(ev["kind"] == "Punt") & (ev["receiver"] == team)]
    fg = g[g["PLAY TYPE"].isin(stm.FGS) & (g["offense"] == team)]
    made = fg["RESULT"] == "Good"
    fg_dist = fg["YARDLINE_100"] + 17
    xp = g[g["PLAY TYPE"].isin(stm.XPS) & (g["offense"] == team) & (g["RESULT"] != "Penalty")]
    two = g[g["PLAY TYPE"].isin(["2 Pt.", "2 Pt. Block", "2 Pt. Defend"]) & (g["offense"] == team)]
    games = g["game_id"].nunique()
    kicks_vs = g[g["PLAY TYPE"].isin(stm.FGS | stm.XPS | {"Punt", "Punt Rec"}) & (g["defense"] == team)]
    blocks = int((kicks_vs["RESULT"] == "Block").sum())
    epa = profiles.team_metrics(profiles.standardize_hudl(g, team), team)["special_teams"]

    def pct(s):
        return (f"{s.mean():.0%} ({int(s.sum())}/{len(s)})", s.mean()) if len(s) else ("–", np.nan)

    def spot(x, sign=1):
        return (ins._yard_label(x.mean()), sign * x.mean()) if len(x) else ("–", np.nan)

    def per_kick(key):
        val, n = epa.get(key, (np.nan, 0))
        return (f"{val:+.2f} ({n})", val) if pd.notna(val) and n else ("–", np.nan)

    rows = {
        ("Field goals", "Made – Att"): ((f"{int(made.sum())}–{len(fg)}", np.nan), "count", None),
        ("Field goals", "FG %"): (pct(made), "pct", 0.05),
        ("Field goals", "Long make"): ((f"{int(fg_dist[made].max())} yds", fg_dist[made].max()) if made.any()
                                       else ("–", np.nan), "yds", 3),
        ("Field goals", "EPA per kick"): (per_kick("fg_epa"), "epa", 0.2),
        ("Tries", "PAT %"): (pct(xp["RESULT"] == "Good"), "pct", 0.03),
        ("Tries", "2-pt %"): (pct(two["RESULT"] == "Good"), "pct", 0.10),
        ("Kickoffs", "Touchback %"): (pct(ko_k["result"] == "Touchback"), "pct", 0.05),
        ("Kickoffs", "Opp. avg start"): (spot(ko_k["start"], -1), "spot", 2),
        ("Kickoffs", "EPA per kick"): (per_kick("ko_epa"), "epa", 0.1),
        ("Kick returns", "Avg start"): (spot(ko_r["start"]), "spot", 2),
        ("Kick returns", "EPA per return"): (per_kick("ko_ret_epa"), "epa", 0.1),
        ("Punting", "Net avg"): ((f"{pu_k['net'].mean():.1f} yds", pu_k["net"].mean()) if len(pu_k) else ("–", np.nan),
                                 "yds", 2),
        ("Punting", "Inside the 20 %"): (pct(pu_k["start"] <= 20), "pct", 0.05),
        ("Punting", "EPA per punt"): (per_kick("punt_epa"), "epa", 0.1),
        ("Punt returns", "Opp. net avg"): ((f"{pu_r['net'].mean():.1f} yds", -pu_r["net"].mean()) if len(pu_r)
                                           else ("–", np.nan), "yds", 2),
        ("Punt returns", "Avg start"): (spot(pu_r["start"]), "spot", 2),
        ("Punt returns", "EPA per return"): (per_kick("punt_ret_epa"), "epa", 0.1),
        ("Blocks", "Kicks blocked / game"): ((f"{blocks / games:.2f}", blocks / games) if games else ("–", np.nan),
                                             "num2", 0.1),
    }
    return {k: (txt, val, kind, thr) for k, ((txt, val), kind, thr) in rows.items()}


def st_table(a: dict, b: dict, sa: int, sb: int) -> tuple[pd.DataFrame, list[int]]:
    rows, tones = [], []
    for (unit, stat), (txt_a, val_a, kind, thr) in a.items():
        txt_b, val_b = b[(unit, stat)][:2]
        chg, tone = "–", 0
        if kind != "count" and pd.notna(val_a) and pd.notna(val_b):
            d = val_b - val_a
            # "Opp." rows are stored negated (so higher = better); show the change in the shown numbers
            shown = -d if stat.startswith("Opp.") else d
            chg = {"pct": f"{shown * 100:+.0f} pts", "yds": f"{shown:+.1f} yds", "spot": f"{shown:+.1f} yds",
                   "epa": f"{shown:+.2f}", "num2": f"{shown:+.2f}"}[kind]
            tone = (1 if d > 0 else -1) if abs(d) >= thr else 0
        rows.append({"Unit": unit, "Stat": stat, str(sa): txt_a, str(sb): txt_b, "Change": chg})
        tones.append(tone)
    return pd.DataFrame(rows), tones


def _styled(t: pd.DataFrame, tones: list[int]):
    return t.style.apply(lambda _: [GOOD if s > 0 else BAD if s < 0 else "" for s in tones], subset=["Change"],
                         axis=0)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render_page(df_all: pd.DataFrame) -> None:
    teams = sorted({t for t in df_all["offense"].dropna().unique() if t}, key=lambda t: (t != v.TEAM, t))
    c = st.columns([2, 1, 1, 2])
    team = c[0].selectbox("Team", teams, key="sc_team")
    seasons = team_seasons(df_all, team)
    if len(seasons) < 2:
        st.info(f"{team} has film from {'only ' + str(seasons[0]) if seasons else 'no season'}, so there's no "
                "second season to compare yet.")
        return
    sa = c[1].selectbox("Season", seasons, index=len(seasons) - 2, key=f"sc_a_{team}")
    sb = c[2].selectbox("Compared to", seasons, index=len(seasons) - 1, key=f"sc_b_{team}")
    same_weeks = c[3].toggle("Same weeks only", key="sc_same_weeks",
                             help="Compare only the weeks both seasons have film for, e.g. Weeks 1-3.")
    ga, gb = season_games(df_all, team, sa), season_games(df_all, team, sb)
    if same_weeks:
        weeks = set(ga["WEEK"]) & set(gb["WEEK"])
        ga, gb = ga[ga["WEEK"].isin(weeks)], gb[gb["WEEK"].isin(weeks)]
    na, nb = ga["game_id"].nunique(), gb["game_id"].nunique()
    if not na or not nb:
        st.info("No games in common weeks for these two seasons." if same_weeks else "No games for one of the seasons.")
        return
    plural = lambda n: f"{n} game{'' if n == 1 else 's'}"
    weeks_note = f" (Weeks {', '.join(str(int(w)) for w in sorted(set(ga['WEEK'])))})" if same_weeks else ""
    st.caption(f"{team}: {plural(na)} of film in {sa} · {plural(nb)} in {sb}{weeks_note}. Green/red in the tables = "
               f"clearly better/worse in {sb} than {sa}; small changes aren't colored.")

    st.subheader("Profiles")
    try:
        bench = profiles.load_benchmarks()
    except FileNotFoundError:
        st.error(f"{profiles.BENCHMARK_PATH} not found, so the radars can't be drawn.")
        bench = None
    if bench is not None:
        stats = profiles.benchmark_stats(bench)
        # Special teams are centered on the average kick across all our film, so both seasons share one center.
        stats = profiles.hs_centered_stats(stats, profiles.hs_special_teams_averages(
            profiles.standardize_hudl(df_all, team)))
        frames = {}
        for s, g in ((sa, ga), (sb, gb)):
            m = profiles.team_metrics(profiles.standardize_hudl(g, team), team)
            frames[s] = {u: profiles.score_team(m, stats, u) for u in profiles.UNITS}
        color = team_color(team)
        for unit, label in (("offense", "Offense"), ("defense", "Defense"), ("special_teams", "Special Teams")):
            for col, s, n in zip(st.columns(2), (sa, sb), (na, nb)):
                with col:
                    profiles.render_radar(f"{team} {label} · {s}", plural(n), [(str(s), frames[s][unit])],
                                          height=520, colors=[color], width=620)
        st.caption(f"Same charts as Team Profiles: each axis is the distance from the {int(bench['season'].iat[0])} "
                   "FBS median in FBS standard deviations (special teams: from the average kick in our film); "
                   "farther out is better. Hollow points are small samples.")

    height = lambda t: 35 * (len(t) + 1) + 3
    for side, title, note in (
            ("offense", "Offense", "Runs and passes on 1st-4th down (penalties left out). Early downs = 1st and 2nd. "
                                   "Explosive = a run of 10+ or a pass of 20+. Points/drive counts what each drive "
                                   "actually scored."),
            ("defense", "Defense", "What opponents did against this defense, so lower is better (green), except "
                                   "Avg 3rd Down Distance, where longer is better.")):
        a, b = unit_stats(ga, team, side), unit_stats(gb, team, side)
        t, tones = unit_table(a, b, sa, sb, side)
        st.subheader(title)
        st.dataframe(_styled(t, tones), hide_index=True, width="stretch", height=height(t))
        st.caption(f"{note} {sa}: {a['_snaps']} snaps, {a['_drives']} drives · {sb}: {b['_snaps']} snaps, "
                   f"{b['_drives']} drives.")
    t, tones = st_table(st_stats(ga, team), st_stats(gb, team), sa, sb)
    st.subheader("Special Teams")
    st.dataframe(_styled(t, tones), hide_index=True, width="stretch", height=height(t))
    st.caption("Kicks are measured by where the next drive started, since kick and return yards are rarely tagged. "
               "EPA per kick is the same as the Special Teams page and the radar (number of kicks in parentheses): "
               "above 0 is good for this team.")
