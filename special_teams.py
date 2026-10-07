"""
special_teams.py — the Special Teams page.

Every kick is matched to the next scrimmage snap in the same game, which is
where the receiving team actually starts. That gives starting field position
after kickoffs, net punting (punt spot to the opponent's first snap), and
returns, without needing KICK YARDS / RET YARDS to be tagged.
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import insights as ins

TEAM = ins.TEAM
KICKOFFS = {"KO", "KO Rec"}
PUNTS = {"Punt", "Punt Rec"}
FGS = {"FG", "FG Block"}
XPS = {"Extra Pt.", "Extra Pt. Block"}
TWO_PTS = {"2 Pt.", "2 Pt. Block", "2 Pt. Defend"}
HS_TOUCHBACK = 20

RESULT_COLORS = {"Downed": "#1D9E75", "Fair Catch": "#378ADD", "Out of Bounds": "#7F77DD", "Touchback": "#888780",
                 "Return": "#EF9F27", "Penalty": "#B4B2A9", "Fumble": "#E24B4A", "Block": "#E24B4A"}


@st.cache_data(show_spinner=False)
def kick_events(df: pd.DataFrame) -> pd.DataFrame:
    """One row per kickoff and punt, with where the receiving team's next snap started."""
    rows = []
    for gid, g in df.groupby("game_id", sort=False):
        g = g.sort_values("PLAY #").reset_index(drop=True)
        opp = ins.other_team(g, TEAM) or ins.opponent_of(gid)
        week = g["WEEK"].iat[0]
        scrim = g.index[g["DN"].isin([1, 2, 3, 4]) & ~g["PLAY TYPE"].isin(KICKOFFS | XPS | TWO_PTS)
                        & g["YARDLINE_100"].notna()].to_numpy()
        for i, r in g[g["PLAY TYPE"].isin(KICKOFFS | PUNTS)].iterrows():
            later = scrim[scrim > i]
            if not len(later):
                continue
            nxt = g.loc[later[0]]
            start = 100 - nxt["YARDLINE_100"]           # yards from the receiver's own goal
            if r["PLAY TYPE"] in KICKOFFS:
                receiver = nxt["offense"]
                kicker = ins.other_team(g, receiver)   # whoever didn't receive it (works for any two teams)
                if kicker is None:
                    continue
                rows.append({"game": gid, "week": week, "opp": opp, "kind": "Kickoff", "kicker": kicker,
                             "receiver": receiver, "QTR": r["QTR"], "spot": np.nan, "start": start,
                             "net": np.nan, "result": r["RESULT"], "ret": r.get("RET YARDS", np.nan)})
            else:
                if r["offense"] == nxt["offense"]:
                    continue                             # fake / muffed / penalty re-kick: not a change of possession
                rows.append({"game": gid, "week": week, "opp": opp, "kind": "Punt", "kicker": r["offense"],
                             "receiver": nxt["offense"], "QTR": r["QTR"], "spot": 100 - r["YARDLINE_100"],
                             "start": start, "net": r["YARDLINE_100"] - start, "result": r["RESULT"],
                             "ret": r.get("RET YARDS", np.nan)})
    return pd.DataFrame(rows)


def _card(label: str, value: str, sub: str) -> str:
    return (f'<div style="background:rgba(128,128,128,.08);border-radius:8px;padding:10px 12px">'
            f'<div style="font-size:12px;opacity:.7">{label}</div>'
            f'<div style="font-size:20px;font-weight:600">{value}</div>'
            f'<div style="font-size:12px;opacity:.7;margin-top:2px">{sub}</div></div>')


def _own(x: float) -> str:
    x = int(round(x))
    return "50" if x == 50 else f"own {x}" if x < 50 else f"opp {100 - x}"


def render_unit_cards(df: pd.DataFrame, ev: pd.DataFrame, team: str = TEAM) -> None:
    ko_k = ev[(ev["kind"] == "Kickoff") & (ev["kicker"] == team)]
    ko_r = ev[(ev["kind"] == "Kickoff") & (ev["receiver"] == team)]
    pu_k = ev[(ev["kind"] == "Punt") & (ev["kicker"] == team)]
    pu_r = ev[(ev["kind"] == "Punt") & (ev["receiver"] == team)]
    fg = df[df["PLAY TYPE"].isin(FGS) & (df["offense"] == team)]
    xp = df[df["PLAY TYPE"].isin(XPS) & (df["offense"] == team) & (df["RESULT"] != "Penalty")]
    fg_dist = fg["YARDLINE_100"] + 17
    made = fg["RESULT"] == "Good"
    long_make = f"long make {int(fg_dist[made].max())} yds" if made.any() else "no makes yet"
    cards = [
        _card("Kickoffs", f"{(ko_k['result'] == 'Touchback').mean():.0%} TB" if len(ko_k) else "–",
              f"{len(ko_k)} kicks · opponents start at their {_own(ko_k['start'].mean()) if len(ko_k) else '–'}"),
        _card("Kick returns", _own(ko_r["start"].mean()).capitalize() if len(ko_r) else "–",
              f"average start · {len(ko_r)} kickoffs received"),
        _card("Punting", f"{pu_k['net'].mean():.1f} net" if len(pu_k) else "–",
              f"{len(pu_k)} punts · {(pu_k['start'] <= 20).mean():.0%} inside the 20" if len(pu_k) else "no punts"),
        _card("Punt returns", _own(pu_r["start"].mean()).capitalize() if len(pu_r) else "–",
              f"average start · opponents net {pu_r['net'].mean():.1f}" if len(pu_r) else "no punts received"),
        _card("Field goals", f"{int(made.sum())}/{len(fg)}" if len(fg) else "–", long_make),
        _card("PATs", f"{int((xp['RESULT'] == 'Good').sum())}/{len(xp)}" if len(xp) else "–", "extra point kicks"),
    ]
    st.html(f'<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px">{"".join(cards)}</div>')


@st.cache_data(show_spinner=False)
def field_position_by_game(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gid in ins.games(df):
        g = df[df["game_id"] == gid]
        opp = ins.other_team(g, TEAM) or ins.opponent_of(gid)
        ours, theirs = ins.drive_summary(g, TEAM), ins.drive_summary(g, opp)
        rows.append({"game": gid, "Game": ins.game_label(df, gid),
                     "us": ours["start"].mean() if len(ours) else np.nan,
                     "them": theirs["start"].mean() if len(theirs) else np.nan,
                     "our_drives": len(ours), "their_drives": len(theirs)})
    return pd.DataFrame(rows)


def render_field_position_battle(df: pd.DataFrame) -> None:
    st.markdown("**Field position battle: average drive start**")
    fp = field_position_by_game(df).dropna(subset=["us", "them"])
    if fp.empty:
        st.info("No drives in the selected games.")
        return
    lo = max(0, min(fp["us"].min(), fp["them"].min()) - 5)
    hi = min(100, max(fp["us"].max(), fp["them"].max()) + 5)
    p = lambda x: (x - lo) / (hi - lo) * 100
    rows = []
    for _, r in fp.iterrows():
        gap = r["us"] - r["them"]
        left, right = sorted([p(r["us"]), p(r["them"])])
        rows.append(
            f'<div style="opacity:.75">{r["Game"]}</div>'
            f'<div style="position:relative;height:22px">'
            f'<span style="position:absolute;top:10px;height:2px;left:{left}%;width:{right - left}%;background:rgba(128,128,128,.5)"></span>'
            f'<span title="Opponent: own {r["them"]:.1f}" style="position:absolute;top:4px;left:{p(r["them"])}%;width:14px;height:14px;margin-left:-7px;border-radius:50%;background:#B4B2A9"></span>'
            f'<span title="Dowling: own {r["us"]:.1f}" style="position:absolute;top:4px;left:{p(r["us"])}%;width:14px;height:14px;margin-left:-7px;border-radius:50%;background:#185FA5"></span></div>'
            f'<div style="font-weight:600;color:{"#1D9E75" if gap >= 0 else "#E24B4A"}">{gap:+.1f}</div>')
    ticks = "".join(f'<span style="position:absolute;left:{p(x)}%;transform:translateX(-50%)">{x}</span>'
                    for x in range(int(np.ceil(lo / 5) * 5), int(hi) + 1, 5))
    st.html(
        '<div style="font-family:inherit;font-size:13px">'
        '<div style="display:flex;gap:16px;font-size:12px;opacity:.75;margin-bottom:6px">'
        '<span>● <span style="color:#185FA5">Dowling</span></span><span>● <span style="opacity:.7">Opponent</span></span>'
        '<span style="margin-left:auto">Yard line from own goal · number = Dowling advantage per drive</span></div>'
        f'<div style="display:grid;grid-template-columns:130px 1fr 56px;gap:8px 12px;align-items:center">{"".join(rows)}'
        f'<div></div><div style="position:relative;height:14px;font-size:11px;opacity:.6">{ticks}</div><div></div></div></div>')
    st.caption("Every possession change counts (kicks, returns, turnovers, downs), but kicks and returns are the "
               "biggest piece. Breakdown by source is in the table below.")
    ev = kick_events(df)
    rows = []
    for _, r in fp.iterrows():
        e = ev[ev["game"] == r["game"]]
        k = lambda kind, who, col: e[(e["kind"] == kind) & (e[who] == TEAM)][col].mean()
        rows.append({"Game": r["Game"], "Drive start (us)": r["us"], "Drive start (them)": r["them"],
                     "Advantage": r["us"] - r["them"],
                     "Our net punt": k("Punt", "kicker", "net"), "Their net punt": k("Punt", "receiver", "net"),
                     "Our KO start": k("Kickoff", "receiver", "start"),
                     "Their KO start": e[(e["kind"] == "Kickoff") & (e["kicker"] == TEAM)]["start"].mean()})
    t = pd.DataFrame(rows).set_index("Game")
    st.dataframe(t.style.format({c: ("{:+.1f}" if c == "Advantage" else "{:.1f}") for c in t.columns}, na_rep="–"),
                 width="stretch")


def render_fg_range(df: pd.DataFrame) -> None:
    import fourth_down_core as fd

    st.markdown("**Field goal range vs the 4th Down Bot's model**")
    fg = df[df["PLAY TYPE"].isin(FGS) & (df["offense"] == TEAM)].copy()   # Dowling's kicks only
    fg["Distance"] = fg["YARDLINE_100"] + 17
    fg["Result"] = np.where(fg["RESULT"] == "Good", "Made", "Missed")
    fg["y"] = np.where(fg["Result"] == "Made", 1.0, 0.0)
    curve = pd.DataFrame({"Distance": np.arange(18, 63)})
    curve["Make %"] = [fd.fg_prob(d - 17) for d in curve["Distance"]]
    line = alt.Chart(curve).mark_line(color="#185FA5", strokeWidth=2.5).encode(
        x=alt.X("Distance:Q", title="Kick distance (yards)", scale=alt.Scale(domain=[18, 62])),
        y=alt.Y("Make %:Q", title="Make probability", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[-0.05, 1.05])),
        tooltip=[alt.Tooltip("Distance:Q"), alt.Tooltip("Make %:Q", format=".0%")])
    pts = alt.Chart(fg).mark_point(size=110, filled=True, opacity=0.9).encode(
        x="Distance:Q", y="y:Q",
        color=alt.Color("Result:N", scale=alt.Scale(domain=["Made", "Missed"], range=["#1D9E75", "#E24B4A"]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Distance", "Result", alt.Tooltip("WEEK:Q", title="Week")])
    layers = [line, pts] if len(fg) else [line]
    st.altair_chart(alt.layer(*layers).properties(height=300), width="stretch")
    if fg.empty:
        st.caption("No Dowling field goal attempts in the selected weeks.")
    st.caption("The curve is the make probability the 4th Down Bot uses for every kick-or-go decision. As the "
               "season adds kicks, this shows whether it's too optimistic or too pessimistic about your kicker.")


def render_punt_map(ev: pd.DataFrame, team: str = TEAM) -> None:
    st.markdown(f"**Punt map: {team} punts**")
    p = ev[(ev["kind"] == "Punt") & (ev["kicker"] == team)].copy()
    if p.empty:
        st.info("No punts in the selected games.")
        return
    p["land"] = 100 - p["start"]               # opponent's start, in the punting team's coordinates
    pos = lambda x: 4 + x * 0.92
    lines = "".join(f'<span style="position:absolute;top:0;bottom:0;left:{pos(x)}%;border-left:'
                    f'{"1px solid rgba(128,128,128,.55)" if x == 50 else "0.5px solid rgba(128,128,128,.3)"}"></span>'
                    for x in range(10, 100, 10))
    rows = []
    for _, r in p.iterrows():
        a, b = pos(r["spot"]), pos(r["land"])
        c = RESULT_COLORS.get(r["result"], "#888780")
        tip = f'Week {int(r["week"])} vs {r["opp"]}: punt from {_own(r["spot"])}, {r["opp"]} took over at their own {r["start"]:.0f} ({r["result"]}, net {r["net"]:.0f})'
        rows.append(
            f'<div style="font-size:12px;opacity:.7">W{int(r["week"])} Q{int(r["QTR"])}</div>'
            f'<div style="position:relative;height:20px;background:rgba(128,128,128,.07);border-radius:4px" title="{tip}">{lines}'
            f'<span style="position:absolute;top:9px;height:2px;left:{min(a, b)}%;width:{abs(b - a)}%;background:{c}"></span>'
            f'<span style="position:absolute;top:5px;left:{a}%;width:10px;height:10px;margin-left:-5px;border-radius:50%;background:#185FA5"></span>'
            f'<span style="position:absolute;top:3px;left:{b}%;margin-left:-6px;width:0;height:0;border-top:7px solid transparent;'
            f'border-bottom:7px solid transparent;border-left:12px solid {c}"></span></div>'
            f'<div style="font-size:12px">{r["net"]:.0f} net · {r["result"]}</div>')
    legend = "".join(f'<span style="display:inline-flex;align-items:center;gap:4px"><span style="width:10px;height:10px;'
                     f'border-radius:2px;background:{c}"></span>{k}</span>'
                     for k, c in RESULT_COLORS.items() if k in set(p["result"]))
    ticks = "".join(f'<span style="position:absolute;left:{pos(x)}%;transform:translateX(-50%)">'
                    f'{"G" if x in (0, 100) else (x if x <= 50 else 100 - x)}</span>' for x in range(0, 101, 10))
    st.html(f'<div style="font-family:inherit"><div style="display:flex;flex-wrap:wrap;gap:12px;font-size:12px;'
            f'opacity:.8;margin-bottom:6px"><span>● <span style="color:#185FA5">punt spot</span></span>{legend}'
            f'<span style="margin-left:auto">Arrow = where the opponent took over · driving left to right</span></div>'
            f'<div style="display:grid;grid-template-columns:56px 1fr 120px;gap:5px 10px;align-items:center">'
            f'<div></div><div style="position:relative;height:14px;font-size:11px;opacity:.6">{ticks}</div><div></div>'
            f'{"".join(rows)}</div></div>')


def render_kick_logs(df: pd.DataFrame, ev: pd.DataFrame) -> None:
    fg = df[df["PLAY TYPE"].isin(FGS | XPS | TWO_PTS)].copy()
    fg["Team"] = fg["offense"]
    fg["Kick"] = fg["PLAY TYPE"].map(lambda x: "Field goal" if x in FGS else "PAT" if x in XPS else "2-point try")
    fg["Distance"] = np.where(fg["Kick"] == "Field goal", fg["YARDLINE_100"] + 17, np.nan)
    with st.expander("Field goal and PAT log"):
        st.dataframe(fg[["WEEK", "QTR", "Team", "Kick", "Distance", "RESULT"]].rename(columns={"RESULT": "Result"})
                     .style.format({"Distance": "{:.0f}"}, na_rep=""), width="stretch", hide_index=True)
    with st.expander("Kickoff and punt log"):
        t = ev.assign(Week=ev["week"].astype(int))[["Week", "opp", "kind", "kicker", "receiver", "start", "net", "result"]]
        t = t.rename(columns={"opp": "Opponent", "kind": "Kick", "kicker": "Kicking team", "receiver": "Receiving team",
                              "start": "Receiver's start (own yd)", "net": "Net punt", "result": "Result"})
        st.dataframe(t.style.format({"Receiver's start (own yd)": "{:.0f}", "Net punt": "{:.0f}"}, na_rep=""),
                     width="stretch", hide_index=True)


@st.cache_data(show_spinner=False)
def _kick_tables(df: pd.DataFrame) -> dict:
    import profiles
    return profiles.standardize_hudl(df, TEAM)


def render_st_epa(df: pd.DataFrame, df_all_games: pd.DataFrame | None = None) -> None:
    """
    Dowling's special teams EPA by unit, from Dowling's point of view, with the high school average next to it.

    Kicks are priced exactly like the Team Profiles special teams axes (profiles.add_st_epa): one neutral
    context (tie game, middle of a half), a kickoff against the touchback, a punt from the punter's 4th-down
    expected points to the receiver's first snap, a field goal against the expected points before the kick.
    The HS average is the same number over every kick in our film (df_all_games: Dowling games and scout film),
    which is also the middle ring of the profile's special teams radar, so the two pages always agree.
    PATs and 2-point tries aren't on the radar; they use the play-by-play EPA.
    """
    import profiles

    st.markdown("**Special teams EPA by unit**")
    df_all_games = df if df_all_games is None else df_all_games
    ours_t, all_t = _kick_tables(df), _kick_tables(df_all_games)
    hs = profiles.hs_special_teams_averages(all_t)
    k, f = ours_t["kicks"], ours_t["fgs"]
    is_k = lambda kind, who: k[(k["kind"] == kind) & (k[who] == TEAM)]["epa"].dropna()
    tries = lambda d: d[d["PLAY TYPE"].isin(XPS | TWO_PTS) & d["epa"].notna()]
    our_tries = tries(df)[tries(df)["offense"] == TEAM]["epa"]
    units = [
        ("Punts", is_k("punt", "kicker"), hs["punt_epa"]),
        ("Punt returns", -is_k("punt", "receiver"), hs["punt_ret_epa"]),
        ("Kickoffs", is_k("ko", "kicker"), hs["ko_epa"]),
        ("Kickoff returns", -is_k("ko", "receiver"), hs["ko_ret_epa"]),
        ("Field goals", f.loc[f["kicker"] == TEAM, "epa"].dropna(), hs["fg_epa"]),
        ("PATs and 2-point tries", our_tries, tries(df_all_games)["epa"].mean()),
    ]
    rows = [{"Unit": name, "Plays": len(e), "EPA per play": e.mean() if len(e) else np.nan, "HS average": avg,
             "vs HS average": (e.mean() - avg) if len(e) and pd.notna(avg) else np.nan}
            for name, e, avg in units]
    t = pd.DataFrame(rows).set_index("Unit")

    def color(col):
        return ["" if pd.isna(x) else "background-color:#E1F5EE;color:#085041" if x >= 0.10 else
                "background-color:#FCEBEB;color:#791F1F" if x <= -0.10 else "" for x in col]

    st.dataframe(t.style.format({"EPA per play": "{:+.2f}", "HS average": "{:+.2f}", "vs HS average": "{:+.2f}"},
                                na_rep="–").apply(color, subset=["vs HS average"]), width="stretch")
    st.caption("EPA from Dowling's point of view. Read the vs HS average column: the raw number on its own "
               "isn't centered on zero. Punts start from a 4th-down spot, so the average high school punt prices "
               "out negative for the punting team and positive for the return team; kickoffs are measured against "
               "a touchback. HS average = every kick in our film (Dowling games and scout film), priced the same "
               "way. These are the same numbers the Team Profiles special teams radar uses.")


def render_special_teams_page(df_games: pd.DataFrame) -> None:
    # Only games Dowling played: scout film of other teams' kicks doesn't say anything about Dowling's units
    # (it's still the baseline for the special teams EPA table's HS average).
    all_games = df_games
    df_games = ins.dowling_only(df_games)
    if df_games.empty:
        st.info("No Dowling games in the selected weeks.")
        return
    ev = kick_events(df_games)
    render_unit_cards(df_games, ev)
    render_field_position_battle(df_games)
    c1, c2 = st.columns(2)
    with c1:
        render_fg_range(df_games)
    with c2:
        render_st_epa(df_games, all_games)
    render_punt_map(ev)
    render_kick_logs(df_games, ev)
    st.caption("KICK YARDS is blank in the data, so net punts and starts come from the punt spot and the receiving "
               "team's first snap. Tagging kick distance in Hudl would add gross punting and kickoff depth.")
