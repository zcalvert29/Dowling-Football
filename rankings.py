"""
rankings.py — the Rankings page: every team in the data, side by side, ranked against each other.

Each cell shows the team's number and, in parentheses, its rank among the teams shown: Off Pass EPA +0.20 (12).
Rank 1 is always the best (lowest for defensive EPA and success rate, highest for most everything else).
Sorting a column sorts by the number, not the text.

Every number comes from the same standard tables as Team Profiles (profiles.standardize_hudl): scrimmage plays
are runs and passes on downs 1-4 without a penalty, and special teams EPA prices every kick with the app's own EP
model. A team's numbers use every game it's in, whichever team filmed it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

import insights as ins
import profiles
import special_teams
import teams as tm

TEAM = ins.TEAM


@dataclass(frozen=True)
class Col:
    label: str
    higher_is_better: bool | None   # None: no rank (counts)
    fmt: str                        # "epa", "pct", "num1", "int", "record", "made"
    help: str


COLUMNS = [
    Col("Games", None, "int", "Games of film with this team in them (scouted or played by Dowling)."),
    Col("Record", True, "record", "Wins and losses in those games, from the final score on film. Ranked by win %."),
    Col("Pts/G", True, "num1", "Points scored per game on film."),
    Col("Pts Allowed/G", False, "num1", "Points allowed per game on film."),
    Col("Net EPA", True, "epa", "Offensive EPA per play minus defensive EPA per play allowed."),
    Col("Off EPA", True, "epa", "Expected points added per offensive play (runs and passes, downs 1-4)."),
    Col("Off Success", True, "pct", "Plays that stay on schedule: 40% / 70% / 100% of the yards needed on 1st / 2nd / "
                                    "3rd-4th."),
    Col("Off Rush EPA", True, "epa", "EPA per run."),
    Col("Off Rush Success", True, "pct", "Success rate on runs."),
    Col("Off Pass EPA", True, "epa", "EPA per pass play (sacks and scrambles included)."),
    Col("Off Pass Success", True, "pct", "Success rate on pass plays."),
    Col("Off Pass Rate", True, "pct", "Share of plays that are passes. Ranked most pass-heavy first (not better or "
                                      "worse)."),
    Col("Off Explosive", True, "pct", "Runs of 10+ yards and passes of 20+."),
    Col("Off 3rd Down", True, "pct", "3rd downs that gained the line to gain."),
    Col("Off Scoring Drives", True, "pct", "Drives that ended in a touchdown or made field goal."),
    Col("Off Turnover", False, "pct", "Interceptions and fumbles lost per play."),
    Col("Def EPA", False, "epa", "EPA per opponent play (lower is better)."),
    Col("Def Success", False, "pct", "Opponent success rate."),
    Col("Def Rush EPA", False, "epa", "EPA per opponent run."),
    Col("Def Rush Success", False, "pct", "Opponent success rate on runs."),
    Col("Def Pass EPA", False, "epa", "EPA per opponent pass."),
    Col("Def Pass Success", False, "pct", "Opponent success rate on passes."),
    Col("Def Explosive", False, "pct", "Opponent runs of 10+ and passes of 20+."),
    Col("Def 3rd Down", False, "pct", "Opponent 3rd downs converted."),
    Col("Def Scoring Drives", False, "pct", "Opponent drives that ended in points."),
    Col("Def Takeaway", True, "pct", "Interceptions and fumbles forced per opponent play."),
    Col("Turnover Margin/G", True, "num1", "Takeaways minus giveaways per game."),
    Col("ST EPA/G", True, "epa", "Special teams EPA per game: kickoffs, punts, and field goals, plus returns "
                                 "(the other team's kick EPA, flipped)."),
    Col("Net Punt Yds", True, "num1", "Yards from the punt spot to the receiving team's first snap (KICK YARDS "
                                      "isn't tagged, so this is net, not gross)."),
    Col("KO Touchback %", True, "pct", "Kickoffs that went for a touchback (onside kicks left out)."),
    Col("Opp Kick Start", False, "num1", "Where opponents start after this team's kickoffs, in yards from their own "
                                         "goal."),
    Col("FG %", True, "made", "Field goals made / attempted. Ranked by make %."),
    Col("Longest FG", True, "int", "Longest made field goal (spot + 17 yards)."),
    Col("PAT %", True, "made", "Extra point kicks made / attempted. Ranked by make %."),
]
BY_LABEL = {c.label: c for c in COLUMNS}
DEFAULT_SORT = "Off EPA"


def _mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan


def final_scores(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team): points for and against, from the filming team's running score."""
    rows = []
    for gid, g in df.groupby("game_id", sort=False):
        pair = ins.teams_in_game(g)
        if len(pair) != 2:
            continue
        filming = str(gid).split("_")[0]
        other = pair[1] if pair[0] == filming else pair[0]
        if filming not in pair:
            continue
        us, them = g["team_score"].max(), g["opponent_score"].max()
        if pd.isna(us) or pd.isna(them):
            continue
        rows += [{"game": gid, "team": filming, "pf": us, "pa": them},
                 {"game": gid, "team": other, "pf": them, "pa": us}]
    return pd.DataFrame(rows, columns=["game", "team", "pf", "pa"])


def team_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team with the raw number for every column (plus FG/PAT makes and attempts for display)."""
    tables = profiles.standardize_hudl(df)
    plays, drives, kicks, fgs = (tables[k] for k in ("plays", "drives", "kicks", "fgs"))
    plays = plays.assign(success=profiles._success(plays["down"], plays["dist"], plays["yards"]),
                         explosive=(((plays["is_rush"] == 1) & (plays["yards"] >= 10))
                                    | ((plays["is_pass"] == 1) & (plays["yards"] >= 20))).astype(float))
    ev = special_teams.kick_events(df)
    if ev.empty:
        ev = pd.DataFrame(columns=["kind", "kicker", "receiver", "result", "start", "net"])
    finals = final_scores(df)
    tries = df[df["PLAY TYPE"].isin(special_teams.XPS) & (df["RESULT"] != "Penalty")]
    fg_plays = df[df["PLAY TYPE"].isin(special_teams.FGS)]

    games_by_team = {}
    for gid, g in df.groupby("game_id", sort=False):
        for t in ins.teams_in_game(g):
            games_by_team.setdefault(t, set()).add(gid)

    rows = []
    for team, gids in games_by_team.items():
        n_games = len(gids)
        o, d = plays[plays["offense"] == team], plays[plays["defense"] == team]
        o_rush, o_pass = o[o["is_rush"] == 1], o[o["is_pass"] == 1]
        d_rush, d_pass = d[d["is_rush"] == 1], d[d["is_pass"] == 1]
        o3, d3 = o[o["down"] == 3], d[d["down"] == 3]
        od, dd = drives[drives["offense"] == team], drives[drives["defense"] == team]
        f = finals[finals["team"] == team]
        wins, losses = int((f["pf"] > f["pa"]).sum()), int((f["pf"] < f["pa"]).sum())
        ties = int((f["pf"] == f["pa"]).sum())

        k_kick, k_recv = kicks[kicks["kicker"] == team], kicks[kicks["receiver"] == team]
        k_fg = fgs[fgs["kicker"] == team]
        st_epa = (k_kick["epa"].sum(min_count=1), -k_recv["epa"].sum(min_count=1), k_fg["epa"].sum(min_count=1))
        st_epa = np.nansum(st_epa) / n_games if any(pd.notna(x) for x in st_epa) else np.nan

        punts = ev[(ev["kind"] == "Punt") & (ev["kicker"] == team)]
        kos = ev[(ev["kind"] == "Kickoff") & (ev["kicker"] == team)]
        fg = fg_plays[fg_plays["offense"] == team]
        fg_made = fg["RESULT"] == "Good"
        pat = tries[tries["offense"] == team]
        pat_made = int((pat["RESULT"] == "Good").sum())

        off_epa, def_epa = _mean(o["epa"]), _mean(d["epa"])
        rows.append({
            "Team": team,
            "Games": n_games,
            "Record": (wins + 0.5 * ties) / len(f) if len(f) else np.nan,
            "_record": f"{wins}-{losses}" + (f"-{ties}" if ties else ""),
            "Pts/G": _mean(f["pf"]), "Pts Allowed/G": _mean(f["pa"]),
            "Net EPA": off_epa - def_epa if pd.notna(off_epa) and pd.notna(def_epa) else np.nan,
            "Off EPA": off_epa, "Off Success": _mean(o["success"]),
            "Off Rush EPA": _mean(o_rush["epa"]), "Off Rush Success": _mean(o_rush["success"]),
            "Off Pass EPA": _mean(o_pass["epa"]), "Off Pass Success": _mean(o_pass["success"]),
            "Off Pass Rate": _mean(o["is_pass"]), "Off Explosive": _mean(o["explosive"]),
            "Off 3rd Down": _mean(o3["yards"] >= o3["dist"]) if len(o3) else np.nan,
            "Off Scoring Drives": _mean(od["scored"].astype(float)), "Off Turnover": _mean(o["turnover"]),
            "Def EPA": def_epa, "Def Success": _mean(d["success"]),
            "Def Rush EPA": _mean(d_rush["epa"]), "Def Rush Success": _mean(d_rush["success"]),
            "Def Pass EPA": _mean(d_pass["epa"]), "Def Pass Success": _mean(d_pass["success"]),
            "Def Explosive": _mean(d["explosive"]),
            "Def 3rd Down": _mean(d3["yards"] >= d3["dist"]) if len(d3) else np.nan,
            "Def Scoring Drives": _mean(dd["scored"].astype(float)), "Def Takeaway": _mean(d["turnover"]),
            "Turnover Margin/G": (d["turnover"].sum() - o["turnover"].sum()) / n_games if len(o) or len(d) else np.nan,
            "ST EPA/G": st_epa,
            "Net Punt Yds": _mean(punts["net"]),
            "KO Touchback %": _mean(kos["result"] == "Touchback") if len(kos) else np.nan,
            "Opp Kick Start": _mean(kos["start"]),
            "FG %": float(fg_made.mean()) if len(fg) else np.nan, "_fg": f"{int(fg_made.sum())}/{len(fg)}",
            "Longest FG": float((fg.loc[fg_made, "YARDLINE_100"] + 17).max()) if fg_made.any() else np.nan,
            "PAT %": pat_made / len(pat) if len(pat) else np.nan, "_pat": f"{pat_made}/{len(pat)}",
        })
    return pd.DataFrame(rows)


def ranks(table: pd.DataFrame) -> pd.DataFrame:
    """Rank of each team in each ranked column (1 = best; ties share the best rank; no number, no rank)."""
    out = pd.DataFrame(index=table.index)
    for c in COLUMNS:
        if c.higher_is_better is None or c.label not in table:
            continue
        out[c.label] = table[c.label].rank(ascending=not c.higher_is_better, method="min")
    return out


def _number(c: Col, x: float, row: pd.Series) -> str:
    if c.fmt == "record":
        return row["_record"]
    if c.fmt == "made":
        return f"{x:.0%} ({row['_fg' if c.label == 'FG %' else '_pat']})"
    if c.fmt == "epa":
        return f"{x:+.2f}"
    if c.fmt == "pct":
        return f"{x:.0%}"
    if c.fmt == "int":
        return f"{int(round(x))}"
    return f"{x:.1f}"


def display_text(table: pd.DataFrame, rk: pd.DataFrame) -> pd.DataFrame:
    """The text shown in each cell: 'number (rank)', or '–' with no number."""
    out = pd.DataFrame(index=table.index)
    for c in COLUMNS:
        vals = []
        for i, row in table.iterrows():
            x = row[c.label]
            if pd.isna(x):
                vals.append("–")
                continue
            text = _number(c, x, row)
            if c.label in rk and pd.notna(rk.at[i, c.label]):
                text += f" ({int(rk.at[i, c.label])})"
            vals.append(text)
        out[c.label] = vals
    return out


def _text_color(hex_color: str) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#FFFFFF" if lum < 0.55 else "#111111"


def styled(table: pd.DataFrame, team_colors: dict[str, str]):
    """A Styler of the numbers (so columns sort by number) that shows 'number (rank)' and each team's color."""
    rk = ranks(table)
    text = display_text(table, rk)
    shown = table[["Team"] + [c.label for c in COLUMNS]].copy()
    sty = shown.style
    for i in shown.index:
        sty = sty.format({c: (lambda _v, s=text.at[i, c]: s) for c in text.columns}, subset=pd.IndexSlice[[i], :])

    def team_css(name):
        color = team_colors.get(name)
        if not color:
            return "font-weight: 600"
        return f"background-color: {color}; color: {_text_color(color)}; font-weight: 600"

    return sty.map(team_css, subset=["Team"])


@st.cache_data(show_spinner=False)
def cached_team_table(df: pd.DataFrame) -> pd.DataFrame:
    return team_table(df)


def render_page(df_all: pd.DataFrame) -> None:
    import good_teams

    st.markdown("Every team in the app, ranked against each other. Each cell is the team's number and, in "
                "parentheses, its rank among the teams shown (**1 = best**: lowest for defensive EPA and success "
                "rate). Click a column header to sort by it. Numbers use every game a team is in, whoever filmed it.")
    seasons = sorted(int(s) for s in df_all["SEASON"].dropna().unique())
    c1, c2, c3 = st.columns([1, 1, 2])
    season = c1.selectbox("Season", seasons[::-1], key="rk_season")
    df = df_all[df_all["SEASON"] == season]
    with st.spinner("Ranking teams…"):
        table = cached_team_table(df)
    if table.empty:
        st.info("No games in this season yet.")
        return
    most = int(table["Games"].max())
    min_games = c2.number_input("Minimum games", 1, most, 1, key="rk_min_games",
                                help="Teams with fewer games of film are left out (and out of the ranks).")
    only = c3.segmented_control("Teams", ["All", "Good teams only"], default="All", key="rk_good")
    table = table[table["Games"] >= min_games]
    if only == "Good teams only":
        good = {t.casefold() for t in good_teams.current()}
        table = table[table["Team"].str.casefold().isin(good)]
    if table.empty:
        st.info("No teams match.")
        return
    table = table.sort_values(DEFAULT_SORT, ascending=False, na_position="last").reset_index(drop=True)

    colors = tm.colors()
    config = {"Team": st.column_config.TextColumn("Team", pinned=True)}
    for c in COLUMNS:
        config[c.label] = st.column_config.Column(c.label, help=c.help)
    st.dataframe(styled(table, colors), hide_index=True, width="stretch", column_config=config,
                 height=min(38 * (len(table) + 1) + 3, 900))
    st.caption(f"{len(table)} teams, sorted by Off EPA. Rank 1 is best in every column (most pass-heavy for Off "
               f"Pass Rate). Small samples rank too, so check Games before reading much into one team.")
    _render_color_editor(table["Team"].tolist(), colors)


def _render_color_editor(team_names: list[str], colors: dict[str, str]) -> None:
    import github_sync

    missing = [t for t in team_names if t not in colors]
    label = "Team colors" + (f" ({len(missing)} not set)" if missing else "")
    with st.expander(label):
        st.caption("Each team's primary color. Saved in teams.csv.")
        saved = tm.load()
        names = sorted(set(team_names) | set(saved["team"]), key=str.casefold)
        current = dict(zip(saved["team"], saved["color"]))
        table = pd.DataFrame({"Team": names, "Color": [current.get(t, "") for t in names]})
        edited = st.data_editor(
            table, hide_index=True, width="stretch", key="rk_colors", disabled=["Team"],
            column_config={"Color": st.column_config.SelectboxColumn(options=list(tm.COLOR_HEX),
                                                                     help="Blank = no color.")})
        new = saved.set_index("team")
        for t, color in zip(edited["Team"], edited["Color"].fillna("")):
            if t in new.index:
                new.loc[t, "color"] = str(color).strip()
            else:
                new.loc[t] = pd.Series({"color": str(color).strip()})
        new = new.reset_index().rename(columns={"index": "team"})
        if dict(zip(new["team"], new["color"])) == {t: current.get(t, "") for t in new["team"]}:
            return
        data = tm.to_csv(new)
        cfg = github_sync.config()
        if cfg is None:
            st.caption("Saving for everyone isn't set up (it needs a `[github]` section in the app's secrets). "
                       "Download teams.csv and replace it in the repo instead.")
            st.download_button("Download teams.csv", data, file_name="teams.csv", mime="text/csv", on_click="ignore")
            return
        if not ins._upload_unlocked():
            return
        if st.button("Save colors for everyone", type="primary"):
            try:
                with st.spinner("Saving to GitHub…"):
                    url = github_sync.commit_files(cfg, lambda read: {"teams.csv": data},
                                                   "Update team colors\n\nSaved from the app's Rankings page.")
            except github_sync.SaveError as e:
                st.error(f"Couldn't save: {e}")
                return
            st.success("Saved. Everyone gets the new colors when the app redeploys in a minute or two."
                       + (f" [See the commit]({url})." if url else ""))
