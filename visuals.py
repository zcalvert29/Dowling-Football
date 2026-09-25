"""
Data prep, metric definitions, and one render function per Tableau sheet.

Each render_* function takes the play-by-play frame (already filtered by the
sidebar Week / Down / Distance selections)
and draws one visual. To split the dashboard into pages later, just call the
render functions you want from each page file.
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

TEAM = "Dowling Catholic"

# ---------------------------------------------------------------------------
# Tableau hard-coded its dimension filters to whatever values existed when the
# sheets were built (e.g. Run Scheme Detail keeps BLUNT but drops STORM, and
# DCHS Formations drops TEXAS). By default this app instead keeps every
# non-null value so new weeks/formations show up automatically. Flip this to
# True to reproduce the Tableau snapshot exactly.
# ---------------------------------------------------------------------------
USE_TABLEAU_SNAPSHOT_FILTERS = False

SNAPSHOT_FILTERS = {
    "dchs_formations": {"OFF FORM": ["ALASKA", "ARIZONA", "DELAWARE", "DUCKS", "EMPTY", "FLORIDA",
                                     "IOWA", "LUKE/RICK", "MICHIGAN", "OREGON", "RICK"]},
    "d_vs_formation": {"OFF FORM": ["ACE", "CLOSED WING", "DEUCE", "DOUBLE WING OPEN", "EMPTY", "FAR",
                                    "PRO", "PRO WING", "TRIPS", "TRIPS PRO", "TRIPS PRO EMPTY"]},
    "d_pass_coverage": {"COVERAGE": ["3", "3 SKATE", "3 TRAP", "5", "6", "IOWA", "IOWA ROBBER", "OMAHA"]},
    "run_scheme": {"RUN SCHEME": ["BLUNT", "BULLDOGS", "DRAW", "HAWKEYES", "HERKY", "IZZY", "MONEY",
                                  "MONSTER", "PANTHERS", "PATRIOTS", "SEATTLE"]},
    "men_in_box": {"MEN IN BOX": [5, 6, 7, 8, 9]},
    "dchs_offense": {"RESULT": ["Complete", "Complete, Fumble", "Complete, TD", "Fumble", "Incomplete",
                                "Interception", "Rush", "Rush, TD", "Sack", "Scramble"]},
}

DISTANCE_ORDER = ["Short (1-3 yards)", "Medium (4-6 yds)", "Long (7-10 yds)", "Extra Long (11+ yds)"]
AIR_YARDS_ORDER = ["Short", "Medium", "Long"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    # Tableau calc: Distance
    dist = df["DIST"]
    df["Distance"] = pd.Categorical(
        np.select(
            [dist.between(1, 3), dist.between(4, 6), dist.between(7, 10), dist > 10],
            DISTANCE_ORDER,
            default=None,
        ),
        categories=DISTANCE_ORDER,
        ordered=True,
    )

    # Tableau calc: Air Yards Bin (only meaningful where AIR YARDS is present)
    air = df["AIR YARDS"]
    df["Air Yards Bin"] = pd.Categorical(
        np.select([air < 10, air < 20, air >= 20], AIR_YARDS_ORDER, default=None),
        categories=AIR_YARDS_ORDER,
        ordered=True,
    )

    # Mixed int/str codes (3, 6, "IOWA") -> consistent strings
    df["COVERAGE"] = df["COVERAGE"].map(lambda v: str(v) if pd.notna(v) else None)
    df["MEN IN BOX"] = df["MEN IN BOX"].astype("Int64")
    df["DN"] = df["DN"].astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Metrics (names match the Tableau Measure Names aliases)
# ---------------------------------------------------------------------------
METRICS = {
    "Plays": lambda g: g["PLAY #"].count(),
    "Pass Rate": lambda g: g["PASS"].mean(),
    "Rush Rate": lambda g: g["RUSH"].mean(),
    "Avg Yards Gained": lambda g: g["GN/LS"].mean(),
    "Success Rate": lambda g: g["success"].mean(),
    "EPA per Play": lambda g: g["epa"].mean(),
    "Chunk Rate": lambda g: g["chunk_play"].mean(),
    "Explosive Rate": lambda g: g["explosive_play"].mean(),
    "3rd Down Conversions": lambda g: g["THIRD_DOWN_CONVERTED"].sum(),
    "3rd Down Conversion Rate": lambda g: g["THIRD_DOWN_CONVERTED"].sum() / g["PLAY #"].count(),
    "4th Down Conversion Rate": lambda g: g["FOURTH_DOWN_CONVERTED"].sum() / g["PLAY #"].count(),
}

PCT = "{:.0%}"
FORMATS = {
    "Plays": "{:,.0f}",
    "3rd Down Conversions": "{:,.0f}",
    "Avg Yards Gained": "{:.1f}",
    "EPA per Play": "{:.2f}",
    **{m: PCT for m in ["Pass Rate", "Rush Rate", "Success Rate", "Chunk Rate", "Explosive Rate",
                        "3rd Down Conversion Rate", "4th Down Conversion Rate"]},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def keep(df: pd.DataFrame, visual: str, col: str) -> pd.DataFrame:
    """Snapshot whitelist if enabled, otherwise just drop nulls."""
    if USE_TABLEAU_SNAPSHOT_FILTERS and col in SNAPSHOT_FILTERS.get(visual, {}):
        return df[df[col].isin(SNAPSHOT_FILTERS[visual][col])]
    return df[df[col].notna()]


def run_pass(df: pd.DataFrame, types=("Run", "Pass")) -> pd.DataFrame:
    return df[df["PLAY TYPE"].isin(types)]


def downs(df: pd.DataFrame, values=(1, 2, 3, 4)) -> pd.DataFrame:
    return df[df["DN"].isin(values)]


def crosstab(df: pd.DataFrame, rows, measures, sort_by_count: bool = False) -> pd.DataFrame:
    """Rows = dimension(s), columns = measures. Mirrors a Tableau Measure Names/Values text table."""
    rows = [rows] if isinstance(rows, str) else list(rows)
    df = df.dropna(subset=rows)
    g = df.groupby(rows, observed=True, sort=True)
    out = pd.DataFrame({m: METRICS[m](g) for m in measures})
    if sort_by_count:
        out = out.loc[g.size().sort_values(ascending=False, kind="stable").index]
    return out


def show_table(title: str, table: pd.DataFrame) -> None:
    st.markdown(f"**{title}**")
    if table.empty:
        st.info("No plays match the current filters.")
        return
    fmt = {c: FORMATS[c] for c in table.columns if c in FORMATS}
    st.dataframe(table.style.format(fmt, na_rep=""), width="stretch")


def bar_chart(data: pd.DataFrame, x: str, y: str, title: str, y_format: str, x_sort=None) -> None:
    st.markdown(f"**{title}**")
    if data.empty:
        st.info("No plays match the current filters.")
        return
    base = alt.Chart(data).encode(
        x=alt.X(f"{x}:O", sort=x_sort, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{y}:Q", axis=alt.Axis(format=y_format)),
        tooltip=[alt.Tooltip(f"{x}:O"), alt.Tooltip(f"{y}:Q", format=y_format)],
    )
    labels = base.mark_text(dy=-8).encode(text=alt.Text(f"{y}:Q", format=y_format))
    st.altair_chart((base.mark_bar() + labels).properties(height=280), width="stretch")


# ---------------------------------------------------------------------------
# Dowling Catholic offense
# ---------------------------------------------------------------------------
def render_dchs_offense(df):
    d = downs(run_pass(df[df["offense"] == TEAM]))
    d = d[d["Distance"].notna()]
    if USE_TABLEAU_SNAPSHOT_FILTERS:
        d = keep(d, "dchs_offense", "RESULT")
    else:
        d = d[~d["RESULT"].isin(["Penalty", "Timeout"])]
    t = crosstab(d, "PLAY TYPE", ["Avg Yards Gained", "Success Rate", "EPA per Play", "Chunk Rate", "Explosive Rate", "Plays"])
    show_table("DCHS O Stats", t)


def render_dchs_offense_tendencies(df):
    d = downs(run_pass(df[df["offense"] == TEAM]))
    t = crosstab(d, ["DN", "Distance"], ["Pass Rate", "Rush Rate", "Success Rate", "EPA per Play", "Plays"])
    show_table("DCHS O Tendencies", t)


def render_dchs_offense_3rd_downs(df):
    d = downs(run_pass(df[df["offense"] == TEAM]), [3])
    t = crosstab(d, "Distance", ["Pass Rate", "Rush Rate", "3rd Down Conversion Rate", "3rd Down Conversions", "Plays"])
    show_table("DCHS O 3rd Downs", t)


def render_dchs_offense_4th_downs(df):
    d = downs(run_pass(df[df["offense"] == TEAM]), [4])
    t = crosstab(d, "Distance", ["Pass Rate", "Rush Rate", "4th Down Conversion Rate", "Plays"])
    show_table("DCHS O 4th Downs", t)


def render_dchs_formations(df):
    d = keep(run_pass(df[df["offense"] == TEAM]), "dchs_formations", "OFF FORM")
    t = crosstab(d, "OFF FORM", ["Pass Rate", "Rush Rate", "Avg Yards Gained", "Success Rate", "EPA per Play",
                                 "Chunk Rate", "Explosive Rate", "Plays"], sort_by_count=True)
    show_table("DCHS Formations", t)


def render_run_scheme_detail(df):
    d = downs(run_pass(df[df["offense"] == TEAM], ["Run"]))
    d = keep(d[d["Distance"].notna()], "run_scheme", "RUN SCHEME")
    t = crosstab(d, "RUN SCHEME", ["Avg Yards Gained", "Success Rate", "EPA per Play", "Chunk Rate", "Explosive Rate", "Plays"], sort_by_count=True)
    show_table("DCHS O Run Scheme Detail", t)


def render_dchs_intended_pass_distance(df):
    d = downs(df[df["offense"] == TEAM])
    counts = d.dropna(subset=["AIR YARDS"]).groupby("Air Yards Bin", observed=True)["AIR YARDS"].count()
    data = (counts / counts.sum()).rename("Pct of Passes").reset_index() if counts.sum() else pd.DataFrame()
    bar_chart(data, "Air Yards Bin", "Pct of Passes", "DCHS O Intended Pass Distance", ".0%", AIR_YARDS_ORDER)


def _men_in_box(df, play_type, side, title, exclude=()):
    d = run_pass(df[df[side] == TEAM], [play_type])
    d = keep(d, "men_in_box", "MEN IN BOX")
    d = d[~d["MEN IN BOX"].isin(exclude)]
    t = crosstab(d, "MEN IN BOX", ["Avg Yards Gained", "Success Rate", "EPA per Play", "Plays"])
    show_table(title, t)


def render_rush_vs_box(df):
    _men_in_box(df, "Run", "offense", "DCHS O Rush Metrics vs Men in the Box", exclude=(3, 10))


def render_pass_vs_box(df):
    _men_in_box(df, "Pass", "offense", "DCHS O Pass Metrics vs Men in the Box")


# ---------------------------------------------------------------------------
# Weekly trends (Dowling Catholic offense, all weeks)
# ---------------------------------------------------------------------------
def _weekly(df, play_type, col, title, fmt):
    d = run_pass(df[df["offense"] == TEAM], [play_type])
    data = d.groupby("WEEK")[col].mean().rename(title).reset_index()
    bar_chart(data, "WEEK", title, title, fmt)


def render_weekly_pass_epa(df):
    _weekly(df, "Pass", "epa", "Weekly Pass EPA per Play", ".2f")


def render_weekly_pass_success(df):
    _weekly(df, "Pass", "success", "Weekly Pass Success", ".0%")


def render_weekly_rush_epa(df):
    _weekly(df, "Run", "epa", "Weekly Rush EPA per Play", ".2f")


def render_weekly_rush_success(df):
    _weekly(df, "Run", "success", "Weekly Rush Success", ".0%")


# ---------------------------------------------------------------------------
# Dowling Catholic defense
# ---------------------------------------------------------------------------
def render_dchs_defense(df):
    d = run_pass(df[df["defense"] == TEAM])
    t = crosstab(d, "PLAY TYPE", ["Avg Yards Gained", "Success Rate", "EPA per Play", "Chunk Rate", "Explosive Rate", "Plays"])
    show_table("DCHS D Stats", t)


def render_d_3rd_downs(df):
    d = downs(run_pass(df[df["defense"] == TEAM]), [3])
    t = crosstab(d, "Distance", ["Pass Rate", "Rush Rate", "3rd Down Conversion Rate", "3rd Down Conversions", "Plays"])
    show_table("DCHS D 3rd Downs", t)


def render_d_rush_vs_box(df):
    _men_in_box(df, "Run", "defense", "DCHS D Rush Metrics vs Men in the Box")


def render_d_pass_coverage(df):
    d = downs(run_pass(df[df["defense"] == TEAM], ["Pass"]))
    d = keep(d[d["Distance"].notna()], "d_pass_coverage", "COVERAGE")
    t = crosstab(d, "COVERAGE", ["Avg Yards Gained", "EPA per Play", "Success Rate", "Chunk Rate", "Explosive Rate", "Plays"],
                 sort_by_count=True)
    show_table("DCHS D Pass Coverage Stats", t)


def render_d_vs_formation(df):
    d = downs(run_pass(df[df["defense"] == TEAM]))
    d = keep(d[d["Distance"].notna()], "d_vs_formation", "OFF FORM")
    t = crosstab(d, "OFF FORM", ["Pass Rate", "Rush Rate", "Avg Yards Gained", "Success Rate", "EPA per Play",
                                 "Chunk Rate", "Explosive Rate", "Plays"], sort_by_count=True)
    show_table("DCHS D vs Formation Stats", t)


# ---------------------------------------------------------------------------
# Opponent offense (opponent picked in the sidebar)
# ---------------------------------------------------------------------------
def render_opp_tendencies(df, opponent):
    d = downs(run_pass(df[df["offense"] == opponent]))
    t = crosstab(d, ["DN", "Distance"], ["Pass Rate", "Rush Rate", "Plays"])
    show_table(f"{opponent} Offensive Tendencies", t)


def render_opp_3rd_downs(df, opponent):
    d = downs(run_pass(df[df["offense"] == opponent]), [3])
    t = crosstab(d, "Distance", ["Pass Rate", "Rush Rate", "3rd Down Conversion Rate", "3rd Down Conversions", "Plays"])
    show_table(f"{opponent} Offense 3rd Downs", t)


def render_opp_4th_downs(df, opponent):
    d = downs(run_pass(df[df["offense"] == opponent]), [4])
    t = crosstab(d, "Distance", ["Pass Rate", "Rush Rate", "4th Down Conversion Rate", "Plays"])
    show_table(f"{opponent} Offense 4th Downs", t)
