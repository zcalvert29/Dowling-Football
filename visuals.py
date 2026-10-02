"""
Data prep, metric definitions, and one render function per Tableau sheet.

Each render_* function takes the play-by-play frame (already filtered by the
sidebar Week / Down / Distance selections)
and draws one visual. To split the dashboard into pages later, just call the
render functions you want from each page file.
"""
from __future__ import annotations

import base64

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
    "d_run_play_results": {"OFF PLAY": ["FOLD", "GIVE @ 1", "GIVE @ 2", "JET", "POWER", "QB COUNTER",
                                        "TACKLE FOLD", "TOSS", "TOWARD"]},
    "d_pass_play_results": {"OFF PLAY": ["BOOT @ 8", "BOOT @ 9", "FLOOD", "HB SCREEN", "LEVELS", "MESH",
                                         "STICK", "TE DELAY", "VERTICAL"]},
    "o_pass_play_results": {"OFF PLAY": [
        "BOUND COVER", "BOUND COVER LOW", "BOUND FEVER", "BOUND HOOK AND GO", "BOUND NILE", "BOUND PETRINO",
        "BOUND PIPE COVER SWITCH", "BOUND PIPE DIVA SWITCH", "BOUND PIPE FLOODS", "BOUND PIPE MILLS COVER",
        "BOUND PIPE MILLS SWITCH", "BOUND SWAP STICK", "BOUND UNCOIL COVER", "BOUND UNCOIL DIVA",
        "BOUND UNCOIL DIVA SWITCH", "BOUND VOLS", "BOUND VOLS SWITCH", "BOUND VOLS SWITCH LOW",
        "BREAK LEAK COVER", "BREAK LEAK COVER SWITCH", "BREAK LEAK DIVA SWITCH", "BREAK Q HAWKEYES H PIPE",
        "BREAK SPRAY JOHN C", "BREAK SWAP STICK", "BREAK UNCOIL DIVA", "FLASH DRAGON", "FLASH LIZARD",
        "HAIL MARY", "HAWKEYES C", "HAWKEYES KEY 2", "HAWKEYES OZZY", "HERKY", "HERKY OZZY", "ICE ALABAMA",
        "MIRROR BULLDOGS", "MIRROR BULLDOGS SLICE EGYPT", "MIRROR PATRIOTS TEAL", "PATRIOTS C",
        "PATRIOTS JAGUARS", "PATRIOTS KEY 3", "PIPE JUKE", "RAIN", "SEARCH JUKE", "SNAP BEAU POST",
        "TEAR ICE COVER", "TECH SEARCH DIVA SWITCH",
    ]},
}

DISTANCE_ORDER = ["Short (1-3 yards)", "Medium (4-6 yds)", "Long (7-10 yds)", "Extra Long (11+ yds)"]
AIR_YARDS_ORDER = ["Short", "Medium", "Long"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    # Re-derive drives with the current rules, so games curated before a fix pick it up without re-curating
    # (e.g. a possession after a defensive touchdown used to merge into the drive that ended in the score).
    import curate_pbp
    if {"game_id", "PLAY #", "drive", "turnover", "success"} <= set(df.columns):
        df = pd.concat([curate_pbp.assign_drives(g.sort_values("PLAY #", kind="stable"))
                        for _, g in df.groupby("game_id", sort=False)]).sort_index()

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

    # Older curated files won't have 'completion' yet; derive it the same
    # way curate_pbp.py does so the pass-zone visuals still work.
    if "completion" not in df.columns:
        res = df["RESULT"].fillna("")
        df["completion"] = np.select(
            [res.str.contains("Incomplete|Interception"), res.str.contains("Complete")], [0, 1], default=np.nan
        )
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
    **{m: PCT for m in ["Pass Rate", "Rush Rate", "Success Rate", "Explosive Rate",
                        "3rd Down Conversion Rate", "4th Down Conversion Rate"]},
}


# Columns shared by the per-play-call "Game Detail" tables
PLAY_RESULT_MEASURES = ["Avg Yards Gained", "Success Rate", "EPA per Play", "Explosive Rate", "Plays"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Set from the app's sidebar. Tables hide rows with fewer plays than this.
MIN_PLAYS = 1
# Rows with fewer plays than this are grayed out (and lose their colors).
LOW_N_ROWS = 10

# A cell only gets colored when it's at least this far from the table's
# overall average, so the eye goes to what's actually different.
COLOR_THRESHOLDS = {
    "Avg Yards Gained": 1.0,
    "Success Rate": 0.05,
    "EPA per Play": 0.10,
    "Explosive Rate": 0.04,
    "3rd Down Conversion Rate": 0.08,
    "4th Down Conversion Rate": 0.10,
}
GOOD_STYLE = "background-color:#E1F5EE;color:#085041"
OK_STYLE = "background-color:#FAEEDA;color:#633806"
BAD_STYLE = "background-color:#FCEBEB;color:#791F1F"

# Offense tables only use red for results that are bad on their own terms;
# anything else short of green is yellow ("fine, not great").
OFFENSE_RED_BELOW = {"EPA per Play": 0.0, "Avg Yards Gained": 4.5}
LOW_N_STYLE = "color:#9A9890;background-color:transparent"

METRIC_HELP = {
    "Plays": "Number of plays in this row.",
    "Pass Rate": "Share of plays that were passes.",
    "Rush Rate": "Share of plays that were runs.",
    "Avg Yards Gained": "Average yards gained per play.",
    "Success Rate": "Share of plays that stayed on schedule: 40% of the yards needed on 1st down, "
                    "70% on 2nd, a first down on 3rd/4th.",
    "EPA per Play": "Expected points added: how many points the average play was worth, based on "
                    "down, distance, and field position. Above 0 is good for the offense.",
    "Explosive Rate": "Share of plays that were a run of 10+ yards or a pass of 20+ yards.",
    "3rd Down Conversions": "3rd downs converted.",
    "3rd Down Conversion Rate": "Share of 3rd downs converted.",
    "4th Down Conversion Rate": "Share of 4th downs converted.",
}


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
    """Rows = dimension(s), columns = measures. Mirrors a Tableau Measure Names/Values text table.

    The table's overall values (all rows combined) are stored in
    out.attrs["overall"]; show_table colors cells against them.
    """
    rows = [rows] if isinstance(rows, str) else list(rows)
    df = df.dropna(subset=rows)
    g = df.groupby(rows, observed=True, sort=True)
    out = pd.DataFrame({m: METRICS[m](g) for m in measures})
    if sort_by_count:
        out = out.loc[g.size().sort_values(ascending=False, kind="stable").index]
    # Plain floats: Streamlit serializes attrs to JSON and can't handle numpy ints.
    out.attrs["overall"] = {m: float(METRICS[m](df)) for m in measures} if len(df) else {}
    return out


def style_table(table: pd.DataFrame, good_high: bool = True, overall: dict | None = None):
    """
    Format a crosstab and color cells green (good for Dowling) or red (bad
    for Dowling) when they're clearly above/below the table's overall
    average. good_high=False flips the colors, for tables where a high
    number is the opponent's offense doing well (defense/scouting pages).
    Rows with fewer than LOW_N_ROWS plays are grayed out.
    """
    overall = table.attrs.get("overall", {}) if overall is None else overall
    fmt = {c: FORMATS[c] for c in table.columns if c in FORMATS}

    def color_col(col):
        thr, ref = COLOR_THRESHOLDS.get(col.name), overall.get(col.name)
        if thr is None or ref is None or pd.isna(ref):
            return [""] * len(col)
        styles = []
        for v in col:
            if pd.isna(v):
                styles.append("")
            elif good_high:
                # Offense rule: green at/above the cutoff; red only for negative
                # EPA or under 4.5 yards; everything else below green is yellow
                # (for EPA/yards) or yellow only when clearly below average
                # (success, explosive, conversion rates never go red).
                floor = OFFENSE_RED_BELOW.get(col.name)
                if floor is not None and v < floor:
                    styles.append(BAD_STYLE)
                elif v >= ref + thr:
                    styles.append(GOOD_STYLE)
                elif floor is not None or v <= ref - thr:
                    styles.append(OK_STYLE)
                else:
                    styles.append("")
            elif abs(v - ref) < thr:
                styles.append("")
            else:
                styles.append(GOOD_STYLE if (v > ref) == good_high else BAD_STYLE)
        return styles

    sty = table.style.format(fmt, na_rep="").apply(color_col, axis=0)
    if "Plays" in table.columns:
        low = (table["Plays"] < LOW_N_ROWS).to_numpy()
        sty = sty.apply(
            lambda row: [LOW_N_STYLE if low[table.index.get_loc(row.name)] else "" for _ in row], axis=1
        )
    return sty


def show_table(title: str, table: pd.DataFrame, good_high: bool = True, caption: str | None = None) -> None:
    st.markdown(f"**{title}**")
    overall = table.attrs.get("overall", {})
    if "Plays" in table.columns and MIN_PLAYS > 1:
        table = table[table["Plays"] >= MIN_PLAYS]
    if table.empty:
        st.info("No plays match the current filters.")
        return
    config = {c: st.column_config.Column(help=METRIC_HELP[c]) for c in table.columns if c in METRIC_HELP}
    st.dataframe(style_table(table, good_high, overall), width="stretch", column_config=config)
    if caption:
        st.caption(caption)


def tag_note(d: pd.DataFrame, col: str, label: str, threshold: float = 0.9) -> None:
    """Caption when a tag the visual depends on is mostly missing."""
    if d.empty:
        return
    tagged = int(d[col].notna().sum())
    if tagged / len(d) < threshold:
        st.caption(f"Only {tagged} of {len(d)} plays here have a {label} tag, so this only covers those plays.")


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


def theme_ink() -> str:
    """Neutral text color matching Streamlit's light/dark theme (for images)."""
    theme = getattr(getattr(st.context, "theme", None), "type", None)
    return "#FAFAFA" if theme == "dark" else "#31333F"


# ---------------------------------------------------------------------------
# Takeaway cards and "use more / use less" chart (work off any crosstab
# that has Plays and EPA per Play columns)
# ---------------------------------------------------------------------------
TAKEAWAY_MIN_PLAYS = 10


def render_takeaways(table: pd.DataFrame, noun: str, good_high: bool = True) -> None:
    """
    Three plain-language cards: most used, best, and struggling (best and
    struggling only consider rows with TAKEAWAY_MIN_PLAYS+ plays).
    noun = what a row is, e.g. "formation", "run scheme", "coverage".
    """
    if table.empty or not {"Plays", "EPA per Play"} <= set(table.columns):
        return
    t = table.dropna(subset=["EPA per Play"])
    if t.empty:
        return
    total = table["Plays"].sum()
    avg = table.attrs.get("overall", {}).get("EPA per Play", t["EPA per Play"].mean())
    name = lambda idx: " · ".join(map(str, idx)) if isinstance(idx, tuple) else str(idx)

    top = t["Plays"].idxmax()
    cards = [("Most used", name(top),
              f"{t.loc[top, 'Plays'] / total:.0%} of plays · {t.loc[top, 'EPA per Play']:+.2f} EPA")]
    sized = t[t["Plays"] >= TAKEAWAY_MIN_PLAYS]
    if not sized.empty:
        best = sized["EPA per Play"].idxmax() if good_high else sized["EPA per Play"].idxmin()
        worst = sized["EPA per Play"].idxmin() if good_high else sized["EPA per Play"].idxmax()
        cards.append((f"Best ({TAKEAWAY_MIN_PLAYS}+ plays)", name(best),
                      f"{sized.loc[best, 'EPA per Play']:+.2f} EPA on {int(sized.loc[best, 'Plays'])} plays"))
        worst_epa = sized.loc[worst, "EPA per Play"]
        if (worst_epa < avg) == good_high and worst != best:
            sr = sized.loc[worst, "Success Rate"] if "Success Rate" in sized else np.nan
            sr_txt = "" if pd.isna(sr) else f" · {sr:.0%} success"
            cards.append(("Struggling", name(worst),
                          f"{worst_epa:+.2f} EPA{sr_txt} on {int(sized.loc[worst, 'Plays'])} plays"))
    else:
        cards.append((f"Best ({TAKEAWAY_MIN_PLAYS}+ plays)", "–", f"No {noun} has {TAKEAWAY_MIN_PLAYS}+ plays yet"))

    cols = st.columns(len(cards))
    for col, (label, value, sub) in zip(cols, cards):
        with col.container(border=True):
            st.caption(label)
            st.markdown(f"**{value}**")
            st.caption(sub)


def render_usage_scatter(table: pd.DataFrame, title: str, good_high: bool = True, noun: str = "Formation") -> None:
    """
    Success rate (x) vs EPA per play (y) for each row of a crosstab, dot
    size = plays. Dashed lines mark the averages, splitting the chart into
    four coaching quadrants. Hover any dot for name, EPA, success, plays.
    """
    need = {"Plays", "EPA per Play", "Success Rate"}
    if table.empty or not need <= set(table.columns):
        return
    data = table.reset_index()
    data = data.rename(columns={data.columns[0]: noun}).dropna(subset=["EPA per Play", "Success Rate"])
    data[noun] = data[noun].astype(str)
    if data.empty:
        return
    overall = table.attrs.get("overall", {})
    avg_epa = overall.get("EPA per Play", data["EPA per Play"].mean())
    avg_sr = overall.get("Success Rate", data["Success Rate"].mean())
    if good_high:
        # Same rule as the offense tables: green at/above the green line,
        # red only for negative EPA, yellow in between.
        green_line = avg_epa + COLOR_THRESHOLDS["EPA per Play"]
        cats = ["Green: above average", "Yellow: positive, below green", "Red: negative EPA"]
        colors = ["#1D9E75", "#E8B923", "#E24B4A"]
        data["vs_avg"] = np.select([data["EPA per Play"] >= max(green_line, 0), data["EPA per Play"] >= 0],
                                   cats[:2], default=cats[2])
    else:
        cats, colors = ["Good for Dowling", "Bad for Dowling"], ["#1D9E75", "#E24B4A"]
        data["vs_avg"] = np.where(data["EPA per Play"] < avg_epa, cats[0], cats[1])
    label_min = max(5, TAKEAWAY_MIN_PLAYS // 2)
    data["Shown label"] = np.where(data["Plays"] >= label_min, data[noun], "")
    data["Sample"] = np.where(data["Plays"] >= TAKEAWAY_MIN_PLAYS, "big", "small")

    st.markdown(f"**{title}**")
    y_lo, y_hi = data["EPA per Play"].min(), data["EPA per Play"].max()
    pad = max((y_hi - y_lo) * 0.12, 0.2)
    y_dom = [min(y_lo, avg_epa) - pad, max(y_hi, avg_epa) + pad]
    x_dom = [max(0.0, min(data["Success Rate"].min(), avg_sr) - 0.08),
             min(1.0, max(data["Success Rate"].max(), avg_sr) + 0.08)]
    x_dom = [0.0, 1.0] if x_dom[1] - x_dom[0] < 0.2 else x_dom

    base = alt.Chart(data).encode(
        x=alt.X("Success Rate:Q", scale=alt.Scale(domain=x_dom), title="Success rate", axis=alt.Axis(format=".0%")),
        y=alt.Y("EPA per Play:Q", scale=alt.Scale(domain=y_dom), title="EPA per play", axis=alt.Axis(format="+.1f")),
    )
    tooltip = [alt.Tooltip(f"{noun}:N"), alt.Tooltip("EPA per Play:Q", format="+.2f"),
               alt.Tooltip("Success Rate:Q", format=".0%"), alt.Tooltip("Plays:Q")]
    points = base.mark_circle(stroke="white", strokeWidth=1).encode(
        size=alt.Size("Plays:Q", scale=alt.Scale(range=[40, 900]), legend=alt.Legend(title="Plays", orient="right")),
        color=alt.Color("vs_avg:N", title=None, scale=alt.Scale(domain=cats, range=colors),
                        legend=alt.Legend(orient="top")),
        opacity=alt.Opacity("Sample:N", scale=alt.Scale(domain=["big", "small"], range=[0.9, 0.4]), legend=None),
        tooltip=tooltip,
    )
    labels = base.mark_text(align="left", dx=10, fontSize=12).encode(text="Shown label", tooltip=tooltip)
    epa_line = alt.Chart(pd.DataFrame({"y": [avg_epa]})).mark_rule(strokeDash=[5, 4], color="#888780").encode(y="y:Q")
    sr_line = alt.Chart(pd.DataFrame({"x": [avg_sr]})).mark_rule(strokeDash=[5, 4], color="#888780").encode(x="x:Q")
    if good_high:
        corners = {("right", "top"): "Working: on schedule and explosive",
                   ("left", "top"): "Boom or bust",
                   ("right", "bottom"): "Steady, few big plays",
                   ("left", "bottom"): "Not working"}
    else:  # opponent offense: their success is bad for Dowling
        corners = {("right", "top"): "They move it at will",
                   ("left", "top"): "Big plays allowed",
                   ("right", "bottom"): "They stay on schedule, no damage",
                   ("left", "bottom"): "We shut it down"}
    corner_layers = []
    for (h, vpos), text in corners.items():
        cx, cy = (x_dom[1] if h == "right" else x_dom[0]), (y_dom[1] if vpos == "top" else y_dom[0])
        corner_layers.append(
            alt.Chart(pd.DataFrame({"x": [cx], "y": [cy], "t": [text]})).mark_text(
                align=h, baseline=vpos, dx=-6 if h == "right" else 6, dy=6 if vpos == "top" else -6,
                color="#888780", fontSize=12,
            ).encode(x="x:Q", y="y:Q", text="t:N")
        )
    chart = alt.layer(epa_line, sr_line, *corner_layers, points, labels).properties(height=380)
    st.altair_chart(chart, width="stretch")
    st.caption(f"Dashed lines: average EPA ({avg_epa:+.2f}) and success rate ({avg_sr:.0%}). Bigger dots = more "
               f"plays. Faded dots have fewer than {TAKEAWAY_MIN_PLAYS} plays and only {label_min}+ play rows are "
               "labeled; hover any dot for the details.")


# ---------------------------------------------------------------------------
# Dowling Catholic offense
#
# *_table functions build a crosstab; render_* functions draw it. Pages
# that also show takeaway cards or a usage chart build the table once and
# pass it to each (render_*(df, table=t)).
# ---------------------------------------------------------------------------
def render_dchs_offense(df):
    d = downs(run_pass(df[df["offense"] == TEAM]))
    d = d[d["Distance"].notna()]
    if USE_TABLEAU_SNAPSHOT_FILTERS:
        d = keep(d, "dchs_offense", "RESULT")
    else:
        d = d[~d["RESULT"].isin(["Penalty", "Timeout"])]
    t = crosstab(d, "PLAY TYPE", ["Avg Yards Gained", "Success Rate", "EPA per Play", "Explosive Rate", "Plays"])
    show_table("DCHS O Stats", t)
    return t


def render_dchs_offense_tendencies(df):
    d = downs(run_pass(df[df["offense"] == TEAM]))
    t = crosstab(d, ["DN", "Distance"], ["Pass Rate", "Rush Rate", "Success Rate", "EPA per Play", "Plays"])
    show_table("DCHS O Tendencies", t)
    return t


def render_dchs_offense_3rd_downs(df):
    d = downs(run_pass(df[df["offense"] == TEAM]), [3])
    t = crosstab(d, "Distance", ["Pass Rate", "Rush Rate", "3rd Down Conversion Rate", "3rd Down Conversions", "Plays"])
    show_table("DCHS O 3rd Downs", t)
    return t


def render_dchs_offense_4th_downs(df):
    d = downs(run_pass(df[df["offense"] == TEAM]), [4])
    t = crosstab(d, "Distance", ["Pass Rate", "Rush Rate", "4th Down Conversion Rate", "Plays"])
    show_table("DCHS O 4th Downs", t)
    return t


def dchs_formations_table(df):
    d = keep(run_pass(df[df["offense"] == TEAM]), "dchs_formations", "OFF FORM")
    return crosstab(d, "OFF FORM", ["Pass Rate", "Rush Rate", "Avg Yards Gained", "Success Rate", "EPA per Play",
                                    "Explosive Rate", "Plays"], sort_by_count=True)


def render_dchs_formations(df, table=None):
    t = dchs_formations_table(df) if table is None else table
    show_table("DCHS Formations", t)
    tag_note(run_pass(df[df["offense"] == TEAM]), "OFF FORM", "formation")
    return t


def run_scheme_table(df):
    d = downs(run_pass(df[df["offense"] == TEAM], ["Run"]))
    d = keep(d[d["Distance"].notna()], "run_scheme", "RUN SCHEME")
    return crosstab(d, "RUN SCHEME", ["Avg Yards Gained", "Success Rate", "EPA per Play", "Explosive Rate", "Plays"],
                    sort_by_count=True)


def render_run_scheme_detail(df, table=None):
    t = run_scheme_table(df) if table is None else table
    show_table("DCHS O Run Scheme Detail", t)
    tag_note(run_pass(df[df["offense"] == TEAM], ["Run"]), "RUN SCHEME", "run scheme (read from OFF PLAY)")
    return t


def o_pass_detail_table(df):
    d = run_pass(df[df["offense"] == TEAM], ["Pass"])
    d = keep(d, "o_pass_play_results", "OFF PLAY")
    return crosstab(d, "OFF PLAY", PLAY_RESULT_MEASURES, sort_by_count=True)


def render_o_pass_game_detail(df, table=None):
    """Tableau sheet: DCHS O Pass Play Results (sorted by play count, like the Tableau shelf sort)."""
    t = o_pass_detail_table(df) if table is None else table
    show_table("DCHS O Pass Game Detail", t)
    tag_note(run_pass(df[df["offense"] == TEAM], ["Pass"]), "OFF PLAY", "play call")
    return t


def render_dchs_intended_pass_distance(df):
    d = downs(df[df["offense"] == TEAM])
    counts = d.dropna(subset=["AIR YARDS"]).groupby("Air Yards Bin", observed=True)["AIR YARDS"].count()
    data = (counts / counts.sum()).rename("Pct of Passes").reset_index() if counts.sum() else pd.DataFrame()
    bar_chart(data, "Air Yards Bin", "Pct of Passes", "DCHS O Intended Pass Distance", ".0%", AIR_YARDS_ORDER)


def _men_in_box(df, play_type, side, title, exclude=(), good_high=True):
    base = run_pass(df[df[side] == TEAM], [play_type])
    d = keep(base, "men_in_box", "MEN IN BOX")
    d = d[~d["MEN IN BOX"].isin(exclude)]
    t = crosstab(d, "MEN IN BOX", ["Avg Yards Gained", "Success Rate", "EPA per Play", "Plays"])
    show_table(title, t, good_high=good_high)
    tag_note(base, "MEN IN BOX", "men-in-box")
    return t


def render_rush_vs_box(df):
    return _men_in_box(df, "Run", "offense", "DCHS O Rush Metrics vs Men in the Box", exclude=(3, 10))


def render_pass_vs_box(df):
    return _men_in_box(df, "Pass", "offense", "DCHS O Pass Metrics vs Men in the Box")


# ---------------------------------------------------------------------------
# Pass zones heatmap (used on DCHS O / DCHS D / Scout Opposing O pass pages)
#
# PASS ZONE numbering, from the quarterback's view:
#   7 8 9   20+ yards
#   4 5 6   10-19 yards
#   1 2 3   0-9 yards
#   L M R
# ---------------------------------------------------------------------------
ZONE_ROWS = [("20+ yds", [7, 8, 9]), ("10–19 yds", [4, 5, 6]), ("0–9 yds", [1, 2, 3])]
ZONE_COLS = [("Left", [1, 4, 7]), ("Middle", [2, 5, 8]), ("Right", [3, 6, 9])]
LOW_N = 10  # zones with fewer attempts than this get a "low n" flag

# key -> (button label, short label, column in the zone stats frame, formatter)
_pct = lambda v: f"{v:.0%}"
_epa = lambda v: f"{v:+.2f}"
PASS_ZONE_METRICS = {
    "Share of throws": ("Share", "share", lambda v: f"{v:.1%}"),
    "Completion %": ("Comp", "comp", _pct),
    "EPA per play": ("EPA", "epa", _epa),
    "Success rate": ("Succ", "success", _pct),
    "Explosive rate": ("Expl", "explosive", _pct),
}

# Blue (below reference) -> gray (at reference) -> red (above reference).
# (background, text color) pairs, index 0..6 = -3..+3.
_ZONE_STOPS = [
    ("#185FA5", "#E6F1FB"), ("#378ADD", "#042C53"), ("#B5D4F4", "#042C53"),
    ("#F1EFE8", "#2C2C2A"),
    ("#F7C1C1", "#501313"), ("#E24B4A", "#501313"), ("#A32D2D", "#FCEBEB"),
]
# Efficiency metrics (comp %, EPA, success, explosive) use red (bad for
# Dowling) -> gray -> green (good for Dowling) instead, ordered bad to good.
_GOOD_BAD_STOPS = [
    ("#A32D2D", "#FCEBEB"), ("#E24B4A", "#501313"), ("#F7C1C1", "#501313"),
    ("#F1EFE8", "#2C2C2A"),
    ("#9FE1CB", "#04342C"), ("#1D9E75", "#04342C"), ("#0F6E56", "#E1F5EE"),
]
_EMPTY_ZONE = ("rgba(128,128,128,0.12)", "inherit")


def _zone_stats(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-zone attempts/share/comp/epa/success/explosive, plus overall averages."""
    d = d.assign(explosive_play=pd.to_numeric(d["explosive_play"], errors="coerce"))
    g = d.groupby("zone")
    stats = pd.DataFrame({
        "att": g.size(),
        "comp": g["completion"].sum() / g["completion"].count(),
        "epa": g["epa"].mean(),
        "success": g["success"].mean(),
        "explosive": g["explosive_play"].mean(),
    }).reindex(range(1, 10))
    stats["att"] = stats["att"].fillna(0).astype(int)
    stats["share"] = stats["att"] / stats["att"].sum()
    overall = {
        "att": len(d),
        "share": 1 / 9,  # reference for share = an even split across 9 zones
        "comp": d["completion"].sum() / d["completion"].count() if d["completion"].count() else np.nan,
        "epa": d["epa"].mean(),
        "success": d["success"].mean(),
        "explosive": d["explosive_play"].mean(),
    }
    return stats, overall


def _zone_color(stats: pd.DataFrame, col: str, ref: float, value: float, good_high: bool = True) -> tuple[str, str]:
    if pd.isna(value) or pd.isna(ref):
        return _EMPTY_ZONE
    # Scale the color range by zones with a real sample, so one 1-for-1
    # deep shot can't wash every other zone out to gray. Low-n zones still
    # get colored, just clipped to the ends of the scale.
    has_att = stats["att"] > 0
    sized = stats.loc[stats["att"] >= LOW_N, col]
    dev = (sized if sized.notna().any() else stats.loc[has_att, col]).sub(ref).abs().max()
    t = (value - ref) / dev if dev and not pd.isna(dev) else 0.0
    t = float(np.clip(t, -1, 1))
    a = abs(t)
    step = 0 if a < 0.2 else 1 if a < 0.5 else 2 if a < 0.8 else 3
    if col == "share":
        return _ZONE_STOPS[3 + int(np.sign(t)) * step]
    sign = int(np.sign(t)) * (1 if good_high else -1)
    return _GOOD_BAD_STOPS[3 + sign * step]


def _fmt(fn, v) -> str:
    return "–" if pd.isna(v) else fn(v)


def _pass_zone_html(stats: pd.DataFrame, overall: dict, metric: str, good_high: bool = True,
                    ink: str = "inherit") -> str:
    short, col, fn = PASS_ZONE_METRICS[metric]
    ref = overall[col]
    ref_label = "even split" if col == "share" else "avg"
    muted = "opacity:.65"

    kpis = [("Attempts", str(overall["att"])), ("Comp %", _fmt(_pct, overall["comp"])),
            ("EPA / play", _fmt(_epa, overall["epa"])), ("Success", _fmt(_pct, overall["success"])),
            ("Explosive", _fmt(_pct, overall["explosive"]))]
    kpi_html = "".join(
        f'<div style="background:rgba(128,128,128,.08);border-radius:8px;padding:6px 10px">'
        f'<div style="font-size:12px;{muted}">{l}</div>'
        f'<div style="font-size:20px;font-weight:500;font-variant-numeric:tabular-nums">{v}</div></div>'
        for l, v in kpis
    )

    def share_of(zones):
        return f'{stats.loc[zones, "share"].sum():.0%}'

    cells = ['<div></div>'] + [
        f'<div style="text-align:center;font-size:13px">{name}<br>'
        f'<span style="font-size:12px;{muted}">{share_of(zs)} of throws</span></div>'
        for name, zs in ZONE_COLS
    ]
    for label, zones in ZONE_ROWS:
        cells.append(
            f'<div style="display:flex;flex-direction:column;justify-content:center;align-items:flex-end;'
            f'text-align:right;font-size:12px;padding-right:6px;{muted}">{label}<span>{share_of(zones)}</span></div>'
        )
        for z in zones:
            row = stats.loc[z]
            att = int(row["att"])
            bg, fg = _zone_color(stats, col, ref, row[col], good_high) if att else _EMPTY_ZONE
            flag = " · low n" if 0 < att < LOW_N else ""
            others = "".join(
                f"<span>{s} {_fmt(f, row[c])}</span>"
                for m, (s, c, f) in PASS_ZONE_METRICS.items() if m != metric
            )
            cells.append(
                f'<div style="background:{bg};color:{fg};border-radius:8px;padding:9px 11px;min-height:116px;'
                f'display:flex;flex-direction:column;justify-content:space-between">'
                f'<div style="display:flex;justify-content:space-between;font-size:12px;opacity:.85">'
                f'<span>Zone {z}</span><span>{att} att{flag}</span></div>'
                f'<div style="font-size:24px;font-weight:500;font-variant-numeric:tabular-nums">{_fmt(fn, row[col])}</div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:1px 8px;font-size:12px;'
                f'font-variant-numeric:tabular-nums">{others}</div></div>'
            )

    swatches = "".join(
        f'<span style="width:22px;height:12px;border-radius:2px;background:{bg};'
        f'border:0.5px solid rgba(128,128,128,.4)"></span>'
        for bg, _ in (_ZONE_STOPS if col == "share" else _GOOD_BAD_STOPS)
    )
    low_txt, high_txt = (f"Below {ref_label}", f"Above {ref_label}") if col == "share" else \
        ("Worse for Dowling", "Better for Dowling")
    return (
        f'<div style="max-width:820px;font-family:inherit;color:{ink}">'
        f'<div style="display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin-bottom:12px">{kpi_html}</div>'
        f'<div style="display:grid;grid-template-columns:62px repeat(3,minmax(0,1fr));gap:4px">{"".join(cells)}</div>'
        '<div style="display:grid;grid-template-columns:62px 1fr;gap:4px;margin-top:4px">'
        f'<div style="font-size:12px;text-align:right;padding-right:6px;{muted}">LOS</div>'
        '<div style="border-top:2px solid rgba(128,128,128,.6);text-align:center;font-size:12px;padding-top:4px">QB</div></div>'
        f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:12px;margin-top:8px">'
        f'<span style="{muted}">{low_txt}</span>{swatches}<span style="{muted}">{high_txt}</span>'
        f'<span style="{muted};margin-left:6px">{"Even split" if col == "share" else "Avg"}: {_fmt(fn, ref)}</span></div>'
        '</div>'
    )


def _zone_frame(df: pd.DataFrame, side: str, team: str) -> pd.DataFrame:
    d = run_pass(df[df[side] == team], ["Pass"]).copy()
    d["zone"] = pd.to_numeric(d["PASS ZONE"], errors="coerce")
    d = d[d["zone"].between(1, 9)]
    d["zone"] = d["zone"].astype(int)
    return d


def pass_zones_html(df: pd.DataFrame, side: str, team: str, metric: str = "Share of throws",
                    good_high: bool = True, ink: str = "inherit") -> str:
    """HTML for the pass-zone heatmap ("" if no zone-tagged passes)."""
    d = _zone_frame(df, side, team)
    if d.empty:
        return ""
    stats, overall = _zone_stats(d)
    return _pass_zone_html(stats, overall, metric, good_high, ink)


def render_pass_zones(df: pd.DataFrame, side: str, team: str, title: str, key: str,
                      good_high: bool = True) -> None:
    """
    Pass-zone heatmap for passes where `side` ("offense" or "defense") == team.
    Only Pass plays with a PASS ZONE of 1-9 are included. good_high=False
    for defense/scouting views (the opponent completing passes is bad for
    Dowling); share of throws is volume, so it always uses blue-red.
    """
    st.markdown(f"**{title}**")
    d = _zone_frame(df, side, team)
    if d.empty:
        st.info("No pass plays with a PASS ZONE tag match the current filters.")
        return
    metric = st.radio("Shade by", list(PASS_ZONE_METRICS), horizontal=True, key=key)
    stats, overall = _zone_stats(d)
    st.html(_pass_zone_html(stats, overall, metric, good_high))
    tag_note(run_pass(df[df[side] == team], ["Pass"]), "PASS ZONE", "pass zone")


# ---------------------------------------------------------------------------
# Run gaps diagram (used on DCHS O / DCHS D / Scout Opposing O run pages)
#
# GAP tag -> lane: A = between C and G, B = between G and T, C = outside
# the T (D and E are folded into C). PLAY DIR (L/R) picks the side.
# ---------------------------------------------------------------------------
# Matched on the first letter, so tags like "E-Alley" still map correctly.
RUN_GAP_MAP = {"A": "A", "B": "B", "C": "C", "D": "C", "E": "C"}

# (direction, gap) -> (bend x, arrow tip x, bottom label). Bends sit just
# behind the line at y=275; arrows finish at y=110.
_RUN_LANES = [
    ("L", "C", 165, 85, "Left C"),
    ("L", "B", 235, 235, "Left B"),
    ("L", "A", 305, 305, "Left A"),
    ("R", "A", 375, 375, "Right A"),
    ("R", "B", 445, 445, "Right B"),
    ("R", "C", 515, 595, "Right C"),
]
_LINEMEN = [("LT", 200), ("LG", 270), ("C", 340), ("RG", 410), ("RT", 480)]
_POS, _NEG, _NONE = "#1D9E75", "#E24B4A", "rgba(128,128,128,.45)"
_LOW_N = "#9A9890"     # lanes with only a few carries: shown, but not colored good/bad
GAP_MIN_CARRIES = 5


def _run_gaps_svg(runs: pd.DataFrame, lanes: dict, ink: str, good_high: bool = True) -> str:
    """`ink` = color for neutral text/lines (matches the Streamlit theme)."""
    font = 'font-family="Source Sans Pro, Segoe UI, Helvetica, Arial, sans-serif"'
    num = f'{font} font-size="14" font-weight="600"'
    lab = f'{font} font-size="12" fill="{ink}" fill-opacity=".7"'

    td = runs["RESULT"].fillna("").str.contains("TD") & ~runs["RESULT"].fillna("").str.contains("Def TD")
    expl = pd.to_numeric(runs["explosive_play"], errors="coerce").sum()
    epa = runs["epa"].mean()
    stats = [
        (f"{len(runs)}", "carries"),
        (f"{runs['GN/LS'].sum():.0f}", "yards"),
        (f"{runs['GN/LS'].mean():.1f}", "yds / carry"),
        (f"{int(td.sum())}", "rush TD"),
        (f"{int(expl)}", "explosive"),
        ("–" if pd.isna(epa) else f"{epa:+.2f}", "EPA / rush"),
    ]
    parts = []
    for i, (value, label) in enumerate(stats):
        x = 80 + i * 104
        parts.append(f'<text x="{x}" y="28" text-anchor="middle" fill="{ink}" {num}>{value}</text>'
                     f'<text x="{x}" y="46" text-anchor="middle" {lab}>{label}</text>')
    parts.append(f'<line x1="40" y1="62" x2="640" y2="62" stroke="{ink}" stroke-opacity=".2"/>')

    max_n = max([n for _, n in lanes.values()] or [1])
    for d, g, bend_x, tip_x, name in _RUN_LANES:
        mean, n = lanes.get((d, g), (np.nan, 0))
        if n == 0 or pd.isna(mean):
            color = _NONE
        elif n < GAP_MIN_CARRIES:
            color = _LOW_N
        else:
            color = _POS if (mean >= 0) == good_high else _NEG
        dash = ' stroke-dasharray="6 6"' if n == 0 else ""
        width = 3 + 6 * n / max_n  # thicker arrow = run there more often
        parts.append(
            f'<path d="M340 308 L{bend_x} 275 L{tip_x} 110" fill="none" stroke="{color}" stroke-width="{width:.1f}" '
            f'stroke-linecap="round" stroke-linejoin="round"{dash}/>'
        )
        # Arrowhead drawn as its own shape (pointing along the last segment)
        # so it doesn't depend on SVG marker support.
        dx, dy = tip_x - bend_x, 110 - 275
        length = (dx * dx + dy * dy) ** 0.5
        ux, uy = dx / length, dy / length
        bx, by = tip_x - 14 * ux, 110 - 14 * uy
        px, py = -uy * 8, ux * 8
        parts.append(
            f'<polygon points="{tip_x + 2 * ux:.1f},{110 + 2 * uy:.1f} {bx + px:.1f},{by + py:.1f} '
            f'{bx - px:.1f},{by - py:.1f}" fill="{color}"/>'
        )
        value = "–" if n == 0 or pd.isna(mean) else f"{mean:+.2f}"
        parts.append(f'<text x="{tip_x}" y="92" text-anchor="middle" fill="{color}" {num}>{value}</text>')
        parts.append(f'<text x="{tip_x}" y="390" text-anchor="middle" fill="{color}" {num}>{n}</text>'
                     f'<text x="{tip_x}" y="408" text-anchor="middle" {lab}>{name}</text>')

    for name, x in _LINEMEN:
        parts.append(f'<circle cx="{x}" cy="240" r="22" fill="#185FA5"/>'
                     f'<text x="{x}" y="240" text-anchor="middle" dominant-baseline="central" fill="#E6F1FB" {num}>{name}</text>')
    parts.append('<circle cx="340" cy="330" r="22" fill="#BA7517"/>'
                 f'<text x="340" y="330" text-anchor="middle" dominant-baseline="central" fill="#FAEEDA" {num}>RB</text>')
    parts.append(f'<line x1="40" y1="366" x2="640" y2="366" stroke="{ink}" stroke-opacity=".2"/>')

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="680" height="425" viewBox="0 0 680 425">'
           f'{"".join(parts)}</svg>')
    # st.html strips inline <svg>, so the drawing is embedded as an image.
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return (f'<img src="data:image/svg+xml;base64,{b64}" alt="Average EPA per rush and carries by run gap" '
            f'style="width:100%;max-width:760px;height:auto">')


def _run_gap_lanes(df: pd.DataFrame, side: str, team: str):
    runs = run_pass(df[df[side] == team], ["Run"])
    runs = runs[runs["RESULT"] != "Penalty"]
    gap = runs["GAP"].astype("string").str.strip().str.upper().str[:1].map(RUN_GAP_MAP)
    direction = runs["PLAY DIR"].astype("string").str.strip().str.upper()
    tagged = runs.assign(lane_gap=gap, lane_dir=direction)
    tagged = tagged[tagged["lane_gap"].notna() & tagged["lane_dir"].isin(["L", "R"])]
    lanes = {k: (g["epa"].mean(), len(g)) for k, g in tagged.groupby(["lane_dir", "lane_gap"])}
    return runs, tagged, lanes


def run_gaps_html(df: pd.DataFrame, side: str, team: str, good_high: bool = True, ink: str = "#31333F") -> str:
    """HTML (an embedded image) for the run-gap diagram ("" if no runs)."""
    runs, _, lanes = _run_gap_lanes(df, side, team)
    return _run_gaps_svg(runs, lanes, ink, good_high) if not runs.empty else ""


def render_run_gaps(df: pd.DataFrame, side: str, team: str, title: str, good_high: bool = True) -> None:
    """
    Run-gap diagram for runs where `side` ("offense" or "defense") == team.
    Top strip uses every Run play (penalties excluded); arrows use runs with
    both a GAP and a PLAY DIR tag. Arrow color/number = average EPA,
    bottom number = carries. Green = good for Dowling, so good_high=False
    on defense/scouting views.
    """
    st.markdown(f"**{title}**")
    runs, tagged, lanes = _run_gap_lanes(df, side, team)
    if runs.empty:
        st.info("No runs match the current filters.")
        return
    st.html(_run_gaps_svg(runs, lanes, theme_ink(), good_high))
    untagged = len(runs) - len(tagged)
    note = (f"Thicker arrows = more carries. Gray arrows have fewer than {GAP_MIN_CARRIES} carries, so they aren't "
            f"colored good or bad.")
    if untagged:
        missing_gap = int(runs["GAP"].isna().sum())
        missing_dir = int(runs["PLAY DIR"].isna().sum())
        which = ("GAP" if missing_gap >= missing_dir else "PLAY DIR")
        note = (f"{untagged} of {len(runs)} runs aren't shown in the arrows (missing GAP on {missing_gap}, PLAY DIR on "
                f"{missing_dir}; tagging {which} would fill in the most). " + note)
    if len(tagged) < 0.25 * len(runs):
        st.warning(f"Only {len(tagged)} of {len(runs)} runs have both a GAP and PLAY DIR tag, so the arrows don't "
                   f"represent this run game yet.", icon="🏷️")
    st.caption(note)


# ---------------------------------------------------------------------------
# Weekly trends (Dowling Catholic offense, all weeks)
# ---------------------------------------------------------------------------
def week_labels(df: pd.DataFrame) -> dict:
    """WEEK -> 'W1 Valley' using the opponent in each week's game_id."""
    labels = {}
    for week, gid in df.dropna(subset=["WEEK"]).groupby("WEEK")["game_id"].first().items():
        parts = str(gid).split("_")
        opp = parts[1] if len(parts) > 1 else ""
        labels[week] = f"W{int(week)} {opp}".strip()
    return labels


def render_weekly_trend(df, col, title, fmt):
    """Rush and pass lines by week, opponent names on the axis, season average as a dashed rule."""
    st.markdown(f"**{title}**")
    d = run_pass(df[df["offense"] == TEAM])
    if d.empty:
        st.info("No plays match the current filters.")
        return
    labels = week_labels(df)
    data = d.groupby(["WEEK", "PLAY TYPE"])[col].mean().rename("value").reset_index()
    data["Week"] = data["WEEK"].map(labels)
    data["Play type"] = data["PLAY TYPE"].map({"Run": "Rush", "Pass": "Pass"})
    order = [labels[w] for w in sorted(labels)]
    avg = d[col].mean()
    lines = alt.Chart(data).mark_line(point=alt.OverlayMarkDef(size=70, filled=True), strokeWidth=2.5).encode(
        x=alt.X("Week:N", sort=order, axis=alt.Axis(labelAngle=0), title=None),
        y=alt.Y("value:Q", axis=alt.Axis(format=fmt), title=None),
        color=alt.Color("Play type:N", scale=alt.Scale(domain=["Pass", "Rush"], range=["#534AB7", "#BA7517"]),
                        legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.StrokeDash("Play type:N", scale=alt.Scale(domain=["Pass", "Rush"], range=[[1, 0], [6, 4]]),
                                  legend=None),
        tooltip=["Week", "Play type", alt.Tooltip("value:Q", format=fmt, title=title)],
    )
    rule = alt.Chart(pd.DataFrame({"y": [avg]})).mark_rule(strokeDash=[2, 4], color="#888780").encode(y="y:Q")
    st.altair_chart(alt.layer(rule, lines).properties(height=300), width="stretch")
    st.caption(f"Dotted line: season average across all plays ({format(avg, fmt.replace('+', ''))}).")


# Kept for backwards compatibility with older page code.
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
# Dowling Catholic defense (good_high=False: a high number is the opponent
# offense doing well, which is bad for Dowling)
# ---------------------------------------------------------------------------
def render_dchs_defense(df):
    d = run_pass(df[df["defense"] == TEAM])
    t = crosstab(d, "PLAY TYPE", ["Avg Yards Gained", "Success Rate", "EPA per Play", "Explosive Rate", "Plays"])
    show_table("DCHS D Stats", t, good_high=False)
    return t


def render_d_3rd_downs(df):
    d = downs(run_pass(df[df["defense"] == TEAM]), [3])
    t = crosstab(d, "Distance", ["Pass Rate", "Rush Rate", "3rd Down Conversion Rate", "3rd Down Conversions", "Plays"])
    show_table("DCHS D 3rd Downs", t, good_high=False)
    return t


def render_d_rush_vs_box(df):
    return _men_in_box(df, "Run", "defense", "DCHS D Rush Metrics vs Men in the Box", good_high=False)


def d_coverage_table(df):
    d = downs(run_pass(df[df["defense"] == TEAM], ["Pass"]))
    d = keep(d[d["Distance"].notna()], "d_pass_coverage", "COVERAGE")
    return crosstab(d, "COVERAGE", ["Avg Yards Gained", "EPA per Play", "Success Rate", "Explosive Rate", "Plays"],
                    sort_by_count=True)


def render_d_pass_coverage(df, table=None):
    t = d_coverage_table(df) if table is None else table
    show_table("DCHS D Pass Coverage Stats", t, good_high=False)
    tag_note(run_pass(df[df["defense"] == TEAM], ["Pass"]), "COVERAGE", "coverage")
    return t


def render_d_run_game_detail(df):
    """Tableau sheet: DCHS D Run Play Results."""
    base = run_pass(df[df["defense"] == TEAM], ["Run"])
    t = crosstab(keep(base, "d_run_play_results", "OFF PLAY"), "OFF PLAY", PLAY_RESULT_MEASURES)
    show_table("DCHS D Run Game Detail", t, good_high=False)
    tag_note(base, "OFF PLAY", "play call")
    return t


def render_d_pass_game_detail(df):
    """Tableau sheet: DCHS D Pass Play Results."""
    base = run_pass(df[df["defense"] == TEAM], ["Pass"])
    t = crosstab(keep(base, "d_pass_play_results", "OFF PLAY"), "OFF PLAY", PLAY_RESULT_MEASURES)
    show_table("DCHS D Pass Game Detail", t, good_high=False)
    tag_note(base, "OFF PLAY", "play call")
    return t


def d_vs_formation_table(df):
    d = downs(run_pass(df[df["defense"] == TEAM]))
    d = keep(d[d["Distance"].notna()], "d_vs_formation", "OFF FORM")
    return crosstab(d, "OFF FORM", ["Pass Rate", "Rush Rate", "Avg Yards Gained", "Success Rate", "EPA per Play",
                                    "Explosive Rate", "Plays"], sort_by_count=True)


def render_d_vs_formation(df, table=None):
    t = d_vs_formation_table(df) if table is None else table
    show_table("DCHS D vs Formation Stats", t, good_high=False)
    tag_note(run_pass(df[df["defense"] == TEAM]), "OFF FORM", "formation")
    return t


# ---------------------------------------------------------------------------
# Opponent offense (opponent picked in the sidebar)
# ---------------------------------------------------------------------------
def render_opp_tendencies(df, opponent):
    d = downs(run_pass(df[df["offense"] == opponent]))
    t = crosstab(d, ["DN", "Distance"], ["Pass Rate", "Rush Rate", "Plays"])
    show_table(f"{opponent} Offensive Tendencies", t, good_high=False)
    return t


def render_opp_3rd_downs(df, opponent):
    d = downs(run_pass(df[df["offense"] == opponent]), [3])
    t = crosstab(d, "Distance", ["Pass Rate", "Rush Rate", "3rd Down Conversion Rate", "3rd Down Conversions", "Plays"])
    show_table(f"{opponent} Offense 3rd Downs", t, good_high=False)
    return t


def render_opp_4th_downs(df, opponent):
    d = downs(run_pass(df[df["offense"] == opponent]), [4])
    t = crosstab(d, "Distance", ["Pass Rate", "Rush Rate", "4th Down Conversion Rate", "Plays"])
    show_table(f"{opponent} Offense 4th Downs", t, good_high=False)
    return t


def opp_formations_table(df, team):
    d = run_pass(df[df["offense"] == team])
    d = d[d["OFF FORM"].notna()]
    return crosstab(d, "OFF FORM", ["Pass Rate", "Rush Rate", "Avg Yards Gained", "Success Rate", "EPA per Play",
                                    "Explosive Rate", "Plays"], sort_by_count=True)


# ---------------------------------------------------------------------------
# Down-and-distance tendency card (opponent scouting and self-scout)
# ---------------------------------------------------------------------------
DD_DOWNS = [1, 2, 3, 4]
TELL_SHARE = 0.75   # 75%+ one way...
TELL_MIN_N = 5      # ...on at least this many plays = a "tell"


# How sure a tendency is. "Runs 100% on 6 plays" and "runs 83% on 35 plays"
# look alike as percentages, but the second is far more trustworthy. The
# grade uses the Wilson lower bound: the lowest one-way share consistent with
# what we've seen (90% confidence). A 29-of-35 tendency is at least ~71% one
# way; a 6-of-6 tendency could plausibly be only ~69%; 5-of-6 only ~49%.
TENDENCY_GRADES = [(0.70, "Solid"), (0.55, "Likely")]  # else "Small sample"


def wilson_lower(k: float, n: float, z: float = 1.645) -> float:
    if n <= 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((center - margin) / denom)


def tendency_grade(k: float, n: float) -> tuple[str, float]:
    """(grade, lower bound) for k of n plays going the majority way."""
    lb = wilson_lower(k, n)
    for cut, name in TENDENCY_GRADES:
        if lb >= cut:
            return name, lb
    return "Small sample", lb


def dd_tendency_html(df: pd.DataFrame, team: str, ink: str = "inherit") -> str:
    """Grid of downs x distance: run/pass split bar, plays, top formation, tell flag."""
    d = downs(run_pass(df[df["offense"] == team]))
    d = d[d["Distance"].notna()]
    if d.empty:
        return ""
    muted = "opacity:.65"
    head = ['<div></div>'] + [
        f'<div style="font-size:12px;text-align:center;{muted}">{lab.split(" (")[0]}<br>'
        f'{lab.split("(")[1].rstrip(")") if "(" in lab else ""}</div>' for lab in DISTANCE_ORDER
    ]
    cells = list(head)
    present_downs = [dn for dn in DD_DOWNS if (d["DN"] == dn).any()]
    for dn in present_downs:
        cells.append(f'<div style="font-weight:600;font-size:14px;display:flex;align-items:center">'
                     f'{ {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}[dn] }</div>')
        for dist in DISTANCE_ORDER:
            g = d[(d["DN"] == dn) & (d["Distance"] == dist)]
            if g.empty:
                cells.append(f'<div style="background:rgba(128,128,128,.08);border-radius:8px;padding:8px 10px;'
                             f'font-size:12px;{muted}">No plays</div>')
                continue
            n = len(g)
            pr = g["PASS"].mean()
            rr = 1 - pr
            lead = f"Pass {pr:.0%}" if pr >= 0.5 else f"Run {rr:.0%}"
            tell = n >= TELL_MIN_N and max(pr, rr) >= TELL_SHARE
            counts = g["OFF FORM"].value_counts()
            form_txt = (f'{str(counts.index[0]).title()} · {counts.iat[0] / n:.0%}' if len(counts) else "–")
            flag = ('<span style="font-size:11px;padding:1px 6px;border-radius:999px;background:#FAEEDA;'
                    'color:#854F0B;margin-left:6px">tell</span>') if tell else ""
            low = " · low n" if n < TELL_MIN_N else ""
            cells.append(
                f'<div style="background:rgba(128,128,128,.08);border-radius:8px;padding:8px 10px;font-size:12px">'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:6px">'
                f'<div style="font-weight:600;font-size:13px">{lead}{flag}</div>'
                f'<div style="font-size:12px;{muted};white-space:nowrap">{n} plays{low}</div></div>'
                f'<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;margin:6px 0">'
                f'<span style="width:{rr * 100:.0f}%;background:#BA7517"></span>'
                f'<span style="width:{pr * 100:.0f}%;background:#534AB7"></span></div>'
                f'<div>{form_txt}</div></div>'
            )
    legend = (
        f'<div style="display:flex;gap:16px;font-size:12px;margin-bottom:8px;{muted}">'
        '<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#BA7517;'
        'margin-right:4px"></span>Run</span>'
        '<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#534AB7;'
        'margin-right:4px"></span>Pass</span>'
        f'<span>tell = {TELL_SHARE:.0%}+ one way on {TELL_MIN_N}+ plays · bottom line = most-used formation and '
        f'its share of the plays in that box</span></div>'
    )
    return (f'<div style="max-width:860px;color:{ink};font-family:inherit">{legend}'
            f'<div style="display:grid;grid-template-columns:48px repeat(4,minmax(0,1fr));gap:6px">'
            f'{"".join(cells)}</div></div>')


def render_dd_tendencies(df: pd.DataFrame, team: str, title: str) -> None:
    st.markdown(f"**{title}**")
    html = dd_tendency_html(df, team)
    if not html:
        st.info("No plays match the current filters.")
        return
    st.html(html)
