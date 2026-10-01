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
    t = crosstab(d, "PLAY TYPE", ["Avg Yards Gained", "Success Rate", "EPA per Play", "Explosive Rate", "Plays"])
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
                                 "Explosive Rate", "Plays"], sort_by_count=True)
    show_table("DCHS Formations", t)


def render_run_scheme_detail(df):
    d = downs(run_pass(df[df["offense"] == TEAM], ["Run"]))
    d = keep(d[d["Distance"].notna()], "run_scheme", "RUN SCHEME")
    t = crosstab(d, "RUN SCHEME", ["Avg Yards Gained", "Success Rate", "EPA per Play", "Explosive Rate", "Plays"], sort_by_count=True)
    show_table("DCHS O Run Scheme Detail", t)


def render_o_pass_game_detail(df):
    """Tableau sheet: DCHS O Pass Play Results (sorted by play count, like the Tableau shelf sort)."""
    d = run_pass(df[df["offense"] == TEAM], ["Pass"])
    d = keep(d, "o_pass_play_results", "OFF PLAY")
    t = crosstab(d, "OFF PLAY", PLAY_RESULT_MEASURES, sort_by_count=True)
    show_table("DCHS O Pass Game Detail", t)


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


def _zone_color(stats: pd.DataFrame, col: str, ref: float, value: float) -> tuple[str, str]:
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
    return _ZONE_STOPS[3 + int(np.sign(t)) * step]


def _fmt(fn, v) -> str:
    return "–" if pd.isna(v) else fn(v)


def _pass_zone_html(stats: pd.DataFrame, overall: dict, metric: str) -> str:
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
            bg, fg = _zone_color(stats, col, ref, row[col]) if att else _EMPTY_ZONE
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
        for bg, _ in _ZONE_STOPS
    )
    return (
        '<div style="max-width:820px;font-family:inherit">'
        f'<div style="display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin-bottom:12px">{kpi_html}</div>'
        f'<div style="display:grid;grid-template-columns:62px repeat(3,minmax(0,1fr));gap:4px">{"".join(cells)}</div>'
        '<div style="display:grid;grid-template-columns:62px 1fr;gap:4px;margin-top:4px">'
        f'<div style="font-size:12px;text-align:right;padding-right:6px;{muted}">LOS</div>'
        '<div style="border-top:2px solid rgba(128,128,128,.6);text-align:center;font-size:12px;padding-top:4px">QB</div></div>'
        f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:12px;margin-top:8px">'
        f'<span style="{muted}">Below {ref_label}</span>{swatches}<span style="{muted}">Above {ref_label}</span>'
        f'<span style="{muted};margin-left:6px">{"Even split" if col == "share" else "Avg"}: {_fmt(fn, ref)}</span></div>'
        '</div>'
    )


def render_pass_zones(df: pd.DataFrame, side: str, team: str, title: str, key: str) -> None:
    """
    Pass-zone heatmap for passes where `side` ("offense" or "defense") == team.
    Only Pass plays with a PASS ZONE of 1-9 are included.
    """
    st.markdown(f"**{title}**")
    d = run_pass(df[df[side] == team], ["Pass"]).copy()
    d["zone"] = pd.to_numeric(d["PASS ZONE"], errors="coerce")
    d = d[d["zone"].between(1, 9)]
    d["zone"] = d["zone"].astype(int)
    if d.empty:
        st.info("No pass plays with a PASS ZONE tag match the current filters.")
        return
    metric = st.radio("Shade by", list(PASS_ZONE_METRICS), horizontal=True, key=key)
    stats, overall = _zone_stats(d)
    st.html(_pass_zone_html(stats, overall, metric))


# ---------------------------------------------------------------------------
# Run gaps diagram (used on DCHS O / DCHS D / Scout Opposing O run pages)
#
# GAP tag -> lane: A = between C and G, B = between G and T, C = outside
# the T (D and E are folded into C). PLAY DIR (L/R) picks the side.
# ---------------------------------------------------------------------------
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


def _run_gaps_svg(runs: pd.DataFrame, lanes: dict) -> str:
    font = 'font-family="inherit"'
    num = f'{font} font-size="14" font-weight="500"'
    lab = f'{font} font-size="12" fill="currentColor" opacity=".7"'

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
        parts.append(f'<text x="{x}" y="28" text-anchor="middle" fill="currentColor" {num}>{value}</text>'
                     f'<text x="{x}" y="46" text-anchor="middle" {lab}>{label}</text>')
    parts.append('<line x1="40" y1="62" x2="640" y2="62" stroke="currentColor" stroke-opacity=".2"/>')

    for d, g, bend_x, tip_x, name in _RUN_LANES:
        mean, n = lanes.get((d, g), (np.nan, 0))
        color = _NONE if n == 0 or pd.isna(mean) else (_POS if mean >= 0 else _NEG)
        dash = ' stroke-dasharray="6 6"' if n == 0 else ""
        parts.append(
            f'<path d="M340 308 L{bend_x} 275 L{tip_x} 110" fill="none" stroke="{color}" stroke-width="5" '
            f'stroke-linecap="round" stroke-linejoin="round"{dash} marker-end="url(#rg-arrow)" '
            f'style="color:{color}"/>'
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
    parts.append('<line x1="40" y1="366" x2="640" y2="366" stroke="currentColor" stroke-opacity=".2"/>')

    marker = ('<defs><marker id="rg-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4" markerHeight="4" '
              'orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" '
              'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>')
    return (f'<div style="max-width:760px"><svg width="100%" viewBox="0 0 680 425" role="img" '
            f'aria-label="Average EPA per rush and carries by run gap">{marker}{"".join(parts)}</svg></div>')


def render_run_gaps(df: pd.DataFrame, side: str, team: str, title: str) -> None:
    """
    Run-gap diagram for runs where `side` ("offense" or "defense") == team.
    Top strip uses every Run play (penalties excluded); arrows use runs with
    both a GAP and a PLAY DIR tag. Arrow color/number = average EPA,
    bottom number = carries.
    """
    st.markdown(f"**{title}**")
    runs = run_pass(df[df[side] == team], ["Run"])
    runs = runs[runs["RESULT"] != "Penalty"]
    if runs.empty:
        st.info("No runs match the current filters.")
        return

    gap = runs["GAP"].astype("string").str.strip().str.upper().map(RUN_GAP_MAP)
    direction = runs["PLAY DIR"].astype("string").str.strip().str.upper()
    tagged = runs.assign(lane_gap=gap, lane_dir=direction)
    tagged = tagged[tagged["lane_gap"].notna() & tagged["lane_dir"].isin(["L", "R"])]
    lanes = {k: (g["epa"].mean(), len(g)) for k, g in tagged.groupby(["lane_dir", "lane_gap"])}

    st.html(_run_gaps_svg(runs, lanes))
    untagged = len(runs) - len(tagged)
    if untagged:
        st.caption(f"{untagged} of {len(runs)} runs are missing a GAP or PLAY DIR tag and aren't shown in the arrows.")


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
    t = crosstab(d, "PLAY TYPE", ["Avg Yards Gained", "Success Rate", "EPA per Play", "Explosive Rate", "Plays"])
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
    t = crosstab(d, "COVERAGE", ["Avg Yards Gained", "EPA per Play", "Success Rate", "Explosive Rate", "Plays"],
                 sort_by_count=True)
    show_table("DCHS D Pass Coverage Stats", t)


def render_d_run_game_detail(df):
    """Tableau sheet: DCHS D Run Play Results."""
    d = run_pass(df[df["defense"] == TEAM], ["Run"])
    d = keep(d, "d_run_play_results", "OFF PLAY")
    t = crosstab(d, "OFF PLAY", PLAY_RESULT_MEASURES)
    show_table("DCHS D Run Game Detail", t)


def render_d_pass_game_detail(df):
    """Tableau sheet: DCHS D Pass Play Results."""
    d = run_pass(df[df["defense"] == TEAM], ["Pass"])
    d = keep(d, "d_pass_play_results", "OFF PLAY")
    t = crosstab(d, "OFF PLAY", PLAY_RESULT_MEASURES)
    show_table("DCHS D Pass Game Detail", t)


def render_d_vs_formation(df):
    d = downs(run_pass(df[df["defense"] == TEAM]))
    d = keep(d[d["Distance"].notna()], "d_vs_formation", "OFF FORM")
    t = crosstab(d, "OFF FORM", ["Pass Rate", "Rush Rate", "Avg Yards Gained", "Success Rate", "EPA per Play",
                                 "Explosive Rate", "Plays"], sort_by_count=True)
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
