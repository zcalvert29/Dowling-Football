"""
pass_chart.py — a pass chart: every throw drawn where it was caught (or
thrown to) on a perspective field, styled like the rest of the dashboard
(Streamlit light/dark theme, the shared green/red/blue, gray stat tiles).

We don't tag exact catch locations, so each throw is placed from two tags:
  * depth  = AIR YARDS (exact), with a hair of jitter so stacked throws separate;
  * side   = PASS ZONE, 1-3 / 4-6 / 7-9 are left / middle / right, placed at a
             random spot inside that third of the field (the same spot every time).
High school hash marks split the field into exact thirds (53'4" from each
sideline), so the left / middle / right zones line up with the hashes.
If AIR YARDS is missing, depth comes from the zone's band (0-9, 10-19, 20+).
If PASS ZONE is missing the throw can't be placed and is left off (it still
counts in the stat line).

Markers: completion = filled green, touchdown = filled blue, incomplete =
hollow ring, interception = filled red.

No Streamlit code here: st.html(pass_chart_html(...)). The field is an SVG
embedded as an image, the same way the run-gap diagram is drawn (st.html
strips inline SVG), so colors are passed in from the page theme.
"""
from __future__ import annotations

import base64
import html
import zlib

import numpy as np
import pandas as pd

FIELD_W = 160 / 3            # 53 1/3 yards
HASH_X = (FIELD_W / 3, 2 * FIELD_W / 3)  # high school hashes: exact thirds
Y_MIN, Y_MAX = -10, 50       # yards from the line of scrimmage shown

THROWN = ("Complete", "Incomplete", "Interception")

# Dashboard colors (insights.GREEN / RED, the app's blue). Lines and field use
# translucent gray so they read on both the light and dark theme.
GREEN, RED, BLUE = "#1D9E75", "#E24B4A", "#185FA5"
OUTCOME_COLOR = {"Complete": GREEN, "Touchdown": BLUE, "Interception": RED}
FIELD_FILL = "rgba(128,128,128,.07)"
SIDELINE_FILL = "rgba(128,128,128,.22)"
LINE = "rgba(128,128,128,.38)"
TICK = "rgba(128,128,128,.32)"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def throws(df: pd.DataFrame, team: str, side: str = "offense") -> pd.DataFrame:
    """Every pass attempt where `side` ("offense" or "defense") == team (sacks, scrambles, penalties left out),
    with chart positions."""
    d = df[(df[side] == team) & (df["PLAY TYPE"].astype(str) == "Pass")].copy()
    res = d["RESULT"].astype(str)
    d = d[res.str.startswith(THROWN)].copy()
    res = d["RESULT"].astype(str)
    d["is_int"] = res.str.startswith("Interception")
    d["is_comp"] = res.str.startswith("Complete")
    d["is_td"] = d["is_comp"] & res.str.contains("TD")
    d["outcome"] = np.select([d["is_int"], d["is_td"], d["is_comp"]], ["Interception", "Touchdown", "Complete"],
                             "Incomplete")

    zone = pd.to_numeric(d["PASS ZONE"], errors="coerce")
    air = pd.to_numeric(d["AIR YARDS"], errors="coerce")
    band_mid = ((zone - 1) // 3).map({0: 5.0, 1: 15.0, 2: 27.0})
    depth = air.fillna(band_mid)
    third = (zone - 1) % 3  # 0 left, 1 middle, 2 right

    # Stable jitter: the same throw lands in the same spot every time the chart is drawn.
    xs, ys = [], []
    for idx, t, dep in zip(d.index, third, depth):
        if pd.isna(t) or pd.isna(dep):
            xs.append(np.nan); ys.append(np.nan); continue
        rng = np.random.default_rng(zlib.crc32(f"{d.at[idx, 'game_id']}|{idx}".encode()))
        lo = t * FIELD_W / 3
        xs.append(lo + rng.uniform(2.2, FIELD_W / 3 - 2.2))
        y = dep + rng.uniform(-0.6, 0.6)
        if dep >= 0:
            y = max(y, 0.0)  # a throw at or past the line never drifts behind it
        ys.append(float(np.clip(y, Y_MIN + 1, Y_MAX - 1)))
    d["x"], d["y"] = xs, ys
    return d


def stat_line(t: pd.DataFrame) -> dict:
    comp = int(t["is_comp"].sum())
    att = len(t)
    yards = int(pd.to_numeric(t.loc[t["is_comp"], "GN/LS"], errors="coerce").fillna(0).sum())
    air = pd.to_numeric(t["AIR YARDS"], errors="coerce").dropna()
    return {"comp": comp, "att": att, "yards": yards, "td": int(t["is_td"].sum()), "int": int(t["is_int"].sum()),
            "adot": float(air.mean()) if len(air) else float("nan"), "plotted": int(t["x"].notna().sum())}


def stat_line(t: pd.DataFrame) -> dict:
    comp = int(t["is_comp"].sum())
    att = len(t)
    yards = int(pd.to_numeric(t.loc[t["is_comp"], "GN/LS"], errors="coerce").fillna(0).sum())
    air = pd.to_numeric(t["AIR YARDS"], errors="coerce").dropna()
    return {"comp": comp, "att": att, "yards": yards, "td": int(t["is_td"].sum()), "int": int(t["is_int"].sum()),
            "adot": float(air.mean()) if len(air) else float("nan"), "plotted": int(t["x"].notna().sum())}


# ---------------------------------------------------------------------------
# Drawing (SVG)
# ---------------------------------------------------------------------------
VB_W, VB_H = 760, 470
_NEAR_W = 600          # field width at the bottom edge, px
_FIELD_TOP, _FIELD_BOT = 16, 452


def _project(x, y):
    """Field (x across, y downfield from LOS) -> SVG px, with a gentle perspective. Returns (px, py, scale)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    v = np.clip((y - Y_MIN) / (Y_MAX - Y_MIN), 0, 1)   # 0 near, 1 far
    scale = 1.0 - 0.36 * v
    frac = 1 - (1 - v) ** 1.18
    px = VB_W / 2 + (x - FIELD_W / 2) / FIELD_W * _NEAR_W * scale
    py = _FIELD_BOT - frac * (_FIELD_BOT - _FIELD_TOP)
    return px, py, scale


def _seg(x0, y0, x1, y1, color, width):
    (a, c), (b, d), _ = _project([x0, x1], [y0, y1])
    return f'<line x1="{a:.1f}" y1="{b:.1f}" x2="{c:.1f}" y2="{d:.1f}" stroke="{color}" stroke-width="{width}"/>'


def _poly(pts, fill):
    px, py, _ = _project([p[0] for p in pts], [p[1] for p in pts])
    return f'<polygon points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py))}" fill="{fill}"/>'


def _field_svg() -> str:
    out = [_poly([(-2.2, Y_MIN), (0, Y_MIN), (0, Y_MAX), (-2.2, Y_MAX)], SIDELINE_FILL),
           _poly([(FIELD_W, Y_MIN), (FIELD_W + 2.2, Y_MIN), (FIELD_W + 2.2, Y_MAX), (FIELD_W, Y_MAX)], SIDELINE_FILL),
           _poly([(0, Y_MIN), (FIELD_W, Y_MIN), (FIELD_W, Y_MAX), (0, Y_MAX)], FIELD_FILL)]
    for y in range(Y_MIN, Y_MAX + 1, 5):
        if y != 0:
            out.append(_seg(0, y, FIELD_W, y, LINE, 1.4 if y % 10 == 0 else 0.8))
        if y > 0 and y % 10 == 0:
            for x, anchor in ((-3.4, "end"), (FIELD_W + 3.4, "start")):
                px, py, s = _project(x, y)
                out.append(f'<text x="{float(px):.1f}" y="{float(py):.1f}" text-anchor="{anchor}" '
                           f'dominant-baseline="central" font-size="{12 * (0.8 + 0.2 * float(s)):.1f}" '
                           f'font-weight="600" fill="currentColor" opacity=".7">+{y}</text>')
        for xc in HASH_X:
            out.append(_seg(xc, y - 0.35, xc, y + 0.35, TICK, 1))
    # One-yard ticks: sidelines and the high school hashes.
    for y in range(Y_MIN + 1, Y_MAX):
        if y % 5 == 0:
            continue
        out.append(_seg(0, y, 0.9, y, TICK, 0.8))
        out.append(_seg(FIELD_W - 0.9, y, FIELD_W, y, TICK, 0.8))
        for xc in HASH_X:
            out.append(_seg(xc - 0.55, y, xc + 0.55, y, TICK, 0.9))
    # Line of scrimmage.
    out.append(_seg(-2.2, 0, FIELD_W + 2.2, 0, BLUE, 3))
    for x, anchor in ((-3.4, "end"), (FIELD_W + 3.4, "start")):
        px, py, _ = _project(x, 0)
        out.append(f'<text x="{float(px):.1f}" y="{float(py):.1f}" text-anchor="{anchor}" dominant-baseline="central" '
                   f'font-size="12" font-weight="600" fill="currentColor" opacity=".7">LOS</text>')
    # Thirds under the field: they match the pass zones.
    for i, name in enumerate(("Left", "Middle", "Right")):
        px, py, _ = _project((i + 0.5) * FIELD_W / 3, Y_MIN)
        out.append(f'<text x="{float(px):.1f}" y="{float(py) + 13:.1f}" text-anchor="middle" font-size="11" '
                   f'fill="currentColor" opacity=".55">{name}</text>')
    return "".join(out)


def _throws_svg(t: pd.DataFrame, ink: str, bg: str) -> str:
    t = t[t["x"].notna()].sort_values("y", ascending=False)  # far throws first, near ones drawn on top
    n = len(t)
    base_r = 9.0 if n <= 40 else 7.5 if n <= 90 else 6.0
    px, py, s = _project(t["x"].to_numpy(), t["y"].to_numpy())
    out = []
    for (_, r), x, y, sc in zip(t.iterrows(), px, py, s):
        rad = base_r * (0.7 + 0.3 * sc)
        o = r["outcome"]
        if o == "Incomplete":
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad:.1f}" fill="{bg}" fill-opacity=".55" '
                       f'stroke="{ink}" stroke-width="2" stroke-opacity=".85"/>')
        else:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad:.1f}" fill="{OUTCOME_COLOR[o]}" '
                       f'stroke="{bg}" stroke-width="1.5"/>')
    return "".join(out)


def _legend_html() -> str:
    dot = lambda style: (f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
                         f'box-sizing:border-box;{style}"></span>')
    items = [(dot(f"background:{GREEN}"), "Complete"), (dot(f"background:{BLUE}"), "Touchdown"),
             (dot("border:2px solid currentColor;opacity:.85"), "Incomplete"), (dot(f"background:{RED}"), "Interception")]
    return ('<div style="display:flex;flex-wrap:wrap;gap:16px;font-size:12px;margin-top:6px">'
            + "".join(f'<span style="display:inline-flex;align-items:center;gap:6px">{d}{lab}</span>' for d, lab in items)
            + "</div>")


def pass_chart_html(df: pd.DataFrame, team: str, ink: str = "#31333F", bg: str = "#FFFFFF",
                    side: str = "offense") -> str:
    """HTML for the pass chart of the throws where `side` == team over the plays in `df` ("" if none).
    ink / bg: the page's text and background colors (visuals.theme_ink / theme_bg)."""
    t = throws(df, team, side)
    offense = team
    if t.empty:
        return ""
    st = stat_line(t)
    muted = "opacity:.65"
    pct = f"{st['comp'] / st['att']:.0%}" if st["att"] else "—"
    adot = "—" if np.isnan(st["adot"]) else f"{st['adot']:.1f}"
    kpis = [("Comp / Att", f"{st['comp']}/{st['att']}"), ("Comp %", pct), ("Yards", f"{st['yards']}"),
            ("TD-INT", f"{st['td']}-{st['int']}"), ("Avg depth", f"{adot} yds")]
    kpi_html = "".join(
        f'<div style="background:rgba(128,128,128,.08);border-radius:8px;padding:6px 10px">'
        f'<div style="font-size:12px;{muted}">{lab}</div>'
        f'<div style="font-size:20px;font-weight:500;font-variant-numeric:tabular-nums">{val}</div></div>'
        for lab, val in kpis)
    notes = ["Depth is air yards; side comes from the pass zone, so a dot's spot within its third is approximate."]
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VB_W} {VB_H}" '
           f'font-family="Source Sans Pro, -apple-system, Segoe UI, Helvetica, Arial, sans-serif" color="{ink}">'
           f'{_field_svg().replace("currentColor", ink)}{_throws_svg(t, ink, bg)}</svg>')
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    if st["plotted"] < st["att"]:
        notes.append(f"{st['att'] - st['plotted']} throw(s) without a pass zone are counted but not drawn.")
    return (
        f'<div style="max-width:820px;font-family:inherit;color:{ink}">'
        f'<div style="display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin-bottom:10px">'
        f'{kpi_html}</div>'
        f'<img src="data:image/svg+xml;base64,{b64}" alt="{html.escape(offense)} pass chart" '
        f'style="width:100%;height:auto;display:block">'
        f'{_legend_html()}'
        f'<div style="font-size:12px;{muted};margin-top:6px">{" ".join(notes)}</div>'
        "</div>"
    )
