"""
Deeper breakdowns built on the same curated play-by-play:

    1. Tendency tree        formation -> run/pass -> direction -> play (icicle chart)
    2. Field & strength     hash x formation strength: run/pass, to the field, to strength
    3. Sequencing           what comes after a run, a pass, a big gain, a loss...
    4. Red zone             trips, points per trip, how it's called inside the 20 / 10 / 5
    5. Season drives        points per drive by starting spot, drive funnel, 3-and-outs
    6. 4th-down trend       how often Dowling went when the model said go, game by game

Every function takes the already-filtered frame the page hands it, so the
sidebar filters work the same way they do everywhere else.
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import insights as ins
import visuals as v

RUN_C, PASS_C, NEUTRAL_C = "#BA7517", "#534AB7", "#6E6C66"
DCHS_C, OPP_C = "#8C1D2C", "#185FA5"
GOOD_BG, BAD_BG = "#E1F5EE", "#FCEBEB"
CARD = "background:rgba(128,128,128,.08);border-radius:8px;padding:8px 10px"
BADGE = {"Solid": ("tell", "#FCEBEB", "#791F1F"), "Likely": ("lean", "#FAEEDA", "#854F0B"),
         "Small sample": ("small sample", "rgba(128,128,128,.15)", "inherit")}
MIN_TELL_SHARE = 0.70  # below this a split isn't called a tendency at all


def _badge(k: int, n: int) -> str:
    """Confidence pill for k of n plays going one way (same grading as the Home page)."""
    if n < v.TELL_MIN_N or k / n < MIN_TELL_SHARE:
        return ""
    label, bg, fg = BADGE[v.tendency_grade(k, n)[0]]
    return (f'<span style="font-size:11px;padding:1px 7px;border-radius:999px;background:{bg};color:{fg};'
            f'margin-left:6px;white-space:nowrap">{label}</span>')


def _split_bar(run_share: float, height: int = 6) -> str:
    return (f'<div style="display:flex;height:{height}px;border-radius:3px;overflow:hidden;margin:5px 0">'
            f'<span style="width:{run_share * 100:.0f}%;background:{RUN_C}"></span>'
            f'<span style="width:{(1 - run_share) * 100:.0f}%;background:{PASS_C}"></span></div>')


LEGEND = (f'<div style="display:flex;gap:14px;font-size:12px;margin:2px 0 8px">'
          f'<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:{RUN_C};'
          f'margin-right:4px"></span>Run</span>'
          f'<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:{PASS_C};'
          f'margin-right:4px"></span>Pass</span>'
          f'<span style="opacity:.7">tell / lean / small sample = how much to trust a 70%+ split</span></div>')


def _team_plays(df: pd.DataFrame, team: str, side: str = "offense") -> pd.DataFrame:
    """Run/pass snaps where `team` is on `side` (offense = their offense, defense = offenses against them)."""
    return v.run_pass(df[df[side] == team]).copy()


# ==============================================================================
# 1. TENDENCY TREE
# ==============================================================================
TREE_LEVELS = ["Formation", "Run / pass", "Direction", "Play"]


def tendency_tree_frame(d: pd.DataFrame, depth: int = 3, min_form_plays: int | None = None) -> pd.DataFrame:
    """
    One row per node of formation -> run/pass -> direction (-> play), laid
    out as stacked rectangles: y0/y1 are cumulative play counts, so each
    node's height is its number of plays. Formations with fewer than
    min_form_plays plays (default: 4, or 3% of the plays if that's more)
    are folded into "Other" so the chart doesn't turn into slivers.
    """
    d = d.copy()
    if min_form_plays is None:
        min_form_plays = max(4, int(np.ceil(0.03 * len(d))))
    d["Formation"] = d["OFF FORM"].astype("string").str.strip().str.title().fillna("Untagged")
    counts = d["Formation"].value_counts()
    d.loc[d["Formation"].isin(counts[counts < min_form_plays].index), "Formation"] = "Other"
    d["Run / pass"] = d["PLAY TYPE"].astype(str)
    d["Direction"] = d["PLAY DIR"].astype("string").str.upper().map({"L": "Left", "R": "Right"}).fillna("No dir.")
    d["Play"] = d["OFF PLAY"].astype("string").str.strip().str.title().fillna("Untagged")
    levels = TREE_LEVELS[:depth]
    total = len(d)
    rows = []
    last = ("Other", "Untagged", "No dir.")  # catch-all buckets sort to the bottom of their parent

    def walk(sub: pd.DataFrame, i: int, y0: float, path: list[str], kind: str) -> None:
        if i == len(levels):
            return
        col = levels[i]
        sizes = sub[col].value_counts()
        order = sorted(sizes.index, key=lambda k: (k in last, -sizes[k]))
        y = y0
        for key in order:
            part = sub[sub[col] == key]
            n = len(part)
            node_kind = key if col == "Run / pass" else kind
            rows.append({
                "level": i, "label": str(key), "n": n, "y0": y, "y1": y + n, "kind": node_kind,
                "path": " › ".join(path + [str(key)]), "share_parent": n / len(sub), "share_total": n / total,
                "run": 1 - part["PASS"].mean(), "success": part["success"].mean(), "epa": part["epa"].mean(),
            })
            walk(part, i + 1, y, path + [str(key)], node_kind)
            y += n

    walk(d, 0, 0, [], "Formation")
    return pd.DataFrame(rows)


def tendency_tree_chart(t: pd.DataFrame, depth: int, height: int = 560) -> alt.Chart:
    total = t.loc[t["level"] == 0, "n"].sum()
    pad = total * 0.0025  # thin gap between stacked boxes
    t = t.assign(
        x0=t["level"] + 0.012, x1=t["level"] + 0.988, ya=t["y0"] + pad, yb=t["y1"] - pad,
        ym=(t["y0"] + t["y1"]) / 2, tx=t["level"] + 0.04,
        # Deeper levels get lighter so the run/pass split stays the loudest thing on the chart.
        op=t["level"].map({0: 1.0, 1: 1.0, 2: 0.72, 3: 0.5}),
        text=[f"{(lab[:20] + '…') if len(lab) > 21 else lab}  {n}" for lab, n in zip(t["label"], t["n"])],
        detail=[f"{p:.0%} of parent · {s:.0%} of all plays" for p, s in zip(t["share_parent"], t["share_total"])],
        runpass=[f"Run {r:.0%} · Pass {1 - r:.0%}" for r in t["run"]],
    )
    y = alt.Y("ya:Q", axis=None, scale=alt.Scale(domain=[0, total], reverse=True, nice=False))
    x = alt.X("x0:Q", axis=alt.Axis(values=[i + 0.5 for i in range(depth)], labelExpr=f"{TREE_LEVELS[:depth]}"
                                    f"[floor(datum.value)]", title=None, orient="top", ticks=False, domain=False,
                                    grid=False, labelFontWeight="bold"),
              scale=alt.Scale(domain=[0, depth], nice=False))
    rect = alt.Chart(t).mark_rect(cornerRadius=2).encode(
        x=x, x2="x1:Q", y=y, y2="yb:Q",
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Formation", "Run", "Pass"],
                                                  range=[NEUTRAL_C, RUN_C, PASS_C]),
                        legend=alt.Legend(orient="top", title=None)),
        opacity=alt.Opacity("op:Q", scale=None, legend=None),
        tooltip=[alt.Tooltip("path:N", title="Path"), alt.Tooltip("n:Q", title="Plays"),
                 alt.Tooltip("detail:N", title="Share"), alt.Tooltip("runpass:N", title="Split"),
                 alt.Tooltip("success:Q", title="Success rate", format=".0%"),
                 alt.Tooltip("epa:Q", title="EPA / play", format="+.2f")],
    )
    # Labels only where the box is tall enough to hold one (~3% of the plays).
    labels = alt.Chart(t[t["n"] >= total * 0.03]).mark_text(align="left", baseline="middle", color="white",
                                                             fontSize=11, fontWeight=500).encode(
        x=alt.X("tx:Q", scale=alt.Scale(domain=[0, depth], nice=False)),
        y=alt.Y("ym:Q", scale=alt.Scale(domain=[0, total], reverse=True, nice=False)), text="text:N")
    return (rect + labels).properties(height=height)


def _tree_takeaways(t: pd.DataFrame, top: int = 3) -> list[str]:
    """Plain-language read of the biggest formations: what they do out of each one."""
    out = []
    forms = t[(t["level"] == 0) & ~t["label"].isin(["Other", "Untagged"])].nlargest(top, "n")
    for _, f in forms.iterrows():
        n, run = int(f["n"]), f["run"]
        kind, share = ("runs", run) if run >= 0.5 else ("throws", 1 - run)
        k = round(share * n)
        line = f"**{f['label']}** ({n} plays): {kind} {share:.0%}"
        dirs = t[(t["level"] == 2) & t["path"].str.startswith(f["path"] + " › ")
                 & (t["kind"] == ("Run" if kind == "runs" else "Pass")) & (t["label"] != "No dir.")]
        if len(dirs) and dirs["n"].sum() >= v.TELL_MIN_N:
            best = dirs.groupby("label")["n"].sum()
            side, m = best.idxmax(), best.max()
            if m / best.sum() >= 0.6:
                line += f", and those go {side.lower()} {m / best.sum():.0%} of the time ({m} of {best.sum()})"
            else:
                line += f", split about evenly left and right"
        badge = _badge(k, n)
        out.append(line + (f" {badge}" if badge else ""))
    return out


def render_tendency_tree(df: pd.DataFrame, team: str, title: str, side: str = "offense", key: str = "tree") -> None:
    st.markdown(f"**{title}**")
    d = _team_plays(df, team, side)
    if len(d) < 5:
        st.info("Not enough plays match the current filters.")
        return
    play_tagged = d["OFF PLAY"].notna().mean()
    options = ["Direction", "Play"] if play_tagged >= 0.5 else ["Direction"]
    depth_label = st.segmented_control("Show down to", options, default="Direction", key=f"{key}_depth") \
        if len(options) > 1 else "Direction"
    depth = 4 if depth_label == "Play" else 3
    t = tendency_tree_frame(d, depth=depth)
    fold = max(4, int(np.ceil(0.03 * len(d))))
    lines = _tree_takeaways(t)
    if lines:
        st.html('<div style="font-size:14px;line-height:1.7">' + "<br>".join(
            _md_bold(x) for x in lines) + "</div>")
    st.altair_chart(tendency_tree_chart(t, depth), width="stretch")
    note = ("Read left to right: each box's height is how many plays it covers. Hover any box for its split, "
            f"success rate and EPA. Formations with under {fold} plays are grouped as Other.")
    if play_tagged < 0.5:
        note += f" Play calls are tagged on only {play_tagged:.0%} of these snaps, so the Play level is hidden."
    st.caption(note)


def _md_bold(s: str) -> str:
    """**x** -> <b>x</b> for lines rendered through st.html."""
    parts = s.split("**")
    return "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))


# ==============================================================================
# 2. FIELD / BOUNDARY x FORMATION STRENGTH
# ==============================================================================
HASHES = [("L", "Left hash"), ("M", "Middle"), ("R", "Right hash")]
STRENGTHS = [("L", "Strength left"), ("BAL", "Balanced"), ("R", "Strength right")]


def _field_strength_flags(d: pd.DataFrame) -> pd.DataFrame:
    """
    to_field: the play went to the wide side (from the left hash the field is
    to the right). to_strength: the play went toward the formation's strength.
    Both are NaN when the tags needed aren't there. Hash, strength, and
    direction all use the same left/right, so these hold whichever way the
    film is shot.
    """
    d = d.copy()
    dir_ = d["PLAY DIR"].astype("string").str.upper()
    has_dir = dir_.isin(["L", "R"])
    field_side = d["HASH"].astype("string").str.upper().map({"L": "R", "R": "L"})
    d["to_field"] = np.where(has_dir & field_side.notna(), (dir_ == field_side).astype(float), np.nan)
    stren = d["OFF STR"].astype("string").str.upper()
    d["to_strength"] = np.where(has_dir & stren.isin(["L", "R"]), (dir_ == stren).astype(float), np.nan)
    return d


def _cell_html(c: pd.DataFrame, title: str = "") -> str:
    n = len(c)
    if n == 0:
        return f'<div style="{CARD};opacity:.45;font-size:12px">{title}<div>no plays</div></div>'
    run = 1 - c["PASS"].mean()
    k = round(max(run, 1 - run) * n)
    lead = f"Run {run:.0%}" if run >= 0.5 else f"Pass {1 - run:.0%}"
    bits = []
    for col, word in (("to_field", "to field"), ("to_strength", "to strength")):
        s = c[col].dropna()
        if len(s):
            share = s.mean()
            side_word = word if share >= 0.5 else word.replace("field", "boundary").replace("to strength",
                                                                                              "away from strength")
            kk = round(max(share, 1 - share) * len(s))
            bits.append(f'{max(share, 1 - share):.0%} {side_word} ({kk}/{len(s)}){_badge(kk, len(s))}')
    faded = "opacity:.55;" if n < v.TELL_MIN_N else ""
    return (f'<div style="{CARD};{faded}font-size:12px;line-height:1.5">'
            + (f'<div style="opacity:.6;font-size:11px">{title}</div>' if title else "")
            + f'<div><b style="font-size:14px">{lead}</b>{_badge(k, n)}<span style="float:right;opacity:.6">'
              f'{n} plays</span></div>{_split_bar(run)}'
            + "".join(f"<div>{b}</div>" for b in bits) + "</div>")


def field_strength_html(d: pd.DataFrame) -> str:
    d = _field_strength_flags(d)
    hash_ = d["HASH"].astype("string").str.upper()
    stren = d["OFF STR"].astype("string").str.upper()
    head = "".join(f'<div style="font-size:12px;font-weight:600;text-align:center">{lab}</div>'
                   for _, lab in STRENGTHS + [("all", "All")])
    body = []
    for h, hlab in HASHES:
        row = [f'<div style="font-size:12px;font-weight:600;align-self:center">{hlab}</div>']
        for s, _ in STRENGTHS:
            row.append(_cell_html(d[(hash_ == h) & (stren == s)]))
        row.append(_cell_html(d[hash_ == h]))
        body.append("".join(row))
    total_row = ['<div style="font-size:12px;font-weight:600;align-self:center">All</div>']
    total_row += [_cell_html(d[stren == s]) for s, _ in STRENGTHS] + [_cell_html(d)]
    grid = ('<div style="display:grid;grid-template-columns:90px repeat(4,minmax(150px,1fr));gap:6px;'
            'min-width:720px">' + '<div></div>' + head + "".join(body) + "".join(total_row) + "</div>")
    return LEGEND + f'<div style="overflow-x:auto">{grid}</div>'


def render_field_strength(df: pd.DataFrame, team: str, title: str, side: str = "offense") -> None:
    st.markdown(f"**{title}**")
    d = _team_plays(df, team, side)
    if d.empty:
        st.info("No plays match the current filters.")
        return
    # Where the formation's strength is set relative to the field: a tell on its own, before the snap.
    hash_ = d["HASH"].astype("string").str.upper()
    stren = d["OFF STR"].astype("string").str.upper()
    sided = d[hash_.isin(["L", "R"]) & stren.isin(["L", "R"])]
    if len(sided) >= 10:
        to_field = (sided["HASH"].str.upper().map({"L": "R", "R": "L"}) == sided["OFF STR"].str.upper())
        k, n = int(to_field.sum()), len(sided)
        share, word = (k / n, "the field") if k >= n - k else (1 - k / n, "the boundary")
        kk = k if word == "the field" else n - k
        st.html(f'<div style="font-size:14px;margin-bottom:4px">On a hash, the formation\'s strength is set to '
                f'<b>{word} {share:.0%}</b> of the time ({kk} of {n} plays){_badge(kk, n)}. '
                f'<span style="opacity:.65">Balanced formations aren\'t counted.</span></div>')
    st.html(field_strength_html(d))
    no_str = d["OFF STR"].isna().mean()
    note = ("Rows = where the ball was spotted; columns = which side the formation's strength was to. "
            "\"To field\" means toward the wide side of the field (the middle hash has no field side); \"to "
            "strength\" means toward the formation's strength. Both count plays with a direction tag.")
    if no_str > 0.1:
        note += f" {no_str:.0%} of these plays have no strength tag and only show in the All column."
    st.caption(note)


# ==============================================================================
# 3. SEQUENCING
# ==============================================================================
def sequence_frame(df_full: pd.DataFrame, team: str, side: str = "offense") -> pd.DataFrame:
    """
    Every run/pass snap with what happened on the snap before it in the same
    drive. Built from the full data (df_full) so the previous play is known
    even when the sidebar filters hide it.
    """
    d = _team_plays(df_full, team, side).sort_values(["game_id", "PLAY #"])
    g = d.groupby(["game_id", "drive"], sort=False)
    d["prev_type"] = g["PLAY TYPE"].shift()
    d["prev_gain"] = g["GN/LS"].shift()
    d["prev_complete"] = g["completion"].shift()
    d["first_of_drive"] = g.cumcount() == 0
    return d


SEQUENCE_ROWS = [
    ("First snap of a drive", lambda d: d["first_of_drive"]),
    ("After a run", lambda d: d["prev_type"] == "Run"),
    ("After a pass", lambda d: d["prev_type"] == "Pass"),
    ("After a completion", lambda d: (d["prev_type"] == "Pass") & (d["prev_complete"] == 1)),
    ("After an incompletion", lambda d: (d["prev_type"] == "Pass") & (d["prev_complete"] == 0)),
    ("After a gain of 10+", lambda d: d["prev_gain"] >= 10),
    ("After no gain or a loss", lambda d: d["prev_type"].notna() & (d["prev_gain"] <= 0)),
    ("New set of downs (not drive start)", lambda d: (d["DN"] == 1) & ~d["first_of_drive"]),
]


def sequence_table(seq: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, rule in SEQUENCE_ROWS:
        s = seq[rule(seq).fillna(False).astype(bool)]
        if len(s) == 0:
            continue
        run = 1 - s["PASS"].mean()
        rows.append({"After": name, "Plays": len(s), "Run": run, "Pass": 1 - run,
                     "Success": s["success"].mean(), "EPA": s["epa"].mean()})
    return pd.DataFrame(rows)


def _two_prop_z(k1, n1, k2, n2) -> float:
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else np.nan
    return (k1 / n1 - k2 / n2) / se if se and se > 0 else 0.0


def sequence_html(t: pd.DataFrame, base_pass: float) -> str:
    rows = []
    for _, r in t.iterrows():
        n, run = int(r["Plays"]), r["Run"]
        k = round(max(run, 1 - run) * n)
        diff = (1 - run) - base_pass
        diff_txt = (f'<span style="opacity:.65">{diff * 100:+.0f} pts pass vs usual</span>'
                    if abs(diff) >= 0.05 else '<span style="opacity:.4">about usual</span>')
        rows.append(
            f'<div style="display:grid;grid-template-columns:minmax(150px,1.3fr) 60px minmax(140px,2fr) '
            f'minmax(130px,1.1fr) 70px 70px;gap:10px;align-items:center;padding:6px 0;'
            f'border-top:0.5px solid rgba(128,128,128,.25);font-size:13px{";opacity:.55" if n < v.TELL_MIN_N else ""}">'
            f'<div>{r["After"]}</div><div style="text-align:right;opacity:.7">{n}</div>'
            f'<div>{_split_bar(run, 8)}<div style="font-size:11px;opacity:.75">Run {run:.0%} · Pass {1 - run:.0%}'
            f'{_badge(k, n)}</div></div><div style="font-size:12px">{diff_txt}</div>'
            f'<div style="text-align:right">{r["Success"]:.0%}</div><div style="text-align:right">{r["EPA"]:+.2f}</div>'
            f'</div>')
    head = ('<div style="display:grid;grid-template-columns:minmax(150px,1.3fr) 60px minmax(140px,2fr) '
            'minmax(130px,1.1fr) 70px 70px;gap:10px;font-size:11px;opacity:.6;padding-bottom:4px">'
            '<div>Previous snap</div><div style="text-align:right">Plays</div><div>Next play</div>'
            '<div>vs. their usual pass rate</div><div style="text-align:right">Success</div>'
            '<div style="text-align:right">EPA</div></div>')
    return LEGEND + f'<div style="overflow-x:auto"><div style="min-width:680px">{head}{"".join(rows)}</div></div>'


def render_sequencing(df: pd.DataFrame, df_full: pd.DataFrame, team: str, title: str, side: str = "offense") -> None:
    """df = filtered plays (decides which 'next' plays count); df_full = unfiltered, for the previous snap."""
    st.markdown(f"**{title}**")
    seq = sequence_frame(df_full, team, side)
    seq = seq[seq.index.isin(df.index)]
    if len(seq) < 10:
        st.info("Not enough plays match the current filters.")
        return
    t = sequence_table(seq)
    base_pass = seq["PASS"].mean()
    # Headline: does the previous play type move the next call?
    a, b = seq[seq["prev_type"] == "Run"], seq[seq["prev_type"] == "Pass"]
    if len(a) >= 10 and len(b) >= 10:
        pa, pb = a["PASS"].mean(), b["PASS"].mean()
        z = _two_prop_z(b["PASS"].sum(), len(b), a["PASS"].sum(), len(a))
        verdict = ("a real pattern at this sample size" if abs(z) >= 1.645 else
                   "could still be noise at this sample size")
        if abs(pb - pa) >= 0.05:
            st.html(f'<div style="font-size:14px;margin-bottom:6px">Throws <b>{pb:.0%}</b> after a pass vs '
                    f'<b>{pa:.0%}</b> after a run ({len(b)} and {len(a)} plays): {verdict}.</div>')
    st.html(sequence_html(t, base_pass))
    st.caption(f"\"Usual\" = {base_pass:.0%} pass across all these plays. Previous snap = the run or pass right before, "
               f"in the same drive (penalties and kicks skipped). Faded rows have under {v.TELL_MIN_N} plays.")


# ==============================================================================
# DRIVES (shared by red zone and the season drive dashboard)
# ==============================================================================
POINTS = {"Touchdown": 7, "FG Made": 3}  # touchdowns counted as 7; the try isn't tracked per drive


def season_drives(df_games: pd.DataFrame, team: str = v.TEAM, side: str = "offense") -> pd.DataFrame:
    """
    Every possession in these games for `team` on offense (side="offense") or
    for whoever played `team` (side="defense"), with how far it got.
    snap_deepest = closest snap to the goal, in yards from the offense's own
    goal line: what red-zone trips use (a 60-yard touchdown run isn't a
    red-zone trip). deepest = the same but 100 for any touchdown: what the
    drive funnel uses, since the ball did cross every line on the way.
    """
    out = []
    for gid in ins.games(df_games):
        g = df_games[df_games["game_id"] == gid]
        teams = [t for t in g["offense"].dropna().unique() if t]
        if team not in teams and team not in g["defense"].unique():
            continue
        offense = team if side == "offense" else next((t for t in teams if t != team), None)
        if offense is None:
            continue
        drives = ins.drive_summary(g, offense)
        if drives.empty:
            continue
        snaps = v.run_pass(g[g["offense"] == offense])
        deepest = (100 - snaps.groupby("drive")["YARDLINE_100"].min()).rename("deepest")
        drives = drives.join(deepest, on="drive")
        drives["snap_deepest"] = np.maximum(drives["deepest"].fillna(drives["start"]), drives["start"])
        drives["deepest"] = np.where(drives["result"] == "Touchdown", 100, drives["snap_deepest"])
        drives["points"] = drives["result"].map(POINTS).fillna(0)
        drives["game_id"], drives["offense"] = gid, offense
        drives["Game"] = ins.game_label(df_games, gid)
        out.append(drives)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _drive_kpis(d: pd.DataFrame) -> dict:
    n = len(d)
    if n == 0:
        return {}
    return {"Drives": n, "Points / drive": d["points"].mean(), "TD rate": (d["result"] == "Touchdown").mean(),
            "Scoring rate": (d["points"] > 0).mean(), "3-and-outs": d["three_and_out"].mean(),
            "Avg start": d["start"].mean(), "Yards / drive": d["yards"].mean(),
            "Turnovers": int(d["result"].isin(["Turnover", "Turnover on Downs"]).sum())}


# ==============================================================================
# 4. RED ZONE
# ==============================================================================
RZ_BANDS = [("Red zone (11–20)", 11, 20), ("Inside the 10 (6–10)", 6, 10), ("Goal line (1–5)", 1, 5)]
RZ_RESULTS = [("Touchdown", "#1D9E75"), ("FG Made", "#E8B923"), ("FG Missed", "#E24B4A"),
              ("Turnover on Downs", "#E24B4A"), ("Turnover", "#A32D2D"), ("End of Half", "#888780"),
              ("End of Game", "#888780")]


def render_red_zone(df: pd.DataFrame, df_games: pd.DataFrame, team: str, side: str, label: str,
                    good_high: bool, key: str) -> None:
    """
    df = filtered plays (play-level splits); df_games = whole games for the
    trip-level numbers (a trip needs the full drive). label names the offense
    being measured, e.g. "DCHS offense" or "Opponents vs DCHS".
    """
    drives = season_drives(df_games, team, side)
    if drives.empty:
        st.info("No drives in the selected games.")
        return
    trips = drives[drives["snap_deepest"] >= 80]
    st.markdown(f"**{label}: red zone trips**")
    if trips.empty:
        st.info("No drives reached the red zone in the selected games.")
        return
    c = st.columns(5)
    n = len(trips)
    td = (trips["result"] == "Touchdown").mean()
    c[0].metric("Trips", n, help="Drives with a snap at or inside the opponent's 20.")
    c[1].metric("Touchdown rate", f"{td:.0%}")
    c[2].metric("Scoring rate", f"{(trips['points'] > 0).mean():.0%}")
    c[3].metric("Points / trip", f"{trips['points'].mean():.1f}", help="Touchdowns counted as 7.")
    c[4].metric("Trips per game", f"{n / drives['game_id'].nunique():.1f}")

    # How trips ended, as one stacked bar.
    res = trips["result"].value_counts()
    seg = "".join(f'<span title="{r}: {res[r]}" style="width:{res[r] / n * 100:.1f}%;background:{col}"></span>'
                  for r, col in RZ_RESULTS if r in res)
    leg = "".join(f'<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;'
                  f'background:{col};margin-right:4px"></span>{r} {res[r]}</span>' for r, col in RZ_RESULTS if r in res)
    st.html(f'<div style="display:flex;height:14px;border-radius:4px;overflow:hidden;margin:4px 0 6px">{seg}</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:12px;font-size:12px">{leg}</div>')

    # Play-level: how it's called as the field shrinks.
    rz = v.run_pass(df[(df[side] == team)])
    rz = rz[rz["YARDLINE_100"] <= 20]
    if rz.empty:
        return
    rows = []
    for name, lo, hi in RZ_BANDS:
        b = rz[rz["YARDLINE_100"].between(lo, hi)]
        if len(b):
            rows.append({"Zone": name, "Plays": len(b), "Run": 1 - b["PASS"].mean(), "Pass": b["PASS"].mean(),
                         "Success Rate": b["success"].mean(), "EPA per Play": b["epa"].mean(),
                         "TD on the play": b["RESULT"].astype(str).str.contains("TD").mean()})
    zt = pd.DataFrame(rows).set_index("Zone")
    st.markdown(f"**{label}: how it's called inside the 20**")
    fmt = {"Run": "{:.0%}", "Pass": "{:.0%}", "Success Rate": "{:.0%}", "EPA per Play": "{:+.2f}",
           "TD on the play": "{:.0%}"}
    st.dataframe(zt.style.format(fmt), width="stretch")
    forms = v.crosstab(rz, "OFF FORM", v.PLAY_RESULT_MEASURES, sort_by_count=True)
    v.show_table("Formations inside the 20", forms, good_high=good_high)
    # Full width, not side by side: half-width tables cut off the Plays column.
    if rz["OFF PLAY"].notna().mean() >= 0.5:
        calls = v.crosstab(rz, "OFF PLAY", v.PLAY_RESULT_MEASURES, sort_by_count=True)
        v.show_table("Play calls inside the 20", calls, good_high=good_high)
    else:
        dirs = rz.assign(Direction=rz["PLAY TYPE"].astype(str) + " " +
                         rz["PLAY DIR"].map({"L": "left", "R": "right"}).fillna("(no dir.)"))
        v.show_table("Run/pass by direction inside the 20",
                     v.crosstab(dirs, "Direction", v.PLAY_RESULT_MEASURES, sort_by_count=True), good_high=good_high)
    st.caption("Trips use whole drives from the selected weeks; the zone, formation and call tables follow every "
               "sidebar filter. TD on the play = share of snaps that scored right there.")


# ==============================================================================
# 5. SEASON DRIVE DASHBOARD
# ==============================================================================
START_BUCKETS = [("Own 1–19", 0, 19), ("Own 20–34", 20, 34), ("Own 35–49", 35, 49), ("Opp 50–21", 50, 79),
                 ("Opp 20 & in", 80, 100)]
FUNNEL = [("All drives", lambda d: pd.Series(True, index=d.index)), ("Crossed midfield", lambda d: d["deepest"] > 50),
          ("Reached the red zone", lambda d: d["deepest"] >= 80), ("Scored", lambda d: d["points"] > 0),
          ("Touchdown", lambda d: d["result"] == "Touchdown")]
SIDES = {"DCHS offense": DCHS_C, "Opponents (DCHS defense)": OPP_C}


def _bucket(start: float) -> str:
    for name, lo, hi in START_BUCKETS:
        if lo <= start <= hi:
            return name
    return START_BUCKETS[-1][0]


def render_season_drives(df_all: pd.DataFrame) -> None:
    o = season_drives(df_all, v.TEAM, "offense").assign(Side="DCHS offense")
    d = season_drives(df_all, v.TEAM, "defense").assign(Side="Opponents (DCHS defense)")
    allx = pd.concat([o, d], ignore_index=True)
    if allx.empty:
        st.info("No drives in the data yet.")
        return
    allx["Start"] = allx["start"].map(_bucket)
    domain, colors = list(SIDES), list(SIDES.values())

    # KPI strip: Dowling offense with the defense's number as the comparison.
    ko, kd = _drive_kpis(o), _drive_kpis(d)
    st.markdown("**Season at a glance** (DCHS offense, with opponents against the DCHS defense underneath)")
    cols = st.columns(6)
    for col, (name, fmt) in zip(cols, [("Points / drive", "{:.2f}"), ("TD rate", "{:.0%}"),
                                       ("Scoring rate", "{:.0%}"), ("3-and-outs", "{:.0%}"),
                                       ("Avg start", None), ("Yards / drive", "{:.0f}")]):
        val = ins._yard_label(ko[name]) if fmt is None else fmt.format(ko[name])
        opp = ins._yard_label(kd[name]) if fmt is None else fmt.format(kd[name])
        col.metric(name, val)
        col.caption(f"Opponents: {opp}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Points per drive by starting field position**")
        b = (allx.groupby(["Side", "Start"]).agg(ppd=("points", "mean"), n=("points", "size")).reset_index())
        b["label"] = [f"{p:.1f} ({n})" for p, n in zip(b["ppd"], b["n"])]
        order = [x[0] for x in START_BUCKETS]
        base = alt.Chart(b).encode(
            x=alt.X("Start:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Side:N", sort=domain),
            color=alt.Color("Side:N", scale=alt.Scale(domain=domain, range=colors),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("Side:N"), alt.Tooltip("Start:N", title="Drive started"),
                     alt.Tooltip("ppd:Q", title="Points / drive", format=".2f"), alt.Tooltip("n:Q", title="Drives")])
        bars = base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            y=alt.Y("ppd:Q", title="Points per drive"), opacity=alt.condition("datum.n >= 5", alt.value(1),
                                                                               alt.value(0.4)))
        text = base.mark_text(dy=-6, fontSize=10).encode(y="ppd:Q", text="label:N", color=alt.value(v.theme_ink()))
        st.altair_chart((bars + text).properties(height=300), width="stretch")
        st.caption("Label = points per drive (drives). Faded bars have under 5 drives. Touchdowns counted as 7.")
    with c2:
        st.markdown("**Drive funnel: how far drives get**")
        f = []
        for side, part in allx.groupby("Side"):
            for i, (stage, rule) in enumerate(FUNNEL):
                k = int(rule(part).sum())
                f.append({"Side": side, "Stage": stage, "i": i, "k": k, "share": k / len(part),
                          "label": f"{k / len(part):.0%} ({k})"})
        f = pd.DataFrame(f)
        stages = [s for s, _ in FUNNEL]
        base = alt.Chart(f).encode(
            y=alt.Y("Stage:N", sort=stages, title=None, axis=alt.Axis(labelLimit=200)), yOffset=alt.YOffset("Side:N", sort=domain),
            color=alt.Color("Side:N", scale=alt.Scale(domain=domain, range=colors), legend=None),
            tooltip=[alt.Tooltip("Side:N"), alt.Tooltip("Stage:N"), alt.Tooltip("k:Q", title="Drives"),
                     alt.Tooltip("share:Q", title="Share of drives", format=".0%")])
        bars = base.mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3).encode(
            x=alt.X("share:Q", title="Share of all drives", axis=alt.Axis(format="%", values=[0, .25, .5, .75, 1]),
                    scale=alt.Scale(domain=[0, 1.18])))
        text = base.mark_text(align="left", dx=4, fontSize=10).encode(x="share:Q", text="label:N",
                                                                       color=alt.value(v.theme_ink()))
        st.altair_chart((bars + text).properties(height=300), width="stretch")
        st.caption("Each bar is a share of all drives, so the drop from one stage to the next shows where drives "
                   "die. Maroon = Dowling's offense, blue = opponents against Dowling's defense.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**3-and-out rate by quarter**")
        q = allx[allx["QTR"].between(1, 4)].groupby(["Side", "QTR"]).agg(
            rate=("three_and_out", "mean"), n=("three_and_out", "size")).reset_index()
        q["Quarter"] = "Q" + q["QTR"].astype(int).astype(str)
        base = alt.Chart(q).encode(
            x=alt.X("Quarter:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset=alt.XOffset("Side:N", sort=domain),
            color=alt.Color("Side:N", scale=alt.Scale(domain=domain, range=colors), legend=None),
            tooltip=[alt.Tooltip("Side:N"), alt.Tooltip("Quarter:N"), alt.Tooltip("rate:Q", format=".0%",
                                                                                  title="3-and-out rate"),
                     alt.Tooltip("n:Q", title="Drives")])
        st.altair_chart((base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            y=alt.Y("rate:Q", title=None, axis=alt.Axis(format="%")))
            + base.mark_text(dy=-6, fontSize=10).encode(y="rate:Q", text=alt.Text("rate:Q", format=".0%"),
                                                        color=alt.value(v.theme_ink()))).properties(height=240),
            width="stretch")
        st.caption("For the defense, higher is better: it's the share of opponent drives that went 3-and-out.")
    with c2:
        st.markdown("**How drives ended**")
        res_order = [("Touchdown", "#1D9E75"), ("FG Made", "#E8B923"), ("FG Missed", "#EF9F27"), ("Punt", "#B4B2A9"),
                     ("Downs", "#E24B4A"), ("Turnover", "#A32D2D"), ("End of half", "#D3D1C7")]
        r = allx["result"].replace({"Turnover on Downs": "Downs", "End of Half": "End of half",
                                    "End of Game": "End of half"})
        html = []
        for side in domain:
            part = r[allx["Side"] == side]
            cnt, n = part.value_counts(), len(part)
            seg = "".join(f'<span title="{k}: {cnt[k]} ({cnt[k] / n:.0%})" style="width:{cnt[k] / n * 100:.2f}%;'
                          f'background:{c}"></span>' for k, c in res_order if k in cnt)
            html.append(f'<div style="font-size:12px;margin:10px 0 3px">{side} <span style="opacity:.6">· {n} drives'
                        f'</span></div><div style="display:flex;height:22px;border-radius:4px;overflow:hidden">{seg}</div>')
        totals = r.value_counts()
        leg = "".join(f'<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;'
                      f'background:{c};margin-right:4px"></span>{k}</span>' for k, c in res_order if k in totals)
        st.html("".join(html) + f'<div style="display:flex;flex-wrap:wrap;gap:12px;font-size:12px;margin-top:10px">'
                f'{leg}</div>')
        st.caption("Hover a segment for the count.")

    st.markdown("**By game**")
    rows = []
    for gid in ins.games(df_all):
        go, gd = _drive_kpis(o[o["game_id"] == gid]), _drive_kpis(d[d["game_id"] == gid])
        if not go and not gd:
            continue
        rows.append({"Game": ins.game_label(df_all, gid),
                     "DCHS drives": go.get("Drives"), "DCHS pts / drive": go.get("Points / drive"),
                     "DCHS TD rate": go.get("TD rate"), "DCHS 3-and-outs": go.get("3-and-outs"),
                     "DCHS avg start": ins._yard_label(go["Avg start"]) if go else "",
                     "Opp drives": gd.get("Drives"), "Opp pts / drive": gd.get("Points / drive"),
                     "Opp 3-and-outs": gd.get("3-and-outs"),
                     "Opp avg start": ins._yard_label(gd["Avg start"]) if gd else ""})
    t = pd.DataFrame(rows).set_index("Game")
    st.dataframe(t.style.format({"DCHS pts / drive": "{:.2f}", "Opp pts / drive": "{:.2f}", "DCHS TD rate": "{:.0%}",
                                 "DCHS 3-and-outs": "{:.0%}", "Opp 3-and-outs": "{:.0%}"}, na_rep=""),
                 width="stretch")
    st.caption("Offensive possessions only: defensive and special-teams touchdowns aren't credited to a drive. "
               "Drives that end on a kneel-down or the clock count as \"End of half\".")


# ==============================================================================
# 6. 4TH-DOWN AGGRESSIVENESS BY GAME
# ==============================================================================
def aggressiveness_frame(t: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """
    Per game, from the 4th-down review table: how often Dowling went when the
    model said go (and it wasn't a toss-up), how often it went when the model
    said kick, and the win probability left on the table. Decisions made with
    the game already decided are left out, same as the ledger.
    """
    t = t[t["category"] != "Decided"]
    rows = []
    for gid in order:
        g = t[t["game"] == gid]
        if g.empty:
            continue
        model_go = g[(g["Model"] == "Go for it") & (g["Strength"] != "TOSS-UP")]
        model_kick = g[(g["Model"] != "Go for it") & (g["Strength"] != "TOSS-UP")]
        rows.append({
            "game": gid, "Game": g["Game"].iat[0], "4th downs": len(g),
            "model_go": len(model_go), "went_on_go": int((model_go["We chose"] == "Go for it").sum()),
            "model_kick": len(model_kick), "went_on_kick": int((model_kick["We chose"] == "Go for it").sum()),
            # Same definition as the ledger: only costly disagreements (toss-ups don't count).
            "wp_left": float(g.loc[g["category"] == "Costly", "WP left (pts)"].sum()),
        })
    a = pd.DataFrame(rows)
    if not a.empty:
        a["go_rate"] = np.where(a["model_go"] > 0, a["went_on_go"] / a["model_go"].replace(0, np.nan), np.nan)
        a["go_text"] = [f"{w} of {m}" if m else "none" for w, m in zip(a["went_on_go"], a["model_go"])]
    return a


def render_aggressiveness(t: pd.DataFrame, order: list[str]) -> None:
    a = aggressiveness_frame(t, order)
    if len(a) < 2:
        return
    st.markdown("**Aggressiveness by game**")
    went, said = int(a["went_on_go"].sum()), int(a["model_go"].sum())
    kick_went, kick_said = int(a["went_on_kick"].sum()), int(a["model_kick"].sum())
    parts = [f"When the model clearly said go, Dowling went <b>{went} of {said}</b>"
             + (f" ({went / said:.0%})" if said else "")]
    if kick_said:
        parts.append(f"when it clearly said kick, Dowling went anyway {kick_went} of {kick_said}")
    st.html(f'<div style="font-size:14px;margin-bottom:6px">{"; ".join(parts)}. '
            f'Win probability left on the table: <b>{a["wp_left"].sum():.1f} pts</b> this season.</div>')
    order_labels = a["Game"].tolist()
    c1, c2 = st.columns(2)
    with c1:
        g = a[a["model_go"] > 0]
        if g.empty:
            st.caption("No clear go-for-it situations yet.")
        else:
            base = alt.Chart(g).encode(x=alt.X("Game:N", sort=order_labels, title=None, axis=alt.Axis(labelAngle=0)),
                                       tooltip=[alt.Tooltip("Game:N"), alt.Tooltip("go_text:N", title="Went / model said go")])
            st.altair_chart((base.mark_bar(color="#1D9E75", cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                y=alt.Y("go_rate:Q", title="Went for it when model said go", axis=alt.Axis(format="%"),
                        scale=alt.Scale(domain=[0, 1.1])))
                + base.mark_text(dy=-6, fontSize=10).encode(y="go_rate:Q", text="go_text:N",
                                                            color=alt.value(v.theme_ink()))).properties(height=230),
                width="stretch")
    with c2:
        base = alt.Chart(a).encode(x=alt.X("Game:N", sort=order_labels, title=None, axis=alt.Axis(labelAngle=0)),
                                   tooltip=[alt.Tooltip("Game:N"), alt.Tooltip("wp_left:Q", format=".1f",
                                                                                title="WP left (pts)"),
                                            alt.Tooltip("4th downs:Q")])
        st.altair_chart((base.mark_bar(color="#E24B4A", cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            y=alt.Y("wp_left:Q", title="Win prob. left on the table (pts)"))
            + base.mark_text(dy=-6, fontSize=10).encode(y="wp_left:Q", text=alt.Text("wp_left:Q", format=".1f"),
                                                        color=alt.value(v.theme_ink()))).properties(height=230),
            width="stretch")
    st.caption("\"Clearly\" = the model's pick beat the next option by at least 1 point of win probability "
               "(not a toss-up). Decisions with the game already decided aren't counted.")
