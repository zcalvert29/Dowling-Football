"""
profiles.py — team profile radar charts benchmarked against FBS.

Every axis shows how far a team's number sits from the FBS median, in FBS
standard deviations (team-to-team spread), oriented so farther out is always
better for that unit. The middle ring is the FBS median.

The same metric code runs on two sources so the comparison is like-for-like:

  Hudl data (our curated play-by-play)  ->  standardize_hudl()
  cfbfastR play-by-play (FBS seasons)   ->  standardize_cfbfastr()
                                            (used by build_cfb_benchmarks.py)

Both produce the same four "standard" tables (plays, drives, kicks, field
goals), and team_metrics() computes every metric from those.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

BENCHMARK_PATH = "cfb_benchmarks.csv"
HS_TOUCHBACK = 20     # high school touchback spot (yards from own goal)
CFB_TOUCHBACK = 25    # college touchback spot
Z_LIMIT = 2.0         # radar runs from -2 SD (center) to +2 SD (edge)


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    higher_is_better: bool
    fmt: str            # "pct", "epa", "num", "yds"
    min_n: int          # below this many plays/attempts/kicks, the point is hollow
    help: str


# Offense and defense share one axis order (clockwise from the top), so each offense axis sits in the same spot
# as the defense axis it's measured against: Scoring Rate <-> Stop Rate, 3rd Down Conv. <-> 3rd Down Stop, etc.
OFFENSE = [
    Metric("success", "Success Rate", True, "pct", 40, "Plays that stay on schedule (40% / 70% / 100% of the yards needed on 1st / 2nd / 3rd-4th)."),
    Metric("scoring", "Scoring Rate", True, "pct", 10, "Drives that ended in points (touchdown or field goal)."),
    Metric("conv3", "3rd Down Conv.", True, "pct", 10, "3rd downs that gained the line to gain."),
    Metric("explosive", "Explosive Rate", True, "pct", 40, "Runs of 10+ yards or passes of 20+ yards."),
    Metric("turnover", "Turnover %", False, "pct", 40, "Interceptions and fumbles per play."),
    Metric("int", "INT Rate", False, "pct", 20, "Interceptions per pass attempt."),
    Metric("sack", "Sack Rate", False, "pct", 20, "Sacks per pass play."),
    Metric("comp", "Completion %", True, "pct", 20, "Completions / attempts (interceptions count as incompletions)."),
    Metric("stuffed", "Stuffed Run %", False, "pct", 25, "Runs that gained 0 or fewer yards."),
    Metric("epa_pass", "EPA per Pass", True, "epa", 20, "Expected points added per pass (sacks and scrambles included)."),
    Metric("epa_rush", "EPA per Rush", True, "epa", 25, "Expected points added per run."),
]

DEFENSE = [
    Metric("success", "Success Rate", False, "pct", 40, "Opponent plays that stayed on schedule."),
    Metric("stop", "Stop Rate", True, "pct", 10, "Opponent drives that ended without points."),
    Metric("conv3", "3rd Down Stop", False, "pct", 10, "Opponent 3rd downs converted (fewer is better)."),
    Metric("explosive", "Explosive Rate", False, "pct", 40, "Opponent runs of 10+ / passes of 20+."),
    Metric("turnover", "Turnover %", True, "pct", 40, "Takeaways per opponent play."),
    Metric("int", "INT Rate", True, "pct", 20, "Interceptions per opponent attempt."),
    Metric("sack", "Sack Rate", True, "pct", 20, "Sacks per opponent pass play."),
    Metric("comp", "Completion %", False, "pct", 20, "Opponent completion rate."),
    Metric("stuffed", "Stuff Rate", True, "pct", 25, "Opponent runs held to 0 or fewer yards."),
    Metric("epa_pass", "EPA per Pass", False, "epa", 20, "Expected points added per opponent pass."),
    Metric("epa_rush", "EPA per Rush", False, "epa", 25, "Expected points added per opponent run."),
]

SPECIAL_TEAMS = [
    Metric("fg", "Field Goals", True, "pct", 3, "Field goal make rate (PATs aren't in the college data, so they're left out)."),
    Metric("ko_cov", "Kickoff", False, "yds", 5, "Where opponents start after our kickoffs, vs the touchback spot (HS 20, college 25)."),
    Metric("ko_ret", "Kick Return", True, "yds", 5, "Where we start after their kickoffs, vs the touchback spot."),
    Metric("punt", "Punt", True, "num", 5, "Net punt: punt spot to where the opponent's next snap is."),
    Metric("punt_ret", "Punt Return", False, "num", 5, "Opponent net punt against us (fewer net yards = better return)."),
]

UNITS = {"offense": OFFENSE, "defense": DEFENSE, "special_teams": SPECIAL_TEAMS}
UNIT_TITLES = {"offense": "offensive profile", "defense": "defensive profile", "special_teams": "special teams"}


# ---------------------------------------------------------------------------
# Standard tables
#
#   plays:  game, offense, defense, is_rush, is_pass, down, dist, yards, epa,
#           completion (NaN unless a pass attempt), interception (NaN unless
#           an attempt), sack (NaN unless a pass play), turnover (0/1)
#   drives: game, offense, defense, scored (bool)
#   kicks:  kind ("ko"/"punt"), kicker, receiver, value
#           (ko: receiver's start minus the touchback spot; punt: net yards)
#   fgs:    kicker, made (bool)
# ---------------------------------------------------------------------------
def _success(down, dist, yards):
    return np.select([down == 1, down == 2], [yards >= 0.4 * dist, yards >= 0.7 * dist],
                     default=yards >= dist).astype(float)


def team_metrics(tables: dict, team: str) -> dict:
    """{unit: {metric_key: (value, n)}} for one team from the standard tables."""
    plays, drives, kicks, fgs = (tables[k] for k in ("plays", "drives", "kicks", "fgs"))
    out = {}
    for unit, side in (("offense", "offense"), ("defense", "defense")):
        p = plays[plays[side] == team]
        rush, pas = p[p["is_rush"] == 1], p[p["is_pass"] == 1]
        y = p["yards"]
        explosive = ((p["is_rush"] == 1) & (y >= 10)) | ((p["is_pass"] == 1) & (y >= 20))
        third = p[p["down"] == 3]
        att = p["completion"].notna()
        m = {
            "success": (_success(p["down"], p["dist"], y).mean() if len(p) else np.nan, len(p)),
            "explosive": (explosive.mean() if len(p) else np.nan, len(p)),
            "epa_rush": (rush["epa"].mean(), len(rush)),
            "epa_pass": (pas["epa"].mean(), len(pas)),
            "ypp": (y.mean(), len(p)),
            "comp": (p["completion"].mean(), int(att.sum())),
            "conv3": ((third["yards"] >= third["dist"]).mean() if len(third) else np.nan, len(third)),
            "sack": (pas["sack"].mean(), len(pas)),
            "int": (p["interception"].mean(), int(att.sum())),
            "turnover": (p["turnover"].mean(), len(p)),
            "stuffed": ((rush["yards"] <= 0).mean() if len(rush) else np.nan, len(rush)),
        }
        if unit == "defense":
            dd = drives[drives["defense"] == team]
            m["stop"] = ((~dd["scored"]).mean() if len(dd) else np.nan, len(dd))
        else:
            od = drives[drives["offense"] == team]
            m["scoring"] = (od["scored"].mean() if len(od) else np.nan, len(od))
        out[unit] = m

    ko_k, ko_r = kicks[(kicks["kind"] == "ko") & (kicks["kicker"] == team)], kicks[(kicks["kind"] == "ko") & (kicks["receiver"] == team)]
    pu_k, pu_r = kicks[(kicks["kind"] == "punt") & (kicks["kicker"] == team)], kicks[(kicks["kind"] == "punt") & (kicks["receiver"] == team)]
    f = fgs[fgs["kicker"] == team]
    out["special_teams"] = {
        "fg": (f["made"].mean() if len(f) else np.nan, len(f)),
        "ko_cov": (ko_k["value"].mean(), len(ko_k)),
        "ko_ret": (ko_r["value"].mean(), len(ko_r)),
        "punt": (pu_k["value"].mean(), len(pu_k)),
        "punt_ret": (pu_r["value"].mean(), len(pu_r)),
    }
    return out


# ---------------------------------------------------------------------------
# Hudl (our curated data) -> standard tables
# ---------------------------------------------------------------------------
_KICKOFFS = {"KO", "KO Rec"}
_TRIES = {"Extra Pt.", "Extra Pt. Block", "2 Pt.", "2 Pt. Block", "2 Pt. Defend"}
_PUNTS = {"Punt", "Punt Rec"}
_FGS = {"FG", "FG Block"}


def standardize_hudl(df: pd.DataFrame, team: str = "Dowling Catholic") -> dict:
    """Standard tables from the curated play-by-play (visuals.load_data output)."""
    import insights as ins  # local import: insights pulls in streamlit

    rp = df[df["PLAY TYPE"].isin(["Run", "Pass"]) & df["DN"].isin([1, 2, 3, 4]) & (df["RESULT"] != "Penalty")]
    is_pass = (rp["PLAY TYPE"] == "Pass").astype(int)
    completion = pd.to_numeric(rp.get("completion"), errors="coerce")
    plays = pd.DataFrame({
        "game": rp["game_id"], "offense": rp["offense"], "defense": rp["defense"],
        "is_rush": (rp["PLAY TYPE"] == "Run").astype(int), "is_pass": is_pass,
        "down": rp["DN"].astype(float), "dist": rp["DIST"].astype(float), "yards": rp["GN/LS"].astype(float),
        "epa": rp["epa"].astype(float), "completion": completion,
        "interception": np.where(completion.notna(), rp["interception"], np.nan),
        "sack": np.where(is_pass == 1, rp["SACK"], np.nan), "turnover": rp["turnover"].astype(float),
    })

    drives, kicks = [], []
    for gid, g in df.groupby("game_id", sort=False):
        g = g.sort_values("PLAY #").reset_index(drop=True)
        # The two teams that actually played. (Pairing every game with Dowling credited Dowling's defense
        # with drives from scout film of other teams' games.)
        game_teams = ins.teams_in_game(g)
        if len(game_teams) != 2:
            continue
        a, b = game_teams
        for off, deff in ((a, b), (b, a)):
            ds = ins.drive_summary(g, off)
            for res in ds.get("result", []):
                drives.append({"game": gid, "offense": off, "defense": deff, "scored": res in ("Touchdown", "FG Made")})
        scrim = g.index[g["DN"].isin([1, 2, 3, 4]) & ~g["PLAY TYPE"].isin(_KICKOFFS | _TRIES)
                        & g["YARDLINE_100"].notna()].to_numpy()
        for i, r in g[g["PLAY TYPE"].isin(_KICKOFFS | _PUNTS)].iterrows():
            later = scrim[scrim > i]
            if not len(later):
                continue
            nxt = g.loc[later[0]]
            start = 100 - nxt["YARDLINE_100"]
            if r["PLAY TYPE"] in _KICKOFFS:
                kicker = b if nxt["offense"] == a else a
                kicks.append({"kind": "ko", "kicker": kicker, "receiver": nxt["offense"], "value": start - HS_TOUCHBACK})
            elif r["offense"] != nxt["offense"]:
                kicks.append({"kind": "punt", "kicker": r["offense"], "receiver": nxt["offense"],
                              "value": r["YARDLINE_100"] - start})
    fg = df[df["PLAY TYPE"].isin(_FGS)]
    fgs = pd.DataFrame({"kicker": fg["offense"], "made": fg["RESULT"] == "Good"})
    return {"plays": plays, "drives": pd.DataFrame(drives, columns=["game", "offense", "defense", "scored"]),
            "kicks": pd.DataFrame(kicks, columns=["kind", "kicker", "receiver", "value"]), "fgs": fgs}


# ---------------------------------------------------------------------------
# cfbfastR -> standard tables (used by build_cfb_benchmarks.py)
# ---------------------------------------------------------------------------
CFBFASTR_COLUMNS = [
    "game_id", "game_play_number", "pos_team", "def_pos_team", "offense_conference", "defense_conference",
    "down", "distance", "yards_to_goal", "yards_gained", "EPA", "rush", "pass", "completion", "sack", "int",
    "fumble_vec", "kickoff_play", "punt_play", "fg_inds", "fg_made", "drive_id", "drive_pts", "penalty_no_play",
]
FBS_CONFERENCES = {"ACC", "American Athletic", "Big 12", "Big Ten", "Conference USA", "FBS Independents",
                   "Mid-American", "Mountain West", "Pac-12", "SEC", "Sun Belt"}


def standardize_cfbfastr(d: pd.DataFrame) -> dict:
    d = d.sort_values(["game_id", "game_play_number"]).reset_index(drop=True)
    d = d[~d["penalty_no_play"].fillna(False).astype(bool)].reset_index(drop=True)
    special = (d["kickoff_play"] == 1) | (d["punt_play"] == 1) | (d["fg_inds"] == 1)
    is_scrim = ((d["rush"] == 1) | (d["pass"] == 1)) & d["down"].between(1, 4) & ~special
    s = d[is_scrim]
    is_pass = (s["pass"] == 1).astype(int)
    attempt = (is_pass == 1) & (s["sack"].fillna(0) != 1)
    plays = pd.DataFrame({
        "game": s["game_id"], "offense": s["pos_team"], "defense": s["def_pos_team"],
        "is_rush": (s["rush"] == 1).astype(int), "is_pass": is_pass, "down": s["down"], "dist": s["distance"],
        "yards": s["yards_gained"], "epa": s["EPA"],
        "completion": np.where(attempt, s["completion"].fillna(0), np.nan),
        "interception": np.where(attempt, s["int"].fillna(0), np.nan),
        "sack": np.where(is_pass == 1, s["sack"].fillna(0), np.nan),
        "turnover": ((s["int"].fillna(0) == 1) | (s["fumble_vec"].fillna(0) == 1)).astype(float),
    })
    dr = d.dropna(subset=["drive_id"]).groupby("drive_id").agg(
        game=("game_id", "first"), offense=("pos_team", "first"), defense=("def_pos_team", "first"),
        pts=("drive_pts", "first"))
    drives = dr.assign(scored=dr["pts"] > 0)[["game", "offense", "defense", "scored"]]

    # Next scrimmage snap after each kickoff/punt (same game).
    idx = pd.Series(np.where(is_scrim, d.index, np.nan), index=d.index)
    next_scrim = idx[::-1].groupby(d["game_id"][::-1]).cummin()[::-1].shift(-1)
    kicks = []
    for kind, mask in (("ko", d["kickoff_play"] == 1), ("punt", d["punt_play"] == 1)):
        k = d[mask].copy()
        k["ni"] = next_scrim[mask]
        k = k.dropna(subset=["ni"])
        n = d.loc[k["ni"].astype(int)]
        k["recv"], k["recv_ytg"], k["ngame"] = n["pos_team"].to_numpy(), n["yards_to_goal"].to_numpy(), n["game_id"].to_numpy()
        k = k[k["ngame"] == k["game_id"]]
        if kind == "ko":
            k["kicker"] = np.where(k["pos_team"] == k["recv"], k["def_pos_team"], k["pos_team"])
            k = k[k["kicker"] != k["recv"]]
            val = (100 - k["recv_ytg"]) - CFB_TOUCHBACK
        else:
            k = k[k["pos_team"] != k["recv"]]
            k["kicker"] = k["pos_team"]
            val = k["yards_to_goal"] - (100 - k["recv_ytg"])
        kicks.append(pd.DataFrame({"kind": kind, "kicker": k["kicker"], "receiver": k["recv"], "value": val}))
    fg = d[d["fg_inds"] == 1]
    fgs = pd.DataFrame({"kicker": fg["pos_team"], "made": fg["fg_made"].fillna(False).astype(bool)})
    return {"plays": plays, "drives": drives, "kicks": pd.concat(kicks, ignore_index=True), "fgs": fgs}


def fbs_teams(d: pd.DataFrame, min_plays: int = 300) -> list[str]:
    """
    Teams whose own offensive snaps are tagged with an FBS conference.
    Only runs/passes count: on kickoffs cfbfastR's pos_team can be the
    receiving team while the conference columns describe the kicking team,
    which would let FCS opponents leak in.
    """
    s = d[((d["rush"] == 1) | (d["pass"] == 1)) & d["down"].between(1, 4)]
    s = s[s["offense_conference"].isin(FBS_CONFERENCES)]
    counts = s["pos_team"].value_counts()
    return sorted(counts[counts >= min_plays].index)


# ---------------------------------------------------------------------------
# Benchmarks and scoring
# ---------------------------------------------------------------------------
def load_benchmarks(path: str = BENCHMARK_PATH) -> pd.DataFrame:
    """Long table: season, unit, metric, team, value, n (one row per FBS team per metric)."""
    return pd.read_csv(path)


def benchmark_stats(bench: pd.DataFrame) -> dict:
    """{(unit, metric): {"median", "std", "values", "season", "n_teams"}}."""
    out = {}
    for (unit, metric), g in bench.groupby(["unit", "metric"]):
        vals = g["value"].dropna().to_numpy()
        out[(unit, metric)] = {"median": float(np.median(vals)), "std": float(np.std(vals, ddof=1)),
                               "values": vals, "season": g["season"].iat[0], "n_teams": len(vals)}
    return out


def _fmt(kind: str, x: float) -> str:
    if x is None or pd.isna(x):
        return "–"
    return {"pct": f"{x:.1%}", "epa": f"{x:+.2f}", "num": f"{x:.1f}", "yds": f"{x:+.1f} yds"}[kind]


def score_team(metrics: dict, stats: dict, unit: str) -> pd.DataFrame:
    """One row per axis: raw value, FBS median, SD from median (better = positive), percentile."""
    rows = []
    for m in UNITS[unit]:
        value, n = metrics[unit].get(m.key, (np.nan, 0))
        st_ = stats.get((unit, m.key))
        if st_ is None:
            continue
        sign = 1 if m.higher_is_better else -1
        z = sign * (value - st_["median"]) / st_["std"] if pd.notna(value) and st_["std"] > 0 else np.nan
        vals = st_["values"]
        pct = np.nan if pd.isna(value) else ((vals < value).mean() + 0.5 * (vals == value).mean())
        pct = pct if m.higher_is_better or pd.isna(pct) else 1 - pct
        rows.append({
            "Metric": m.label, "key": m.key, "value": value, "Value": _fmt(m.fmt, value),
            "FBS median": _fmt(m.fmt, st_["median"]), "SD vs median": z, "Better than": pct,
            "n": int(n), "low_n": int(n) < m.min_n, "help": m.help,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Radar chart (Chart.js in a components iframe, so hover tooltips work)
# ---------------------------------------------------------------------------
TEAM_COLORS = ["#8C1D2C", "#185FA5"]  # Dowling maroon, comparison blue


def _esc(x) -> str:
    return (str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def radar_html(title: str, subtitle: str, teams: list[tuple[str, pd.DataFrame]], ink: str = "#31333F",
               height: int = 600, colors: list[str] | None = None, width: int = 760,
               interactive: bool = True) -> str:
    """
    Self-contained SVG radar (no JavaScript libraries, so nothing can fail to
    load). teams = [(name, score_team() frame), ...]; the first sets the axes.
    Hovering a point shows its numbers.
    """
    labels = teams[0][1]["Metric"].tolist()
    n_ax = len(labels)
    colors = colors or TEAM_COLORS
    W, H = width, height - 90        # room above the SVG for the title and legend
    cx, cy, R = W / 2, H / 2, H / 2 - 42
    dark = ink == "#FAFAFA"
    grid = "rgba(250,250,250,.18)" if dark else "rgba(49,51,63,.16)"
    median_c = "rgba(250,250,250,.6)" if dark else "rgba(49,51,63,.6)"
    ang = [np.deg2rad(-90 + 360 * i / n_ax) for i in range(n_ax)]

    def pt(z, i):
        r = (min(max(z, -Z_LIMIT), Z_LIMIT) + Z_LIMIT) / (2 * Z_LIMIT) * R
        return cx + r * np.cos(ang[i]), cy + r * np.sin(ang[i])

    parts = []
    for z in range(-int(Z_LIMIT) + 1, int(Z_LIMIT) + 1):          # rings
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(z, i) for i in range(n_ax)))
        parts.append(f'<polygon points="{pts}" fill="none" stroke="{median_c if z == 0 else grid}" '
                     f'stroke-width="{2 if z == 0 else 1}"/>')
    for i in range(n_ax):                                            # spokes + labels
        x, y = pt(Z_LIMIT, i)
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{grid}"/>')
        lx, ly = cx + (R + 16) * np.cos(ang[i]), cy + (R + 16) * np.sin(ang[i])
        c = np.cos(ang[i])
        anchor = "middle" if abs(c) < 0.3 else ("start" if c > 0 else "end")
        base = "auto" if np.sin(ang[i]) < -0.3 else ("hanging" if np.sin(ang[i]) > 0.3 else "central")
        parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" dominant-baseline="{base}" '
                     f'font-size="13" fill="{ink}">{_esc(labels[i])}</text>')
    for z in range(-int(Z_LIMIT) + 1, int(Z_LIMIT) + 1):           # ring labels on the top spoke
        x, y = pt(z, 0)
        txt = "FBS median" if z == 0 else f"{z:+d} SD"
        parts.append(f'<text x="{x + 4:.1f}" y="{y - 3:.1f}" font-size="10" fill="#378ADD">{txt}</text>')

    for k, (name, t) in enumerate(teams):                            # team shapes
        t = t.set_index("Metric").reindex(labels)
        color = colors[k % len(colors)]
        zs = t["SD vs median"].astype(float).fillna(-Z_LIMIT).tolist()
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(z, i) for i, z in enumerate(zs)))
        parts.append(f'<polygon points="{poly}" fill="{color}" fill-opacity="{0.28 if k == 0 else 0.16}" '
                     f'stroke="{color}" stroke-width="2.5" stroke-linejoin="round"'
                     f'{"" if k == 0 else " stroke-dasharray=\"7 5\""}/>')
        for i, (metric, r) in enumerate(t.iterrows()):
            z = r["SD vs median"]
            x, y = pt(-Z_LIMIT if pd.isna(z) else z, i)
            low = bool(r["low_n"]) if pd.notna(r["low_n"]) else True
            fill = "#ffffff" if low else color
            ztxt = "–" if pd.isna(z) else f"{z:+.1f} SD vs FBS median" + (" (off the chart)" if abs(z) > Z_LIMIT else "")
            pct = "" if pd.isna(r["Better than"]) else f"Better than {r['Better than']:.0%} of FBS teams"
            nn = int(r["n"]) if pd.notna(r["n"]) else 0
            tip = "|".join([f"{name} · {metric}", f"{r['Value']}  (FBS median {r['FBS median']})", ztxt, pct,
                            f"{nn} plays/attempts" + (" · small sample" if low else "")])
            attrs = f'fill="{fill}" stroke="{color}" stroke-width="2" class="pt" data-tip="{_esc(tip)}"'
            if pd.notna(z) and abs(z) > Z_LIMIT:                     # pinned to the edge: triangle
                a = ang[i]
                tri = [(x + 8 * np.cos(a), y + 8 * np.sin(a)),
                       (x + 6 * np.cos(a + 2.4), y + 6 * np.sin(a + 2.4)),
                       (x + 6 * np.cos(a - 2.4), y + 6 * np.sin(a - 2.4))]
                parts.append(f'<polygon points="{" ".join(f"{px:.1f},{py:.1f}" for px, py in tri)}" {attrs}/>')
            else:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" {attrs}/>')

    legend = "" if len(teams) == 1 else "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin:0 9px">'
        f'<span style="width:18px;border-top:2.5px {"solid" if i == 0 else "dashed"} {colors[i % len(colors)]}"></span>'
        f'{_esc(n)}</span>' for i, (n, _) in enumerate(teams))
    if not interactive:
        return (f'<div style="font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:{ink}">'
                f'<div style="text-align:center;font-size:15px;font-weight:600">{_esc(title)}</div>'
                f'<div style="text-align:center;font-size:12px;opacity:.7;margin:2px 0 4px">{_esc(subtitle)}</div>'
                f'<svg viewBox="0 0 {W} {H}" style="width:100%;display:block;margin:0 auto">{"".join(parts)}</svg></div>')
    return f"""
<div style="font-family:'Source Sans Pro',-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:{ink};position:relative">
  <div style="text-align:center;font-size:17px;font-weight:600">{_esc(title)}</div>
  <div style="text-align:center;font-size:12px;opacity:.7;margin:2px 0 4px">{_esc(subtitle)}</div>
  <div style="text-align:center;font-size:12px">{legend}</div>
  <svg viewBox="0 0 {W} {H}" style="width:100%;max-width:{W}px;display:block;margin:0 auto">{"".join(parts)}</svg>
  <div id="tip" style="position:absolute;display:none;pointer-events:none;background:{'#262730' if dark else '#ffffff'};
       color:{ink};border:1px solid {grid};border-radius:6px;padding:6px 9px;font-size:12px;line-height:1.45;
       box-shadow:0 2px 6px rgba(0,0,0,.15);max-width:280px"></div>
</div>
<script>
const tip = document.getElementById("tip");
document.querySelectorAll(".pt").forEach(el => {{
  el.style.cursor = "pointer";
  el.addEventListener("mousemove", e => {{
    const lines = el.dataset.tip.split("|").filter(Boolean);
    tip.innerHTML = "<b>" + lines[0] + "</b><br>" + lines.slice(1).join("<br>");
    tip.style.display = "block";
    const box = tip.parentElement.getBoundingClientRect();
    let x = e.clientX - box.left + 14, y = e.clientY - box.top + 14;
    if (x + 290 > box.width) x = e.clientX - box.left - 290;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }});
  el.addEventListener("mouseleave", () => tip.style.display = "none");
}});
</script>"""


def render_radar(title: str, subtitle: str, teams: list[tuple[str, pd.DataFrame]], height: int = 600,
                 colors: list[str] | None = None, width: int = 760) -> None:
    import streamlit as st
    import streamlit.components.v1 as components

    if not teams or teams[0][1].empty:
        st.info("No benchmark data available for this chart.")
        return
    theme = getattr(getattr(st.context, "theme", None), "type", None)
    ink = "#FAFAFA" if theme == "dark" else "#31333F"
    components.html(radar_html(title, subtitle, teams, ink, height, colors, width), height=height)
    with st.expander("See the numbers"):
        for name, t in teams:
            shown = t[["Metric", "Value", "FBS median", "SD vs median", "Better than", "n"]].copy()
            shown["Sample"] = np.where(t["low_n"], "small", "")
            st.markdown(f"**{name}**")

            def color(col):
                return ["" if pd.isna(x) else
                        "background-color:#E1F5EE;color:#085041" if x >= 0.5 else
                        "background-color:#FCEBEB;color:#791F1F" if x <= -0.5 else "" for x in col]

            st.dataframe(shown.style.format({"SD vs median": "{:+.1f}", "Better than": "{:.0%}"}, na_rep="–")
                         .apply(color, subset=["SD vs median"]), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def profile_charts(df_plays: pd.DataFrame, df_games: pd.DataFrame, opponent: str, team: str = "Dowling Catholic",
                   bench_path: str = BENCHMARK_PATH, cache=None) -> dict | None:
    """
    Everything the profile charts need: {"season", "n_ours", "n_theirs", "rows"}, where rows is three
    (left, right) pairs of (title, subtitle, team name, score_team frame, color): Dowling O | opponent D,
    Dowling D | opponent O, Dowling ST | opponent ST. None if the benchmark file is missing.
    """
    try:
        bench = load_benchmarks(bench_path)
    except FileNotFoundError:
        return None
    stats = benchmark_stats(bench)
    std = cache(standardize_hudl) if cache else standardize_hudl
    plays_tables, game_tables = std(df_plays, team), std(df_games, team)
    # Offense/defense from filtered plays; drives and kicks from whole games.
    tables = {"plays": plays_tables["plays"], "drives": game_tables["drives"],
              "kicks": game_tables["kicks"], "fgs": game_tables["fgs"]}
    ours, theirs = team_metrics(tables, team), team_metrics(tables, opponent)
    games_of = lambda t: df_games.loc[(df_games["offense"] == t) | (df_games["defense"] == t), "game_id"].nunique()
    n_ours, n_theirs = games_of(team), games_of(opponent)
    plural = lambda n: f"{n} game{'' if n == 1 else 's'}"
    pairs = [(("offense", "Offense"), ("defense", "Defense")), (("defense", "Defense"), ("offense", "Offense")),
             (("special_teams", "Special Teams"), ("special_teams", "Special Teams"))]
    rows = [((f"{team} {ol}", plural(n_ours), team, score_team(ours, stats, ou), TEAM_COLORS[0]),
             (f"{opponent} {tl}", plural(n_theirs), opponent, score_team(theirs, stats, tu), TEAM_COLORS[1]))
            for (ou, ol), (tu, tl) in pairs]
    return {"season": int(bench["season"].iat[0]), "n_ours": n_ours, "n_theirs": n_theirs, "rows": rows}


def profiles_report_html(df_plays: pd.DataFrame, df_games: pd.DataFrame, opponent: str,
                         team: str = "Dowling Catholic") -> str:
    """The six radars as static HTML (no hover script) for the printable scouting report."""
    p = profile_charts(df_plays, df_games, opponent, team)
    if p is None:
        return ""
    cells = []
    for left, right in p["rows"]:
        for title, sub, name, frame, color in (left, right):
            if frame.empty:
                cells.append("<div></div>")
                continue
            cells.append(radar_html(title, sub, [(name, frame)], "#31333F", 520, [color], 620, interactive=False))
    return ('<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px 16px">' + "".join(
        f'<div style="break-inside:avoid">{c}</div>' for c in cells) + "</div>"
            f'<p style="font-size:12px;color:#666">Each axis = distance from the {p["season"]} FBS median in FBS '
            f'standard deviations (±{Z_LIMIT:.0f}); farther out is better. Hollow points are small samples.</p>')


def render_profiles_page(df_plays: pd.DataFrame, df_games: pd.DataFrame, opponent: str,
                         team: str = "Dowling Catholic", bench_path: str = BENCHMARK_PATH) -> None:
    """
    df_plays: sidebar-filtered plays (offense/defense axes).
    df_games: whole games for the selected weeks (special teams and drives
              need complete games, so down/situation filters don't apply).
    """
    import streamlit as st

    p = profile_charts(df_plays, df_games, opponent, team, bench_path,
                       cache=st.cache_data(show_spinner="Scoring against FBS..."))
    if p is None:
        st.error(f"{bench_path} not found. Run build_cfb_benchmarks.py to create it, then commit it with the app.")
        return
    n_ours, n_theirs = p["n_ours"], p["n_theirs"]
    st.caption(f"Each axis = how far the number is from the {p['season']} FBS median, in FBS standard deviations "
               f"(the spread between FBS teams). The middle ring is the median; farther out is always better. Offense "
               f"and defense axes sit in the same spots (scoring rate across from stop rate, and so on). Hollow "
               f"points are small samples; triangles are beyond ±{Z_LIMIT:.0f} SD and pinned to the edge. Hover any "
               f"point for the numbers. {team}: {n_ours} game{'' if n_ours == 1 else 's'} · {opponent}: {n_theirs} "
               f"game{'' if n_theirs == 1 else 's'} of film.")
    for left, right in p["rows"]:
        for col, (title, sub, name, frame, color) in zip(st.columns(2), (left, right)):
            with col:
                render_radar(title, sub, [(name, frame)], height=520, colors=[color], width=620)
    st.caption(
        "Same definitions on both sides: our curated data for the team, cfbfastR play-by-play for FBS. "
        "Kickoffs are measured from the touchback spot (HS 20, college 25) so the different rules don't decide the "
        "axis. Punting and field goals still reflect real high school vs college differences (shorter punts, shorter "
        "kicks), so read those two axes with that in mind."
    )
