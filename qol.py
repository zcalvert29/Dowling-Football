"""
qol.py — quality-of-life pieces:

  check_password()       optional login gate (set app_password in secrets)
  render_home()          Monday-morning home page
  render_play_finder()   search every play by any combination of tags
  render_sideline()      phone-sized tendency predictor for game day
  notes_box()            coach notes per opponent / game (Google Sheet if connected)
  render_glossary()      "How to read this app"
  validate_plays()       likely tagging mistakes (used by Add a Game and Tagging Coverage)
"""
from __future__ import annotations

import hmac
from datetime import datetime
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st

import insights as ins
import visuals as v

TEAM = v.TEAM


def _secret(key, default=None):
    """st.secrets lookup that doesn't crash when there's no secrets file."""
    try:
        return st.secrets[key]
    except Exception:  # noqa: BLE001 — missing file or missing key
        return default


# ---------------------------------------------------------------------------
# Login gate
# ---------------------------------------------------------------------------
def check_password() -> bool:
    """
    If app_password is set in .streamlit/secrets.toml (or the Streamlit Cloud
    secrets box), ask for it once per session. With no password set, the app
    stays open, so nothing changes until you add one.
    """
    expected = _secret("app_password")
    if not expected or st.session_state.get("_authed"):
        return True
    st.title("Dowling Scouting Dashboard")
    with st.form("login"):
        pw = st.text_input("Password", type="password")
        ok = st.form_submit_button("Sign in", type="primary")
    if ok:
        if hmac.compare_digest(str(pw), str(expected)):
            st.session_state["_authed"] = True
            st.rerun()
        st.error("That password isn't right.")
    return False


# ---------------------------------------------------------------------------
# Tagging checks
# ---------------------------------------------------------------------------
KNOWN_PLAY_TYPES = {"Run", "Pass", "KO", "KO Rec", "Punt", "Punt Rec", "Fake Punt", "FG", "FG Block", "Extra Pt.",
                    "Extra Pt. Block", "2 Pt.", "2 Pt. Block", "2 Pt. Defend"}


def validate_plays(df: pd.DataFrame) -> pd.DataFrame:
    """Likely tagging mistakes, one row per issue: game, PLAY #, issue."""
    issues = []
    for gid, g in df.groupby("game_id", sort=False):
        g = g.sort_values("PLAY #").reset_index(drop=True)
        label = ins.game_label(df, gid) if "WEEK" in df else gid
        add = lambda r, msg: issues.append({"Game": label, "PLAY #": r["PLAY #"], "QTR": r.get("QTR"), "Issue": msg})

        dup = g[g["PLAY #"].duplicated(keep=False)]
        for _, r in dup.drop_duplicates("PLAY #").iterrows():
            add(r, "PLAY # appears more than once")
        for _, r in g[g["PLAY TYPE"].notna() & ~g["PLAY TYPE"].isin(KNOWN_PLAY_TYPES)].iterrows():
            add(r, f"Unrecognized PLAY TYPE '{r['PLAY TYPE']}'")
        scrim = g[g["PLAY TYPE"].isin(["Run", "Pass"])]
        for _, r in scrim[scrim["DN"].isna() | scrim["DIST"].isna() | scrim["YARD LN"].isna()].iterrows():
            add(r, "Run/pass missing DN, DIST, or YARD LN")
        for _, r in g[g["YARD LN"].abs() > 50].iterrows():
            add(r, f"YARD LN {r['YARD LN']:.0f} is outside -50 to 50")
        for col in ("drive_result", "SERIES_RESULT"):
            if col in g:
                for _, r in g[g[col] == "Error"].drop_duplicates("drive" if col == "drive_result" else "SERIES").iterrows():
                    add(r, f"{col} came out as Error (drive/series couldn't be classified)")

        # Kickoff direction: whoever scored last kicks (after a safety, the team that gave it up kicks).
        prev_t = prev_o = 0
        last_scorer, last_was_safety = None, False
        for i, r in g.iterrows():
            t, o = r.get("team_score", prev_t), r.get("opponent_score", prev_o)
            is_try = r["PLAY TYPE"] in ("Extra Pt.", "Extra Pt. Block", "2 Pt.", "2 Pt. Block", "2 Pt. Defend")
            if pd.notna(t) and pd.notna(o) and (t, o) != (prev_t, prev_o):
                if not is_try:  # the try belongs to whoever scored the touchdown; it doesn't change the kicker
                    last_scorer = "us" if t > prev_t else "them"
                    last_was_safety = "Safety" in str(r.get("RESULT", ""))
                prev_t, prev_o = t, o
            if i > 0 and g.loc[i - 1, "QTR"] == 2 and r["QTR"] == 3:
                last_scorer = None  # second-half kickoff: can't tell who kicks from the score
            pt = r["PLAY TYPE"]
            if pt in ("KO", "KO Rec") and last_scorer:
                we_kick = (last_scorer == "us") != last_was_safety
                expected = "KO" if we_kick else "KO Rec"
                if pt != expected:
                    add(r, f"Tagged {pt}, but {'Dowling' if last_scorer == 'us' else 'the opponent'} just scored, "
                           f"so it should be {expected}")
                last_scorer = None
            # Tries need a touchdown right before them (ignoring timeouts/penalty re-tries).
            if pt in ("Extra Pt.", "Extra Pt. Block", "2 Pt.", "2 Pt. Block", "2 Pt. Defend"):
                j = i - 1
                while j >= 0 and (g.loc[j, "PLAY TYPE"] in ("Extra Pt.", "Extra Pt. Block", "2 Pt.", "2 Pt. Block",
                                                            "2 Pt. Defend") or str(g.loc[j, "RESULT"]) in ("Timeout", "Penalty")):
                    j -= 1
                if j < 0 or "TD" not in str(g.loc[j, "RESULT"]):
                    add(r, f"{pt} with no touchdown right before it")
                else:
                    ours = pt in ("Extra Pt.", "2 Pt.")
                    td_ours = (g.loc[j, "defense"] if "Def TD" in str(g.loc[j, "RESULT"]) else g.loc[j, "offense"]) == TEAM
                    if "offense" in g and ours != td_ours:
                        add(r, f"{pt} after a {'Dowling' if td_ours else 'opponent'} touchdown: the tag says it's "
                               f"{'our' if ours else 'their'} try")
                    elif "offense" in g and (r["offense"] == TEAM) != ours:
                        add(r, f"{pt} is credited to {r['offense']} in the curated file. Rebuild with the current "
                               "curate_pbp.py, which credits tries after defensive touchdowns correctly")
            # A missed field goal can't be followed by the kicking team kicking off (same half).
            if pt in ("FG", "FG Block") and str(r["RESULT"]) == "No Good" and i + 1 < len(g) and "offense" in g:
                nxt = g.loc[i + 1]
                same_half = (r["QTR"] in (1, 2)) == (nxt["QTR"] in (1, 2))
                if nxt["PLAY TYPE"] in ("KO", "KO Rec") and same_half:
                    kicker = ins.other_team(g, nxt["offense"]) if nxt["offense"] in ins.game_teams(gid) else None
                    if kicker == r["offense"]:
                        add(r, "FG tagged No Good, but the kicking team kicks off next: it was probably good")
    return pd.DataFrame(issues, columns=["Game", "PLAY #", "QTR", "Issue"])


def render_validation(df: pd.DataFrame, title: str = "Possible tagging mistakes") -> None:
    st.markdown(f"**{title}**")
    issues = validate_plays(df)
    if issues.empty:
        st.success("No likely tagging mistakes found.")
    else:
        st.caption(f"{len(issues)} things worth a second look in Hudl. Use PLAY # to find each one.")
        st.dataframe(issues, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Coach notes
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _notes_sheet():
    info, url = _secret("gcp_service_account"), _secret("notes_sheet_url")
    if not info or not url:
        return None
    import gspread  # only needed once notes are connected

    sh = gspread.service_account_from_dict(dict(info)).open_by_url(url)
    try:
        return sh.worksheet("notes")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet("notes", rows=1000, cols=4)
        ws.append_row(["scope", "key", "updated", "text"])
        return ws


def _load_notes() -> pd.DataFrame:
    ws = _notes_sheet()
    if ws is None:
        rows = st.session_state.setdefault("_notes", [])
        return pd.DataFrame(rows, columns=["scope", "key", "updated", "text"])
    return pd.DataFrame(ws.get_all_records(), columns=["scope", "key", "updated", "text"])


def latest_note(scope: str, key: str) -> str:
    try:
        n = _load_notes()
    except Exception:  # noqa: BLE001
        return ""
    n = n[(n["scope"] == scope) & (n["key"].astype(str) == str(key))]
    return "" if n.empty else str(n.iloc[-1]["text"])


def notes_box(scope: str, key: str, label: str) -> None:
    """A notes area for one opponent or game. Saves to a Google Sheet when connected."""
    connected = _notes_sheet() is not None
    with st.expander(f"Coach notes: {label}", expanded=bool(latest_note(scope, key))):
        current = latest_note(scope, key)
        text = st.text_area("Notes", value=current, key=f"note_{scope}_{key}", height=120,
                            label_visibility="collapsed", placeholder="e.g. They shifted to Trips late vs SEP")
        if st.button("Save note", key=f"save_{scope}_{key}", disabled=text == current):
            row = [scope, str(key), datetime.now().strftime("%Y-%m-%d %H:%M"), text]
            if connected:
                _notes_sheet().append_row(row)
            else:
                st.session_state.setdefault("_notes", []).append(row)
            st.success("Saved.")
        if not connected:
            st.caption("Notes only last for this session until a Google Sheet is connected "
                       "(add gcp_service_account and notes_sheet_url to the app's secrets).")


# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------
# Badge for each confidence grade: (label, color).
GRADE_BADGE = {"Solid": ("Tell", "red"), "Likely": ("Lean", "orange"), "Small sample": ("Small sample", "gray")}


def _opponent_tells(df: pd.DataFrame, team: str, n: int = 3) -> list[tuple[str, str, str]]:
    """
    Up to n run/pass tendencies (75%+ one way on 5+ plays), most trustworthy
    first, plus the opponent's best formation. Each is (badge, color, text);
    the badge says how much to trust it (see visuals.tendency_grade).
    """
    d = v.downs(v.run_pass(df[df["offense"] == team]))
    d = d[d["Distance"].notna()]
    tells = []
    for (dn, dist), g in d.groupby(["DN", "Distance"], observed=True):
        if len(g) < v.TELL_MIN_N:
            continue
        pr = g["PASS"].mean()
        share = max(pr, 1 - pr)
        if share >= v.TELL_SHARE:
            kind = "Throws" if pr >= 0.5 else "Runs"
            k = int(round(share * len(g)))
            grade, lb = v.tendency_grade(k, len(g))
            form = g["OFF FORM"].mode()
            form_txt = f", mostly out of {str(form.iat[0]).title()}" if len(form) else ""
            dist_txt = str(dist).split(" (")[0].lower()
            badge, color = GRADE_BADGE[grade]
            tells.append((lb, badge, color, f"{kind} {share:.0%} on {['', '1st', '2nd', '3rd', '4th'][int(dn)]} & "
                                            f"{dist_txt}{form_txt} ({k} of {len(g)} plays)"))
    out = [(b_, c_, t) for _, b_, c_, t in sorted(tells, key=lambda x: -x[0])[:n]]
    f = v.opp_formations_table(df, team)
    f = f[f["Plays"] >= 5]
    if not f.empty:
        best = f["EPA per Play"].idxmax()
        plays = int(f.loc[best, "Plays"])
        small = plays < v.LOW_N_ROWS
        out.append(("Threat" + (" · small sample" if small else ""), "gray" if small else "orange",
                    f"Best formation: {str(best).title()}, {f.loc[best, 'EPA per Play']:+.2f} EPA on {plays} plays"))
    return out


def _attention_items(df: pd.DataFrame) -> list[tuple[str, str]]:
    items = []
    gs = ins.games(df)
    if gs:
        cov = ins.tagging_coverage(df[df["game_id"] == gs[-1]])
        if not cov.empty:
            low = [c for c, x in cov.iloc[0].items() if pd.notna(x) and x < 0.5]
            if low:
                items.append(("danger", f"{ins.game_label(df, gs[-1])} is missing most of: {', '.join(low).lower()}"))
    runs = df[df["PLAY TYPE"] == "Run"]
    if len(runs):
        gap = runs["GAP"].notna().mean()
        if gap < 0.7:
            items.append(("warning", f"Gap tagged on {gap:.0%} of runs this season"))
    issues = validate_plays(df)
    if len(issues):
        items.append(("warning", f"{len(issues)} possible tagging mistakes (see Tagging Coverage)"))
    return items


def _unit_line(d: pd.DataFrame) -> dict:
    rp = v.run_pass(d)
    return {"EPA / play": rp["epa"].mean(), "Success": rp["success"].mean(),
            "Explosive": pd.to_numeric(rp["explosive_play"], errors="coerce").mean()}


def render_home(df: pd.DataFrame, opponents: list[str], pages: dict) -> None:
    """
    df = every game loaded. The "Up next" card reads the opponent's film from any game it's in (that's the
    point of scouting film); everything about Dowling (last game vs season, tagging checks, data-through date)
    uses only games Dowling played.
    """
    st.title("Dowling Scouting Dashboard")
    st.caption(ins.last_updated_text(df))
    dchs = ins.dowling_only(df)
    default_opp = _secret("next_opponent") or st.session_state.get("f_opp") or (opponents[0] if opponents else None)
    st.session_state.setdefault("f_opp", default_opp if default_opp in opponents else (opponents[0] if opponents else None))

    c1, c2 = st.columns([3, 2])
    with c1:
        with st.container(border=True):
            opp = st.selectbox("Up next", opponents, key="f_opp",
                               help="Sets the opponent for every Scouting page too. An admin can set a default "
                                    "with next_opponent in the app's secrets.")
            n_games = df.loc[(df["offense"] == opp) | (df["defense"] == opp), "game_id"].nunique()
            st.caption(f"{n_games} game(s) of film on {opp}")
            tells = _opponent_tells(df, opp)
            for badge, color, text in tells:
                st.markdown(f":{color}-badge[{badge}] {text}")
            if tells:
                st.caption("**Tell** = solid sample. **Lean** = probably real, worth a check on film. "
                           "**Small sample** = could easily be noise.")
            note = latest_note("opponent", opp)
            if note:
                st.info(f"Staff note: {note}")
            b1, b2, b3 = st.columns(3)
            b1.page_link(pages["report"], label="Scouting report", icon=":material/description:")
            b2.page_link(pages["matchup"], label="Matchup", icon=":material/compare_arrows:")
            b3.page_link(pages["sideline"], label="Sideline mode", icon=":material/smartphone:")
    with c2:
        with st.container(border=True):
            st.markdown("**Needs attention**")
            items = _attention_items(dchs)
            if not items:
                st.caption("Nothing flagged.")
            for kind, text in items:
                st.markdown(f":{'red' if kind == 'danger' else 'orange'}[●] {text}")
            st.page_link(pages["tagging"], label="Tagging coverage", icon=":material/sell:")

    gs = ins.games(dchs)
    if len(gs) >= 2:
        last, rest = dchs[dchs["game_id"] == gs[-1]], dchs[dchs["game_id"] != gs[-1]]
        st.markdown(f"**Last game vs season average** ({ins.game_label(dchs, gs[-1])} compared to Dowling's other "
                    f"{len(gs) - 1} games)")
        cols = st.columns(6)
        o_last, o_rest = _unit_line(last[last["offense"] == TEAM]), _unit_line(rest[rest["offense"] == TEAM])
        d_last, d_rest = _unit_line(last[last["defense"] == TEAM]), _unit_line(rest[rest["defense"] == TEAM])
        fmt = {"EPA / play": lambda x: f"{x:+.2f}", "Success": lambda x: f"{x:.0%}", "Explosive": lambda x: f"{x:.0%}"}
        dfmt = {"EPA / play": lambda x: f"{x:+.2f}", "Success": lambda x: f"{x * 100:+.0f} pts",
                "Explosive": lambda x: f"{x * 100:+.0f} pts"}
        for i, k in enumerate(fmt):
            cols[i].metric(f"Off. {k}", fmt[k](o_last[k]), dfmt[k](o_last[k] - o_rest[k]))
            cols[i + 3].metric(f"Def. {k} allowed", fmt[k](d_last[k]), dfmt[k](d_last[k] - d_rest[k]),
                               delta_color="inverse")

    st.markdown("**Jump to**")
    q = st.columns(5)
    for col, (key, label, icon) in zip(q, [("recap", "Game recap", ":material/sports_football:"),
                                           ("selfscout", "Self-scout", ":material/visibility:"),
                                           ("st", "Special teams", ":material/sports:"),
                                           ("finder", "Play finder", ":material/search:"),
                                           ("glossary", "How to read this", ":material/help:")]):
        col.page_link(pages[key], label=label, icon=icon)


# ---------------------------------------------------------------------------
# Play finder
# ---------------------------------------------------------------------------
ST_PLAY_TYPES = {"KO", "KO Rec", "Punt", "Punt Rec", "Fake Punt", "FG", "FG Block", "Extra Pt.", "Extra Pt. Block",
                 "2 Pt.", "2 Pt. Block", "2 Pt. Defend"}


def render_play_finder(df: pd.DataFrame) -> None:
    st.caption("Filter every play by any combination of tags. Picking a situation leaves out special teams plays.")
    c = st.columns(4)
    teams = sorted(df["offense"].dropna().unique(), key=lambda t: (t != TEAM, t))
    offense = c[0].multiselect("Offense", teams, key="pf_off")
    weeks = c[1].multiselect("Week", sorted(df["WEEK"].dropna().unique()), key="pf_week",
                             format_func=lambda w: v.week_labels(df).get(w, f"W{w}"))
    downs = c[2].multiselect("Down", [1, 2, 3, 4], key="pf_down")
    dist = c[3].multiselect("Distance", v.DISTANCE_ORDER, key="pf_dist")
    c = st.columns(4)
    ptype = c[0].multiselect("Play type", sorted(df["PLAY TYPE"].dropna().unique()), key="pf_type")
    base = df[df["offense"].isin(offense)] if offense else df
    form = c[1].multiselect("Formation", sorted(base["OFF FORM"].dropna().astype(str).unique()), key="pf_form")
    call = c[2].text_input("Play call contains", key="pf_call")
    result_col = c[3]  # filled in below, once the other filters are applied, so it only lists results that exist
    c = st.columns(4)
    hash_ = c[0].multiselect("Hash", sorted(df["HASH"].dropna().unique()), key="pf_hash")
    situation = c[1].selectbox("Situation", list(ins.SITUATIONS), key="pf_sit")
    explosive = c[2].checkbox("Explosive plays only", key="pf_expl")
    negative = c[3].checkbox("Negative plays only", key="pf_neg")

    d = base
    for col, sel in (("WEEK", weeks), ("DN", downs), ("Distance", dist), ("PLAY TYPE", ptype),
                     ("OFF FORM", form), ("HASH", hash_)):
        if sel:
            d = d[d[col].astype(str).isin([str(s) for s in sel])] if col == "OFF FORM" else d[d[col].isin(sel)]
    if call:
        d = d[d["OFF PLAY"].astype(str).str.contains(call, case=False, na=False)]
    d = ins.apply_situation(d, situation)
    if situation != "All plays":
        d = d[~d["PLAY TYPE"].isin(ST_PLAY_TYPES)]
    if explosive:
        d = d[pd.to_numeric(d["explosive_play"], errors="coerce") == 1]
    if negative:
        d = d[d["GN/LS"] < 0]
    picked = st.session_state.get("pf_result_sel", [])
    options = sorted(set(d["RESULT"].dropna().astype(str)) | set(picked))
    result = result_col.multiselect("Result", options, key="pf_result_sel",
                                    help="Only results that show up with the other filters are listed.")
    if result:
        d = d[d["RESULT"].astype(str).isin(result)]

    rp = v.run_pass(d)
    m = st.columns(5)
    m[0].metric("Plays found", len(d))
    m[1].metric("Pass rate", f"{rp['PASS'].mean():.0%}" if len(rp) else "–")
    m[2].metric("EPA / play", f"{rp['epa'].mean():+.2f}" if len(rp) else "–")
    m[3].metric("Success", f"{rp['success'].mean():.0%}" if len(rp) else "–")
    m[4].metric("Avg gain", f"{rp['GN/LS'].mean():.1f}" if len(rp) else "–")
    cols = ["game_id", "QTR", "DN", "DIST", "YARD LN", "HASH", "offense", "PLAY TYPE", "OFF FORM",
            "OFF PLAY", "PLAY DIR", "RESULT", "GN/LS", "epa"]
    t = d[[c for c in cols if c in d]].copy()
    succ = pd.to_numeric(d["success"], errors="coerce")
    t["Success"] = np.select([succ == 1, succ == 0], ["Yes", "No"], default="")
    mine = ins.dowling_game_ids(df)
    t["game_id"] = t["game_id"].map(lambda g: ins.game_label(df, g) if g in mine else ins.scout_game_label(df, g))
    t = t.rename(columns={"game_id": "Game", "offense": "Offense", "epa": "EPA"})
    for c_ in ("DN", "DIST", "QTR"):
        t[c_] = t[c_].astype("Int64")
    st.dataframe(t.style.format({"EPA": "{:+.2f}", "YARD LN": "{:.0f}", "GN/LS": "{:.0f}"}, na_rep=""),
                 width="stretch", hide_index=True, height=480)
    st.caption("The download button in the table's top-right corner exports these plays to CSV.")


# ---------------------------------------------------------------------------
# Sideline mode
# ---------------------------------------------------------------------------
def _bucket(dist: float) -> str:
    return "s" if dist <= 3 else "m" if dist <= 6 else "l" if dist <= 10 else "x"


def render_sideline(df: pd.DataFrame, opponents: list[str], pages: dict) -> None:
    st.caption("Built for a phone or the press-box tablet: pick what you see, get their tendency.")
    opp = st.selectbox("Opponent", opponents, key="f_opp")
    plays = v.run_pass(df[df["offense"] == opp])
    dn = st.segmented_control("Down", [1, 2, 3, 4], default=1, key="sl_dn",
                              format_func=lambda x: ["", "1st", "2nd", "3rd", "4th"][x])
    dist = st.segmented_control("Distance", ["s", "m", "l", "x"], default="l", key="sl_dist",
                                format_func={"s": "1–3", "m": "4–6", "l": "7–10", "x": "11+"}.get)
    hash_ = st.segmented_control("Hash", ["L", "M", "R"], key="sl_hash",
                                 format_func={"L": "Left", "M": "Middle", "R": "Right"}.get)
    forms = plays["OFF FORM"].value_counts().index.astype(str).tolist()
    form = st.selectbox("Formation", ["Any"] + forms, key="sl_form", format_func=lambda f: f.title())

    m = plays
    if dn:
        m = m[m["DN"] == dn]
    if dist:
        m = m[m["DIST"].map(_bucket) == dist]
    if hash_:
        m = m[m["HASH"] == hash_]
    if form != "Any":
        m = m[m["OFF FORM"].astype(str) == form]

    with st.container(border=True):
        if m.empty:
            st.markdown("### No matching plays")
            st.caption("Loosen a filter (clear the hash or set formation to Any).")
        else:
            pr = m["PASS"].mean()
            call = f"Pass {pr:.0%}" if pr >= 0.5 else f"Run {1 - pr:.0%}"
            st.markdown(f"## {call}")
            st.progress(float(1 - pr), text=f"Run {1 - pr:.0%} · Pass {pr:.0%} · {len(m)} plays")
            dirs = m["PLAY DIR"].dropna()
            dirs = dirs[dirs.isin(["L", "R"])]
            if len(dirs):
                st.caption(f"Direction: left {(dirs == 'L').mean():.0%} · right {(dirs == 'R').mean():.0%}")
            share = max(pr, 1 - pr)
            grade, lb = v.tendency_grade(round(share * len(m)), len(m))
            if len(m) < v.TELL_MIN_N or grade == "Small sample":
                st.warning(f"Small sample: could easily be noise (only sure it's at least {lb:.0%} one way).")
            elif grade == "Solid" and share >= v.TELL_SHARE:
                st.success(f"Tell: {share:.0%} one way, and the sample backs it up (at least {lb:.0%}).")
            elif share >= v.TELL_SHARE:
                st.info(f"Lean: {share:.0%} one way, but only sure it's at least {lb:.0%}. Check the film.")
            top = m["OFF FORM"].value_counts().head(3)
            if form == "Any" and len(top):
                st.caption("Most common looks here: " + ", ".join(f"{str(f).title()} ({n})" for f, n in top.items()))
    c = st.columns(3)
    c[0].page_link(pages["fourth"], label="4th Down Bot", icon="🏈")
    c[1].page_link(pages["go2"], label="Go for 2 Bot", icon="🎯")
    if "home" in pages:
        c[2].page_link(pages["home"], label="Full dashboard", icon=":material/dashboard:")
    with st.expander("Full down & distance card"):
        v.render_dd_tendencies(df, opp, f"{opp} down & distance tendencies")


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------
GLOSSARY = """
### The numbers
- **EPA (expected points added):** how many points a play was worth, comparing the chance of scoring before and after the snap from down, distance, and field position. Above 0 helped the offense.
- **Success:** a play that stays on schedule: 40% of the yards needed on 1st down, 70% on 2nd, a first down on 3rd or 4th.
- **Explosive:** a run of 10+ yards or a pass of 20+ yards.
- **Stuffed / stuff rate:** a run that gains 0 or fewer yards.
- **Stop rate:** share of opponent drives that end without points.
- **Win probability (WP):** the chance of winning from a situation, from the same model the 4th Down Bot uses. "Points" of WP are percentage points.

### Colors
- **Dowling offense tables:** green = clearly above the table's average. Red only for negative EPA, under 4.5 yards per play, or under 40% success. Anything in between is uncolored.
- **Defense and scouting tables:** green is good for Dowling, red is bad, compared to the table's average.
- **Gray rows** have fewer than 10 plays: small samples.

### Team Profiles (radar charts)
Each axis is how far a number sits from the FBS median, in FBS standard deviations. The middle ring is the median; farther out is always better. Special teams axes are centered on the average kick in our own film instead (high school punts and kicks are shorter than college ones), using the same per-kick EPA as the Special Teams page. Triangles are past ±2 SD; hollow points are small samples.

### 4th downs
- **Toss-up:** the best option beats the next one by under 1 point of win probability, so either call is fine.
- **Estimated clock:** the data has no game clock, so each quarter's plays are spread evenly across 12 minutes, then scaled to the model's 15-minute quarters.
- **Neutral site:** home/away isn't in the data, so reviews average the two.

### Tendencies
- **Tell / Lean / Small sample:** a tendency is 75%+ one way (run or pass) on 5+ plays. The badge says how much to trust it, based on the lowest one-way share the plays are consistent with (90% confidence): **Tell** if that's 70%+ (e.g. 29 of 35), **Lean** if 55%+ (e.g. 6 of 6), otherwise **Small sample** (e.g. 5 of 6).
- **Strong / Lean / Mixed** (self-scout): 80%+, 70–79%, under 70% one way.

### Data notes
- Everything comes from Hudl tags. The **Tagging Coverage** page shows what's missing; charts built on a mostly blank tag say so underneath.
- Special teams starts and net punts come from where the receiving team's next snap was, since kick distance isn't tagged.
"""


def render_glossary() -> None:
    st.markdown(GLOSSARY)


# ---------------------------------------------------------------------------
# Shareable links for the game-day bots: the inputs live in the URL, so a
# link texted from the press box opens to the exact same situation.
# ---------------------------------------------------------------------------
def _parse_param(raw: str, kind, allowed):
    """Convert a URL value; None if it's malformed or out of range."""
    try:
        if kind is bool:
            val = raw in ("1", "true", "yes")
        elif kind is int:
            val = int(float(raw))
        else:
            val = str(raw)
    except (TypeError, ValueError):
        return None
    if isinstance(allowed, tuple) and kind is int and not allowed[0] <= val <= allowed[1]:
        return None
    if isinstance(allowed, list) and val not in allowed:
        return None
    return val


def apply_link_params(spec: list[tuple], flag: str) -> None:
    """
    Before the widgets are drawn: load a shared link's values into session
    state, once per visit. spec rows are (param, state_key, type, default,
    allowed), where allowed is a (min, max) range, a list of options, or None.
    Bad or out-of-range values are ignored rather than crashing the page.
    """
    # Defaults live here instead of on the widgets, so a widget never gets a
    # value from both its default and session state (Streamlit warns on that).
    for _param, key, _kind, default, _allowed in spec:
        st.session_state.setdefault(key, default)
    if st.session_state.get(flag):
        return
    st.session_state[flag] = True
    qp = st.query_params
    for param, key, kind, _default, allowed in spec:
        if param in qp:
            val = _parse_param(qp[param], kind, allowed)
            if val is not None:
                st.session_state[key] = val


def write_link_params(spec: list[tuple], values: dict) -> str:
    """After the widgets: put every non-default value in the URL. Returns the full link."""
    params = {}
    for param, key, kind, default, _allowed in spec:
        val = values.get(key, default)
        if val is None or val == default:
            continue
        params[param] = ("1" if val else "0") if kind is bool else str(val)
    if dict(st.query_params) != params:
        st.query_params.from_dict(params)
    try:
        base = (st.context.url or "").split("?")[0]
    except Exception:
        base = ""
    return base + ("?" + "&".join(f"{k}={quote(str(v))}" for k, v in params.items()) if params else "")


def share_link_box(link: str, what: str = "this situation") -> None:
    if not link:
        return
    with st.expander(f"Share {what}"):
        st.code(link, language=None, wrap_lines=True)
        st.caption("Copy with the button on the right. The link opens straight to these inputs. Your browser's "
                   "address bar always has the same link.")
