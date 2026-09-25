"""
Dowling Catholic Scouting Dashboard (Streamlit port of Dowling_Scouting_Dashboard.twb)
plus the 4th Down Bot and Go for 2 Bot.

Run with:  streamlit run app.py

Folder layout (everything sits next to this file):
    app.py              entrypoint: navigation + scouting sidebar filters
    visuals.py          scouting data prep and render_* functions
    Fourth_Downs.py     4th Down Bot page
    Go_for_2.py         Go for 2 Bot page
    model_utils.py      EP/WP model loading used by the 4th Down Bot
    curated-pbp.xlsx    play-by-play data
    *.json              trained model files (optional)
"""
import streamlit as st

import visuals as v

DATA_PATH = "curated-pbp.xlsx"

st.set_page_config(page_title="Dowling Scouting Dashboard", layout="wide")

# Keep scouting filter selections when you visit a Game Day page and come back.
# Streamlit drops widget state for widgets that aren't drawn on the current page;
# re-assigning the value here keeps it alive.
for _k in list(st.session_state.keys()):
    if _k.startswith("f_"):
        st.session_state[_k] = st.session_state[_k]


# ---- Shared page text -----------------------------------------------------
BENCHMARKS = {
    # "team"/"offense" pages: elite offensive numbers. "defense": elite defensive numbers.
    "offense": dict(success="> 50%", pass_epa="> 0.35", rush_epa="> 0.2", pass_chunk="> 17%",
                    rush_chunk="> 13%", pass_expl="> 15%", rush_expl="> 8%"),
    "defense": dict(success="< 35%", pass_epa="< -0.05", rush_epa="< -0.12", pass_chunk="< 13%",
                    rush_chunk="< 5%", pass_expl="< 8%", rush_expl="< 3%"),
}

DEFINITIONS = """
**Definitions:**

**Success:** a 1st down play that gains 40% of yards for the first down, a 2nd down play that gains 60% of yards for the first down, and a 3rd/4th down play that gains a first down are successful plays. An elite College Football {unit} has success rates {success}.

**EPA:** Expected Points Added, which measures how many points a play was worth by comparing your chances of scoring before and after the snap, based on down, distance, and field position. An elite College Football {unit} has a Pass EPA per Play {pass_epa} and a Rush EPA per Play {rush_epa}.

**Chunk Rate:** % of plays that gain between 10 and 19 yards. An elite College Football {unit} has a Passing Chunk Rate {pass_chunk} and a Rushing Chunk Rate {rush_chunk}.

**Explosive Rate:** % of plays that gain 20+ yards. An elite College Football {unit} has a Passing Explosive Rate {pass_expl} and a Rushing Explosive Rate {rush_expl}.
"""


def page_header(title: str, definitions: str | None = None) -> None:
    """definitions: None (no text), "team", "offense", or "defense"."""
    st.title(title)
    if definitions:
        bench = BENCHMARKS["defense" if definitions == "defense" else "offense"]
        text = DEFINITIONS.format(unit=definitions, **bench)
        st.caption(text.replace("<", "\\<").replace(">", "\\>"))  # keep < and > literal in Markdown


# ---- Pages ----------------------------------------------------------------
def dchs_offense():
    page_header("DCHS Offense", definitions="offense")
    v.render_dchs_offense(df)
    v.render_dchs_offense_tendencies(df)
    v.render_dchs_offense_3rd_downs(df_any_down)
    v.render_dchs_offense_4th_downs(df_any_down)
    v.render_dchs_formations(df)


def dchs_o_run_game():
    page_header("DCHS O Run Game", definitions="offense")
    v.render_run_scheme_detail(df)
    v.render_rush_vs_box(df)


def dchs_o_pass_game():
    page_header("DCHS O Pass Game")
    v.render_dchs_intended_pass_distance(df)
    v.render_pass_vs_box(df)


def dchs_o_weekly_trends():
    page_header("DCHS O Weekly Trends")
    c1, c2 = st.columns(2)
    with c1:
        v.render_weekly_pass_epa(df_all_weeks)
        v.render_weekly_rush_epa(df_all_weeks)
    with c2:
        v.render_weekly_pass_success(df_all_weeks)
        v.render_weekly_rush_success(df_all_weeks)


def dchs_d_overview():
    page_header("DCHS D Overview", definitions="defense")
    v.render_dchs_defense(df)
    v.render_d_3rd_downs(df_any_down)
    v.render_d_vs_formation(df)


def dchs_d_run_game():
    page_header("DCHS D Run Game", definitions="defense")
    v.render_d_rush_vs_box(df)


def dchs_d_pass_game():
    page_header("DCHS D Pass Game")
    v.render_d_pass_coverage(df)


def scout_opposing_offense():
    page_header("Scout Opposing Offense", definitions="team")
    v.render_opp_tendencies(df, opponent)
    v.render_opp_3rd_downs(df_any_down, opponent)
    v.render_opp_4th_downs(df_any_down, opponent)


SCOUTING_PAGES = {
    "DCHS Offense": [
        st.Page(dchs_offense, title="DCHS Offense", url_path="dchs-offense", default=True),
        st.Page(dchs_o_run_game, title="DCHS O Run Game", url_path="dchs-o-run-game"),
        st.Page(dchs_o_pass_game, title="DCHS O Pass Game", url_path="dchs-o-pass-game"),
        st.Page(dchs_o_weekly_trends, title="DCHS O Weekly Trends", url_path="dchs-o-weekly-trends"),
    ],
    "DCHS Defense": [
        st.Page(dchs_d_overview, title="DCHS D Overview", url_path="dchs-d-overview"),
        st.Page(dchs_d_run_game, title="DCHS D Run Game", url_path="dchs-d-run-game"),
        st.Page(dchs_d_pass_game, title="DCHS D Pass Game", url_path="dchs-d-pass-game"),
    ],
    "Scouting": [
        st.Page(scout_opposing_offense, title="Scout Opposing Offense", url_path="scout-opposing-offense"),
    ],
}
GAME_DAY_PAGES = {
    "Game Day": [
        st.Page("Fourth_Downs.py", title="4th Down Bot", icon="🏈", url_path="fourth-down-bot"),
        st.Page("Go_for_2.py", title="Go for 2 Bot", icon="🎯", url_path="go-for-2-bot"),
    ],
}

pg = st.navigation({**SCOUTING_PAGES, **GAME_DAY_PAGES})


# ---- Scouting sidebar filters (only drawn on scouting pages) ---------------
def dropdown_multiselect(label, options, key, fmt=str):
    """Compact dropdown: a button that opens a checklist, with a one-line summary.
    Everything is checked by default."""
    keys = {o: f"{key}_{o}" for o in options}
    for k in keys.values():
        st.session_state.setdefault(k, True)

    def set_all(value):
        for k in keys.values():
            st.session_state[k] = value

    with st.popover(label, width="stretch"):
        c1, c2 = st.columns(2)
        c1.button("Select all", key=f"btn_{key}_all", on_click=set_all, args=(True,), width="stretch")
        c2.button("Clear", key=f"btn_{key}_none", on_click=set_all, args=(False,), width="stretch")
        selected = [o for o in options if st.checkbox(fmt(o), key=keys[o])]
    if len(selected) == len(options):
        summary = "All"
    elif not selected:
        summary = "None selected"
    else:
        summary = ", ".join(fmt(o) for o in selected)
    st.caption(f"{label}: {summary}")
    return selected


def filtered(use_week: bool = True, use_down: bool = True):
    """Apply the sidebar filters. A filter with everything selected is skipped,
    so plays with a blank Down or Distance aren't dropped unless you narrow it."""
    d = df_all
    if use_week:
        d = d[d["WEEK"].isin(sel_weeks)]
    if use_down and set(sel_downs) != set(ALL_DOWNS):
        d = d[d["DN"].isin(sel_downs)]
    if set(sel_dist) != set(v.DISTANCE_ORDER):
        d = d[d["Distance"].isin(sel_dist)]
    return d


if any(pg is p for group in SCOUTING_PAGES.values() for p in group):
    df_all = v.load_data(DATA_PATH)
    ALL_DOWNS = [1, 2, 3, 4]
    ALL_WEEKS = sorted(df_all["WEEK"].dropna().unique().tolist())
    OPPONENTS = sorted(o for o in df_all["offense"].dropna().unique() if o != v.TEAM)
    st.session_state.setdefault("f_opp", "SEP" if "SEP" in OPPONENTS else OPPONENTS[0])

    with st.sidebar:
        st.header("Filters")
        sel_downs = dropdown_multiselect("Down", ALL_DOWNS, key="f_down")
        sel_dist = dropdown_multiselect("Distance", v.DISTANCE_ORDER, key="f_dist")
        sel_weeks = dropdown_multiselect("Week", ALL_WEEKS, key="f_week", fmt=lambda w: f"Week {w}")
        # Opponent only matters on the Scouting section's pages
        if any(pg is p for p in SCOUTING_PAGES["Scouting"]):
            opponent = st.selectbox("Opponent", OPPONENTS, key="f_opp")

    df = filtered()                          # most visuals
    df_any_down = filtered(use_down=False)   # 3rd/4th down tables set their own down
    df_all_weeks = filtered(use_week=False)  # weekly trend charts show every week

pg.run()
