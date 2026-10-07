"""
Dowling Catholic Scouting Dashboard (Streamlit port of Dowling_Scouting_Dashboard.twb)
plus game review, self-scout, data tools, and the 4th Down / Go for 2 bots.

Run with:  streamlit run app.py

Folder layout (everything sits next to this file):
    app.py              entrypoint: navigation, sidebar filters, page layouts
    visuals.py          scouting data prep, tables, and charts
    insights.py         game recap, self-scout, matchup, report, data tools, WP / 4th-down review
    profiles.py         team profile radar charts benchmarked against FBS
    build_cfb_benchmarks.py  builds cfb_benchmarks.csv from cfbfastR play-by-play (run locally)
    cfb_benchmarks.csv  FBS team-by-team metric table the radar charts compare against
    curate_pbp.py       curation pipeline (used by the "Add a game" page)
    fourth_down_core.py 4th Down Bot decision math (shared with the 4th-down review)
    Fourth_Downs.py     4th Down Bot page
    Go_for_2.py         Go for 2 Bot page
    model_utils.py      EP/WP model loading used by the bots and win probability
    curated-pbp.xlsx    play-by-play data
    *.json              trained model files (optional)
"""
import streamlit as st

import breakdowns as bd
import insights as ins
import profiles
import qol
import special_projects
import special_teams
import visuals as v

DATA_PATH = "curated-pbp.xlsx"

st.set_page_config(page_title="Dowling Scouting Dashboard", layout="wide")

# Optional login: set app_password in the app's secrets to require it.
if not qol.check_password():
    st.stop()

# Keep filter selections when you visit a page without the filters and come
# back. Streamlit drops widget state for widgets that aren't drawn on the
# current page; re-assigning the value here keeps it alive.
for _k in list(st.session_state.keys()):
    if _k.startswith(("f_", "g_")):
        st.session_state[_k] = st.session_state[_k]


# ---- Shared page text -----------------------------------------------------
DEFINITIONS = """
**Success:** a 1st down play that gains 40% of the yards needed, a 2nd down play that gains 70%, or a 3rd/4th down play that gains a first down.

**EPA (expected points added):** how many points a play was worth, comparing the offense's chances of scoring before and after the snap based on down, distance, and field position. Above 0 is good for the offense.

**Explosive rate:** share of plays that are a run of 10+ yards or a pass of 20+ yards.

**Colors on Dowling offense tables:** green = clearly above the table's average. Red is only for results that are bad on their own: negative EPA, under 4.5 yards per play, or under a 40% success rate. Anything in between is left uncolored.

**Colors on DCHS defense tables:** green is good for Dowling and red is bad, compared to the table's average, so green = the opponent's offense did worse.

**Colors when scouting an opponent's offense:** from their side, with the same rules as Dowling's offense tables: green = it works for them, red = it doesn't.

Gray rows have fewer than 10 plays, so treat them as small samples. Hover a column name for its definition.
"""


def page_header(title: str, show_definitions: bool = True, show_filters: bool = True) -> None:
    st.title(title)
    if show_filters and "sel_weeks" in globals():
        chips = active_filter_chips()
        st.html(
            '<div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:12px">'
            '<span style="opacity:.6;margin-right:2px">Showing</span>'
            + "".join(f'<span style="padding:3px 10px;border-radius:999px;border:0.5px solid rgba(128,128,128,.4);'
                      f'background:rgba(128,128,128,.08)">{c}</span>' for c in chips)
            + "</div>"
        )
    if show_definitions:
        with st.expander("What do these numbers mean?"):
            st.markdown(DEFINITIONS)


# ---- Pages: DCHS offense -----------------------------------------------------
def dchs_offense():
    page_header("DCHS Offense")
    forms = v.dchs_formations_table(df)
    v.render_takeaways(forms, "formation")
    v.render_dchs_offense(df)
    v.render_dchs_offense_tendencies(df)
    v.render_dchs_offense_3rd_downs(df_any_down)
    v.render_dchs_offense_4th_downs(df_any_down)
    v.render_dchs_formations(df, table=forms)
    v.render_usage_scatter(forms, "DCHS formations: EPA vs success", noun="Formation")


def dchs_o_run_game():
    page_header("DCHS O Run Game")
    v.render_run_gaps(df, "offense", v.TEAM, "DCHS O Run Gaps")
    schemes = v.run_scheme_table(df)
    v.render_takeaways(schemes, "run scheme")
    v.render_run_scheme_detail(df, table=schemes)
    v.render_usage_scatter(schemes, "DCHS run schemes: EPA vs success", noun="Run scheme", min_plays=10)
    v.render_rush_vs_box(df)


def dchs_o_pass_game():
    page_header("DCHS O Pass Game")
    v.render_pass_zones(df, "offense", v.TEAM, "DCHS O Pass Zones", key="pz_dchs_o")
    calls = v.o_pass_detail_table(df)
    v.render_takeaways(calls, "pass play")
    v.render_o_pass_game_detail(df, table=calls)
    v.render_usage_scatter(calls, "DCHS pass plays: EPA vs success", noun="Pass play")
    v.render_dchs_intended_pass_distance(df)
    v.render_pass_vs_box(df)


def dchs_o_weekly_trends():
    page_header("DCHS O Weekly Trends")
    v.render_weekly_trend(df_all_weeks, "epa", "EPA per play by week", "+.2f")
    v.render_weekly_trend(df_all_weeks, "success", "Success rate by week", ".0%")


def dchs_self_scout():
    page_header("DCHS Self-Scout")
    st.caption("What an opponent scouting Dowling's film would see. Strong tendencies are worth breaking "
               "before someone builds a game plan around them.")
    tabs = st.tabs(["Tendencies", "Tendency tree", "Field & strength", "Sequencing"])
    with tabs[0]:
        ins.render_self_scout(df, df_any_down)
    with tabs[1]:
        bd.render_tendency_tree(df, v.TEAM, "DCHS formation tree", key="tree_dchs")
    with tabs[2]:
        bd.render_field_strength(df, v.TEAM, "DCHS by hash and formation strength")
    with tabs[3]:
        bd.render_sequencing(df, _week_games(), v.TEAM, "What DCHS calls after...")


def dchs_o_red_zone():
    page_header("DCHS O Red Zone")
    bd.render_red_zone(df, _week_games(), v.TEAM, "offense", "DCHS offense", good_high=True, key="rz_o")


def dchs_d_red_zone():
    page_header("DCHS D Red Zone")
    bd.render_red_zone(df, _week_games(), v.TEAM, "defense", "Opponents vs DCHS defense", good_high=False,
                       key="rz_d")


def _week_games():
    """Whole games from the selected weeks: for anything that needs full drives or the previous snap."""
    return df_all[df_all["WEEK"].isin(sel_weeks)]


# ---- Pages: DCHS defense -----------------------------------------------------
def dchs_d_overview():
    page_header("DCHS D Overview")
    forms = v.d_vs_formation_table(df)
    v.render_takeaways(forms, "opponent formation", good_high=False)
    v.render_dchs_defense(df)
    v.render_d_3rd_downs(df_any_down)
    v.render_d_vs_formation(df, table=forms)
    v.render_usage_scatter(forms, "Opponent formations vs DCHS D: EPA vs success", good_high=False,
                           noun="Formation", min_plays=10)


def dchs_d_run_game():
    page_header("DCHS D Run Game")
    v.render_run_gaps(df, "defense", v.TEAM, "DCHS D Run Gaps", good_high=False)
    v.render_d_rush_vs_box(df)
    v.render_d_call_tables(df, "Run")
    v.render_d_run_game_detail(df)


def dchs_d_pass_game():
    page_header("DCHS D Pass Game")
    v.render_pass_zones(df, "defense", v.TEAM, "DCHS D Pass Zones", key="pz_dchs_d", good_high=False)
    cov = v.d_coverage_table(df)
    v.render_takeaways(cov, "coverage", good_high=False)
    v.render_d_pass_coverage(df, table=cov)
    v.render_usage_scatter(cov, "DCHS coverages: EPA vs success allowed", good_high=False, noun="Coverage")
    ins.render_coverage_by_formation(df)
    v.render_d_call_tables(df, "Pass")
    v.render_d_pass_game_detail(df)


# ---- Pages: scouting -----------------------------------------------------------
def scout_opposing_offense():
    page_header("Scout Opposing Offense")
    tendencies, tree, field, seq, field_pos, run_game, pass_game, red_zone = st.tabs(
        ["Tendencies", "Tendency tree", "Field & strength", "Sequencing", "Field position", "Run game", "Pass game",
         "Red zone"])
    with tendencies:
        v.render_dd_tendencies(df_any_down, opponent, f"{opponent} down & distance tendencies")
        v.render_opp_tendencies(df, opponent)
        v.render_opp_look_tendencies(df, opponent)
        v.render_opp_3rd_downs(df_any_down, opponent)
        v.render_opp_4th_downs(df_any_down, opponent)
        ins.render_down_calls(df_any_down, opponent, 3)
        ins.render_down_calls(df_any_down, opponent, 4)
        ins.render_best_plays(df, opponent)
    with tree:
        bd.render_tendency_tree(df, opponent, f"{opponent} formation tree", key="tree_opp",
                                min_form_plays=v.SCOUT_MIN_SAMPLE)
    with field:
        bd.render_field_strength(df, opponent, f"{opponent} by hash and formation strength")
    with seq:
        bd.render_sequencing(df, _week_games(), opponent, f"What {opponent} calls after...")
    with field_pos:
        bd.render_field_position(df, _week_games(), opponent, f"{opponent} offense by field position")
    with run_game:
        # Everything on this page is colored from the scouted team's side (v.SCOUT_GOOD_HIGH).
        v.render_run_gaps(df, "offense", opponent, f"{opponent} O Run Gaps", good_high=v.SCOUT_GOOD_HIGH,
                          hide_below=v.SCOUT_MIN_SAMPLE)
        v.render_opp_play_calls(df, opponent, "Run")
    with pass_game:
        v.render_pass_zones(df, "offense", opponent, f"{opponent} O Pass Zones", key="pz_opp_o",
                            good_high=v.SCOUT_GOOD_HIGH,
                            min_att=v.SCOUT_MIN_SAMPLE)
        v.render_opp_play_calls(df, opponent, "Pass")
    with red_zone:
        bd.render_red_zone(df, _week_games(), opponent, "offense", f"{opponent} offense", good_high=v.SCOUT_GOOD_HIGH,
                           key="rz_opp")


def matchup():
    page_header("Matchup")
    ins.render_matchup(df, opponent)


def team_profiles():
    page_header("Team Profiles", show_definitions=False)
    # Offense/defense axes follow every sidebar filter; special teams and
    # drives need whole games, so they only follow the Week filter.
    profiles.render_profiles_page(df, df_all[df_all["WEEK"].isin(sel_weeks)], opponent)


def scouting_report():
    page_header("Scouting Report", show_definitions=False)
    qol.notes_box("opponent", opponent, opponent)
    html = ins.build_report_html(df, df_any_down, opponent, " · ".join(active_filter_chips()),
                                 notes=qol.latest_note("opponent", opponent), df_games=_week_games())
    st.download_button("Download printable report", html, file_name=f"{opponent}_scouting_report.html",
                       mime="text/html", type="primary")
    st.caption("Opens in any browser; use Print → Save as PDF for the binder.")
    st.divider()
    st.subheader("Team profiles")
    profiles.render_profiles_page(df, _week_games(), opponent)
    st.divider()
    v.render_dd_tendencies(df_any_down, opponent, "Down & distance tendencies")
    c1, c2 = st.columns(2)
    with c1:
        v.render_run_gaps(df, "offense", opponent, "Run game by gap", good_high=v.SCOUT_GOOD_HIGH,
                          hide_below=v.SCOUT_MIN_SAMPLE)
    with c2:
        v.render_pass_zones(df, "offense", opponent, "Where they throw", key="pz_report", good_high=v.SCOUT_GOOD_HIGH,
                            min_att=v.SCOUT_MIN_SAMPLE)
    forms = v.opp_formations_table(df, opponent)
    v.show_table("Formations", forms, good_high=v.SCOUT_GOOD_HIGH)
    c1, c2 = st.columns(2)
    with c1:
        v.render_opp_play_calls(df, opponent, "Run")
    with c2:
        v.render_opp_play_calls(df, opponent, "Pass")
    ins.render_best_plays(df, opponent)


# ---- Pages: game review --------------------------------------------------------
GAME_KINDS = ["Dowling games", "Scout games"]


def _game_picker(allow_all: bool) -> tuple[str | None, str]:
    """
    Dowling games or scout film first, then the game. Returns (game_id or None for all Dowling games, team): the
    team is the side the page is told from, Dowling for Dowling games and the first team in the game_id for scout
    film. Scout games are labeled with both teams ("W1 Johnston vs Waukee").
    """
    kind = st.segmented_control("Games", GAME_KINDS, key="g_kind", default=GAME_KINDS[0]) or GAME_KINDS[0]
    if kind == GAME_KINDS[0]:
        options = ins.games(ins.dowling_only(df_all))[::-1]  # newest first
        if allow_all:
            options = options + ["__all__"]
        pick = st.selectbox("Game", options, key="g_game_all" if allow_all else "g_game",
                            format_func=lambda g: "All Dowling games" if g == "__all__" else ins.game_label(df_all, g))
        return (None if pick == "__all__" else pick), v.TEAM
    options = ins.games(ins.scout_only(df_all))[::-1]
    if not options:
        st.info("No scout games loaded yet.")
        st.stop()
    pick = st.selectbox("Game", options, key="g_game_scout", format_func=lambda g: ins.scout_game_label(df_all, g))
    return pick, ins.game_teams(pick)[0]


def game_recap():
    page_header("Game Recap", show_filters=False)
    c1, c2 = st.columns([2, 3])
    with c1:
        gid, team = _game_picker(allow_all=True)
    with c2:
        half = st.radio("Show", ["Full game", "First half", "Second half"], horizontal=True, key="g_half")
    if gid is None:
        st.caption("All Dowling games: drive charts and the scoreboard need a single game, so this shows the "
                   "head-to-head numbers and play logs for the whole season.")
        ins.render_game_recap(ins.dowling_only(df_all), None, half)
        return
    scout = team != v.TEAM
    label = ins.scout_game_label(df_all, gid) if scout else ins.game_label(df_all, gid)
    qol.notes_box("game", gid, label)
    ins.render_game_recap(df_all, gid, half, team=team)


def season_drives_page():
    page_header("Season Drives", show_filters=False)
    st.caption("Every possession this season: where drives start, how far they get, and how they end.")
    bd.render_season_drives(df_all)


def special_projects_page():
    page_header("Special Projects", show_definitions=False, show_filters=False)
    special_projects.render_first_play_study(df_all)


def win_prob_fourth_downs():
    page_header("Win Probability & 4th Downs", show_definitions=False, show_filters=False)
    gid, team = _game_picker(allow_all=True)
    if gid is None:
        ins.render_fourth_down_review(ins.dowling_only(df_all))
        return
    g = df_all[df_all["game_id"] == gid]
    other = ins.other_team(g, team) or ins.game_teams(gid)[1]
    ins.render_win_probability(g, other, team=team)
    if team == v.TEAM:
        ins.render_fourth_down_review(g)
        return
    # Scout film: both teams' 4th downs, one display each.
    for side in (team, other):
        st.subheader(f"{side} 4th downs")
        ins.render_fourth_down_review(g, team=side)


# ---- Pages: special teams -------------------------------------------------------
def special_teams_page():
    page_header("Special Teams", show_definitions=False)
    st.caption("Special teams use whole games, so only the Week filter applies here.")
    special_teams.render_special_teams_page(df_all[df_all["WEEK"].isin(sel_weeks)])


# ---- Pages: home and tools -------------------------------------------------------
def home():
    qol.render_home(df_all, OPPONENTS, PAGE_LINKS)


def play_finder():
    page_header("Play Finder", show_definitions=False, show_filters=False)
    qol.render_play_finder(df_all)


def sideline_mode():
    st.title("Sideline Mode")
    qol.render_sideline(df_all, OPPONENTS, PAGE_LINKS)


def glossary():
    st.title("How to Read This App")
    qol.render_glossary()


# ---- Pages: data ---------------------------------------------------------------
def tagging_coverage():
    page_header("Tagging Coverage", show_definitions=False, show_filters=False)
    st.caption("Every chart depends on Hudl tags. This shows which ones are missing so they can be filled in "
               "while the film is fresh.")
    ins.render_tagging_coverage(df_all)
    qol.render_validation(df_all)


def add_game():
    page_header("Add a Game", show_definitions=False, show_filters=False)
    ins.render_add_game(DATA_PATH)


HOME_PAGE = st.Page(home, title="Home", icon=":material/home:", url_path="home", default=True)
FILTERED_PAGES = {
    "DCHS Offense": [
        st.Page(dchs_offense, title="DCHS Offense", url_path="dchs-offense"),
        st.Page(dchs_o_run_game, title="DCHS O Run Game", url_path="dchs-o-run-game"),
        st.Page(dchs_o_pass_game, title="DCHS O Pass Game", url_path="dchs-o-pass-game"),
        st.Page(dchs_o_weekly_trends, title="DCHS O Weekly Trends", url_path="dchs-o-weekly-trends"),
        st.Page(dchs_self_scout, title="DCHS Self-Scout", url_path="dchs-self-scout"),
        st.Page(dchs_o_red_zone, title="DCHS O Red Zone", url_path="dchs-red-zone"),
    ],
    "DCHS Defense": [
        st.Page(dchs_d_overview, title="DCHS D Overview", url_path="dchs-d-overview"),
        st.Page(dchs_d_run_game, title="DCHS D Run Game", url_path="dchs-d-run-game"),
        st.Page(dchs_d_pass_game, title="DCHS D Pass Game", url_path="dchs-d-pass-game"),
        st.Page(dchs_d_red_zone, title="DCHS D Red Zone", url_path="dchs-d-red-zone"),
    ],
    "Special Teams": [
        st.Page(special_teams_page, title="Special Teams", url_path="special-teams"),
    ],
    "Scouting": [
        st.Page(scout_opposing_offense, title="Scout Opposing Offense", url_path="scout-opposing-offense"),
        st.Page(matchup, title="Matchup", url_path="matchup"),
        st.Page(team_profiles, title="Team Profiles", url_path="team-profiles"),
        st.Page(scouting_report, title="Scouting Report", url_path="scouting-report"),
    ],
}
OTHER_PAGES = {
    "Special Projects": [
        st.Page(special_projects_page, title="First Play of the Drive", url_path="special-projects"),
    ],
    "Tools": [
        st.Page(play_finder, title="Play Finder", icon=":material/search:", url_path="play-finder"),
        st.Page(sideline_mode, title="Sideline Mode", icon=":material/smartphone:", url_path="sideline"),
        st.Page(glossary, title="How to Read This", icon=":material/help:", url_path="how-to-read"),
    ],
    "Game Review": [
        st.Page(game_recap, title="Game Recap", url_path="game-recap"),
        st.Page(season_drives_page, title="Season Drives", url_path="season-drives"),
        st.Page(win_prob_fourth_downs, title="Win Probability & 4th Downs", url_path="win-probability"),
    ],
    "Game Day": [
        st.Page("Fourth_Downs.py", title="4th Down Bot", icon="🏈", url_path="fourth-down-bot"),
        st.Page("Go_for_2.py", title="Go for 2 Bot", icon="🎯", url_path="go-for-2-bot"),
    ],
    "Data": [
        st.Page(tagging_coverage, title="Tagging Coverage", url_path="tagging-coverage"),
        st.Page(add_game, title="Add a Game", url_path="add-a-game"),
    ],
}
# Kept under its old name for anything that still refers to it.
SCOUTING_PAGES = FILTERED_PAGES

_all_pages = {p.url_path: p for group in {**FILTERED_PAGES, **OTHER_PAGES}.values() for p in group}
PAGE_LINKS = {
    "report": _all_pages["scouting-report"], "matchup": _all_pages["matchup"], "sideline": _all_pages["sideline"],
    "tagging": _all_pages["tagging-coverage"], "recap": _all_pages["game-recap"],
    "selfscout": _all_pages["dchs-self-scout"], "st": _all_pages["special-teams"], "finder": _all_pages["play-finder"],
    "glossary": _all_pages["how-to-read"], "fourth": _all_pages["fourth-down-bot"], "go2": _all_pages["go-for-2-bot"],
    "home": HOME_PAGE,
}

pg = st.navigation({"": [HOME_PAGE], **FILTERED_PAGES, **OTHER_PAGES}, expanded=True)


def _is_phone() -> bool:
    """Best guess from the browser's user agent (tablets count as phones' bigger cousins: no)."""
    try:
        ua = (st.context.headers.get("User-Agent") or "").lower()
    except Exception:
        return False
    return any(t in ua for t in ("iphone", "android", "mobile")) and "ipad" not in ua


# On a phone, the first page of a visit is Sideline Mode: that's what a coach
# on the field needs. Only the first load is redirected, so tapping Home in the
# menu afterwards still goes Home. A link with anything after the "?" (a shared
# filtered view, or just ?full=1) skips it.
if not st.session_state.get("_landed"):
    st.session_state["_landed"] = True
    if pg is HOME_PAGE and _is_phone() and not st.query_params:
        st.switch_page(_all_pages["sideline"])


# ---- Sidebar filters (only drawn on the DCHS and Scouting pages) --------------
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


def _summary(selected, options, fmt=str, all_text="All"):
    if set(selected) == set(options):
        return all_text
    if not selected:
        return "None"
    return ", ".join(fmt(o) for o in selected)


def active_filter_chips() -> list[str]:
    week_labels = v.week_labels(df_all)
    chips = [
        _summary(sel_weeks, ALL_WEEKS, lambda w: week_labels.get(w, f"W{w}").split(" ")[0], "All weeks"),
        _summary(sel_downs, ALL_DOWNS, lambda d: {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}[d], "All downs"),
        _summary(sel_dist, v.DISTANCE_ORDER, lambda s: s.split(" (")[0], "All distances"),
    ]
    if situation != "All plays":
        chips.append(situation)
    if min_plays > 1:
        chips.append(f"Rows with {min_plays}+ plays")
    if any(pg is p for p in FILTERED_PAGES["Scouting"]):
        chips.append(f"Opponent: {opponent}")
    return chips


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
    return ins.apply_situation(d, situation)


SITUATION_NAMES = list(ins.SITUATIONS)


def _apply_query_params():
    """On first load, set the filters from the URL (so a shared link opens to the same view)."""
    if st.session_state.get("_qp_applied"):
        return
    st.session_state["_qp_applied"] = True
    qp = st.query_params

    def set_multi(param, key, options, to_value):
        if param not in qp:
            return
        chosen = set()
        for raw in qp[param].split(","):
            try:
                chosen.add(to_value(raw))
            except (ValueError, IndexError):
                pass
        for o in options:
            st.session_state[f"{key}_{o}"] = o in chosen

    set_multi("down", "f_down", ALL_DOWNS, int)
    set_multi("dist", "f_dist", v.DISTANCE_ORDER, lambda i: v.DISTANCE_ORDER[int(i)])
    set_multi("week", "f_week", ALL_WEEKS, lambda w: type(ALL_WEEKS[0])(float(w)) if ALL_WEEKS else w)
    if qp.get("opp") in OPPONENTS:
        st.session_state["f_opp"] = qp["opp"]
    if qp.get("sit", "").isdigit() and int(qp["sit"]) < len(SITUATION_NAMES):
        st.session_state["f_sit"] = SITUATION_NAMES[int(qp["sit"])]


def _write_query_params():
    params = {}
    if set(sel_downs) != set(ALL_DOWNS):
        params["down"] = ",".join(str(d) for d in sel_downs)
    if set(sel_dist) != set(v.DISTANCE_ORDER):
        params["dist"] = ",".join(str(v.DISTANCE_ORDER.index(s)) for s in sel_dist)
    if set(sel_weeks) != set(ALL_WEEKS):
        params["week"] = ",".join(str(int(w)) for w in sel_weeks)
    if situation != "All plays":
        params["sit"] = str(SITUATION_NAMES.index(situation))
    if any(pg is p for p in FILTERED_PAGES["Scouting"]):
        params["opp"] = opponent
    if dict(st.query_params) != params:
        st.query_params.from_dict(params)


def _reset_filters():
    for k in list(st.session_state.keys()):
        if k.startswith(("f_down_", "f_dist_", "f_week_")):
            st.session_state[k] = True
    st.session_state["f_sit"] = "All plays"
    st.session_state["f_minplays"] = 1


df_all = st.cache_data(ins.add_game_state, show_spinner=False)(v.load_data(DATA_PATH))
st.sidebar.caption(ins.last_updated_text(df_all))
OPPONENTS = sorted(o for o in df_all["offense"].dropna().unique() if o != v.TEAM)
if st.session_state.get("f_opp") not in OPPONENTS:
    # Default opponent: next_opponent from the app's secrets (set weekly), else SEP, else the first one.
    _next = qol._secret("next_opponent")
    st.session_state["f_opp"] = _next if _next in OPPONENTS else "SEP" if "SEP" in OPPONENTS else OPPONENTS[0]

if any(pg is p for group in FILTERED_PAGES.values() for p in group):
    ALL_DOWNS = [1, 2, 3, 4]
    ALL_WEEKS = sorted(df_all["WEEK"].dropna().unique().tolist())
    st.session_state.setdefault("f_sit", "All plays")
    st.session_state.setdefault("f_minplays", 1)
    _apply_query_params()

    with st.sidebar:
        st.header("Filters")
        sel_downs = dropdown_multiselect("Down", ALL_DOWNS, key="f_down")
        sel_dist = dropdown_multiselect("Distance", v.DISTANCE_ORDER, key="f_dist")
        sel_weeks = dropdown_multiselect("Week", ALL_WEEKS, key="f_week", fmt=lambda w: f"Week {w}")
        situation = st.selectbox("Situation", SITUATION_NAMES, key="f_sit")
        if situation == "Neutral":
            st.caption(f"Score within {ins.NEUTRAL_MARGIN} at the snap, and not the last 2 minutes of either half "
                       f"(estimated clock, same as the 4th-down page).")
        min_plays = st.slider("Hide table rows with fewer than … plays", 1, 15, key="f_minplays")
        # Opponent only matters on the Scouting section's pages
        if any(pg is p for p in FILTERED_PAGES["Scouting"]):
            opponent = st.selectbox("Opponent", OPPONENTS, key="f_opp")
        st.button("Reset filters", on_click=_reset_filters, width="stretch")

    v.MIN_PLAYS = min_plays
    _write_query_params()
    df = filtered()                          # most visuals
    df_any_down = filtered(use_down=False)   # 3rd/4th down tables set their own down
    df_all_weeks = filtered(use_week=False)  # weekly trend charts show every week

pg.run()
