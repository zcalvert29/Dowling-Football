"""
4th Down Bot — wired to real trained EP/WP models (model_utils.py)
=====================================================================
Presentation modeled on Ben Baldwin's nfl4th / @ben_bot_baldwin. See
model_utils.py for model loading, feature engineering, and the EP->WP
chaining logic. If the real model JSON files aren't found (see
export_models_to_json.R), this runs on heuristic fallbacks instead — the
sidebar shows which mode is active.
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


import model_utils as mu
import qol
import conversion_model as cm

# This page is built for a narrow layout; the scouting pages use "wide".
st.set_page_config(layout="centered")

# Decision math (conversion/FG/punt models and evaluate_options) lives in
# fourth_down_core.py so other pages can reuse it without running this page.
from fourth_down_core import *  # noqa: F401,F403


def field_spot_label(yards_to_goal, off_abbr, def_abbr):
    if yards_to_goal == 50:
        return "at the 50"
    elif yards_to_goal > 50:
        return f"at the {off_abbr} {100 - yards_to_goal}"
    else:
        return f"at the {def_abbr} {yards_to_goal}"


# ==============================================================================
# STREAMLIT UI
# ==============================================================================

st.title("🏈 4th Down Bot")
st.caption("Presentation modeled on Ben Baldwin's nfl4th / @ben_bot_baldwin.")

if mu.USING_REAL_MODELS:
    st.success("Using your trained EP/WP models.", icon="✅")
else:
    st.warning(
        "Trained model files not found — running on heuristic fallback curves. "
        "Run export_models_to_json.R and copy the JSON files into this folder to activate the real models.",
        icon="⚠️"
    )

# Every input, its URL name, default, and allowed values. The URL keeps the
# non-default ones, so a shared link opens to the same situation.
SITES = ["Offense is home", "Offense is away", "Neutral"]
LINK_SPEC = [
    ("us", "fd_off_score", int, 0, (0, 99)), ("them", "fd_def_score", int, 0, (0, 99)),
    ("q", "fd_quarter", int, 1, [1, 2, 3, 4]), ("min", "fd_minutes", int, 11, (0, 12)),
    ("sec", "fd_seconds", int, 30, (0, 59)), ("side", "fd_side", str, "Own", ["Own", "Opp"]),
    ("yl", "fd_yard_line", int, 35, (1, 50)), ("togo", "fd_distance", int, 2, (1, 30)),
    ("our_to", "fd_off_to", int, 3, [0, 1, 2, 3]), ("their_to", "fd_def_to", int, 3, [0, 1, 2, 3]),
    ("site", "fd_site", str, SITES[0], SITES), ("off", "fd_off_abbr", str, "DCHS", None),
    ("def", "fd_def_abbr", str, "OPP", None), ("wind", "fd_wind_speed", int, 0, (0, 40)),
    ("wdir", "fd_wind_dir", str, "Into", ["Into", "With", "Crosswind"]),
    ("rain", "fd_rain", bool, False, None), ("snow", "fd_snow", bool, False, None),
    ("oteam", "fd_off_team", str, "Dowling Catholic", None), ("dteam", "fd_def_team", str, "Average", None),
    ("pat", "fd_pat", int, round(BOT_SETTINGS["p_xp"] * 100), (50, 100)),
    ("range", "fd_fg_range", int, int(BOT_SETTINGS["max_fg_distance"]), (20, 65)),
    ("guard", "fd_guard", bool, bool(BOT_SETTINGS["own_end"]), None),
    ("edge", "fd_guard_edge", str, f'{BOT_SETTINGS["own_min_edge"]:g}', ["1", "1.5", "2", "3"]),
    ("nogo", "fd_nogo", bool, BOT_SETTINGS["no_go"] is not None, None),
    ("ng1", "fd_nogo_1", int, (BOT_SETTINGS["no_go"] or STAFF_NO_GO_DEFAULT)[0][2], (1, 30)),
    ("ng2", "fd_nogo_2", int, (BOT_SETTINGS["no_go"] or STAFF_NO_GO_DEFAULT)[1][2], (1, 30)),
    ("ng3", "fd_nogo_3", int, (BOT_SETTINGS["no_go"] or STAFF_NO_GO_DEFAULT)[2][2], (1, 30)),
]


CONV = cm.fitted()  # same fit the season review uses
TEAMS = ["Average"] + sorted(set(CONV.offense) | set(CONV.defense))
qol.apply_link_params(LINK_SPEC, "_fd_link_applied")

# Inputs live on the page, not the sidebar: on a phone the sidebar is
# collapsed, so a coach would otherwise see a recommendation for a situation
# they never entered.
with st.container(border=True):
    c1, c2 = st.columns(2)
    off_score = c1.number_input("Our score", 0, 99, key="fd_off_score")
    def_score = c2.number_input("Their score", 0, 99, key="fd_def_score")
    c1, c2, c3 = st.columns([2, 1, 1])
    quarter = c1.segmented_control("Quarter", [1, 2, 3, 4], key="fd_quarter",
                                   format_func=lambda q: f"Q{q}", width="stretch") or 1
    minutes = c2.number_input("Min left", 0, 12, key="fd_minutes")
    seconds = c3.number_input("Sec", 0, 59, key="fd_seconds")
    c1, c2, c3 = st.columns([2, 1, 1])
    side = c1.segmented_control("Ball on", ["Own", "Opp"], key="fd_side", width="stretch") or "Own"
    yard_line = c2.number_input("Yard line", 1, 50, key="fd_yard_line",
                                help="Own 35 = your 35. Opp 10 = their 10. The 50 is either.")
    distance = c3.number_input("To go", 1, 30, key="fd_distance")
    yards_to_goal = 100 - yard_line if side == "Own" else yard_line

    with st.expander("Timeouts, site, teams & weather"):
        c1, c2 = st.columns(2)
        off_timeouts = c1.segmented_control("Our timeouts", [0, 1, 2, 3], key="fd_off_to")
        def_timeouts = c2.segmented_control("Their timeouts", [0, 1, 2, 3], key="fd_def_to")
        off_timeouts = 3 if off_timeouts is None else off_timeouts
        def_timeouts = 3 if def_timeouts is None else def_timeouts
        site_label = st.segmented_control("Site", SITES, key="fd_site", width="stretch",
                                          help="The win probability model includes home-field advantage.") \
            or "Offense is home"
        c1, c2 = st.columns(2)
        off_abbr = c1.text_input("Offense", key="fd_off_abbr").upper()[:4]
        def_abbr = c2.text_input("Defense", key="fd_def_abbr").upper()[:4]
        c1, c2 = st.columns(2)
        wind_speed = c1.number_input("Wind (mph)", 0, 40, help="No modeled effect below 10 mph.",
                                     key="fd_wind_speed")
        wind_direction = c2.selectbox("Wind direction", ["Into", "With", "Crosswind"], key="fd_wind_dir")
        c1, c2 = st.columns(2)
        rain = c1.checkbox("Rain", key="fd_rain")
        snow = c2.checkbox("Snow", key="fd_snow")
        st.caption("Matchup: adjusts the conversion chance for these two units, from our film. "
                   "Teams with few snaps stay close to average.")
        c1, c2 = st.columns(2)
        off_team = c1.selectbox("Offense unit", TEAMS, key="fd_off_team")
        def_team = c2.selectbox("Defense unit", TEAMS, key="fd_def_team")
        c1, c2 = st.columns(2)
        pat_pct = c1.number_input("PAT make rate (%)", 50, 100, key="fd_pat",
                                  help="Used when a conversion is a touchdown. Our film: 89% overall, Dowling 92%.")
        fg_range = c2.number_input("Kicker's range (yds)", 20, 65, key="fd_fg_range",
                                   help="Longest field goal you'd attempt. Beyond it, the bot won't consider a kick.")

    with st.expander("Guardrails"):
        guard_on = st.checkbox("Own-end guardrails", key="fd_guard",
                               help="In our own territory, a go call becomes the kick if it's a toss-up, wins by "
                                    "less than the bar below, or flips when our estimates move 10 points. "
                                    "Guardrails are off when we trail in the last 5:00.")
        guard_edge = float(st.segmented_control("Go must win by (WP pts) in our own end", ["1", "1.5", "2", "3"],
                                                key="fd_guard_edge") or BOT_SETTINGS["own_min_edge"])
        nogo_on = st.checkbox("Staff no-go table", key="fd_nogo",
                              help="Never go for it on these distances or longer, whatever the math says.")
        c1, c2, c3 = st.columns(3)
        ng1 = c1.number_input("Own 1-20: 4th &", 1, 30, key="fd_nogo_1", disabled=not nogo_on)
        ng2 = c2.number_input("Own 21-40: 4th &", 1, 30, key="fd_nogo_2", disabled=not nogo_on)
        ng3 = c3.number_input("Own 41-50: 4th &", 1, 30, key="fd_nogo_3", disabled=not nogo_on)
        st.caption("A guardrail never hides the math: the page shows what it changed and what that costs. "
                   "Changes here are for this situation only; the season review uses the staff settings "
                   "in fourth_down_core.BOT_SETTINGS.")

# "home" / "away" / "neutral" from the offense's point of view. After a change
# of possession, evaluate_options() flips home/away for the other team.
SITE = {"Offense is home": "home", "Offense is away": "away", "Neutral": "neutral"}[site_label]

if minutes == 12:
    seconds = 0  # quarters are 12:00 max; 12:30 isn't a real clock time

# The WP model was trained on 15-minute college quarters, so the high school
# clock is scaled by 15/12 (see fourth_down_core.model_seconds_remaining).
seconds_remaining_in_game = model_seconds_remaining(quarter, minutes * 60 + seconds)
score_diff = off_score - def_score
conv_shift = CONV.team_shift(off_team, def_team)
MODEL_KW = dict(conv_logit_shift=conv_shift, p_xp=pat_pct / 100, max_fg_distance=fg_range)
NO_GO = ((80, 99, ng1), (60, 79, ng2), (50, 59, ng3)) if nogo_on else None
GUARD_KW = dict(own_end=guard_on, own_min_edge=guard_edge, no_go=NO_GO)

result = evaluate_site(SITE, yards_to_goal, distance, score_diff, seconds_remaining_in_game,
                       off_timeouts=off_timeouts, def_timeouts=def_timeouts,
                       wind_speed=wind_speed, wind_direction=wind_direction,
                       rain=rain, snow=snow, **MODEL_KW)
wp = result["wp"]
ranked = sorted(wp.items(), key=lambda kv: -kv[1])
best_option, best_wp = ranked[0]
# Sometimes going for it is the only option: no punts inside their 35, and a
# field goal past the kicker's range is off the table (e.g. 4th & 7 at their
# 34 = a 51-yard kick with the default 50-yard range).
only_option = len(ranked) == 1
second_option, second_wp = ranked[1] if not only_option else (None, best_wp)
margin_pts = (best_wp - second_wp) * 100
tier = "ONLY OPTION" if only_option else mu.strength_tier(margin_pts)

# Robustness: does the call hold if the conversion and FG estimates are each
# off by 10 points either way? If not, it's never labeled better than LEAN.
_res = evaluate_many(yards_to_goal, distance, score_diff, seconds_remaining_in_game,
                     off_timeouts, def_timeouts, site_flag(SITE), wind_speed, wind_direction, rain, snow,
                     **MODEL_KW)
_rob = robustness(_res)
is_robust = bool(_rob["robust"][0])
if not only_option and not is_robust and tier in ("VERY STRONG", "STRONG"):
    tier = "LEAN"

# Guardrails: may turn a go call into the kick. Keep the math's call for the note.
_g = guarded_calls(_res, yards_to_goal, distance, score_diff=score_diff,
                   seconds_remaining=seconds_remaining_in_game, **GUARD_KW)
guard_rule = str(_g["rule"][0])
math_option, math_margin = best_option, margin_pts
if guard_rule:
    best_option = str(_g["call"][0])
    best_wp = wp[best_option]
    tier = "STAFF RULE" if guard_rule.startswith("Staff") else "GUARDRAIL"

emoji = {"Go for it": "👉", "Field goal": "🦵", "Punt": "🏈"}[best_option]

# ---- Verdict (big and wrapping, so it reads on a phone) ----
spot = field_spot_label(yards_to_goal, off_abbr, def_abbr)
clock_str = f"{minutes}:{seconds:02d}"
tier_color = {"VERY STRONG": "#0F6E56", "STRONG": "#1D9E75", "LEAN": "#BA7517", "TOSS-UP": "#888780",
              "GUARDRAIL": "#534AB7", "STAFF RULE": "#534AB7", "ONLY OPTION": "#5F5E5A"}[tier]
verdict = "TOSS-UP" if tier == "TOSS-UP" else best_option.upper()
if guard_rule:
    sub = (f"{guard_rule}. The math alone says {math_option.lower()} by +{math_margin:.1f} WP pts, "
           f"so this costs {float(_g['cost'][0]):.1f} by the bot's numbers.")
elif only_option:
    _why = ["no punt inside their 35"] if yards_to_goal <= PUNT_MIN_YTG else []
    _why.append(f"a {yards_to_goal + 17}-yard field goal is past the kicker's range ({fg_range} yds)")
    sub = f"Only option: {' and '.join(_why)}."
elif tier == "TOSS-UP":
    sub = f"{best_option} by a hair over {second_option.lower()} (+{margin_pts:.1f} WP pts). Either call is fine."
else:
    sub = (f"{tier.title()} · +{margin_pts:.1f} win probability points over {second_option.lower()}"
           + ("" if is_robust else " · flips if our estimates are 10 pts off"))
st.html(
    f'<div style="text-align:center;padding:6px 0 2px">'
    f'<div style="font-size:14px;opacity:.7">4th &amp; {distance} {spot} · Q{quarter} {clock_str} · '
    f'{off_abbr} {off_score}, {def_abbr} {def_score}</div>'
    f'<div style="font-size:clamp(34px,9vw,52px);font-weight:700;line-height:1.15;margin:6px 0">{emoji} {verdict}</div>'
    f'<div style="display:inline-block;padding:3px 12px;border-radius:999px;background:{tier_color};color:#fff;'
    f'font-size:14px">{sub}</div></div>'
)
_link = qol.write_link_params(LINK_SPEC, {k: st.session_state.get(k) for _, k, *_ in LINK_SPEC})
qol.share_link_box(_link)
with st.expander("Copy as text"):
    st.code(f"---> {def_abbr} ({def_score}) @ {off_abbr} ({off_score}) <---\n"
            f"{off_abbr} has 4th & {distance} {spot}\nQ{quarter} {clock_str} remaining\n\n"
            f"Recommendation ({tier}): {emoji} {best_option}"
            + (f" ({guard_rule}; math says {math_option} +{math_margin:.1f} WP)" if guard_rule
               else "" if only_option else f" (+{margin_pts:.1f} WP)"),
            language=None)

# ---- gt-style results table (rbsdm/nfl4th style: success prob + WP on each branch) ----
st.write("#### Win probability by option")
options_detail = result["options"]
rows = []
for opt, w in wp.items():
    d = options_detail[opt]
    if d["success_prob"] is None:
        success_pct, wp_success, wp_fail = "—", "—", "—"
    else:
        success_pct = f"{d['success_prob']*100:.0f}%"
        wp_success = f"{d['wp_success']*100:.1f}%"
        wp_fail = f"{d['wp_fail']*100:.1f}%"
    rows.append({
        "Option": opt,
        "Win Probability": f"{w*100:.1f}%",
        "Success %": success_pct,
        "WP if Success": wp_success,
        "WP if Fail": wp_fail,
        "vs. best (pts)": f"{(w - best_wp)*100:+.1f}",
    })
table_df = pd.DataFrame(rows)
table_df.loc[table_df["Option"] == best_option, "Option"] = "👉 " + best_option
st.dataframe(table_df, hide_index=True, width="stretch")
st.caption(
    "\"Success %\" is the conversion/make probability driving that option; "
    "\"WP if Success\"/\"WP if Fail\" are the win probabilities on each branch "
    "(Punt has no success/fail split in this model)."
)

weather_pts = weather_adjustment(wind_speed, wind_direction, rain, snow) * 100
weather_note = f" (weather: {weather_pts:+.1f} pts)" if weather_pts != 0 else ""
st.caption(
    f"Estimated 4th & {distance} conversion rate: **{result['p_conv']*100:.0f}%** · "
    f"Estimated FG make rate from here: **{result['p_fg']*100:.0f}%**{weather_note}"
)

# Break-even: how often you'd need to convert for going to tie the best kick.
# This is the number to argue with: if you trust your 4th & short offense
# more (or less) than the generic conversion curve, compare against this.
_be = float(break_even_conversion(_res)[0])
if not np.isnan(_be):
    _kick = "field goal" if np.nan_to_num(_res["wp_fg"][0], nan=-1) >= np.nan_to_num(_res["wp_punt"][0], nan=-1) \
        else "punt"
    _p = result["p_conv"]
    _verdict = ("comfortably above it" if _p - _be >= 0.10 else "above it" if _p >= _be
                else "below it" if _be - _p < 0.10 else "well below it")
    st.info(
        f"**Break-even:** going for it beats the {_kick} if you convert at least **{_be:.0%}** of the time. "
        f"The model's estimate for 4th & {distance} is {_p:.0%}, {_verdict}. If you think your offense "
        f"converts this more or less often than that, compare your number to {_be:.0%}.",
        icon="⚖️",
    )

# ---- Decision chart: every field position x distance, one batched model call ----
st.divider()
st.write("#### Decision chart")
st.caption(f"The model's call for every spot on the field and every distance, at the current score, clock, "
           f"timeouts, site, and weather. The ringed square is the current situation: 4th & {distance} {spot}. "
           f"Hover any square for the numbers.")

CALL_COLORS = {"Go for it": "#55A868", "Field goal": "#4C72B0", "Punt": "#C44E52", "Toss-up": "#BDBDBD",
               "Field goal (guardrail)": "#A9BCDD", "Punt (guardrail)": "#E3A6A8"}


@st.cache_data(show_spinner=False)
def build_decision_chart(score_diff_, seconds_remaining_, off_timeouts_, def_timeouts_, site_,
                         wind_speed_, wind_direction_, rain_, snow_, conv_shift_=0.0, p_xp_=DEFAULT_P_XP,
                         fg_range_=MAX_FG_KICK_DISTANCE, guard_on_=True, guard_edge_=OWN_MIN_EDGE,
                         no_go_=None) -> pd.DataFrame:
    # Full 1-yard resolution: ~1,900 situations priced in a single model call.
    Y, D = np.meshgrid(np.arange(1, 100), np.arange(1, 21))
    Y, D = Y.ravel(), D.ravel()
    keep = D <= Y
    Y, D = Y[keep], D[keep]
    res = evaluate_many(Y, D, score_diff_, seconds_remaining_, off_timeouts_, def_timeouts_, site_flag(site_),
                        wind_speed_, wind_direction_, rain_, snow_, conv_logit_shift=conv_shift_, p_xp=p_xp_,
                        max_fg_distance=fg_range_)
    g = guarded_calls(res, Y, D, own_end=guard_on_, own_min_edge=guard_edge_, no_go=no_go_,
                      score_diff=score_diff_, seconds_remaining=seconds_remaining_)
    call = np.where(g["rule"] != "", np.char.add(g["call"], " (guardrail)"),
                    np.where(g["toss_up"], "Toss-up", g["call"]))
    return pd.DataFrame({
        "ytg": Y, "dist": D, "x": 100 - Y,
        "call": call, "rule": np.where(g["rule"] != "", g["rule"], "—"),
        "margin": g["margin"],
        "go": res["wp_go"], "fg": res["wp_fg"], "punt": res["wp_punt"],
        "p_conv": res["p_conv"], "break_even": break_even_conversion(res),
        "spot": [field_spot_label(int(y), "own", "opp").replace("at the ", "") for y in Y],
    })


grid = build_decision_chart(score_diff, seconds_remaining_in_game, off_timeouts, def_timeouts, SITE,
                            wind_speed, wind_direction, rain, snow, conv_shift, pat_pct / 100, fg_range,
                            guard_on, guard_edge, NO_GO)
grid = grid.assign(label=lambda g: "4th & " + g["dist"].astype(str) + " at " + g["spot"])

pct = lambda f, t: alt.Tooltip(f, format=".0%", title=t)
heat = alt.Chart(grid).mark_rect().encode(
    x=alt.X("x:O", title=f"Field position ({off_abbr} goal ← → {def_abbr} goal)",
            axis=alt.Axis(values=list(range(10, 100, 10)), labelExpr="datum.value == 50 ? '50' : datum.value < 50 "
                          "? 'Own ' + datum.value : 'Opp ' + (100 - datum.value)", labelAngle=0)),
    y=alt.Y("dist:O", title="Yards to go", sort="descending"),
    color=alt.Color("call:N", title=None, scale=alt.Scale(domain=list(CALL_COLORS), range=list(CALL_COLORS.values())),
                    legend=alt.Legend(orient="top")),
    tooltip=[alt.Tooltip("label:N", title="Situation"), alt.Tooltip("call:N", title="Call"),
             alt.Tooltip("rule:N", title="Guardrail"),
             alt.Tooltip("margin:Q", format=".1f", title="Margin (WP pts)"),
             pct("go:Q", "WP go"), pct("fg:Q", "WP field goal"), pct("punt:Q", "WP punt"),
             pct("p_conv:Q", "Conversion chance"), pct("break_even:Q", "Break-even conversion")],
)
here = alt.Chart(pd.DataFrame({"x": [100 - yards_to_goal], "dist": [min(distance, 20)]})).mark_rect(
    fill=None, stroke="black", strokeWidth=2.5).encode(x="x:O", y=alt.Y("dist:O", sort="descending"))
st.altair_chart((heat + here).properties(height=360), width="stretch")
st.caption("Gray = toss-up: the top two options are within 1 point of win probability. "
           "Pale red/blue = a go call a guardrail turned into a punt/field goal (hover for why).")

st.divider()
st.caption(
    "**Notes:** these are ESTIMATES; please use accordingly. On 4th & 1, the model "
    "can't know whether it's a long 1 or a short 1 — shorter distance favors going. "
    "Do not use this in overtime. Use with EXTREME CAUTION in the final minute of a "
    "half as clock-management dynamics aren't fully captured here."
)
with st.expander("Model notes & limitations"):
    st.markdown(f"""
- EP and WP come from your trained cfbfastR models when `cfb_ep_model.json` /
  `cfb_wp_model_truth.json` are present (see export_models_to_json.R);
  otherwise this falls back to heuristic curves. Currently:
  **{"real trained models" if mu.USING_REAL_MODELS else "heuristic fallback"}**.
- Conversion probability is fit to our own high school film
  (conversion_model.py: {CONV.n_plays} third- and fourth-down snaps), with an
  optional matchup adjustment for the two units on the field. FG probability
  and punt distance are still hand-calibrated. The break-even conversion rate
  tells you how much that matters: if your real conversion rate is on the
  same side of break-even, the call doesn't change.
- Field position is priced in points first (EP model), then converted to
  win probability at the 25 (fourth_down_core.wp_new_possession). The WP
  model on its own was nearly flat in field position, which made a turnover
  deep in our own end look cheap.
- A missed field goal is the other team's ball at their 20 (NFHS touchback),
  not at the spot of the kick.
- Labels: a call that flips when the conversion or FG chance moves 10
  points either way is never labeled better than LEAN.
- Guardrails (expander above): in our own territory a go call becomes the
  kick if it's a toss-up, wins by less than the bar, or flips when our
  estimates move 10 points. The staff no-go table, when on, overrides the
  math by distance. The verdict always says when a guardrail fired and what
  it costs by the bot's own numbers.
- After a change of possession (turnover on downs, punt, score) the timeouts
  swap sides, and a new first down inside the 10 is 1st & goal.
- FG probability weather adjustment (hand-calibrated, applied on top of the
  distance-based curve, then clipped to [0, 100]%): rain −5 pts, snow −10 pts.
  Wind has no effect below 10 mph; at/above that, it scales linearly with
  total mph at −5 pts/10 mph into the wind (so exactly −5 pts at 10 mph, −10
  pts at 20 mph, etc.), +3 pts/10 mph with the wind, and −8 pts/10 mph on a
  crosswind.
- "Go for it" success and "field goal make" outcomes are evaluated by
  chaining your real EP model's output into your real WP model, the same
  way cfbfastR's own pipeline does internally.
- Team-strength/talent gap is not modeled — a real deployment for college
  football should account for it given the wide talent variance in the sport.
""")
