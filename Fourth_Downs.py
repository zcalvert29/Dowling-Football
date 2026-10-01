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

with st.sidebar:
    st.header("Game Situation")
    off_abbr = st.text_input("Offense (team with the ball)", "DCHS", key="fd_off_abbr").upper()[:4]
    def_abbr = st.text_input("Defense", "VHS", key="fd_def_abbr").upper()[:4]
    off_score = st.number_input(f"{off_abbr} score", 0, 99, 0, key="fd_off_score")
    def_score = st.number_input(f"{def_abbr} score", 0, 99, 0, key="fd_def_score")
    quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=0, key="fd_quarter")
    minutes = st.number_input("Minutes remaining in quarter", 0, 12, 11, key="fd_minutes")
    seconds = st.number_input("Seconds", 0, 59, 30, key="fd_seconds")
    yards_to_goal = st.number_input("Yards to opponent's goal line", 1, 99, 65,
                                     help="1 = at the goal line, 99 = pinned at own 1", key="fd_ytg")
    distance = st.number_input("Yards to go for 1st down", 1, 30, 2, key="fd_distance")
    off_timeouts = st.selectbox("Offense timeouts remaining", [0, 1, 2, 3], index=3, key="fd_off_to")
    def_timeouts = st.selectbox("Defense timeouts remaining", [0, 1, 2, 3], index=3, key="fd_def_to")
    site_label = st.selectbox("Site", ["Offense is home", "Offense is away", "Neutral"], index=0, key="fd_site",
                              help="The win probability model includes home-field advantage.")

    st.header("Weather")
    wind_speed = st.number_input("Wind speed (mph)", 0, 40, 0,
                                  help="No modeled effect at or below 10 mph.", key="fd_wind_speed")
    wind_direction = st.selectbox("Wind direction", ["Into", "With", "Crosswind"], index=0, key="fd_wind_dir")
    rain = st.checkbox("Rain", key="fd_rain")
    snow = st.checkbox("Snow", key="fd_snow")

# "home" / "away" / "neutral" from the offense's point of view. After a change
# of possession, evaluate_options() flips home/away for the other team.
SITE = {"Offense is home": "home", "Offense is away": "away", "Neutral": "neutral"}[site_label]

if minutes == 12:
    seconds = 0  # quarters are 12:00 max; 12:30 isn't a real clock time

# The WP model was trained on 15-minute college quarters, so the high school
# clock is scaled by 15/12 (see fourth_down_core.model_seconds_remaining).
seconds_remaining_in_game = model_seconds_remaining(quarter, minutes * 60 + seconds)
score_diff = off_score - def_score

result = evaluate_site(SITE, yards_to_goal, distance, score_diff, seconds_remaining_in_game,
                       off_timeouts=off_timeouts, def_timeouts=def_timeouts,
                       wind_speed=wind_speed, wind_direction=wind_direction,
                       rain=rain, snow=snow)
wp = result["wp"]
ranked = sorted(wp.items(), key=lambda kv: -kv[1])
best_option, best_wp = ranked[0]
second_option, second_wp = ranked[1]
margin_pts = (best_wp - second_wp) * 100
tier = mu.strength_tier(margin_pts)

emoji = {"Go for it": "👉", "Field goal": "🦵", "Punt": "🏈"}[best_option]

# ---- "Tweet" style card ----
st.divider()
spot = field_spot_label(yards_to_goal, off_abbr, def_abbr)
clock_str = f"{minutes}:{seconds:02d}"
tweet_lines = f"""
```
---> {def_abbr} ({def_score}) @ {off_abbr} ({off_score}) <---
{off_abbr} has 4th & {distance} {spot}
Q{quarter} {clock_str} remaining

Recommendation ({tier}): {emoji} {best_option} (+{margin_pts:.1f} WP)
```
"""
st.markdown(tweet_lines)

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
_res = evaluate_many(yards_to_goal, distance, score_diff, seconds_remaining_in_game,
                     off_timeouts, def_timeouts, site_flag(SITE), wind_speed, wind_direction, rain, snow)
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

CALL_COLORS = {"Go for it": "#55A868", "Field goal": "#4C72B0", "Punt": "#C44E52", "Toss-up": "#BDBDBD"}


@st.cache_data(show_spinner=False)
def build_decision_chart(score_diff_, seconds_remaining_, off_timeouts_, def_timeouts_, site_,
                         wind_speed_, wind_direction_, rain_, snow_) -> pd.DataFrame:
    # Full 1-yard resolution: ~1,900 situations priced in a single model call.
    Y, D = np.meshgrid(np.arange(1, 100), np.arange(1, 21))
    Y, D = Y.ravel(), D.ravel()
    keep = D <= Y
    Y, D = Y[keep], D[keep]
    res = evaluate_many(Y, D, score_diff_, seconds_remaining_, off_timeouts_, def_timeouts_, site_flag(site_),
                        wind_speed_, wind_direction_, rain_, snow_)
    calls = best_calls(res)
    return pd.DataFrame({
        "ytg": Y, "dist": D, "x": 100 - Y,
        "call": np.where(calls["toss_up"], "Toss-up", calls["best"]),
        "margin": calls["margin"],
        "go": res["wp_go"], "fg": res["wp_fg"], "punt": res["wp_punt"],
        "p_conv": res["p_conv"], "break_even": break_even_conversion(res),
        "spot": [field_spot_label(int(y), "own", "opp").replace("at the ", "") for y in Y],
    })


grid = build_decision_chart(score_diff, seconds_remaining_in_game, off_timeouts, def_timeouts, SITE,
                            wind_speed, wind_direction, rain, snow)
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
             alt.Tooltip("margin:Q", format=".1f", title="Margin (WP pts)"),
             pct("go:Q", "WP go"), pct("fg:Q", "WP field goal"), pct("punt:Q", "WP punt"),
             pct("p_conv:Q", "Conversion chance"), pct("break_even:Q", "Break-even conversion")],
)
here = alt.Chart(pd.DataFrame({"x": [100 - yards_to_goal], "dist": [min(distance, 20)]})).mark_rect(
    fill=None, stroke="black", strokeWidth=2.5).encode(x="x:O", y=alt.Y("dist:O", sort="descending"))
st.altair_chart((heat + here).properties(height=360), width="stretch")
st.caption("Gray = toss-up: the top two options are within 1 point of win probability.")

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
- Conversion probability, FG probability, and punt distance are still
  hand-calibrated heuristics — no trained model exists for those yet. The
  break-even conversion rate tells you how much that matters: if your real
  conversion rate is on the same side of break-even, the call doesn't change.
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
