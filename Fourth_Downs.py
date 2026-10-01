"""
4th Down Bot — wired to real trained EP/WP models (model_utils.py)
=====================================================================
Presentation modeled on Ben Baldwin's nfl4th / @ben_bot_baldwin. See
model_utils.py for model loading, feature engineering, and the EP->WP
chaining logic. If the real model JSON files aren't found (see
export_models_to_json.R), this runs on heuristic fallbacks instead — the
sidebar shows which mode is active.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# ---- Decision chart (heatmap across distance x field position) ----
st.divider()
st.write("#### Decision chart")
st.caption(f"Go-for-it recommendation across field position and distance, at the current score/time. "
           f"The dot marks the current situation: 4th & {distance} {spot}.")

@st.cache_data(show_spinner="Building decision chart...")
def build_decision_chart(score_diff_, seconds_remaining_, off_timeouts_, def_timeouts_, site_,
                          wind_speed_, wind_direction_, rain_, snow_):
    # Coarser grid than a naive 1-yard sweep — cuts model calls from 250+ down to
    # ~90 while still giving a clear picture of the go/kick/punt boundaries.
    ytg_grid_ = np.arange(1, 100, 8)
    dist_grid_ = np.arange(1, 21, 3)
    Z_ = np.zeros((len(dist_grid_), len(ytg_grid_)))
    for i, d in enumerate(dist_grid_):
        for j, y in enumerate(ytg_grid_):
            r = evaluate_site(site_, y, min(d, y), score_diff_, seconds_remaining_,
                              off_timeouts=off_timeouts_, def_timeouts=def_timeouts_,
                              wind_speed=wind_speed_, wind_direction=wind_direction_,
                              rain=rain_, snow=snow_)
            ranked_ = sorted(r["wp"].values(), reverse=True)
            best = max(r["wp"], key=r["wp"].get)
            toss_up = len(ranked_) > 1 and (ranked_[0] - ranked_[1]) * 100 < 1
            Z_[i, j] = 2 if toss_up else {"Go for it": 1, "Field goal": 0, "Punt": -1}[best]
    return ytg_grid_, dist_grid_, Z_


# Cached on the situation variables that actually change the grid — so dragging
# the current down/distance, editing team names, or tweaking the score/clock
# elsewhere on a rerun that doesn't touch these values won't recompute it.
ytg_grid, dist_grid, Z = build_decision_chart(
    score_diff, seconds_remaining_in_game, off_timeouts, def_timeouts, SITE,
    wind_speed, wind_direction, rain, snow
)

fig, ax = plt.subplots(figsize=(7, 4))
cmap = plt.matplotlib.colors.ListedColormap(["#C44E52", "#4C72B0", "#55A868", "#BDBDBD"])
ax.pcolormesh(ytg_grid, dist_grid, Z, cmap=cmap, vmin=-1.5, vmax=2.5, shading="nearest")
ax.plot(yards_to_goal, distance, "o", color="white", markeredgecolor="black", markersize=10)
ax.invert_xaxis()
ax.set_xlabel("Yards to opponent's goal (own goal ← → opp goal)")
ax.set_ylabel("Yards to go")
ax.set_title("Green = Go for it · Blue = Field goal · Red = Punt · Gray = toss-up (under 1 WP pt)", fontsize=10)
st.pyplot(fig)

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
  hand-calibrated heuristics — no trained model exists for those yet.
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
