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

st.set_page_config(page_title="4th Down Bot", page_icon="🏈", layout="centered")

# ==============================================================================
# HEURISTIC SUB-MODELS (no trained model exists for these — conversion rate,
# FG accuracy, and punt distance are hand-calibrated curves either way)
# ==============================================================================

def conversion_prob(distance):
    distance = max(distance, 0.5)
    b, d0 = 0.262, 5.0
    return 1 / (1 + np.exp(b * (distance - d0)))


WIND_THRESHOLD_MPH = 10  # wind has no modeled effect below this speed

# Rate of FG probability change per 10 mph of wind, by direction — applied to
# TOTAL wind speed (not excess above the threshold) once at/above the
# threshold, so the full rate is already in effect exactly at 10 mph (e.g.
# Into hits -5% right at 10 mph, then another -5% by 20 mph, etc.), and scales
# linearly and continuously above that (so 10 mph and 12 mph differ). Into
# and crosswind hurt, tailwind helps — crosswind hurts more than a pure
# headwind since it's a lateral-miss problem kickers have less way to correct
# for than a straight distance loss.
WIND_RATE_PER_10MPH = {
    "Into": -0.05,
    "With": 0.03,
    "Crosswind": -0.08,
}


def wind_adjustment(wind_speed, wind_direction):
    if wind_speed < WIND_THRESHOLD_MPH:
        return 0.0
    rate = WIND_RATE_PER_10MPH.get(wind_direction, 0.0)
    return rate * (wind_speed / 10)


def weather_adjustment(wind_speed, wind_direction, rain, snow):
    # Total additive adjustment (probability units, e.g. -0.05 = -5 points)
    # to apply on top of a clean/no-weather FG probability estimate.
    adj = wind_adjustment(wind_speed, wind_direction)
    if rain:
        adj -= 0.05
    if snow:
        adj -= 0.10
    return adj


def fg_prob(yards_to_goal, wind_speed=0, wind_direction="Into", rain=False, snow=False):
    # Gives 85% chance at 30 yards, 65% chance at 40 yards, and 35% chance at 50 yards
    # (before any weather adjustment).
    # No hard floor here anymore — let the logistic curve keep decaying for very
    # long attempts instead of pinning everything past 55 yards to a flat 2%
    # (that flat floor was actually propping up the make-probability for kicks
    # that should be essentially impossible, e.g. from your own 10-yard line).
    kick_distance = yards_to_goal + 17
    c = 0.1156
    base = 1 / (1 + np.exp(c * (kick_distance - 45)))
    adjusted = base + weather_adjustment(wind_speed, wind_direction, rain, snow)
    return float(np.clip(adjusted, 0.0, 1.0))


def punt_net_yards(yards_to_goal):
    if yards_to_goal > 60:
        return 35.0 # 35 net yards on punt
    return float(np.clip(yards_to_goal - 20, 5, 40))


# ==============================================================================
# 4TH DOWN OPTION EVALUATION — now calling the real (or fallback) WP model
# ==============================================================================

# Longest field goal ever converted at any level is in the high-60s of yards;
# beyond that it's not a real coaching option. Excluding it here (same pattern
# as the Punt cutoff below) is what actually fixes the "FG recommended from
# your own 10" bug — before, a kick this long still got priced with a
# probability instead of being taken off the table entirely.
MAX_FG_KICK_DISTANCE = 62  # yards_to_goal + 17

def evaluate_options(yards_to_goal, distance, score_diff, seconds_remaining,
                      off_timeouts=3, def_timeouts=3, is_home_pos=1,
                      wind_speed=0, wind_direction="Into", rain=False, snow=False):
    half_seconds = min(seconds_remaining, 1800)
    p_conv = conversion_prob(distance)

    def wp_of(yards_to_goal_, down_, distance_, score_diff_, is_home_pos_):
        wp, _ = mu.predict_wp_chained(
            score_diff=score_diff_, yards_to_goal=yards_to_goal_, down=down_,
            distance=distance_, half_seconds=half_seconds,
            game_seconds_remaining=seconds_remaining,
            off_timeouts=off_timeouts, def_timeouts=def_timeouts,
            is_home_pos=is_home_pos_
        )
        return wp

    # ---- GO FOR IT ----
    if distance >= yards_to_goal:
        # Touchdown -> ensuing kickoff, OPPONENT takes over at their own 25.
        # Must flip both the score-diff sign and the perspective (1 - is_home_pos),
        # then take (1 - their WP) to get our WP back — same pattern as the
        # fail/miss/punt branches below, not the "we keep the ball" pattern.
        wp_go_success = 1 - wp_of(75, 1, 10, -(score_diff + 7), 1 - is_home_pos)
    else:
        wp_go_success = wp_of(yards_to_goal - distance, 1, 10, score_diff, is_home_pos)
    opp_ytg_on_fail = 100 - yards_to_goal
    wp_go_fail = 1 - wp_of(opp_ytg_on_fail, 1, 10, -score_diff, 1 - is_home_pos)
    wp_go = p_conv * wp_go_success + (1 - p_conv) * wp_go_fail

    # ---- FIELD GOAL ----
    kick_distance = yards_to_goal + 17
    p_fg = fg_prob(yards_to_goal, wind_speed, wind_direction, rain, snow)
    # Made FG -> ensuing kickoff, OPPONENT takes over at their own 25. Same
    # perspective-flip as above — this was a real bug in an earlier draft
    # (it kept the scoring team's own perspective instead of flipping to the
    # opponent who actually gets the ball next).
    wp_fg_make = 1 - wp_of(75, 1, 10, -(score_diff + 3), 1 - is_home_pos)
    opp_ytg_on_miss = 100 - yards_to_goal
    wp_fg_miss = 1 - wp_of(opp_ytg_on_miss, 1, 10, -score_diff, 1 - is_home_pos)
    wp_fg = p_fg * wp_fg_make + (1 - p_fg) * wp_fg_miss

    # ---- PUNT ----
    net = punt_net_yards(yards_to_goal)
    opp_ytg_after_punt = float(np.clip(100 - (yards_to_goal - net), 1, 99))
    wp_punt = 1 - wp_of(opp_ytg_after_punt, 1, 10, -score_diff, 1 - is_home_pos)

    options = {
        "Go for it": {"wp": wp_go, "success_prob": p_conv,
                       "wp_success": wp_go_success, "wp_fail": wp_go_fail},
    }
    if kick_distance <= MAX_FG_KICK_DISTANCE:
        options["Field goal"] = {"wp": wp_fg, "success_prob": p_fg,
                                  "wp_success": wp_fg_make, "wp_fail": wp_fg_miss}
    if yards_to_goal > 35:
        # Punt has no modeled success/fail split — it's a single outcome here.
        options["Punt"] = {"wp": wp_punt, "success_prob": None,
                            "wp_success": None, "wp_fail": None}

    wp = {k: v["wp"] for k, v in options.items()}
    return {"wp": wp, "options": options, "p_conv": p_conv, "p_fg": p_fg}


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
    off_abbr = st.text_input("Offense (team with the ball)", "DCHS").upper()[:4]
    def_abbr = st.text_input("Defense", "VHS").upper()[:4]
    off_score = st.number_input(f"{off_abbr} score", 0, 99, 0)
    def_score = st.number_input(f"{def_abbr} score", 0, 99, 0)
    quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=0)
    minutes = st.number_input("Minutes remaining in quarter", 0, 15, 11)
    seconds = st.number_input("Seconds", 0, 59, 30)
    yards_to_goal = st.number_input("Yards to opponent's goal line", 1, 99, 65,
                                     help="1 = at the goal line, 99 = pinned at own 1")
    distance = st.number_input("Yards to go for 1st down", 1, 30, 2)
    off_timeouts = st.selectbox("Offense timeouts remaining", [0, 1, 2, 3], index=3)
    def_timeouts = st.selectbox("Defense timeouts remaining", [0, 1, 2, 3], index=3)
    is_home = st.checkbox("Offense is the home team", value=True)

    st.header("Weather")
    wind_speed = st.number_input("Wind speed (mph)", 0, 40, 0,
                                  help="No modeled effect at or below 10 mph.")
    wind_direction = st.selectbox("Wind direction", ["Into", "With", "Crosswind"], index=0)
    rain = st.checkbox("Rain")
    snow = st.checkbox("Snow")

quarters_left_after_this = 4 - quarter
seconds_remaining_in_game = quarters_left_after_this * 15 * 60 + minutes * 60 + seconds
score_diff = off_score - def_score

result = evaluate_options(yards_to_goal, distance, score_diff, seconds_remaining_in_game,
                           off_timeouts, def_timeouts, is_home_pos=int(is_home),
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
st.dataframe(table_df, hide_index=True, use_container_width=True)
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
def build_decision_chart(score_diff_, seconds_remaining_, off_timeouts_, def_timeouts_, is_home_pos_,
                          wind_speed_, wind_direction_, rain_, snow_):
    # Coarser grid than a naive 1-yard sweep — cuts model calls from 250+ down to
    # ~90 while still giving a clear picture of the go/kick/punt boundaries.
    ytg_grid_ = np.arange(1, 100, 8)
    dist_grid_ = np.arange(1, 21, 3)
    Z_ = np.zeros((len(dist_grid_), len(ytg_grid_)))
    for i, d in enumerate(dist_grid_):
        for j, y in enumerate(ytg_grid_):
            r = evaluate_options(y, min(d, y), score_diff_, seconds_remaining_,
                                  off_timeouts_, def_timeouts_, is_home_pos_,
                                  wind_speed=wind_speed_, wind_direction=wind_direction_,
                                  rain=rain_, snow=snow_)
            best = max(r["wp"], key=r["wp"].get)
            Z_[i, j] = {"Go for it": 1, "Field goal": 0, "Punt": -1}[best]
    return ytg_grid_, dist_grid_, Z_


# Cached on the situation variables that actually change the grid — so dragging
# the current down/distance, editing team names, or tweaking the score/clock
# elsewhere on a rerun that doesn't touch these values won't recompute it.
ytg_grid, dist_grid, Z = build_decision_chart(
    score_diff, seconds_remaining_in_game, off_timeouts, def_timeouts, int(is_home),
    wind_speed, wind_direction, rain, snow
)

fig, ax = plt.subplots(figsize=(7, 4))
cmap = plt.matplotlib.colors.ListedColormap(["#C44E52", "#4C72B0", "#55A868"])
ax.pcolormesh(ytg_grid, dist_grid, Z, cmap=cmap, vmin=-1, vmax=1, shading="nearest")
ax.plot(yards_to_goal, distance, "o", color="white", markeredgecolor="black", markersize=10)
ax.invert_xaxis()
ax.set_xlabel("Yards to opponent's goal (own goal ← → opp goal)")
ax.set_ylabel("Yards to go")
ax.set_title("Green = Go for it · Blue = Field goal · Red = Punt", fontsize=10)
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
