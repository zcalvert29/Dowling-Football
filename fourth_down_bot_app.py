"""
4th Down Bot — Streamlit Prototype
====================================
Modeled on the NYT / Advanced Football Analytics "4th Down Bot" (Burke & Quealy,
2013): for each 4th down, compare Go For It / Field Goal / Punt using Expected
Points for most of the game, then switch to Win Probability in the last ~10
minutes, since EP-optimal and WP-optimal calls diverge late (e.g. a 2-point
deficit late favors a risky TD shot over a "safe" FG that EP alone prefers).

--------------------------------------------------------------------------
HOOKING UP YOUR REAL cfbfastR-TRAINED MODELS (recommended before real use)
--------------------------------------------------------------------------
This prototype uses hand-calibrated heuristic curves (see MODELS section
below) so it runs standalone with no dependencies beyond streamlit/numpy/
pandas. To use the actual models from cfb_ep_model.R / cfb_wp_model.R:

  1. In R, after training, save in the portable Booster format:
       xgb.save(ep_xgb, "cfb_ep_model.json")
       xgb.save(wp_xgb, "cfb_wp_model.json")
     (xgboost's model format is language-agnostic — a model trained in R's
     xgboost package loads directly into Python's xgboost package.)

  2. In this file, replace the `ep_from_features()` / `wp_from_features()`
     functions with something like:

       import xgboost as xgb
       _ep_booster = xgb.Booster()
       _ep_booster.load_model("cfb_ep_model.json")

       def ep_from_features(yards_to_goal, down, distance, ...):
           dmat = xgb.DMatrix(np.array([[...features in training order...]]))
           probs = _ep_booster.predict(dmat)  # shape (1, 7)
           return float(probs @ POINT_VALUES)  # POINT_VALUES vector matches
                                                # the class order used in R

  3. Feature order/names must exactly match what model.matrix() produced in
     R — mismatches will silently produce garbage predictions, not errors.
--------------------------------------------------------------------------
"""

import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="4th Down Bot", page_icon="🏈", layout="centered")

# ==============================================================================
# HEURISTIC SUB-MODELS (placeholders — see docstring above to swap in real ones)
# ==============================================================================

def conversion_prob(distance):
    """P(converting 4th down) as a function of yards to go.
    Logistic curve calibrated loosely to published 4th-down conversion
    research: ~74% at 4th & 1, ~50% around 4th & 5, tailing off by 4th & 10+.
    """
    distance = max(distance, 0.5)
    b, d0 = 0.262, 5.0
    return 1 / (1 + np.exp(b * (distance - d0)))


def fg_prob(yards_to_goal):
    """P(made field goal) as a function of yards to opponent's goal line.
    Kick distance = yards_to_goal + 17 (10 yd end zone + ~7 yd snap depth).
    Logistic calibrated to roughly: 99% at 20yd, 85% at 40yd, 50% at 50yd.
    Kicks beyond ~62 yards are effectively unattemptable in this model.
    """
    kick_distance = yards_to_goal + 17
    if kick_distance > 68:
        return 0.02
    c = 0.1736
    return 1 / (1 + np.exp(c * (kick_distance - 50)))


def ep_from_position(yards_to_goal):
    """Approximate Expected Points as a function of field position alone
    (yards_to_goal: 100 = own goal line, 0 = opponent's goal line).
    Piecewise-linear interpolation through published-style EP anchor points.
    This ignores down/distance — your real xgboost EP model captures that
    nuance; this is a field-position-only stand-in for the prototype.
    """
    anchor_ytg = [99, 90, 75, 60, 50, 40, 25, 10, 5, 1]
    anchor_ep  = [-1.6, -0.9, 0.4, 1.2, 2.0, 2.7, 3.7, 4.9, 5.5, 6.4]
    return float(np.interp(yards_to_goal, anchor_ytg[::-1], anchor_ep[::-1]))


def punt_net_yards(yards_to_goal):
    """Approximate net punt yardage, shrinking near the opponent's goal
    to avoid unrealistic touchbacks (can't net more than ~yards_to_goal-20
    without it going into the end zone).
    """
    if yards_to_goal > 60:
        return 40.0
    return float(np.clip(yards_to_goal - 20, 5, 40))


def wp_from_features(score_diff, seconds_remaining, ep_estimate):
    """Simplified in-game Win Probability heuristic: logistic function of
    time-adjusted score differential, nudged by field-position EP. This is
    a stand-in for your trained xgboost WP model — swap in real predictions
    before using this for anything beyond illustration.
    """
    minutes_left = max(seconds_remaining / 60, 0.1)
    time_adj_margin = score_diff / np.sqrt(minutes_left + 1)
    k = 0.42
    logit = k * time_adj_margin + 0.05 * ep_estimate
    return float(1 / (1 + np.exp(-logit)))


POINT_VALUES_NOTE = "Heuristic models above are illustrative — replace with trained cfbfastR EP/WP outputs for real decisions."


# ==============================================================================
# 4TH DOWN OPTION EVALUATION
# ==============================================================================

def evaluate_options(yards_to_goal, distance, score_diff, seconds_remaining):
    """Returns a dict with EP and WP for each of the 3 options, plus the
    breakeven conversion probability the coach needs for 'go for it' to
    beat the better of the other two options.
    """
    p_conv = conversion_prob(distance)

    # ---- GO FOR IT ----
    if distance >= yards_to_goal:
        ep_success = 7.0  # touchdown
    else:
        ep_success = ep_from_position(yards_to_goal - distance)
    opp_ytg_on_fail = 100 - yards_to_goal
    ep_fail = -ep_from_position(opp_ytg_on_fail)
    ep_go = p_conv * ep_success + (1 - p_conv) * ep_fail

    # ---- FIELD GOAL ----
    p_fg = fg_prob(yards_to_goal)
    ep_fg_make = 3.0
    opp_ytg_on_miss = 100 - yards_to_goal
    ep_fg_miss = -ep_from_position(opp_ytg_on_miss)
    ep_fg = p_fg * ep_fg_make + (1 - p_fg) * ep_fg_miss

    # ---- PUNT ----
    net = punt_net_yards(yards_to_goal)
    opp_ytg_after_punt = 100 - (yards_to_goal - net)
    opp_ytg_after_punt = float(np.clip(opp_ytg_after_punt, 1, 99))
    ep_punt = -ep_from_position(opp_ytg_after_punt)

    # ---- Win probabilities for each option (used in late-game mode) ----
    wp_go_success = wp_from_features(score_diff + (7 if distance >= yards_to_goal else 0),
                                      seconds_remaining, ep_success)
    wp_go_fail = wp_from_features(score_diff, seconds_remaining, ep_fail)
    wp_go = p_conv * wp_go_success + (1 - p_conv) * wp_go_fail

    wp_fg_make = wp_from_features(score_diff + 3, seconds_remaining, ep_fg_make)
    wp_fg_miss = wp_from_features(score_diff, seconds_remaining, ep_fg_miss)
    wp_fg = p_fg * wp_fg_make + (1 - p_fg) * wp_fg_miss

    wp_punt = wp_from_features(score_diff, seconds_remaining, ep_punt)

    # ---- Breakeven conversion probability (vs best of FG/punt, in EP terms) ----
    best_alt_ep = max(ep_fg, ep_punt)
    denom = (ep_success - ep_fail)
    breakeven = (best_alt_ep - ep_fail) / denom if denom > 0 else np.nan
    breakeven = float(np.clip(breakeven, 0, 1))

    return {
        "ep": {"Go for it": ep_go, "Field goal": ep_fg, "Punt": ep_punt},
        "wp": {"Go for it": wp_go, "Field goal": wp_fg, "Punt": wp_punt},
        "p_conv": p_conv,
        "p_fg": p_fg,
        "breakeven": breakeven,
    }


# ==============================================================================
# STREAMLIT UI
# ==============================================================================

st.title("🏈 4th Down Bot (Prototype)")
st.caption(
    "Modeled on the NYT/Advanced Football Analytics 4th Down Bot. "
    "Maximizes Expected Points for most of the game, switches to "
    "Win Probability in the last 10 minutes."
)

with st.sidebar:
    st.header("Game Situation")
    quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=2)
    minutes = st.number_input("Minutes remaining in quarter", 0, 15, 7)
    seconds = st.number_input("Seconds", 0, 59, 30)
    yards_to_goal = st.slider(
        "Yards to opponent's goal line", 1, 99, 45,
        help="1 = at the goal line, 99 = pinned at own 1"
    )
    distance = st.slider("Yards to go for 1st down", 1, 30, 4)
    score_diff = st.number_input(
        "Score differential (your team − opponent)", -50, 50, 0,
        help="Positive = you're ahead"
    )
    off_timeouts = st.selectbox("Your timeouts remaining", [0, 1, 2, 3], index=3)
    def_timeouts = st.selectbox("Opponent timeouts remaining", [0, 1, 2, 3], index=3)

# Convert to seconds remaining IN THE GAME (regulation), for late-game switch logic
quarters_left_after_this = 4 - quarter
seconds_remaining_in_game = quarters_left_after_this * 15 * 60 + minutes * 60 + seconds

result = evaluate_options(yards_to_goal, distance, score_diff, seconds_remaining_in_game)

LATE_GAME_THRESHOLD_SECONDS = 10 * 60
is_late_game = seconds_remaining_in_game <= LATE_GAME_THRESHOLD_SECONDS

decision_basis = "wp" if is_late_game else "ep"
values = result[decision_basis]
recommendation = max(values, key=values.get)

# ---- Headline recommendation ----
st.divider()
mode_label = "Win Probability (late-game mode)" if is_late_game else "Expected Points"
st.subheader(f"Recommendation: **{recommendation}**")
st.caption(f"Based on {mode_label} — {seconds_remaining_in_game // 60}:"
           f"{seconds_remaining_in_game % 60:02d} remaining in regulation")

col1, col2, col3 = st.columns(3)
for col, option in zip([col1, col2, col3], ["Go for it", "Field goal", "Punt"]):
    with col:
        is_best = option == recommendation
        st.metric(
            label=option + (" ⭐" if is_best else ""),
            value=f"{result['ep'][option]:+.2f} EP",
            delta=f"{result['wp'][option]*100:.1f}% WP" if is_late_game else None,
        )

# ---- Breakdown chart ----
st.divider()
st.write("#### Expected Points by option")
chart_df = pd.DataFrame({"EP": result["ep"]})
st.bar_chart(chart_df)

if is_late_game:
    st.write("#### Win Probability by option (late-game decision basis)")
    wp_df = pd.DataFrame({"WP": {k: v * 100 for k, v in result["wp"].items()}})
    st.bar_chart(wp_df)

# ---- Breakeven analysis ----
st.divider()
st.write("#### Breakeven analysis")
st.write(
    f"Estimated conversion probability at {distance} yards to go: "
    f"**{result['p_conv']*100:.0f}%**"
)
st.write(
    f"Breakeven conversion probability (rate at which 'go for it' ties the "
    f"better of field goal / punt in EP terms): **{result['breakeven']*100:.0f}%**"
)
if result['p_conv'] > result['breakeven']:
    st.success("Estimated conversion rate clears the breakeven — 'go for it' is favored on the numbers.")
else:
    st.info("Estimated conversion rate is below breakeven — the kicking/punting option is favored on the numbers.")

if yards_to_goal + 17 > 68:
    st.warning("Field goal attempt is beyond realistic range — treat the FG option as unattemptable here.")

st.divider()
with st.expander("Model notes & limitations"):
    st.markdown(f"""
- **Sub-models are heuristic placeholders**, not your trained cfbfastR models.
  {POINT_VALUES_NOTE}
- Field-position EP curve ignores down (only 4th-down situations are modeled
  here anyway, so this mostly matters for the "success" outcome after a
  conversion, which is treated as a fresh 1st-and-10).
- Win probability heuristic doesn't account for timeouts remaining, team
  strength, or conference/talent gap — all things your real WP model should
  include.
- The 10-minute EP→WP switch threshold mirrors the original NYT Bot's design
  but is a simplification; some later research (Yam & Lopez 2019) argues for
  a smoother blend rather than a hard cutoff.
- This tool ignores injury risk, weather, personnel matchups, and all the
  "coach's gut" factors that real decision-making legitimately weighs
  alongside the numbers.
""")
