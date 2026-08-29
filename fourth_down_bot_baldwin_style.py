"""
4th Down Bot — Ben Baldwin (nfl4th) Style Prototype
=====================================================
Presentation modeled on Ben Baldwin's nfl4th package / @ben_bot_baldwin
Twitter bot / rbsdm.com Shiny calculator, rather than the older NYT Bot:

  - Everything is reported as WIN PROBABILITY (no EP/WP mode-switching —
    Baldwin's model is WP-based throughout the game, which is the key
    methodological difference from the 2013-era NYT Bot).
  - Tweet-style recommendation line: "Recommendation (STRENGTH): Go for it
    (+X.X WP)", matching the actual @ben_bot_baldwin post format.
  - Strength tiers based on nfl4th's documented go_boost thresholds
    (see nfl4th's "4th down research" article): >=4 pts "Definitely go",
    1-4 "Probably go", -1 to 1 "Toss-up", -4 to -1 "Probably kick",
    <=-4 "Definitely kick". Extended here with a "Very strong" tier at
    >=10 pts to mirror the real bot's occasional "YOU BETTER DO THIS" posts.
  - A decision-boundary heatmap (distance x field position), styled after
    the chart on Baldwin's site, with the current situation plotted as a dot.
  - Blunt disclaimer footer, styled after the real rbsdm.com calculator's
    notes ("these are ESTIMATES", 4th-and-1 distance ambiguity caveat,
    don't use in OT, caution in the final minute).

--------------------------------------------------------------------------
HOOKING UP YOUR REAL cfbfastR-TRAINED MODELS (recommended before real use)
--------------------------------------------------------------------------
This prototype uses hand-calibrated heuristic curves (see MODELS section
below) so it runs standalone. To use your actual trained models:

  1. In R: xgb.save(wp_xgb, "cfb_wp_model.json")
     (xgboost's Booster format is language-agnostic — loads directly in
     Python's xgboost package, no retraining needed.)

  2. Replace `wp_from_features()` below with a real prediction call:
       import xgboost as xgb
       _wp_booster = xgb.Booster()
       _wp_booster.load_model("cfb_wp_model.json")

       def wp_from_features(yards_to_goal, down, distance, ...):
           dmat = xgb.DMatrix(np.array([[...features, matching training order...]]))
           return float(_wp_booster.predict(dmat)[0])

  3. Feature order/names must exactly match model.matrix() in R — silent
     garbage predictions on mismatch, not an error.
--------------------------------------------------------------------------
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="4th Down Bot", page_icon="🏈", layout="centered")

# ==============================================================================
# HEURISTIC SUB-MODELS (placeholders — see docstring above to swap in real ones)
# ==============================================================================

def conversion_prob(distance):
    """P(converting 4th down). Logistic curve loosely calibrated to
    published research: ~74% at 4th & 1, ~50% around 4th & 5."""
    distance = max(distance, 0.5)
    b, d0 = 0.262, 5.0
    return 1 / (1 + np.exp(b * (distance - d0)))


def fg_prob(yards_to_goal):
    """P(made FG). Kick distance = yards_to_goal + 17. Calibrated roughly
    to 99% at 20yd kick, 85% at 40yd, 50% at 50yd. >68yd treated as
    unattemptable, matching nfl4th's approach of not extrapolating wildly
    beyond the range of observed kicks."""
    kick_distance = yards_to_goal + 17
    if kick_distance > 68:
        return 0.02
    c = 0.1736
    return 1 / (1 + np.exp(c * (kick_distance - 50)))


def ep_from_position(yards_to_goal):
    """Field-position-only Expected Points, used internally as an input
    signal to the WP heuristic below (real nfl4th-style models feed EP
    into WP the same way)."""
    anchor_ytg = [99, 90, 75, 60, 50, 40, 25, 10, 5, 1]
    anchor_ep  = [-1.6, -0.9, 0.4, 1.2, 2.0, 2.7, 3.7, 4.9, 5.5, 6.4]
    return float(np.interp(yards_to_goal, anchor_ytg[::-1], anchor_ep[::-1]))


def punt_net_yards(yards_to_goal):
    """Approximate net punt yardage, shrinking near the opponent's goal
    to avoid unrealistic touchbacks."""
    if yards_to_goal > 60:
        return 40.0
    return float(np.clip(yards_to_goal - 20, 5, 40))


def wp_from_features(score_diff, seconds_remaining, ep_estimate):
    """Simplified WP heuristic used for ALL of the game (no EP/WP mode
    switch) — this is the key structural difference from the NYT-bot
    version of this app. Swap for your trained xgboost WP model."""
    minutes_left = max(seconds_remaining / 60, 0.1)
    time_adj_margin = score_diff / np.sqrt(minutes_left + 1)
    k = 0.42
    logit = k * time_adj_margin + 0.05 * ep_estimate
    return float(1 / (1 + np.exp(-logit)))


# ==============================================================================
# 4TH DOWN OPTION EVALUATION (WP for all three options, always)
# ==============================================================================

def evaluate_options(yards_to_goal, distance, score_diff, seconds_remaining):
    p_conv = conversion_prob(distance)

    # ---- GO FOR IT ----
    if distance >= yards_to_goal:
        ep_success = 7.0
    else:
        ep_success = ep_from_position(yards_to_goal - distance)
    opp_ytg_on_fail = 100 - yards_to_goal
    ep_fail = -ep_from_position(opp_ytg_on_fail)

    wp_go_success = wp_from_features(score_diff + (7 if distance >= yards_to_goal else 0),
                                      seconds_remaining, ep_success)
    wp_go_fail = wp_from_features(score_diff, seconds_remaining, ep_fail)
    wp_go = p_conv * wp_go_success + (1 - p_conv) * wp_go_fail

    # ---- FIELD GOAL ----
    p_fg = fg_prob(yards_to_goal)
    opp_ytg_on_miss = 100 - yards_to_goal
    ep_fg_miss = -ep_from_position(opp_ytg_on_miss)
    wp_fg_make = wp_from_features(score_diff + 3, seconds_remaining, 3.0)
    wp_fg_miss = wp_from_features(score_diff, seconds_remaining, ep_fg_miss)
    wp_fg = p_fg * wp_fg_make + (1 - p_fg) * wp_fg_miss

    # ---- PUNT ----
    net = punt_net_yards(yards_to_goal)
    opp_ytg_after_punt = float(np.clip(100 - (yards_to_goal - net), 1, 99))
    ep_punt = -ep_from_position(opp_ytg_after_punt)
    wp_punt = wp_from_features(score_diff, seconds_remaining, ep_punt)

    wp = {"Go for it": wp_go, "Field goal": wp_fg, "Punt": wp_punt}
    return {"wp": wp, "p_conv": p_conv, "p_fg": p_fg}


def strength_tier(margin_pts):
    """margin_pts = WP gain (percentage points) of best option over
    second-best. Tiers below follow nfl4th's documented go_boost cutoffs
    (>=4 / 1-4 / -1 to 1 / -4 to -1 / <=-4), extended with a 'very strong'
    band at 10+ to mirror the real bot's rare emphatic posts."""
    if margin_pts >= 10:
        return "VERY STRONG"
    elif margin_pts >= 4:
        return "STRONG"
    elif margin_pts >= 1:
        return "LEAN"
    else:
        return "TOSS-UP"


def field_spot_label(yards_to_goal, off_abbr, def_abbr):
    """Mimics Baldwin's 'at the BAL 40' style field position label."""
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
st.caption("Presentation modeled on Ben Baldwin's nfl4th / @ben_bot_baldwin — win-probability-based, throughout the game.")

with st.sidebar:
    st.header("Game Situation")
    off_abbr = st.text_input("Offense (team with the ball)", "BAL").upper()[:4]
    def_abbr = st.text_input("Defense", "TEN").upper()[:4]
    off_score = st.number_input(f"{off_abbr} score", 0, 99, 17)
    def_score = st.number_input(f"{def_abbr} score", 0, 99, 13)
    quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=2)
    minutes = st.number_input("Minutes remaining in quarter", 0, 15, 7)
    seconds = st.number_input("Seconds", 0, 59, 30)
    yards_to_goal = st.slider("Yards to opponent's goal line", 1, 99, 40,
                               help="1 = at the goal line, 99 = pinned at own 1")
    distance = st.slider("Yards to go for 1st down", 1, 25, 2)

quarters_left_after_this = 4 - quarter
seconds_remaining_in_game = quarters_left_after_this * 15 * 60 + minutes * 60 + seconds
score_diff = off_score - def_score

result = evaluate_options(yards_to_goal, distance, score_diff, seconds_remaining_in_game)
wp = result["wp"]
ranked = sorted(wp.items(), key=lambda kv: -kv[1])
best_option, best_wp = ranked[0]
second_option, second_wp = ranked[1]
margin_pts = (best_wp - second_wp) * 100
tier = strength_tier(margin_pts)

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

# ---- gt-style results table ----
st.write("#### Win probability by option")
table_df = pd.DataFrame({
    "Option": list(wp.keys()),
    "Win Probability": [f"{v*100:.1f}%" for v in wp.values()],
    "vs. best (pts)": [f"{(v - best_wp)*100:+.1f}" for v in wp.values()],
})
table_df.loc[table_df["Option"] == best_option, "Option"] = "👉 " + best_option
st.dataframe(table_df, hide_index=True, use_container_width=True)

st.caption(
    f"Estimated 4th & {distance} conversion rate: **{result['p_conv']*100:.0f}%** · "
    f"Estimated FG make rate from here: **{result['p_fg']*100:.0f}%**"
)

# ---- Decision chart (heatmap across distance x field position) ----
st.divider()
st.write("#### Decision chart")
st.caption(f"Go-for-it recommendation across field position and distance, at the current score/time. "
           f"The dot marks the current situation: 4th & {distance} {spot}.")

ytg_grid = np.arange(1, 100, 2)
dist_grid = np.arange(1, 21, 1)
Z = np.zeros((len(dist_grid), len(ytg_grid)))

for i, d in enumerate(dist_grid):
    for j, y in enumerate(ytg_grid):
        r = evaluate_options(y, min(d, y), score_diff, seconds_remaining_in_game)
        best = max(r["wp"], key=r["wp"].get)
        Z[i, j] = {"Go for it": 1, "Field goal": 0, "Punt": -1}[best]

fig, ax = plt.subplots(figsize=(7, 4))
cmap = plt.matplotlib.colors.ListedColormap(["#4C72B0", "#DD8452", "#C44E52"])
ax.pcolormesh(ytg_grid, dist_grid, Z, cmap=cmap, vmin=-1, vmax=1, shading="nearest")
ax.plot(yards_to_goal, distance, "o", color="white", markeredgecolor="black", markersize=10)
ax.invert_xaxis()
ax.set_xlabel("Yards to opponent's goal (own goal ← → opp goal)")
ax.set_ylabel("Yards to go")
ax.set_title("Blue = Punt · Orange = Field goal · Red = Go for it", fontsize=10)
st.pyplot(fig)

# ---- Disclaimer footer, styled after rbsdm.com's calculator notes ----
st.divider()
st.caption(
    "**Notes:** these are ESTIMATES; please use accordingly. On 4th & 1, the model "
    "can't know whether it's a long 1 or a short 1 — shorter distance favors going. "
    "Do not use this in overtime. Use with EXTREME CAUTION in the final minute of a "
    "half as clock-management dynamics aren't fully captured here."
)
with st.expander("Model notes & limitations"):
    st.markdown("""
- **Presentation** is modeled on Ben Baldwin's nfl4th / @ben_bot_baldwin, including
  the WP-throughout-the-game approach (no EP/WP mode switch), the strength-tier
  language, and the decision-chart heatmap.
- **Sub-models are heuristic placeholders**, not trained cfbfastR models — see the
  docstring at the top of the source file for how to swap in real xgboost models.
- The real nfl4th also models: possibility of a first down via defensive penalty
  on 4th-down attempts, blocked/returned punts, and roof type for FG probability.
  None of that is included here.
- Team-strength/talent gap is not modeled — a real deployment for college football
  should account for it given the wide talent variance across the sport.
""")
