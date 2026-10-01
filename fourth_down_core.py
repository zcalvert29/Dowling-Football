"""
fourth_down_core.py — the 4th Down Bot's decision math, with no Streamlit
code, so other pages (e.g. the 4th-down decision review) can reuse it.
Moved here unchanged from Fourth_Downs.py, which now imports it.
"""
import numpy as np

import model_utils as mu


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
