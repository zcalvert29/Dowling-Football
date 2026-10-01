"""
fourth_down_core.py — the 4th Down Bot's decision math, with no Streamlit
code, so other pages (e.g. the 4th-down decision review) can reuse it.
Fourth_Downs.py imports it; the 4th-down review on the Win Probability page
uses the same functions.
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


# Net punt from deep in your own territory. Measured from the curated data:
# through week 5, Dowling netted 31.5 per punt and opponents 30.9, so the old
# college-style 35 overstated what a high school punt is worth.
DEEP_PUNT_NET = 32.0


def punt_net_yards(yards_to_goal):
    if yards_to_goal > 60:
        return DEEP_PUNT_NET
    return float(np.clip(yards_to_goal - 20, 5, 40))


# ==============================================================================
# CLOCK: high school quarters are 12 minutes; the WP model was trained on
# 15-minute college quarters. Scale the whole high school clock by 15/12 so the
# model sees the same share of the game remaining (e.g. 6:00 left in a high
# school 4th quarter = 7:30 left in a college 4th quarter).
# ==============================================================================
HS_QUARTER_SECONDS = 12 * 60
MODEL_QUARTER_SECONDS = 15 * 60
CLOCK_SCALE = MODEL_QUARTER_SECONDS / HS_QUARTER_SECONDS


def model_seconds_remaining(quarter, hs_seconds_left_in_quarter):
    """High school clock (quarter 1-4, seconds left in it) -> model game seconds remaining."""
    q = min(max(int(quarter), 1), 4)
    return (4 - q) * MODEL_QUARTER_SECONDS + float(hs_seconds_left_in_quarter) * CLOCK_SCALE


def half_seconds_from_game(seconds_remaining):
    """Seconds left in the current half, from game seconds remaining (model clock)."""
    half = MODEL_QUARTER_SECONDS * 2
    return seconds_remaining - half if seconds_remaining > half else seconds_remaining


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
    # Seconds left in the HALF. (This used to be min(game seconds, 1800), which
    # told the model a full half remained for every 2nd-quarter snap.)
    half_seconds = half_seconds_from_game(seconds_remaining)
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


# ==============================================================================
# Neutral site: the WP model includes home-field advantage. When home/away
# isn't known (the 4th-down review) average the two so neither team gets it.
# ==============================================================================
def evaluate_neutral(yards_to_goal, distance, score_diff, seconds_remaining, **kwargs):
    kwargs.pop("is_home_pos", None)
    home = evaluate_options(yards_to_goal, distance, score_diff, seconds_remaining, is_home_pos=1, **kwargs)
    away = evaluate_options(yards_to_goal, distance, score_diff, seconds_remaining, is_home_pos=0, **kwargs)
    options = {}
    for k, h in home["options"].items():
        a = away["options"][k]
        avg = lambda x, y: None if x is None else (x + y) / 2
        options[k] = {f: avg(h[f], a[f]) for f in h}
    return {"wp": {k: v["wp"] for k, v in options.items()}, "options": options,
            "p_conv": home["p_conv"], "p_fg": home["p_fg"]}


def evaluate_site(site, yards_to_goal, distance, score_diff, seconds_remaining, **kwargs):
    """site: "home", "away", or "neutral" (from the offense's point of view)."""
    if site == "neutral":
        return evaluate_neutral(yards_to_goal, distance, score_diff, seconds_remaining, **kwargs)
    return evaluate_options(yards_to_goal, distance, score_diff, seconds_remaining,
                            is_home_pos=1 if site == "home" else 0, **kwargs)
