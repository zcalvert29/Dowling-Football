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
# 4TH DOWN OPTION EVALUATION
#
# evaluate_many() is the engine: it takes arrays of situations and prices
# every option for all of them with ONE batched model call, so a full
# 1-yard decision chart (thousands of situations) takes milliseconds.
# evaluate_options() / evaluate_neutral() / evaluate_site() keep their old
# single-situation interface on top of it.
#
# Two fixes over the original per-situation version:
#   * Timeouts swap with possession. After a turnover on downs, punt, or
#     score, the other team is on offense, so THEIR offense timeouts are our
#     defense timeouts. (Before, both sides kept the same counts, which only
#     mattered when the counts were different.)
#   * First-and-goal distance. A new first down inside the 10 is 1st & goal,
#     so the distance is the yards to the goal line, not 10. The EP model was
#     trained on real goal-to-go snaps and never saw "1st & 10 at the 3".
# ==============================================================================

# Longest field goal ever converted at any level is in the high-60s of yards;
# beyond that it's not a real coaching option, so it's taken off the table
# instead of being priced with a tiny probability.
MAX_FG_KICK_DISTANCE = 62  # yards_to_goal + 17
PUNT_MIN_YTG = 35          # no punts from inside the opponent's 35
OPTIONS = ("Go for it", "Field goal", "Punt")


def conversion_prob_many(distance):
    d = np.maximum(np.asarray(distance, dtype=float), 0.5)
    return 1 / (1 + np.exp(0.262 * (d - 5.0)))


def fg_prob_many(yards_to_goal, wind_speed=0, wind_direction="Into", rain=False, snow=False):
    kick_distance = np.asarray(yards_to_goal, dtype=float) + 17
    base = 1 / (1 + np.exp(0.1156 * (kick_distance - 45)))
    return np.clip(base + weather_adjustment(wind_speed, wind_direction, rain, snow), 0.0, 1.0)


def punt_net_yards_many(yards_to_goal):
    y = np.asarray(yards_to_goal, dtype=float)
    return np.where(y > 60, DEEP_PUNT_NET, np.clip(y - 20, 5, 40))


def _first_down_distance(ytg):
    """1st & 10, or 1st & goal inside the 10."""
    return np.minimum(10.0, ytg)


def evaluate_many(yards_to_goal, distance, score_diff, seconds_remaining,
                  off_timeouts=3, def_timeouts=3, is_home_pos=1,
                  wind_speed=0, wind_direction="Into", rain=False, snow=False):
    """
    Price every 4th-down option for many situations at once.

    yards_to_goal, distance, score_diff, seconds_remaining, the timeouts and
    is_home_pos can be scalars or arrays (broadcast together); is_home_pos
    can also be 0.5 for a neutral site (home and away averaged). Weather is
    one value for the whole batch.

    Returns a dict of flat arrays: wp_go, wp_fg, wp_punt (NaN where the
    option isn't available), the success/fail branches, p_conv, and p_fg.
    """
    ytg, dist, sd, secs, oto, dto, home = np.broadcast_arrays(
        *(np.asarray(x, dtype=float).ravel() for x in (
            yards_to_goal, distance, score_diff, seconds_remaining,
            off_timeouts, def_timeouts, is_home_pos)))
    n = len(ytg)
    half = np.where(secs > MODEL_QUARTER_SECONDS * 2, secs - MODEL_QUARTER_SECONDS * 2, secs)

    # The possible next snaps. "same" = we still have the ball; "flip" = the
    # other team has it, so score, home/away, and timeouts all flip.
    touchdown = dist >= ytg
    conv_ytg = np.where(touchdown, 75.0, np.maximum(ytg - dist, 1.0))
    after_punt = np.clip(100 - (ytg - punt_net_yards_many(ytg)), 1, 99)
    states = [
        # name,        ytg after,   score diff (offense's view),   flip?
        ("go_success", conv_ytg,     np.where(touchdown, -(sd + 7), sd), touchdown),
        ("turnover",   100 - ytg,    -sd,                           np.ones(n, bool)),
        ("fg_make",    np.full(n, 75.0), -(sd + 3),                 np.ones(n, bool)),
        ("punt",       after_punt,   -sd,                           np.ones(n, bool)),
    ]

    def run(home_flag):
        cols = {k: [] for k in ("sd", "ytg", "oto", "dto", "home")}
        for _, y, s, flip in states:
            cols["ytg"].append(y)
            cols["sd"].append(s)
            cols["oto"].append(np.where(flip, dto, oto))
            cols["dto"].append(np.where(flip, oto, dto))
            cols["home"].append(np.where(flip, 1 - home_flag, home_flag))
        cat = {k: np.concatenate(v) for k, v in cols.items()}
        wp, _ = mu.predict_wp_chained_batch(
            score_diff=cat["sd"], yards_to_goal=cat["ytg"], down=1,
            distance=_first_down_distance(cat["ytg"]), half_seconds=np.tile(half, len(states)),
            game_seconds_remaining=np.tile(secs, len(states)),
            off_timeouts=cat["oto"], def_timeouts=cat["dto"], is_home_pos=cat["home"])
        wp = wp.reshape(len(states), n)
        flips = np.stack([f for *_, f in states])
        return np.where(flips, 1 - wp, wp)  # back to OUR win probability

    # Neutral site (0.5) = average of the home and away answers.
    is_neutral = home == 0.5
    if is_neutral.all():
        out = (run(np.ones(n)) + run(np.zeros(n))) / 2
    elif is_neutral.any():
        out = np.where(is_neutral, (run(np.ones(n)) + run(np.zeros(n))) / 2, run(home))
    else:
        out = run(home)
    wp_go_success, wp_turnover, wp_fg_make, wp_punt = out

    p_conv = conversion_prob_many(dist)
    p_fg = fg_prob_many(ytg, wind_speed, wind_direction, rain, snow)
    fg_ok = ytg + 17 <= MAX_FG_KICK_DISTANCE
    punt_ok = ytg > PUNT_MIN_YTG
    return {
        "wp_go": p_conv * wp_go_success + (1 - p_conv) * wp_turnover,
        "wp_fg": np.where(fg_ok, p_fg * wp_fg_make + (1 - p_fg) * wp_turnover, np.nan),
        "wp_punt": np.where(punt_ok, wp_punt, np.nan),
        "wp_go_success": wp_go_success, "wp_go_fail": wp_turnover,
        "wp_fg_make": np.where(fg_ok, wp_fg_make, np.nan), "wp_fg_miss": np.where(fg_ok, wp_turnover, np.nan),
        "p_conv": p_conv, "p_fg": p_fg,
    }


def best_calls(res, toss_up_pts=1.0):
    """
    From evaluate_many output: per situation, the best option's name, its WP,
    the margin over the next-best (WP points), and whether it's a toss-up.
    """
    wp = np.column_stack([res["wp_go"], res["wp_fg"], res["wp_punt"]])
    filled = np.where(np.isnan(wp), -np.inf, wp)
    order = np.argsort(-filled, axis=1)
    best_i = order[:, 0]
    best_wp = filled[np.arange(len(wp)), best_i]
    second = filled[np.arange(len(wp)), order[:, 1]]
    margin = np.where(np.isfinite(second), (best_wp - second) * 100, 100.0)
    return {"best": np.array(OPTIONS)[best_i], "best_wp": best_wp, "margin": margin,
            "toss_up": margin < toss_up_pts}


def break_even_conversion(res):
    """
    The conversion rate at which going for it ties the best kick, per
    situation (NaN when there's no kicking option). If the model's
    conversion chance is above this, going is the call.
    """
    alt = np.fmax(res["wp_fg"], res["wp_punt"])
    gap = res["wp_go_success"] - res["wp_go_fail"]
    with np.errstate(invalid="ignore", divide="ignore"):
        be = (alt - res["wp_go_fail"]) / gap
    return np.where(np.isnan(alt) | (gap <= 0), np.nan, np.clip(be, 0, 1))


def site_flag(site):
    return {"home": 1.0, "away": 0.0, "neutral": 0.5}[site]


def _to_options(res, i=0):
    """One situation from evaluate_many, in the old evaluate_options dict format."""
    f = lambda k: None if np.isnan(res[k][i]) else float(res[k][i])
    options = {"Go for it": {"wp": f("wp_go"), "success_prob": float(res["p_conv"][i]),
                             "wp_success": f("wp_go_success"), "wp_fail": f("wp_go_fail")}}
    if f("wp_fg") is not None:
        options["Field goal"] = {"wp": f("wp_fg"), "success_prob": float(res["p_fg"][i]),
                                 "wp_success": f("wp_fg_make"), "wp_fail": f("wp_fg_miss")}
    if f("wp_punt") is not None:
        # Punt has no modeled success/fail split; it's a single outcome here.
        options["Punt"] = {"wp": f("wp_punt"), "success_prob": None, "wp_success": None, "wp_fail": None}
    return {"wp": {k: v["wp"] for k, v in options.items()}, "options": options,
            "p_conv": float(res["p_conv"][i]), "p_fg": float(res["p_fg"][i])}


def evaluate_options(yards_to_goal, distance, score_diff, seconds_remaining,
                     off_timeouts=3, def_timeouts=3, is_home_pos=1,
                     wind_speed=0, wind_direction="Into", rain=False, snow=False):
    """One situation. is_home_pos: 1 = offense is home, 0 = away."""
    res = evaluate_many(yards_to_goal, distance, score_diff, seconds_remaining, off_timeouts, def_timeouts,
                        is_home_pos, wind_speed, wind_direction, rain, snow)
    return _to_options(res)


def evaluate_neutral(yards_to_goal, distance, score_diff, seconds_remaining, **kwargs):
    """Neutral site: the WP model includes home-field advantage, so when
    home/away isn't known (the 4th-down review) average the two."""
    kwargs.pop("is_home_pos", None)
    return evaluate_options(yards_to_goal, distance, score_diff, seconds_remaining, is_home_pos=0.5, **kwargs)


def evaluate_site(site, yards_to_goal, distance, score_diff, seconds_remaining, **kwargs):
    """site: "home", "away", or "neutral" (from the offense's point of view)."""
    return evaluate_options(yards_to_goal, distance, score_diff, seconds_remaining,
                            is_home_pos=site_flag(site), **kwargs)


# ==============================================================================
# EXTRA POINT vs TWO-POINT TRY (used by the Go for 2 Bot's model check)
#
# After the try the scoring team kicks off, so every outcome is priced as the
# OTHER team's ball at their own 25 with the new margin (same pattern as a
# made field goal above). Unlike ESPN's cheat sheet, the make rates are inputs,
# so a high school PAT that's 85-90% instead of the NFL's ~95% is accounted for.
# ==============================================================================
def evaluate_try(margin_before_try, seconds_remaining, p_xp, p_two,
                 off_timeouts=3, def_timeouts=3, is_home_pos=1.0):
    """
    margin_before_try: our score minus theirs after the TD's 6 points.
    seconds_remaining: model clock (see model_seconds_remaining).
    Returns WP of kicking, of going for 2, WP for each resulting margin, and
    the 2-point success rate at which the two choices tie.
    """
    m = float(margin_before_try)
    margins = np.array([m, m + 1, m + 2])
    half = half_seconds_from_game(seconds_remaining)

    def after(home_flag):
        # Opponent's ball at their 25: flip score, site, and timeouts.
        wp_opp, _ = mu.predict_wp_chained_batch(
            score_diff=-margins, yards_to_goal=75, down=1, distance=10, half_seconds=half,
            game_seconds_remaining=seconds_remaining, off_timeouts=def_timeouts,
            def_timeouts=off_timeouts, is_home_pos=1 - home_flag)
        return 1 - wp_opp

    wp0, wp1, wp2 = (after(1.0) + after(0.0)) / 2 if is_home_pos == 0.5 else after(float(is_home_pos))
    wp_kick = p_xp * wp1 + (1 - p_xp) * wp0
    wp_go = p_two * wp2 + (1 - p_two) * wp0
    gap = wp2 - wp0
    break_even = float(np.clip((wp_kick - wp0) / gap, 0, 1)) if gap > 1e-9 else float("nan")
    return {"wp_kick": float(wp_kick), "wp_go": float(wp_go), "wp_if": {m: float(wp0), m + 1: float(wp1),
            m + 2: float(wp2)}, "break_even_two": break_even}
