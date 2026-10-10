"""
fourth_down_core.py — the 4th Down Bot's decision math, with no Streamlit
code, so other pages (e.g. the 4th-down decision review) can reuse it.
Fourth_Downs.py imports it; the 4th-down review on the Win Probability page
uses the same functions.
"""
import numpy as np

import model_utils as mu


# ==============================================================================
# CONVERSION RATE — the original hand-drawn curve, re-fit to our high school
# film (conversion_model.py). On 339 tagged 3rd/4th-down runs and passes
# at 10 yards or less through week 7, the original curve said 74% on
# 4th & 1; the film says 64%. It overstated short yardage and understated
# 6-10 yards (the curve is flatter in high school).
#
# Field position matters too: our film converts about 7 points under the
# curve in our own territory and about 5 over it in theirs. HS_CONV_OWN is
# that effect (log-odds, applied with a smooth ramp across midfield); with
# the team adjustments in the same fit it comes out at about -6 points of
# conversion rate on 4th & 6. Refit with conversion_model.fit() as games
# are added and copy a, b, c here.
# ==============================================================================
COLLEGE_CONV_SLOPE, COLLEGE_CONV_MID = 0.262, 5.0
HS_CONV_A, HS_CONV_B, HS_CONV_OWN = 0.034, 0.103, -0.269   # conversion_model.fit(), through week 7


def own_side(yards_to_goal):
    """0 in the opponent's territory, 1 in ours, smooth across midfield (~0.5 at the 50)."""
    return 1 / (1 + np.exp(-(np.asarray(yards_to_goal, dtype=float) - 50) / 5))


def conversion_prob_many(distance, logit_shift=0.0, curve="hs", yards_to_goal=None):
    """Chance a 4th-down try moves the chains.

    curve="hs" (default) is the high school fit; curve="college" is the
    original hand-drawn curve. yards_to_goal applies the field-position
    effect (high school curve only); without it the rate is for a snap at
    midfield. logit_shift moves it for a specific offense and defense
    (ConversionModel.team_shift): +0.25 is roughly +6 points near 50%.
    """
    d = np.maximum(np.asarray(distance, dtype=float), 0.5)
    z = -COLLEGE_CONV_SLOPE * (d - COLLEGE_CONV_MID)
    if curve == "hs":
        z = z + HS_CONV_A + HS_CONV_B * (d - COLLEGE_CONV_MID)
        z = z + HS_CONV_OWN * own_side(50 if yards_to_goal is None else yards_to_goal)
    return 1 / (1 + np.exp(-(z + logit_shift)))


def conversion_prob(distance, logit_shift=0.0, curve="hs", yards_to_goal=None):
    return float(conversion_prob_many(distance, logit_shift, curve, yards_to_goal))


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


# Field goal make rate by kick distance (yards_to_goal + 17), before weather.
# Straight lines between these points; flat at 95% inside 20 yards, and the
# 50-to-60 slope continues past 60 until it reaches zero (about 61 yards).
FG_CURVE_YARDS = (20, 30, 40, 50, 60, 61.43)
FG_CURVE_MAKE = (0.95, 0.80, 0.60, 0.40, 0.05, 0.0)


def _fg_base(kick_distance):
    return np.interp(np.asarray(kick_distance, dtype=float), FG_CURVE_YARDS, FG_CURVE_MAKE)


def fg_prob(yards_to_goal, wind_speed=0, wind_direction="Into", rain=False, snow=False):
    adjusted = _fg_base(yards_to_goal + 17) + weather_adjustment(wind_speed, wind_direction, rain, snow)
    return float(np.clip(adjusted, 0.0, 1.0))


# Net punt from our own 40 or deeper (closer in, punts are capped at the 20).
DEEP_PUNT_NET = 35.0


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

# Beyond the kicker's range a field goal is taken off the table. The default
# is where the staff's FG curve reaches zero, so the curve decides; set it
# lower (max_fg_distance, or on the page) for a kicker with less leg.
MAX_FG_KICK_DISTANCE = 61  # yards_to_goal + 17; FG_CURVE hits 0% at ~61.4
PUNT_MIN_YTG = 35          # no punts from inside the opponent's 35
OPTIONS = ("Go for it", "Field goal", "Punt")


def fg_prob_many(yards_to_goal, wind_speed=0, wind_direction="Into", rain=False, snow=False):
    base = _fg_base(np.asarray(yards_to_goal, dtype=float) + 17)
    return np.clip(base + weather_adjustment(wind_speed, wind_direction, rain, snow), 0.0, 1.0)


def punt_net_yards_many(yards_to_goal):
    y = np.asarray(yards_to_goal, dtype=float)
    return np.where(y > 60, DEEP_PUNT_NET, np.clip(y - 20, 5, 40))


def _first_down_distance(ytg):
    """1st & 10, or 1st & goal inside the 10."""
    return np.minimum(10.0, ytg)


# ------------------------------------------------------------------------------
# PRICING A NEW POSSESSION
#
# Every 4th-down outcome leaves someone with 1st down somewhere. The question
# is what that's worth in win probability.
#
# "raw" (the original method) asks the WP model directly. The shipped WP model
# is nearly flat in field position: tied with 6:00 left, the other team at our
# 9 came out only ~2.5 WP points worse for us than the other team at our 41,
# and the 1 was priced the same as the 15. That made a turnover deep in our
# own end look cheap and pushed the bot to go for it there.
#
# "ep_anchor" (the default now) splits the job between the two models:
#   * SCORE: the WP model, asked about 1st & 10 at the 25 (where it has the
#     most data, every kickoff) with the real score margin. Made monotone
#     across margins, so a bigger lead never lowers win probability.
#   * FIELD POSITION: priced in points by the EP model, which handles field
#     position well, as EP(here) - EP(own 25). Those points are converted to
#     win probability with a smooth logistic curve fit to the WP model's
#     own score curve at this clock, timeouts and site:
#
#     WP(here) = WP_25(margin) + [L(margin + EP(here) - EP(25)) - L(margin)]
#
# So a yard of field position always counts, by about what the EP model
# says it's worth in points, and the score structure the WP model learned
# (down 3 vs down 4 late) is kept.
# ------------------------------------------------------------------------------
FIELD_PRICING = "ep_anchor"
ANCHOR_YTG = 75.0
_MARGINS = np.arange(-50, 51, dtype=float)
_FIT_MARGINS = np.arange(-24, 25, dtype=float)


def _isotonic_increasing(y):
    """Pool-adjacent-violators: the closest non-decreasing sequence to y."""
    vals, wts = [], []
    for v in y:
        vals.append(float(v)); wts.append(1.0)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            wts[-2] = w
            vals.pop(); wts.pop()
    return np.repeat(vals, np.array(wts, dtype=int))


def _logistic_fit(wp_row):
    """Least-squares logistic in margin, fit in log-odds, weighted toward close games."""
    sel = (_MARGINS >= _FIT_MARGINS[0]) & (_MARGINS <= _FIT_MARGINS[-1])
    w = np.clip(wp_row[sel], 0.005, 0.995)
    z = np.log(w / (1 - w))
    sw = np.sqrt(np.exp(-np.abs(_FIT_MARGINS) / 10))
    A = np.column_stack([np.ones_like(_FIT_MARGINS), _FIT_MARGINS]) * sw[:, None]
    a, b = np.linalg.lstsq(A, z * sw, rcond=None)[0]
    return a, max(b, 1e-3)


def _wp_by_margin(secs, oto, dto, home):
    """For each unique (clock, timeouts, site): WP at 1st & 10 from the 25 over
    a grid of score margins (made monotone), plus a smooth logistic fit to it.
    Returns (key index per row, curves, logistic params)."""
    keys = np.column_stack([secs, oto, dto, home])
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    k, m = len(uniq), len(_MARGINS)
    u_secs, u_oto, u_dto, u_home = (np.repeat(uniq[:, j], m) for j in range(4))
    half = np.where(u_secs > MODEL_QUARTER_SECONDS * 2, u_secs - MODEL_QUARTER_SECONDS * 2, u_secs)
    wp, _ = mu.predict_wp_chained_batch(
        score_diff=np.tile(_MARGINS, k), yards_to_goal=ANCHOR_YTG, down=1, distance=10,
        half_seconds=half, game_seconds_remaining=u_secs,
        off_timeouts=u_oto, def_timeouts=u_dto, is_home_pos=u_home)
    rows = wp.reshape(k, m)
    curves = np.vstack([_isotonic_increasing(r) for r in rows])
    params = np.array([_logistic_fit(r) for r in rows])
    return inv.ravel(), curves, params


_YTG_GRID = np.arange(1, 100, dtype=float)


def _ep_curve(sd, half, oto, dto, ytg):
    """EP at 1st down (1st & goal inside the 10), made monotone in field position
    (the EP model has a few small bumps, e.g. around the 34) and interpolated
    to any yard line."""
    keys = np.column_stack([sd, half, oto, dto])
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    k, g = len(uniq), len(_YTG_GRID)
    _, ep = mu.predict_wp_chained_batch(np.repeat(uniq[:, 0], g), np.tile(_YTG_GRID, k), 1,
                                        np.tile(_first_down_distance(_YTG_GRID), k), np.repeat(uniq[:, 1], g),
                                        np.repeat(uniq[:, 1], g), np.repeat(uniq[:, 2], g),
                                        np.repeat(uniq[:, 3], g), 0)
    # Closer to the goal is never worth fewer points: isotonic on the reversed grid.
    curves = np.vstack([_isotonic_increasing(row[::-1])[::-1] for row in ep.reshape(k, g)])
    y = np.clip(np.asarray(ytg, dtype=float), 1, 99)
    return _interp_rows(y, curves, inv.ravel())


def _interp_rows(y, curves, idx):
    lo = np.clip(np.floor(y).astype(int) - 1, 0, len(_YTG_GRID) - 1)
    hi = np.minimum(lo + 1, len(_YTG_GRID) - 1)
    frac = y - _YTG_GRID[lo]
    return (1 - frac) * curves[idx, lo] + frac * curves[idx, hi]


def wp_new_possession(score_diff, yards_to_goal, seconds_remaining, off_timeouts=3, def_timeouts=3,
                      is_home_pos=1, field_pricing=None):
    """Win probability for the team that just got the ball, 1st down at
    yards_to_goal (1st & goal inside the 10). Arrays broadcast together."""
    field_pricing = field_pricing or FIELD_PRICING
    sd, ytg, secs, oto, dto, home = np.broadcast_arrays(
        *(np.asarray(x, dtype=float).ravel() for x in (
            score_diff, yards_to_goal, seconds_remaining, off_timeouts, def_timeouts, is_home_pos)))
    half = np.where(secs > MODEL_QUARTER_SECONDS * 2, secs - MODEL_QUARTER_SECONDS * 2, secs)
    if field_pricing == "raw":
        wp, _ = mu.predict_wp_chained_batch(sd, ytg, 1, _first_down_distance(ytg), half, secs, oto, dto, home)
        return wp

    # Field position in points: EP here minus EP at the 25, same clock, score and timeouts.
    delta = _ep_curve(sd, half, oto, dto, ytg) - _ep_curve(sd, half, oto, dto, np.full(len(sd), ANCHOR_YTG))

    # Score level from the WP model at the 25 (interpolated if the margin isn't whole).
    idx, curves, params = _wp_by_margin(secs, oto, dto, home)
    pos = np.clip(sd - _MARGINS[0], 0, len(_MARGINS) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(_MARGINS) - 1)
    frac = pos - lo
    level = (1 - frac) * curves[idx, lo] + frac * curves[idx, hi]

    # Field position effect from the smooth curve.
    a, b = params[idx, 0], params[idx, 1]
    smooth = lambda x: 1 / (1 + np.exp(-(a + b * x)))
    return np.clip(level + smooth(sd + delta) - smooth(sd), 0.001, 0.999)


# Where the ball goes after each kind of kick (yards to goal for the team receiving it).
KICKOFF_YTG = 75.0  # our film: receiving teams start at their own 26 on average
FG_MISS_YTG = 80.0  # NFHS: a missed FG that breaks the goal-line plane is a touchback at the 20
DEFAULT_P_XP = 0.90  # PAT make rate across our film (89%, 101 tries); Dowling 92%


def evaluate_many(yards_to_goal, distance, score_diff, seconds_remaining,
                  off_timeouts=3, def_timeouts=3, is_home_pos=1,
                  wind_speed=0, wind_direction="Into", rain=False, snow=False,
                  conv_logit_shift=0.0, p_xp=DEFAULT_P_XP, field_pricing=None,
                  conv_curve="hs", fg_miss_rule="nfhs", max_fg_distance=None,
                  conv_field_position=True):
    """
    Price every 4th-down option for many situations at once.

    yards_to_goal, distance, score_diff, seconds_remaining, the timeouts and
    is_home_pos can be scalars or arrays (broadcast together); is_home_pos
    can also be 0.5 for a neutral site (home and away averaged). Weather,
    conv_logit_shift (the offense-vs-defense adjustment from
    conversion_model) and p_xp are one value for the whole batch.

    field_pricing: "ep_anchor" (default) or "raw" (the original method).
    conv_curve: "hs" (default) or "college" (the original curve).
    fg_miss_rule: "nfhs" (miss = their ball at their 20) or "spot" (original).
    max_fg_distance: the kicker's range in yards (default MAX_FG_KICK_DISTANCE).
    conv_field_position: False ignores field position in the conversion
    chance (rates as if at midfield), for comparison.

    Returns a dict of flat arrays: wp_go, wp_fg, wp_punt (NaN where the
    option isn't available), the success/fail branches, p_conv, and p_fg.
    """
    ytg, dist, sd, secs, oto, dto, home = np.broadcast_arrays(
        *(np.asarray(x, dtype=float).ravel() for x in (
            yards_to_goal, distance, score_diff, seconds_remaining,
            off_timeouts, def_timeouts, is_home_pos)))
    n = len(ytg)

    # The WP model misjudges the first minute of each half (kickoff plays were
    # dropped from its training data): there, the team with the ball at its own
    # 25 comes out WORSE than the team without it. Price those states as if a
    # minute had run, which is about when a first 4th down can happen anyway.
    first_minute = 60 * CLOCK_SCALE
    for start in (4 * MODEL_QUARTER_SECONDS, 2 * MODEL_QUARTER_SECONDS):
        secs = np.where((secs > start - first_minute) & (secs <= start), start - first_minute, secs)

    # The possible next snaps. "flip" = the other team has it, so score,
    # home/away, and timeouts all flip.
    touchdown = dist >= ytg
    conv_ytg = np.where(touchdown, KICKOFF_YTG, np.maximum(ytg - dist, 1.0))
    after_punt = np.clip(100 - (ytg - punt_net_yards_many(ytg)), 1, 99)
    after_fg_miss = np.full(n, FG_MISS_YTG) if fg_miss_rule == "nfhs" else 100 - ytg
    yes, no = np.ones(n, bool), np.zeros(n, bool)
    states = [
        # name,           ytg after,          score diff (new offense's view),   flip?
        ("go_xp",         conv_ytg,           np.where(touchdown, -(sd + 7), sd), touchdown),
        ("go_no_xp",      conv_ytg,           np.where(touchdown, -(sd + 6), sd), touchdown),
        ("turnover",      100 - ytg,          -sd,                                yes),
        ("fg_make",       np.full(n, KICKOFF_YTG), -(sd + 3),                     yes),
        ("fg_miss",       after_fg_miss,      -sd,                                yes),
        ("punt",          after_punt,         -sd,                                yes),
    ]
    k = len(states)

    def run(home_flag):
        y = np.concatenate([s[1] for s in states])
        m = np.concatenate([s[2] for s in states])
        flip = np.concatenate([s[3] for s in states])
        wp = wp_new_possession(
            m, y, np.tile(secs, k),
            off_timeouts=np.where(flip, np.tile(dto, k), np.tile(oto, k)),
            def_timeouts=np.where(flip, np.tile(oto, k), np.tile(dto, k)),
            is_home_pos=np.where(flip, 1 - np.tile(home_flag, k), np.tile(home_flag, k)),
            field_pricing=field_pricing)
        return np.where(flip, 1 - wp, wp).reshape(k, n)  # back to OUR win probability

    # Neutral site (0.5) = average of the home and away answers.
    is_neutral = home == 0.5
    if is_neutral.all():
        out = (run(np.ones(n)) + run(np.zeros(n))) / 2
    elif is_neutral.any():
        out = np.where(is_neutral, (run(np.ones(n)) + run(np.zeros(n))) / 2, run(home))
    else:
        out = run(home)
    wp_xp, wp_no_xp, wp_turnover, wp_fg_make, wp_fg_miss, wp_punt = out
    wp_go_success = np.where(touchdown, p_xp * wp_xp + (1 - p_xp) * wp_no_xp, wp_xp)

    p_conv = conversion_prob_many(dist, conv_logit_shift, conv_curve,
                                  ytg if conv_field_position else None)
    p_fg = fg_prob_many(ytg, wind_speed, wind_direction, rain, snow)
    fg_ok = ytg + 17 <= (max_fg_distance or MAX_FG_KICK_DISTANCE)
    punt_ok = ytg > PUNT_MIN_YTG
    return {
        "wp_go": p_conv * wp_go_success + (1 - p_conv) * wp_turnover,
        "wp_fg": np.where(fg_ok, p_fg * wp_fg_make + (1 - p_fg) * wp_fg_miss, np.nan),
        "wp_punt": np.where(punt_ok, wp_punt, np.nan),
        "wp_go_success": wp_go_success, "wp_go_fail": wp_turnover,
        "wp_fg_make": np.where(fg_ok, wp_fg_make, np.nan), "wp_fg_miss": np.where(fg_ok, wp_fg_miss, np.nan),
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


def robustness(res, conv_band=0.10, fg_band=0.10):
    """
    Does the call survive if our conversion and field-goal estimates are off?
    Re-scores every option with the conversion chance and FG make chance each
    moved up and down by the band (all 9 combinations; the WP branches don't
    change). Returns, per situation, whether the best call is the same in all
    9 ("robust") and the smallest margin the best call keeps (WP points,
    negative when it loses somewhere).
    """
    base = best_calls(res)["best"]
    ok = np.ones(len(base), bool)
    worst = np.full(len(base), np.inf)
    for dc in (-conv_band, 0.0, conv_band):
        for df in (-fg_band, 0.0, fg_band):
            pc = np.clip(res["p_conv"] + dc, 0, 1)
            pf = np.clip(res["p_fg"] + df, 0, 1)
            wp = np.column_stack([
                pc * res["wp_go_success"] + (1 - pc) * res["wp_go_fail"],
                pf * res["wp_fg_make"] + (1 - pf) * res["wp_fg_miss"],
                res["wp_punt"]])
            filled = np.where(np.isnan(wp), -np.inf, wp)
            best_i = np.array([OPTIONS.index(b) for b in base])
            mine = filled[np.arange(len(base)), best_i]
            others = filled.copy()
            others[np.arange(len(base)), best_i] = -np.inf
            gap = (mine - others.max(axis=1)) * 100
            gap = np.where(np.isfinite(gap), gap, 100.0)
            worst = np.minimum(worst, gap)
            ok &= gap >= 0
    return {"robust": ok, "worst_margin": worst}


# ==============================================================================
# GUARDRAILS — rules on top of the math, so a close go call deep in our own
# end doesn't get recommended. Every override is labeled with its reason and
# what it costs by the bot's own numbers; the raw math is always kept.
#
# Own-end rules (on by default), for go calls in our own territory only:
#   * a toss-up (edge under 1 point) goes to the kick;
#   * going must win by at least own_min_edge WP points (default 1);
#   * going must survive the robustness check (conversion and FG chances
#     each moved 10 points either way).
# Neither applies when we're trailing in the last 5:00 of the game (high
# school clock) — punting there to protect field position gives the game away.
# Staff no-go table (off by default): never go on 4th & N or longer in a
# zone, whatever the math says. Zones are (fewest yards to goal, most yards
# to goal, N): (80, 99, 4) = our 1 to our 20.
# ==============================================================================
OWN_MIN_EDGE = 1.0
LATE_TRAILING_HS_SECONDS = 5 * 60   # guardrails off when trailing with this little left
STAFF_NO_GO_DEFAULT = (
    (80, 99, 4),   # own 1-20: never go on 4th & 4 or longer
    (60, 79, 7),   # own 21-40: 4th & 7 or longer
    (50, 59, 10),  # own 41 to midfield: 4th & 10 or longer
)


def _spot(ytg):
    ytg = int(round(ytg))
    return "midfield" if ytg == 50 else f"our {100 - ytg}" if ytg > 50 else f"their {ytg}"


def guarded_calls(res, yards_to_goal, distance, own_end=True, own_min_edge=OWN_MIN_EDGE,
                  require_robust=True, no_go=None, toss_up_pts=1.0, score_diff=None, seconds_remaining=None):
    """
    The bot's call after guardrails. Returns flat arrays:
      call       the recommendation (a guardrail turns "Go for it" into the best kick)
      raw_call   what the math alone says
      margin     the raw call's edge over the next-best option (WP points)
      toss_up    raw edge under toss_up_pts (and no guardrail fired)
      rule       "" or the reason a guardrail changed the call
      cost       WP points the guardrail gives up by the bot's own numbers (0 if none)
      call_wp    win probability of the recommended call
    score_diff / seconds_remaining (model clock), when given, switch the
    guardrails off for situations where we trail late.
    """
    n = len(res["wp_go"])
    ytg = np.broadcast_to(np.asarray(yards_to_goal, dtype=float).ravel(), (n,)) if np.ndim(yards_to_goal) == 0 \
        else np.asarray(yards_to_goal, dtype=float).ravel()
    dist = np.broadcast_to(np.asarray(distance, dtype=float).ravel(), (n,)) if np.ndim(distance) == 0 \
        else np.asarray(distance, dtype=float).ravel()
    raw = best_calls(res, toss_up_pts)
    robust = robustness(res)["robust"]
    fg = np.nan_to_num(res["wp_fg"], nan=-1.0)
    punt = np.nan_to_num(res["wp_punt"], nan=-1.0)
    kick_name = np.where(fg >= punt, "Field goal", "Punt")
    kick_wp = np.fmax(fg, punt)
    late_trailing = np.zeros(n, bool)
    if score_diff is not None and seconds_remaining is not None:
        sd = np.broadcast_to(np.asarray(score_diff, dtype=float).ravel(), (n,)) if np.ndim(score_diff) == 0 \
            else np.asarray(score_diff, dtype=float).ravel()
        secs = np.broadcast_to(np.asarray(seconds_remaining, dtype=float).ravel(), (n,)) \
            if np.ndim(seconds_remaining) == 0 else np.asarray(seconds_remaining, dtype=float).ravel()
        late_trailing = (sd < 0) & (secs <= LATE_TRAILING_HS_SECONDS * CLOCK_SCALE)

    call = raw["best"].astype(object).copy()
    rule = np.full(n, "", dtype=object)
    for i in range(n):
        if call[i] != "Go for it" or kick_wp[i] < 0 or late_trailing[i]:
            continue  # only "go" calls are guarded, only with a kick to fall back on, never when trailing late
        why = ""
        if no_go:
            for shallow, deep, min_dist in no_go:
                if shallow <= ytg[i] <= deep and dist[i] >= min_dist:
                    why = (f"Staff rule: no go on 4th & {min_dist}+ from our {100 - deep} to "
                           f"{'midfield' if shallow == 50 else 'our ' + str(100 - shallow)}")
                    break
        if not why and own_end and ytg[i] > 50:
            m = raw["margin"][i]
            if m < toss_up_pts:
                why = "Toss-up in our own end goes to the kick"
            elif m < own_min_edge:
                why = f"Go edge (+{np.floor(m * 10) / 10:.1f}) is under the {own_min_edge:g}-point bar for our own end"
            elif require_robust and not robust[i]:
                why = "Go flips if our conversion or kick estimates are 10 points off"
        if why:
            call[i] = kick_name[i]
            rule[i] = why
    call_wp = np.where(call == "Go for it", res["wp_go"], np.where(call == "Field goal", res["wp_fg"], res["wp_punt"]))
    return {"call": call.astype(str), "raw_call": raw["best"], "margin": raw["margin"],
            "toss_up": raw["toss_up"] & (rule == ""), "rule": rule.astype(str),
            "cost": np.where(rule != "", (raw["best_wp"] - call_wp) * 100, 0.0), "call_wp": call_wp}


# ==============================================================================
# THE BOT'S SETTINGS — one place for the staff's choices. The 4th Down Bot page
# starts from these (and can change them for a single situation), and the
# season review on the Win Probability page always uses them, so the review
# never grades a call the bot wouldn't make on game day.
# Change them here, e.g. BOT_SETTINGS["no_go"] = STAFF_NO_GO_DEFAULT to turn
# the staff no-go table on everywhere.
# ==============================================================================
BOT_SETTINGS = {
    "p_xp": DEFAULT_P_XP,                    # PAT make rate
    "max_fg_distance": MAX_FG_KICK_DISTANCE,  # kicker's range, yards
    "own_end": True,                         # own-end guardrails on
    "own_min_edge": OWN_MIN_EDGE,            # go must win by this much in our own end
    "no_go": None,                           # staff no-go table: None (off) or zones like STAFF_NO_GO_DEFAULT
}
MODEL_KEYS = ("p_xp", "max_fg_distance")
GUARD_KEYS = ("own_end", "own_min_edge", "no_go")


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
                     wind_speed=0, wind_direction="Into", rain=False, snow=False, **kwargs):
    """One situation. is_home_pos: 1 = offense is home, 0 = away. Extra kwargs
    (conv_logit_shift, p_xp, field_pricing, ...) go to evaluate_many."""
    res = evaluate_many(yards_to_goal, distance, score_diff, seconds_remaining, off_timeouts, def_timeouts,
                        is_home_pos, wind_speed, wind_direction, rain, snow, **kwargs)
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
