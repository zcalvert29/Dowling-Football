"""
Checks for the 4th-down / 2-point math. Run from the repo root:

    pip install pytest
    pytest tests

These don't pin exact win probabilities (those change whenever the models
are retrained). They check that the batched code agrees with the one-at-a-time
code, and that the decision logic behaves the way football says it should.
"""
import numpy as np
import pandas as pd
import pytest

import fourth_down_core as fd
import model_utils as mu

Q3_START = fd.model_seconds_remaining(3, 10 * 60)  # 10:00 left in Q3 (clear of the first-minute clamp)


def test_batch_prediction_matches_single():
    cases = [(0, 40, 4, 4, 900, 2700, 3, 3, 1), (7, 75, 1, 10, 1800, 1800, 1, 2, 0), (-3, 8, 1, 8, 60, 60, 0, 3, 1)]
    batch, _ = mu.predict_wp_chained_batch(*[np.array(c) for c in zip(*cases)])
    single = [mu.predict_wp_chained(*c)[0] for c in cases]
    np.testing.assert_allclose(batch, single, atol=1e-6)


def test_evaluate_options_matches_evaluate_many():
    ytg, dist, sd = np.array([65, 30, 5, 90]), np.array([2, 4, 3, 10]), np.array([0, -3, 7, -14])
    many = fd.evaluate_many(ytg, dist, sd, Q3_START, is_home_pos=1)
    for i in range(len(ytg)):
        one = fd.evaluate_options(ytg[i], dist[i], sd[i], Q3_START, is_home_pos=1)["wp"]
        assert one["Go for it"] == pytest.approx(many["wp_go"][i])
        for name, key in (("Field goal", "wp_fg"), ("Punt", "wp_punt")):
            if name in one:
                assert one[name] == pytest.approx(many[key][i])
            else:
                assert np.isnan(many[key][i])


def test_neutral_is_average_of_home_and_away():
    h = fd.evaluate_site("home", 45, 3, 0, Q3_START)["wp"]
    a = fd.evaluate_site("away", 45, 3, 0, Q3_START)["wp"]
    n = fd.evaluate_site("neutral", 45, 3, 0, Q3_START)["wp"]
    for k in n:
        assert n[k] == pytest.approx((h[k] + a[k]) / 2)


def test_option_availability():
    assert "Punt" not in fd.evaluate_site("home", 30, 5, 0, Q3_START)["wp"]       # inside the 35
    assert "Field goal" not in fd.evaluate_site("home", 60, 5, 0, Q3_START)["wp"]  # 77-yard kick
    assert "Field goal" not in fd.evaluate_site("home", 45, 5, 0, Q3_START)["wp"]  # 62 yards: past the curve
    assert set(fd.evaluate_site("home", 38, 5, 0, Q3_START)["wp"]) == {"Go for it", "Field goal", "Punt"}  # 55 yds
    r = fd.evaluate_many(38, 5, 0, Q3_START, is_home_pos=1, max_fg_distance=50)   # a shorter-range kicker
    assert np.isnan(r["wp_fg"][0])


def test_timeouts_swap_after_change_of_possession():
    """Our offense with 1 timeout vs their 3: after a punt THEY have the ball
    with 3 and we have 1. Swapping our counts should be a mirror image."""
    r = fd.evaluate_many(70, 8, 0, Q3_START, off_timeouts=1, def_timeouts=3, is_home_pos=1)
    wp_opp = fd.wp_new_possession(0, 100 - (70 - fd.DEEP_PUNT_NET), Q3_START,
                                  off_timeouts=3, def_timeouts=1, is_home_pos=0)
    assert r["wp_punt"][0] == pytest.approx(1 - wp_opp[0])


def test_break_even_is_where_go_ties_the_kick():
    r = fd.evaluate_many(np.array([65, 40, 20]), np.array([2, 4, 6]), 0, Q3_START, is_home_pos=0.5)
    be = fd.break_even_conversion(r)
    alt = np.fmax(r["wp_fg"], r["wp_punt"])
    go_at_be = be * r["wp_go_success"] + (1 - be) * r["wp_go_fail"]
    ok = (be > 0) & (be < 1)
    np.testing.assert_allclose(go_at_be[ok], alt[ok], atol=1e-9)


def test_shorter_distance_never_makes_going_worse():
    dist = np.arange(1, 16)
    r = fd.evaluate_many(55, dist, 0, Q3_START, is_home_pos=0.5)
    assert np.all(np.diff(r["wp_go"]) <= 1e-9)


def test_two_point_logic():
    late = fd.model_seconds_remaining(4, 5 * 60)
    down2 = fd.evaluate_try(-2, late, p_xp=0.9, p_two=0.45, is_home_pos=0.5)
    assert down2["wp_go"] > down2["wp_kick"]                 # down 2: go to tie
    tied = fd.evaluate_try(0, fd.model_seconds_remaining(2, 600), p_xp=0.95, p_two=0.40, is_home_pos=0.5)
    assert tied["wp_kick"] > tied["wp_go"]                   # tied early, good kicker: kick
    worse_kicker = fd.evaluate_try(-1, late, p_xp=0.70, p_two=0.45, is_home_pos=0.5)
    better_kicker = fd.evaluate_try(-1, late, p_xp=0.95, p_two=0.45, is_home_pos=0.5)
    assert worse_kicker["break_even_two"] < better_kicker["break_even_two"]


# --- Checks on the shipped WP model itself. If a retrain breaks these, look at
# model_training/evaluate.py (including the late-game and goal-line columns) before shipping it.

def _direct_wp(rows):
    import xgboost as xgb
    return mu._wp_booster.predict(xgb.DMatrix(rows.to_numpy(np.float32), feature_names=mu.WP_FEATURE_ORDER))


@pytest.mark.parametrize("feature, direction", [("yards_to_goal", -1), ("distance", -1), ("ep", 1)])
def test_field_position_constraints_hold(feature, direction):
    """The model's guarantees: holding everything else fixed, more yards to the goal or to go never helps,
    and more expected points never hurts."""
    rng = np.random.default_rng(1)
    values = {"yards_to_goal": np.arange(1, 100), "distance": np.arange(1, 31), "ep": np.linspace(-3, 7, 60)}[feature]
    for _ in range(100):
        sd = int(rng.integers(-21, 22)); game = int(rng.integers(30, 3600)); down = int(rng.integers(1, 5))
        base = {"ep": float(rng.uniform(-2, 6)), "score_diff": sd, "yards_to_goal": int(rng.integers(1, 100)),
                **{f"down{k}": float(k == down) for k in (1, 2, 3, 4)}, "distance": int(rng.integers(1, 20)),
                "half_seconds": min(game, 1800), "game_seconds_remaining": game, "off_timeouts": int(rng.integers(0, 4)),
                "def_timeouts": int(rng.integers(0, 4)), "is_home_pos": int(rng.integers(0, 2)),
                "score_diff_time_ratio": sd / (game / 60 + 1)}
        rows = pd.DataFrame([base] * len(values))[mu.WP_FEATURE_ORDER]
        rows[feature] = values
        assert (np.diff(_direct_wp(rows)) * direction >= -1e-6).all()


def test_converting_never_worse_than_failing():
    """A successful 4th-down try should never leave the offense worse off than a failed one."""
    Y, D, S, T = np.meshgrid(np.arange(15, 100, 2), np.arange(1, 11), np.arange(-21, 22, 7),
                             [3000, 2000, 1200, 600], indexing="ij")
    keep = D <= Y - 10
    r = fd.evaluate_many(Y[keep], D[keep], S[keep], T[keep], is_home_pos=0.5)
    assert (r["wp_go_success"] >= r["wp_go_fail"] - 1e-9).all()


def test_goal_line_no_big_reversals():
    """Through the full chain (EP model -> WP model), win probability shouldn't jump as the offense moves away
    from the goal. The original model jumped as much as 42 points inside the 10; the shipped one stays under 3."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        ytg = np.arange(1, 100)
        wp, _ = mu.predict_wp_chained_batch(int(rng.integers(-21, 22)), ytg, int(rng.integers(1, 5)), np.minimum(10, ytg),
                                            int(rng.integers(30, 1800)), int(rng.integers(30, 3600)),
                                            is_home_pos=int(rng.integers(0, 2)))
        assert np.diff(wp).max() < 0.03


def test_own_goal_line_punts_on_long_yardage():
    assert fd.best_calls(fd.evaluate_many(99, 15, 0, Q3_START, is_home_pos=1))["best"][0] == "Punt"


# --- Checks added with the field-position, NFHS and conversion fixes (Oct 2026).

@pytest.mark.parametrize("quarter, clock, margin", [(1, 720, 0), (2, 360, -7), (3, 600, 3), (4, 360, 0), (4, 120, -3)])
def test_field_position_is_worth_something(quarter, clock, margin):
    """The bug behind '4th & 6 from our own 9 is a go': the other team at our 9 was priced barely worse
    for us than the other team at our 41. A new possession's value must fall steadily as it moves away
    from the goal, and the 9 vs the 41 must be a real gap."""
    secs = fd.model_seconds_remaining(quarter, clock)
    ytg = np.arange(1, 100)
    wp = fd.wp_new_possession(margin, ytg, secs, is_home_pos=1)
    assert (np.diff(wp) <= 1e-9).all()
    nine, forty_one = fd.wp_new_possession(margin, [9, 41], secs, is_home_pos=1)
    assert nine - forty_one >= 0.03


def test_raw_pricing_still_available():
    r_new = fd.evaluate_many(91, 6, 0, Q3_START, is_home_pos=1)
    r_old = fd.evaluate_many(91, 6, 0, Q3_START, is_home_pos=1, field_pricing="raw")
    assert r_new["wp_go_fail"][0] < r_old["wp_go_fail"][0]   # a turnover at our 9 costs more now


def test_missed_field_goal_is_a_touchback_at_the_20():
    """NFHS: a miss that reaches the end zone is their ball at their 20, wherever it was kicked from."""
    r = fd.evaluate_many(np.array([10, 20, 30]), 8, 0, Q3_START, is_home_pos=1)
    np.testing.assert_allclose(r["wp_fg_miss"], r["wp_fg_miss"][0])
    spot = fd.evaluate_many(30, 8, 0, Q3_START, is_home_pos=1, fg_miss_rule="spot")
    assert r["wp_fg_miss"][2] > spot["wp_fg_miss"][0]   # a miss from their 30: their 20 beats their 30 for us


def test_pat_rate_matters_on_touchdowns_only():
    sure = fd.evaluate_many(np.array([2, 40]), np.array([2, 2]), -7, fd.model_seconds_remaining(4, 120),
                            is_home_pos=1, p_xp=1.0)
    shaky = fd.evaluate_many(np.array([2, 40]), np.array([2, 2]), -7, fd.model_seconds_remaining(4, 120),
                             is_home_pos=1, p_xp=0.7)
    assert sure["wp_go_success"][0] > shaky["wp_go_success"][0]   # 4th & goal: TD needs the PAT to tie
    assert sure["wp_go_success"][1] == pytest.approx(shaky["wp_go_success"][1])


def test_conversion_curve_and_matchup_shift():
    p = fd.conversion_prob_many(np.arange(1, 21))
    assert (np.diff(p) < 0).all()
    assert fd.conversion_prob(2, logit_shift=0.3) > fd.conversion_prob(2) > fd.conversion_prob(2, logit_shift=-0.3)


def test_conversion_lower_in_our_own_territory():
    own, mid, theirs = fd.conversion_prob_many(4, yards_to_goal=np.array([80, 50, 20]))
    assert own < mid < theirs
    assert np.ptp(fd.conversion_prob_many(4, yards_to_goal=np.array([10, 30]))) < 0.01     # flat away from midfield
    r = fd.evaluate_many(np.array([80, 20]), 4, 0, Q3_START, is_home_pos=1)
    assert r["p_conv"][0] < r["p_conv"][1]


def test_conversion_model_fit():
    import conversion_model as cm
    m = cm.fit(pd.read_excel("curated-pbp.xlsx"))
    assert m.n_plays > 300
    assert (np.diff(m.prob(np.arange(1, 21))) < 0).all()
    assert m.c < 0                                       # harder to convert in our own territory
    assert abs(m.team_shift("Nobody", "Nobody Else")) == 0


def test_robustness_flags_close_calls():
    r = fd.evaluate_many(np.array([55, 91]), np.array([1, 15]), 0, Q3_START, is_home_pos=0.5)
    rob = fd.robustness(r)
    assert rob["robust"][1]                       # 4th & 15 at our own 9: punt no matter what
    calls = fd.best_calls(r)
    assert ((rob["worst_margin"] <= calls["margin"] + 1e-9)).all()


@pytest.mark.parametrize("ytg, dist, expected", [
    (50, 1, "Go for it"),     # 4th & 1 at midfield
    (1, 1, "Go for it"),      # 4th & goal at the 1
    (75, 10, "Punt"),         # 4th & 10 at our own 25
    (80, 12, "Punt"),         # 4th & 12 at our own 20
    (20, 15, "Field goal"),   # 4th & 15 at their 20 (37-yard kick)
])
def test_football_sanity_early_tied(ytg, dist, expected):
    r = fd.evaluate_many(ytg, dist, 0, fd.model_seconds_remaining(1, 600), is_home_pos=0.5)
    assert fd.best_calls(r)["best"][0] == expected


# --- Guardrails (Oct 2026).

def test_own_end_guardrail_turns_close_go_into_punt():
    secs = fd.model_seconds_remaining(2, 360)
    Y, D = np.array([91, 91, 30]), np.array([6, 2, 3])          # 4th & 6 at our 9, 4th & 2 at our 9, 4th & 3 at their 30
    r = fd.evaluate_many(Y, D, 0, secs, is_home_pos=0.5)
    g = fd.guarded_calls(r, Y, D, own_min_edge=3.0)
    assert g["raw_call"][0] == "Go for it" and g["call"][0] == "Punt" and g["rule"][0] and g["cost"][0] > 0
    assert g["call"][1] == "Go for it" and g["rule"][1] == ""   # clear short-yardage go stays
    assert g["call"][2] == "Go for it" and g["rule"][2] == ""   # their territory is never guarded
    off = fd.guarded_calls(r, Y, D, own_end=False)
    assert (off["call"] == off["raw_call"]).all()


def test_staff_no_go_rules():
    secs = fd.model_seconds_remaining(2, 360)
    # 4th & 2 at our 9, 4th & 3 at our 20, 4th & 3 at our 35, 4th & 4 at our 38, 4th & 4 at our 45
    Y, D = np.array([91, 80, 65, 62, 55]), np.array([2, 3, 3, 4, 4])
    r = fd.evaluate_many(Y, D, 0, secs, is_home_pos=0.5, conv_logit_shift=1.0)  # strong offense: math says go
    g = fd.guarded_calls(r, Y, D, own_end=False, no_go=fd.STAFF_NO_GO)
    assert (g["raw_call"] == "Go for it").all()
    assert g["call"][0] == "Go for it"                                       # 2 yards: allowed
    assert g["call"][1] == "Punt" and "25 and in" in g["rule"][1]            # 3+ inside our 25
    assert g["call"][2] == "Go for it"                                       # 3 yards at our 35: allowed
    assert g["call"][3] == "Punt" and "40 and in" in g["rule"][3]            # 4+ inside our 40
    assert g["call"][4] == "Go for it"                                       # our 45: outside the rules
    assert fd.BOT_SETTINGS["no_go"] == fd.STAFF_NO_GO                        # on everywhere by default


def test_guardrails_never_touch_kicks():
    secs = fd.model_seconds_remaining(2, 360)
    Y, D = np.arange(36, 100, 3), np.full(22, 15)
    r = fd.evaluate_many(Y, D, 0, secs, is_home_pos=0.5)
    g = fd.guarded_calls(r, Y, D, no_go=fd.STAFF_NO_GO_DEFAULT)
    kicks = g["raw_call"] != "Go for it"
    assert (g["call"][kicks] == g["raw_call"][kicks]).all()


def test_guardrails_off_when_trailing_late():
    late = fd.model_seconds_remaining(4, 180)
    Y, D = np.array([70, 70]), np.array([7, 7])
    r = fd.evaluate_many(Y, D, np.array([-3, 3]), late, is_home_pos=0.5)
    g = fd.guarded_calls(r, Y, D, no_go=fd.STAFF_NO_GO_DEFAULT, score_diff=np.array([-3, 3]), seconds_remaining=late)
    assert g["rule"][0] == ""                                  # down 3 with 3:00 left: no guardrail
    if g["raw_call"][1] == "Go for it":
        assert g["rule"][1] != ""                              # up 3: the table still applies


def test_review_matches_the_bot():
    """The season review must recommend exactly what the 4th Down Bot would, row by row."""
    import conversion_model as cm
    import insights as ins
    import visuals as v
    df = v.load_data("curated-pbp.xlsx")
    review = getattr(ins.fourth_down_review, "__wrapped__", ins.fourth_down_review)
    t = review(ins.dowling_only(df))
    conv = cm.fitted()
    for _, r in t.iterrows():
        g = df[df["game_id"] == r["game"]]
        opp = r["opp"]
        sd = int(r["score_text"].split()[1].split("-")[0]) - int(r["score_text"].split()[1].split("-")[1])
        q, clock = r["Est. clock"].split()
        m, s_ = map(int, clock.split(":"))
        secs = fd.model_seconds_remaining(int(q[1]), m * 60 + s_)
        res = fd.evaluate_many(r["ytg"], max(r["dist"], 1), sd, secs, is_home_pos=0.5,
                               conv_logit_shift=conv.team_shift(r["team"], opp),
                               **{k: fd.BOT_SETTINGS[k] for k in fd.MODEL_KEYS})
        bot = fd.guarded_calls(res, r["ytg"], max(r["dist"], 1), score_diff=sd, seconds_remaining=secs,
                               **{k: fd.BOT_SETTINGS[k] for k in fd.GUARD_KEYS})
        assert bot["call"][0] == r["Model"], r["Situation"]


def test_bot_page_handles_go_as_the_only_option():
    """4th & 7 at their 34: no punt inside the 35 and a 51-yard kick is past the default range."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("../Fourth_Downs.py", default_timeout=120)
    for k, val in dict(fd_side="Opp", fd_yard_line=34, fd_distance=7, fd_quarter=3, fd_minutes=6,
                       fd_off_score=7, fd_def_score=3).items():
        at.session_state[k] = val
    at.run()
    assert not at.exception
