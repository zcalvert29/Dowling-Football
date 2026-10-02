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

Q3_START = fd.model_seconds_remaining(3, 12 * 60)


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
    assert set(fd.evaluate_site("home", 40, 5, 0, Q3_START)["wp"]) == {"Go for it", "Field goal", "Punt"}


def test_timeouts_swap_after_change_of_possession():
    """Our offense with 1 timeout vs their 3: after a punt THEY have the ball
    with 3 and we have 1. Swapping our counts should be a mirror image."""
    r = fd.evaluate_many(70, 8, 0, Q3_START, off_timeouts=1, def_timeouts=3, is_home_pos=1)
    wp_opp, _ = mu.predict_wp_chained_batch(0, 100 - (70 - fd.DEEP_PUNT_NET), 1, 10,
                                            fd.half_seconds_from_game(Q3_START), Q3_START,
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
