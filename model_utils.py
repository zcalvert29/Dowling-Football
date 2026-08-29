"""
model_utils.py — shared model loading, feature engineering, and small
helpers used by both pages of the Streamlit app.

REAL MODELS: loads the trained xgboost models if their JSON files (exported
via export_models_to_json.R) are present in the working directory. If they
aren't found, falls back to the same hand-calibrated heuristic curves used
in the original prototype, so the app still runs for demoing/UI purposes.
Check USING_REAL_MODELS to see which mode is active.

FEATURE ENCODING — this is the part that has to exactly match what R's
model.matrix() produced during training in cfb_ep_model.R / cfb_wp_model.R,
or predictions will be silently wrong (xgboost predicts positionally against
a raw matrix; it won't error on a column-order mismatch, it'll just give you
a confidently wrong number). Both training formulas had `down` as the ONLY
factor variable and no intercept (-1 in the formula) — R's model.matrix()
gives the first (and here, only) factor FULL dummy encoding in that case
(one column per level: down1, down2, down3, down4), not the usual
treatment-contrast k-1 encoding. That's replicated by hand below.

IF PREDICTIONS LOOK WRONG: the first thing to check is feature order. After
loading a booster, `booster.feature_names` will show you what it was trained
with (if R attached colnames to the training matrix, which model.matrix()
does automatically) — compare that against EP_FEATURE_ORDER / WP_FEATURE_ORDER
below and fix whichever is out of sync.
"""

import os
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

EP_MODEL_PATH = "cfb_ep_model.json"
WP_TRUTH_MODEL_PATH = "cfb_wp_model_truth.json"
WP_CFB_MODEL_PATH = "cfb_wp_model_cfb.json"   # loaded for reference/comparison, not used in recommendations


def _load_booster(path):
    if _XGB_AVAILABLE and os.path.exists(path):
        booster = xgb.Booster()
        booster.load_model(path)
        return booster
    return None


_ep_booster = _load_booster(EP_MODEL_PATH)
_wp_booster = _load_booster(WP_TRUTH_MODEL_PATH)
_wp_cfb_booster = _load_booster(WP_CFB_MODEL_PATH)

USING_REAL_MODELS = _ep_booster is not None and _wp_booster is not None


# ==============================================================================
# EP MODEL — feature order matches cfb_ep_model.R's Model A formula:
#   ~ yards_to_goal + down + distance + half_seconds + score_diff +
#     off_timeouts + def_timeouts + under_2min + goal_to_go - 1
# ==============================================================================
EP_FEATURE_ORDER = ["yards_to_goal", "down1", "down2", "down3", "down4",
                     "distance", "half_seconds", "score_diff",
                     "off_timeouts", "def_timeouts", "under_2min", "goal_to_go"]


def _down_dummies(down):
    return [1 if down == d else 0 for d in (1, 2, 3, 4)]


def build_ep_features(yards_to_goal, down, distance, half_seconds, score_diff,
                       off_timeouts=3, def_timeouts=3):
    under_2min = 1 if half_seconds <= 120 else 0
    goal_to_go = 1 if distance >= yards_to_goal else 0
    row = [yards_to_goal, *_down_dummies(down), distance, half_seconds,
           score_diff, off_timeouts, def_timeouts, under_2min, goal_to_go]
    return pd.DataFrame([row], columns=EP_FEATURE_ORDER)


def _heuristic_ep(yards_to_goal, **_ignored):
    """Fallback used only when the real model isn't available. Same
    field-position-only curve as the original prototype."""
    anchor_ytg = [99, 90, 75, 60, 50, 40, 25, 10, 5, 1]
    anchor_ep = [-1.6, -0.9, 0.4, 1.2, 2.0, 2.7, 3.7, 4.9, 5.5, 6.4]
    return float(np.interp(yards_to_goal, anchor_ytg[::-1], anchor_ep[::-1]))


def predict_ep(yards_to_goal, down, distance, half_seconds, score_diff,
               off_timeouts=3, def_timeouts=3):
    if _ep_booster is not None:
        feats = build_ep_features(yards_to_goal, down, distance, half_seconds,
                                   score_diff, off_timeouts, def_timeouts)
        dmat = xgb.DMatrix(feats.values, feature_names=EP_FEATURE_ORDER)
        return float(_ep_booster.predict(dmat)[0])
    return _heuristic_ep(yards_to_goal)


# ==============================================================================
# WP MODEL — feature order matches cfb_wp_model.R's formula:
#   ~ ep + score_diff + yards_to_goal + down + distance + half_seconds +
#     game_seconds_remaining + off_timeouts + def_timeouts + is_home_pos +
#     score_diff_time_ratio - 1
# ==============================================================================
WP_FEATURE_ORDER = ["ep", "score_diff", "yards_to_goal", "down1", "down2",
                     "down3", "down4", "distance", "half_seconds",
                     "game_seconds_remaining", "off_timeouts", "def_timeouts",
                     "is_home_pos", "score_diff_time_ratio"]


def build_wp_features(ep, score_diff, yards_to_goal, down, distance,
                       half_seconds, game_seconds_remaining,
                       off_timeouts=3, def_timeouts=3, is_home_pos=0):
    score_diff_time_ratio = score_diff / (game_seconds_remaining / 60 + 1)
    row = [ep, score_diff, yards_to_goal, *_down_dummies(down), distance,
           half_seconds, game_seconds_remaining, off_timeouts, def_timeouts,
           is_home_pos, score_diff_time_ratio]
    return pd.DataFrame([row], columns=WP_FEATURE_ORDER)


def _heuristic_wp(score_diff, game_seconds_remaining, ep, **_ignored):
    """Fallback used only when the real model isn't available."""
    minutes_left = max(game_seconds_remaining / 60, 0.1)
    time_adj_margin = score_diff / np.sqrt(minutes_left + 1)
    logit = 0.42 * time_adj_margin + 0.05 * ep
    return float(1 / (1 + np.exp(-logit)))


def predict_wp(ep, score_diff, yards_to_goal, down, distance, half_seconds,
               game_seconds_remaining, off_timeouts=3, def_timeouts=3,
               is_home_pos=0, use_cfb_model=False):
    booster = _wp_cfb_booster if use_cfb_model else _wp_booster
    if booster is not None:
        feats = build_wp_features(ep, score_diff, yards_to_goal, down, distance,
                                   half_seconds, game_seconds_remaining,
                                   off_timeouts, def_timeouts, is_home_pos)
        dmat = xgb.DMatrix(feats.values, feature_names=WP_FEATURE_ORDER)
        return float(booster.predict(dmat)[0])
    return _heuristic_wp(score_diff, game_seconds_remaining, ep)


def predict_wp_chained(score_diff, yards_to_goal, down, distance, half_seconds,
                        game_seconds_remaining, off_timeouts=3, def_timeouts=3,
                        is_home_pos=0):
    """The main entry point most callers want: computes EP first, then feeds
    it into the WP model, exactly the way cfbfastR's own pipeline chains the
    two (WP model takes EP as an input feature)."""
    ep = predict_ep(yards_to_goal, down, distance, half_seconds, score_diff,
                     off_timeouts, def_timeouts)
    wp = predict_wp(ep, score_diff, yards_to_goal, down, distance, half_seconds,
                     game_seconds_remaining, off_timeouts, def_timeouts, is_home_pos)
    return wp, ep


# ==============================================================================
# SMALL SHARED HELPERS (used by both pages)
# ==============================================================================

def strength_tier(margin_pts):
    """margin_pts = WP gain (percentage points) of the recommended option
    over the next-best. Tiers follow nfl4th's documented go_boost cutoffs."""
    if margin_pts >= 10:
        return "VERY STRONG"
    elif margin_pts >= 4:
        return "STRONG"
    elif margin_pts >= 1:
        return "LEAN"
    else:
        return "TOSS-UP"


def score_category(deficit):
    """Describes a point deficit in terms of what kind of score closes it —
    this is the actionable distinction (can a made FG do it, or do you need
    a TD), not just a possession count.
    3 = the FG cutoff (a made field goal ties or takes the lead).
    8 = the ceiling for a single scoring drive (TD + successful 2-pt try)."""
    if deficit <= 0:
        return "leading or tied"
    elif deficit <= 3:
        return "field goal range \u2014 a made FG ties or takes the lead"
    elif deficit <= 8:
        return "touchdown range \u2014 needs at least a TD to tie or take the lead"
    else:
        return "multi-score range"
