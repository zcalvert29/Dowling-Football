"""
Train WP models with cfb_wp_model.R's exact settings (fit 2014-23, early-stop on 2024, 2025 held out).

Variants, so the effect of each change can be measured:
  replica        - same rows and settings as the R script (includes the flipped kickoff rows)
  ko_fixed       - kickoff rows dropped
  ko_fixed_mono  - kickoff rows dropped + monotone constraints on every directional feature
  d6             - kickoff rows dropped, depth 6
  d6_ytg         - depth 6, yards_to_goal constrained only
  shipped        - depth 6, the three field-position features constrained (EP, yards_to_goal,
                   distance); score and clock left free  <- the model the app ships
The app's cfb_wp_model_truth.json is wp_shipped.json.

Why this one:
  * Constraining EVERY directional feature (ko_fixed_mono) cost little overall but made late,
    one-score games clearly worse (2025 late-game log loss 0.4655 vs ~0.458), exactly where 4th-down
    calls are decided. The damage comes from constraining score and clock, so those stay free.
  * Constraining yards_to_goal alone (d6_ytg) wasn't enough: inside the 10, distance and EP change
    along with field position, so win probability could still rise moving away from the goal
    (+5.7 pts from 2nd & goal at the 3 to the 4). Adding EP and distance fixes that at no measurable
    cost (2024 validation late-game log loss 0.4580 for both).
Always check evaluate.py's late-game and goal-line columns, not just overall log loss, before shipping.
"""
import os, sys, time, numpy as np, pandas as pd, xgboost as xgb
from prep import DATA_DIR

FEATS = ["ep", "score_diff", "yards_to_goal", "down1", "down2", "down3", "down4", "distance", "half_seconds",
         "game_seconds_remaining", "off_timeouts", "def_timeouts", "is_home_pos", "score_diff_time_ratio"]
# +1 = higher is better for the offense, -1 = worse, 0 = no constraint. Same order as FEATS.
MONOTONE_ALL = (1, 1, -1, 0, 0, 0, 0, -1, 0, 0, 1, -1, 1, 1)
MONOTONE_YTG_ONLY = (0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
MONOTONE_FIELD_POSITION = (1, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0)  # EP, yards_to_goal, distance: the shipped model

_SPLITS = {}


def splits():
    if not _SPLITS:
        data = pd.concat([pd.read_parquet(os.path.join(DATA_DIR, f"model_{y}.parquet")) for y in range(2014, 2026)],
                         ignore_index=True)
        _SPLITS["fit"] = data[data.season.between(2014, 2023)]
        _SPLITS["val"] = data[data.season == 2024]
        _SPLITS["test"] = data[(data.season == 2025) & (data.is_ko == 0)]  # every model is scored on clean rows
    return _SPLITS["fit"], _SPLITS["val"], _SPLITS["test"]

def dm(d):
    return xgb.DMatrix(d[FEATS].to_numpy(np.float32), label=d["label_win"].to_numpy(), feature_names=FEATS)

def train(name, drop_ko, monotone=None, depth=4):
    """monotone: None or a tuple like MONOTONE_FIELD_POSITION. depth 4 = the R script's setting."""
    fit, val, _ = splits()
    f, v = (fit[fit.is_ko == 0], val[val.is_ko == 0]) if drop_ko else (fit, val)
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": depth, "eta": 0.03,
              "subsample": 0.8, "colsample_bytree": 0.7, "tree_method": "hist", "nthread": 1, "seed": 42}
    if monotone:
        params["monotone_constraints"] = "(" + ",".join(map(str, monotone)) + ")"
    t = time.time()
    b = xgb.train(params, dm(f), 1500, evals=[(dm(v), "val")], early_stopping_rounds=30, verbose_eval=False)
    b.save_model(os.path.join(DATA_DIR, f"wp_{name}.json"))
    print(f"{name}: {len(f):,} fit rows, best iter {b.best_iteration}, {time.time() - t:.0f}s", flush=True)

if __name__ == "__main__":
    fit, val, test = splits()
    print(f"fit {len(fit):,} (kickoff rows {int(fit.is_ko.sum()):,}) · val {len(val):,} · test {len(test):,}", flush=True)
    variants = [("replica", False, None, 4), ("ko_fixed", True, None, 4), ("ko_fixed_mono", True, MONOTONE_ALL, 4),
                ("d6", True, None, 6), ("d6_ytg", True, MONOTONE_YTG_ONLY, 6),
                ("shipped", True, MONOTONE_FIELD_POSITION, 6)]
    only = sys.argv[1:]  # e.g. `python train.py shipped` to rebuild just the app's model
    for name, ko, mono, depth in variants:
        if not only or name in only:
            train(name, ko, mono, depth)
    test.to_parquet(os.path.join(DATA_DIR, "test_2025_clean.parquet"))
    print("DONE", flush=True)
