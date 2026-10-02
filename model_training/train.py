"""
Train WP models with cfb_wp_model.R's exact settings (fit 2014-23, early-stop on 2024, 2025 held out).

Three variants, so the effect of each fix can be measured:
  replica        - same rows and settings as the R script (includes the flipped kickoff rows)
  ko_fixed       - kickoff rows dropped
  ko_fixed_mono  - kickoff rows dropped + monotone constraints  <- the model the app ships
The app's cfb_wp_model_truth.json is wp_ko_fixed_mono.json.
"""
import os, sys, time, numpy as np, pandas as pd, xgboost as xgb
from prep import DATA_DIR

FEATS = ["ep", "score_diff", "yards_to_goal", "down1", "down2", "down3", "down4", "distance", "half_seconds",
         "game_seconds_remaining", "off_timeouts", "def_timeouts", "is_home_pos", "score_diff_time_ratio"]
# +1 = higher is better for the offense, -1 = worse, 0 = no constraint. Same order as FEATS.
MONOTONE = (1, 1, -1, 0, 0, 0, 0, -1, 0, 0, 1, -1, 1, 1)

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

def train(name, drop_ko, monotone):
    fit, val, _ = splits()
    f, v = (fit[fit.is_ko == 0], val[val.is_ko == 0]) if drop_ko else (fit, val)
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 4, "eta": 0.03,
              "subsample": 0.8, "colsample_bytree": 0.7, "tree_method": "hist", "nthread": 1, "seed": 42}
    if monotone:
        params["monotone_constraints"] = "(" + ",".join(map(str, MONOTONE)) + ")"
    t = time.time()
    b = xgb.train(params, dm(f), 800, evals=[(dm(v), "val")], early_stopping_rounds=30, verbose_eval=False)
    b.save_model(os.path.join(DATA_DIR, f"wp_{name}.json"))
    print(f"{name}: {len(f):,} fit rows, best iter {b.best_iteration}, {time.time() - t:.0f}s", flush=True)

if __name__ == "__main__":
    fit, val, test = splits()
    print(f"fit {len(fit):,} (kickoff rows {int(fit.is_ko.sum()):,}) · val {len(val):,} · test {len(test):,}", flush=True)
    for name, ko, mono in [("replica", False, False), ("ko_fixed", True, False), ("ko_fixed_mono", True, True)]:
        train(name, ko, mono)
    test.to_parquet(os.path.join(DATA_DIR, "test_2025_clean.parquet"))
    print("DONE", flush=True)
