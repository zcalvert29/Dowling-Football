"""
Score each WP model on the clean 2025 holdout, then run the app's 4th-down engine with it.

Read the LATE-GAME columns before shipping a model, not just overall log loss. Most plays happen
when the game is far from decided, so the overall number can hide a model that's worse exactly where
4th-down calls are made: the last minutes of close games. (That's how an earlier version slipped
through: 0.2% worse overall, but clearly worse late, and it turned a strong go into a toss-up.)
  late log loss   - last 6 min (model clock) of one-score games (1-8 points)
  up 6, they have ball - leading team's win rate when up 6 and the trailing team has the ball late;
                         compare to the "actual" row
  goal-line rises - of 300 random situations, how many have win probability rising by 1+ point
                    somewhere as the offense moves AWAY from the goal (should be near 0)
"""
import os, sys, numpy as np, pandas as pd, xgboost as xgb
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
import model_utils as mu, fourth_down_core as fd
from train import FEATS
from prep import DATA_DIR

test = pd.read_parquet(os.path.join(DATA_DIR, "test_2025_clean.parquet"))
X = xgb.DMatrix(test[FEATS].to_numpy(np.float32), feature_names=FEATS); y = test["label_win"].to_numpy()
MODELS = {"deployed (cfb_wp_model_truth.json)": os.path.join(REPO, "cfb_wp_model_truth.json"),
          **{label: os.path.join(DATA_DIR, f) for label, f in [
              ("replica", "wp_replica.json"), ("kickoff fix", "wp_ko_fixed.json"),
              ("kickoff fix + all monotone", "wp_ko_fixed_mono.json"), ("depth 6", "wp_d6.json"),
              ("depth 6 + yards_to_goal only", "wp_d6_ytg.json"),
              ("depth 6 + field-position features (shipped)", "wp_shipped.json")]}}
MODELS = {k: p for k, p in MODELS.items() if os.path.exists(p)}

def load(p):
    b = xgb.Booster(); b.load_model(p); return b

def ece(p, y, bins=20):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(p[idx == k].mean() - y[idx == k].mean()) * (idx == k).mean() for k in range(bins) if (idx == k).any())

LATE = ((test["game_seconds_remaining"] <= 360) & test["score_diff"].abs().between(1, 8)).to_numpy()
UP6 = LATE & (test["score_diff"] == -6).to_numpy()  # offense down 6, so the leader is on defense


def logloss(p, m=None):
    yy, pp = (y, p) if m is None else (y[m], p[m])
    pp = np.clip(pp, 1e-6, 1 - 1e-6)
    return float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)))


Q3 = fd.model_seconds_remaining(3, 720)
Y, D, S, T = np.meshgrid(np.arange(15, 100, 2), np.arange(1, 11), np.arange(-21, 22, 7), [3000, 2000, 1200, 600], indexing="ij")
keep = D <= Y - 10
rows = []
for name, path in MODELS.items():
    b = load(path)
    p = b.predict(X)
    ll = logloss(p)
    mu._wp_booster = b  # the app's engine now uses this model
    r = fd.evaluate_many(Y[keep], D[keep], S[keep], T[keep], is_home_pos=0.5)
    conv_worse = float((r["wp_go_success"] < r["wp_go_fail"]).mean())
    # field position: opponent's ball, tie game, Q3 start; WP should fall as they get closer to our goal
    opp_wp, _ = mu.predict_wp_chained_batch(0, np.arange(1, 100), 1, np.minimum(10, np.arange(1, 100)),
                                            fd.half_seconds_from_game(Q3), Q3, is_home_pos=0)
    fp_viol = int((np.diff(opp_wp) > 0.002).sum())  # rises by >0.2 pts while moving AWAY from the goal
    lead35, _ = mu.predict_wp_chained_batch(35, 75, 1, 10, 120, 120, is_home_pos=0)
    rng = np.random.default_rng(0)
    rises = []
    for _ in range(300):
        ytg = np.arange(1, 100)
        wp, _ = mu.predict_wp_chained_batch(int(rng.integers(-21, 22)), ytg, int(rng.integers(1, 5)), np.minimum(10, ytg),
                                            int(rng.integers(30, 1800)), int(rng.integers(30, 3600)),
                                            is_home_pos=int(rng.integers(0, 2)))
        rises.append(np.diff(wp).max())
    rises = np.array(rises)
    rows.append({"model": name, "log loss": ll, "late log loss": logloss(p, LATE),
                 "up 6, they have ball": float((1 - p[UP6]).mean()), "Brier": np.mean((p - y) ** 2),
                 "calib. error": ece(p, y),
                 "convert < fail": conv_worse, "field-pos reversals": fp_viol,
                 "goal-line rises": int((rises > 0.01).sum()), "worst rise (pts)": float(rises.max() * 100), "up 35, 2:00 left": float(lead35[0]),
                 "WP at opp 1 / 10 / 33": " / ".join(f"{opp_wp[i]:.2f}" for i in (0, 9, 32))})
out = pd.DataFrame(rows).set_index("model")
print(f"Actual: leader with a late 6-point lead, trailing team has the ball, won {(1 - y[UP6]).mean():.1%} "
      f"({int(UP6.sum()):,} plays in 2025)\n")
pd.set_option("display.width", 200)
print(out.to_string(formatters={"log loss": "{:.4f}".format, "late log loss": "{:.4f}".format,
                                "up 6, they have ball": "{:.1%}".format, "Brier": "{:.4f}".format,
                                "worst rise (pts)": "{:.1f}".format, "calib. error": "{:.4f}".format,
                                "convert < fail": "{:.1%}".format, "up 35, 2:00 left": "{:.3f}".format}))
