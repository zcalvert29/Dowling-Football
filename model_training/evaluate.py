"""Score each WP model on the clean 2025 holdout, then run the app's 4th-down engine with it."""
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
              ("kickoff fix + monotone", "wp_ko_fixed_mono.json")]}}
MODELS = {k: p for k, p in MODELS.items() if os.path.exists(p)}

def load(p):
    b = xgb.Booster(); b.load_model(p); return b

def ece(p, y, bins=20):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(p[idx == k].mean() - y[idx == k].mean()) * (idx == k).mean() for k in range(bins) if (idx == k).any())

Q3 = fd.model_seconds_remaining(3, 720)
Y, D, S, T = np.meshgrid(np.arange(15, 100, 2), np.arange(1, 11), np.arange(-21, 22, 7), [3000, 2000, 1200, 600], indexing="ij")
keep = D <= Y - 10
rows = []
for name, path in MODELS.items():
    b = load(path)
    p = b.predict(X)
    ll = -np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    mu._wp_booster = b  # the app's engine now uses this model
    r = fd.evaluate_many(Y[keep], D[keep], S[keep], T[keep], is_home_pos=0.5)
    conv_worse = float((r["wp_go_success"] < r["wp_go_fail"]).mean())
    # field position: opponent's ball, tie game, Q3 start; WP should fall as they get closer to our goal
    opp_wp, _ = mu.predict_wp_chained_batch(0, np.arange(1, 100), 1, np.minimum(10, np.arange(1, 100)),
                                            fd.half_seconds_from_game(Q3), Q3, is_home_pos=0)
    fp_viol = int((np.diff(opp_wp) > 0.002).sum())  # rises by >0.2 pts while moving AWAY from the goal
    lead35, _ = mu.predict_wp_chained_batch(35, 75, 1, 10, 120, 120, is_home_pos=0)
    rows.append({"model": name, "log loss": ll, "Brier": np.mean((p - y) ** 2), "calib. error": ece(p, y),
                 "convert < fail": conv_worse, "field-pos reversals": fp_viol, "up 35, 2:00 left": float(lead35[0]),
                 "WP at opp 1 / 10 / 33": " / ".join(f"{opp_wp[i]:.2f}" for i in (0, 9, 32))})
out = pd.DataFrame(rows).set_index("model")
pd.set_option("display.width", 200)
print(out.to_string(formatters={"log loss": "{:.4f}".format, "Brier": "{:.4f}".format, "calib. error": "{:.4f}".format,
                                "convert < fail": "{:.1%}".format, "up 35, 2:00 left": "{:.3f}".format}))
