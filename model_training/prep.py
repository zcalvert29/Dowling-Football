"""
Python replication of cfb_wp_model.R steps 1-4 (filtering, labels, features), with the kickoff fix.

    python model_training/download.py            # cfbfastR play-by-play, 2014-2025 (~1.2 GB)
    python model_training/prep.py 2014 ... 2025  # one model_YYYY.parquet per season
    python model_training/train.py               # fits the WP models
    python model_training/evaluate.py            # compares them on the 2025 holdout

Files live in DATA_DIR (env var, default cfb_data/ next to this folder).
"""
import os, sys, numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cfb_data"))

COLS = ["game_id", "id_play", "period", "down", "distance", "yards_to_goal", "TimeSecsRem", "pos_team",
        "offense_score", "defense_score", "offense_timeouts", "defense_timeouts", "play_type", "season",
        "ep_before", "wp_before", "home", "drive_end_offense_score", "drive_end_defense_score"]
DROP_TYPES = ["Kickoff", "Timeout", "End Period", "End of Half", "End of Game", "Extra Point Good",
              "Extra Point Missed", "Two Point Rush", "Two Point Pass", "Penalty"]


def finals_from_pbp(d):
    """Final score per game. R pulls these from the CFBD API; here they come from the play-by-play.
    Only scrimmage plays (down 1-4, any period including OT) are used, because on kickoffs pos_team and
    the score columns come from different teams' points of view. Each team's final is the highest score
    it reached, counting both the score at the snap and at the end of the drive (so a score on the
    last drive of the game still counts)."""
    s = d[d["down"].isin([1, 2, 3, 4]) & ~d["play_type"].astype(str).str.contains("Kickoff")]
    home_has = s["pos_team"] == s["home"]
    off_hi = np.fmax(s["offense_score"], s["drive_end_offense_score"])
    def_hi = np.fmax(s["defense_score"], s["drive_end_defense_score"])
    f = pd.DataFrame({"game_id": s["game_id"], "h": np.where(home_has, off_hi, def_hi),
                      "a": np.where(home_has, def_hi, off_hi)}).groupby("game_id").max()
    return f.rename(columns={"h": "home_final", "a": "away_final"})


def prep_season(path):
    raw = pd.read_parquet(path, columns=COLS)
    fin = finals_from_pbp(raw)
    d = raw[raw["down"].isin([1, 2, 3, 4]) & (raw["period"] <= 4) & ~raw["play_type"].isin(DROP_TYPES)
            & raw["yards_to_goal"].notna() & raw["TimeSecsRem"].notna() & raw["ep_before"].notna()]
    d = d.sort_values(["game_id", "id_play"]).join(fin, on="game_id")
    d = d[d["home_final"] != d["away_final"]]
    home_won = (d["home_final"] > d["away_final"]).astype(int)
    label = np.where(d["pos_team"] == d["home"], home_won, 1 - home_won)
    trem = d["TimeSecsRem"].astype(float)
    game_secs = np.minimum(trem + (d["period"] <= 2) * 1800, 3600)
    sd_raw = (d["offense_score"] - d["defense_score"]).astype(float)
    out = pd.DataFrame({
        "season": d["season"].astype(int), "label_win": label, "cfb_wp_before": d["wp_before"],
        "ep": d["ep_before"], "score_diff": sd_raw.clip(-50, 50), "yards_to_goal": d["yards_to_goal"],
        **{f"down{k}": (d["down"] == k).astype(float) for k in (1, 2, 3, 4)},
        "distance": np.minimum(d["distance"], 30), "half_seconds": np.minimum(trem, 1800),
        "game_seconds_remaining": game_secs, "off_timeouts": d["offense_timeouts"],
        "def_timeouts": d["defense_timeouts"], "is_home_pos": (d["pos_team"] == d["home"]).astype(int),
        "score_diff_time_ratio": sd_raw / (game_secs / 60 + 1),
        # Kickoff rows carry down=1 but their score is from the kicking team's view while pos_team is the
        # receiver, so score_diff (and the label's perspective) is flipped. The R filter only drops
        # play_type == "Kickoff"; flagged here so training can drop all of them.
        "is_ko": d["play_type"].astype(str).str.contains("Kickoff").astype(int),
    }).dropna(subset=["label_win", "ep", "score_diff", "yards_to_goal", "distance", "half_seconds",
                      "game_seconds_remaining", "off_timeouts", "def_timeouts", "is_home_pos",
                      "score_diff_time_ratio"])
    return out.astype({c: "float32" for c in out.columns if c not in ("season", "is_ko")})


def _prep_one(yr):
    m = prep_season(os.path.join(DATA_DIR, f"pbp_{yr}.parquet"))
    m.to_parquet(os.path.join(DATA_DIR, f"model_{yr}.parquet"))
    late = m[(m.game_seconds_remaining < 120) & (m.score_diff.abs() >= 9) & (m.is_ko == 0)]
    agree = ((late.score_diff > 0) == (late.label_win == 1)).mean()
    print(yr, len(m), f"label sanity (late, 9+ pt lead wins): {agree:.3f}", f"cfb wp vs label corr: "
          f"{np.corrcoef(m.cfb_wp_before.fillna(.5), m.label_win)[0,1]:.3f}")


if __name__ == "__main__":
    for yr in sys.argv[1:] or range(2014, 2026):
        _prep_one(yr)
