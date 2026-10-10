"""
conversion_model.py — how often a 4th-down try converts, fit to our own
high school film (no Streamlit code).

The original bot used one hand-drawn curve on distance only (74% on 4th & 1,
50% on 4th & 5, 21% on 4th & 10) for every team. On our tagged 3rd/4th
downs it was too optimistic on short yardage (64% observed on 4th/3rd & 1)
and far too pessimistic on long yardage (22% observed at 11+ yards).

The model here starts from that curve and lets the film correct it:

    logit p = logit(original curve at d) + a + b*(d - 5) + c*own_side(yards to goal)
              + offense[team] + defense[team]

* a, b re-shape the curve to high school football (a shifts its level,
  b flattens or steepens its slope), with a weak ridge
  prior that pulls them toward zero (the original curve) when data is thin.
* c is the field-position effect. own_side() is 0 in the opponent's
  territory and 1 in ours, with a smooth 10-yard ramp around midfield so a
  yard either side of the 50 doesn't flip a call. Our film converts about
  7 points under the curve in our own territory and 5 over it in theirs;
  without this term those cancel and the curve sits in the middle.
  (A goal-to-go term was tried and added nothing on 15 snaps.)
* offense/defense are per-team shifts with a strong ridge prior
  (TEAM_PRIOR_SD), so a team needs a real sample before it moves far
  from average. A team we've never seen gets 0 (league average).

Fit on every 3rd and 4th down run or pass at 10 yards or less in
curated-pbp.xlsx (both are "move the chains or else" downs; 3rd & long is
left out because offenses often aren't trying to convert). Refit whenever
games are added.
"""
import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

# Ridge prior standard deviations, in log-odds.
CURVE_PRIOR_SD = 1.0   # the shape adjustment (a, b)
TEAM_PRIOR_SD = 0.35
FIELD_PRIOR_SD = 0.5    # the own-territory effect   # each team's offense or defense shift (~±8 pts of conversion rate at 50%)
MAX_DIST = 10   # the range where 4th-down decisions live; 3rd & long is a different game


def own_side(yards_to_goal):
    """0 in the opponent's territory, 1 in ours, smooth across midfield (~0.5 at the 50)."""
    return 1 / (1 + np.exp(-(np.asarray(yards_to_goal, dtype=float) - 50) / 5))


def college_curve_logit(distance):
    """The original hand-calibrated curve, in log-odds: p = 1 / (1 + exp(0.262 * (d - 5)))."""
    d = np.maximum(np.asarray(distance, dtype=float), 0.5)
    return -0.262 * (d - 5.0)


@dataclass
class ConversionModel:
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    offense: dict = field(default_factory=dict)
    defense: dict = field(default_factory=dict)
    n_plays: int = 0

    def team_shift(self, offense=None, defense=None):
        """Log-odds shift for this offense against this defense (0 for teams not in the data)."""
        return self.offense.get(offense, 0.0) + self.defense.get(defense, 0.0)

    def prob(self, distance, offense=None, defense=None, yards_to_goal=50):
        d = np.maximum(np.asarray(distance, dtype=float), 0.5)
        z = (college_curve_logit(d) + self.a + self.b * (d - 5.0) + self.c * own_side(yards_to_goal)
             + self.team_shift(offense, defense))
        return 1 / (1 + np.exp(-z))


def conversion_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """3rd and 4th down runs and passes, with whether they moved the chains."""
    d = pbp[pbp["DN"].isin([3, 4]) & pbp["PLAY TYPE"].astype(str).isin(["Run", "Pass"])
            & pbp["DIST"].between(1, MAX_DIST)].copy()
    d["converted"] = np.where(d["DN"] == 3, d["THIRD_DOWN_CONVERTED"], d["FOURTH_DOWN_CONVERTED"]).astype(float)
    return d[["offense", "defense", "DN", "DIST", "YARDLINE_100", "converted"]].dropna()


def fit(pbp: pd.DataFrame, curve_prior_sd=CURVE_PRIOR_SD, team_prior_sd=TEAM_PRIOR_SD) -> ConversionModel:
    """Ridge-penalized logistic regression by Newton's method (numpy only)."""
    d = conversion_plays(pbp)
    if len(d) == 0:
        return ConversionModel()
    offs, defs = sorted(d["offense"].unique()), sorted(d["defense"].unique())
    dist = d["DIST"].to_numpy(float)
    X = np.column_stack([np.ones(len(d)), dist - 5.0, own_side(d["YARDLINE_100"].to_numpy(float)),
                         (d["offense"].to_numpy()[:, None] == np.array(offs)).astype(float),
                         (d["defense"].to_numpy()[:, None] == np.array(defs)).astype(float)])
    y = d["converted"].to_numpy(float)
    offset = college_curve_logit(dist)
    prec = np.r_[[1 / curve_prior_sd ** 2] * 2, 1 / FIELD_PRIOR_SD ** 2,
                 [1 / team_prior_sd ** 2] * (len(offs) + len(defs))]

    beta = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-(offset + X @ beta)))
        grad = X.T @ (y - p) - prec * beta
        hess = (X * (p * (1 - p))[:, None]).T @ X + np.diag(prec)
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    k = len(offs)
    return ConversionModel(a=float(beta[0]), b=float(beta[1]), c=float(beta[2]),
                           offense=dict(zip(offs, map(float, beta[3:3 + k]))),
                           defense=dict(zip(defs, map(float, beta[3 + k:]))),
                           n_plays=len(d))


def calibration_table(pbp: pd.DataFrame, model: ConversionModel, by="distance") -> pd.DataFrame:
    """Observed vs. predicted conversion rate by distance bucket or field zone (for checking the fit)."""
    d = conversion_plays(pbp)
    d["college_curve"] = 1 / (1 + np.exp(-college_curve_logit(d["DIST"])))
    d["model"] = model.prob(d["DIST"], None, None, d["YARDLINE_100"])
    if by == "zone":
        d["bucket"] = pd.cut(d["YARDLINE_100"], [0, 20, 50, 80, 100],
                             labels=["their 1-20", "their 21-50", "our 21-49", "our 1-20"])
    else:
        d["bucket"] = pd.cut(d["DIST"], [0, 1, 2, 3, 5, 7, 10], labels=["1", "2", "3", "4-5", "6-7", "8-10"])
    return d.groupby("bucket", observed=True).agg(
        plays=("converted", "size"), observed=("converted", "mean"),
        original_curve=("college_curve", "mean"), high_school_curve=("model", "mean")).reset_index()


DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curated-pbp.xlsx")


@lru_cache(maxsize=4)
def _fit_cached(path, mtime):
    try:
        return fit(pd.read_excel(path))
    except Exception:
        return ConversionModel()


def fitted(path=DATA_PATH) -> ConversionModel:
    """The model fit to the data file, refit automatically when the file changes.
    The 4th Down Bot and the 4th-down review both use this, so they agree."""
    try:
        return _fit_cached(path, os.path.getmtime(path))
    except OSError:
        return ConversionModel()
