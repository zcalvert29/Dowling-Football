import os
from typing import Optional
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import re


# ---------------------------------------------------------------------------
# PLAY TYPE groupings (from the filming team's perspective, i.e. `team`)
# ---------------------------------------------------------------------------
KICKOFF_TYPES = {"KO", "KO Rec"}          # KO = team kicks off, KO Rec = team receives
PUNT_TYPES = {"Punt", "Punt Rec"}
XP_TYPES = {"Extra Pt.", "Extra Pt. Block"}
TWO_PT_TYPES = {"2 Pt.", "2 Pt. Block", "2 Pt. Defend"}
FG_TYPES = {"FG", "FG Block"}

# Kicking plays that don't represent a live down/distance situation. They
# get fixed ep values instead of the down/distance formula (see
# calculate_ep_epa), and they're excluded from the SERIES increment logic.
# Anything else with ODK == "K" (punts, fake punts, field goals, blocked
# kicks, etc.) still has a real down/distance/field-position context and
# gets a formula ep value.
NON_SCRIMMAGE_KICK_TYPES = KICKOFF_TYPES | XP_TYPES | TWO_PT_TYPES

# Fixed ep for conversion tries, from the trying team's perspective.
XP_EP = 0.9
TWO_PT_EP = 1.0

# Points for the touchdown itself, not counting the try that follows.
TD_POINTS = 6.0


def assign_offense_defense(team, opponent, plays):
    """
    Add 'offense' and 'defense' columns to the plays DataFrame based on
    'ODK' and, for kicking plays, 'PLAY TYPE'.

    Kickoffs are assigned directly from the PLAY TYPE, since it already
    says which side kicked:
      - "KO"     -> `team` kicked, so `opponent` is the offense (receiving)
      - "KO Rec" -> `team` received, so `team` is the offense
    That means the opening kickoff and second-half kickoff no longer need
    to be seeded manually.

    PATs and 2-point tries are also assigned from the PLAY TYPE: "Extra Pt." /
    "2 Pt." are `team`'s tries, and the Block/Defend versions are the
    opponent's. (Copying the previous offense credited the PAT after a
    defensive touchdown to the team that just gave up the touchdown.)

    Every other kicking play (punts, fake punts, FGs, and blocks of those)
    belongs to the team that was on offense immediately before it. "Punt Rec" is included there: its DN/DIST always
    continues the kicking team's stalled drive (e.g. DN=4, DIST=17
    matching their prior 3rd-and-13), so it stays attributed to the team
    that was actually driving, not the team about to receive.

    An unrecognized 'K' PLAY TYPE also inherits the previous offense, but
    prints a warning so it can be added to the list explicitly.

    Rows with ODK == "S" and rows with RESULT == "Timeout" are dropped
    before anything else, so they never reach the later steps.
    """
    df = plays.copy()
    # Drop rows that aren't real plays: ODK == "S" rows, and timeouts
    # (which would otherwise count as 0-yard plays and can start phantom series).
    df = df[(df["ODK"] != "S") & (df["RESULT"] != "Timeout")].reset_index(drop=True)

    SAME_AS_PREV_OFFENSE = {
        "Punt", "Punt Rec", "Fake Punt", "FG", "FG Block",
    }
    # Tries are named from the filming team's side, like kickoffs: "Extra Pt." /
    # "2 Pt." = team's try, the Block/Defend versions = opponent's try. Using
    # the name (not the previous play) matters after a defensive touchdown,
    # when the team that scored is the team that was just on defense.
    OUR_TRIES = {"Extra Pt.", "2 Pt."}
    THEIR_TRIES = {"Extra Pt. Block", "2 Pt. Block", "2 Pt. Defend"}

    offense_col, defense_col = [], []
    prev_offense, prev_defense = None, None

    for _, row in df.iterrows():
        odk = row["ODK"]
        if odk == "O":
            off, defn = team, opponent
        elif odk == "D":
            off, defn = opponent, team
        elif odk == "K":
            pt = row["PLAY TYPE"]
            if pt == "KO":            # team kicks, opponent receives
                off, defn = opponent, team
            elif pt == "KO Rec":      # team receives
                off, defn = team, opponent
            elif pt in OUR_TRIES:
                off, defn = team, opponent
            elif pt in THEIR_TRIES:
                off, defn = opponent, team
            elif pt in SAME_AS_PREV_OFFENSE:
                off, defn = prev_offense, prev_defense
            else:
                print(f"WARNING: unrecognized K play type {pt!r} at PLAY # {row['PLAY #']}; "
                      f"using previous offense")
                off, defn = prev_offense, prev_defense
        else:
            off, defn = None, None

        offense_col.append(off)
        defense_col.append(defn)
        if off is not None:
            prev_offense, prev_defense = off, defn

    df["offense"] = offense_col
    df["defense"] = defense_col

    return df


def add_success_metrics(plays):
    """
    Add 'success' and 'explosive_play' columns to a plays DataFrame.

    success:
        1st down: gain >= 40% of DIST
        2nd down: gain >= 70% of DIST
        3rd/4th down: gain >= DIST

    explosive_play:
        1 if a rush ("Run" in PLAY TYPE) gains 10+ yards, or a pass
        ("Pass" in PLAY TYPE) gains 20+ yards; 0 otherwise.
        None for kicking plays (ODK == "K").

    Parameters
    ----------
    plays : pd.DataFrame
        Must contain columns 'DN', 'DIST', 'GN/LS', 'ODK', and 'PLAY TYPE'.

    Returns
    -------
    pd.DataFrame
        A copy of `plays` with the two new columns added.
    """
    df = plays.copy()

    def _get_success(row):
        if row["ODK"] == "K":
            return None

        dn = row["DN"]
        dist = row["DIST"]
        gain = row["GN/LS"]

        if pd.isna(dn) or pd.isna(dist) or pd.isna(gain):
            return None

        if dn == 1:
            if dist == 0:
                return None
            return 1 if gain >= 0.40 * dist else 0
        elif dn == 2:
            if dist == 0:
                return None
            return 1 if gain >= 0.70 * dist else 0
        elif dn in (3, 4):
            return 1 if gain >= dist else 0
        else:
            return None  # unexpected DN value

    df["success"] = df.apply(_get_success, axis=1)

    is_kick = df["ODK"] == "K"
    is_rush = df["PLAY TYPE"].str.contains("Run", na=False)
    is_pass = df["PLAY TYPE"].str.contains("Pass", na=False)

    explosive = (
        (is_rush & (df["GN/LS"] >= 10))
        | (is_pass & (df["GN/LS"] >= 20))
    )

    df["explosive_play"] = np.where(is_kick, None, explosive.astype(int))

    return df


def add_score_columns(plays, team, opponent):
    """
    Add cumulative 'team_score' and 'opponent_score' columns, plus
    'offense_score' and 'defense_score', to the plays DataFrame by
    processing plays in order of 'PLAY #'.

    Scoring handled: offensive TD (6), defensive TD (6), extra point (1),
    two-point conversion (2), field goal (3), safety (2). For XP, 2-point,
    and FG rows (including the "Block"/"Defend" versions, which are the
    other team's attempt), RESULT == "Good" means the attempt succeeded
    and the points go to that row's offense.

    'offense_score' / 'defense_score' just mirror 'team_score' /
    'opponent_score' depending on which side is listed in 'offense' for
    that row, so the same play's score can be read from either team's
    perspective without a lookup elsewhere.

    Parameters
    ----------
    plays : pd.DataFrame
        Must contain columns 'PLAY #', 'RESULT', 'PLAY TYPE', 'offense', 'defense'.
    team : str
    opponent : str

    Returns
    -------
    pd.DataFrame
        A copy of `plays`, sorted by 'PLAY #' ascending, with 'team_score',
        'opponent_score', 'offense_score', and 'defense_score' columns
        added (running totals as of each play).
    """
    df = plays.sort_values("PLAY #", ascending=True).reset_index(drop=True)

    team_score = 0
    opponent_score = 0

    team_scores_col = []
    opponent_scores_col = []
    offense_scores_col = []
    defense_scores_col = []

    def credit(side, points):
        nonlocal team_score, opponent_score
        if side == team:
            team_score += points
        elif side == opponent:
            opponent_score += points

    for _, row in df.iterrows():
        result = row["RESULT"] if pd.notna(row["RESULT"]) else ""
        play_type = row["PLAY TYPE"]
        offense = row["offense"]
        defense = row["defense"]

        # Defensive touchdown (pick-six, fumble return, etc.)
        if "Def TD" in result:
            credit(defense, 6)
        # Offensive touchdown (only if it's not a "Def TD")
        elif "TD" in result:
            credit(offense, 6)

        # Extra point / two-point conversion / field goal (incl. the
        # blocked/defended versions, credited to that row's offense when Good)
        if result == "Good":
            if play_type in XP_TYPES:
                credit(offense, 1)
            elif play_type in TWO_PT_TYPES:
                credit(offense, 2)
            elif play_type in FG_TYPES:
                credit(offense, 3)

        # Safety
        if "Safety" in result:
            credit(defense, 2)

        team_scores_col.append(team_score)
        opponent_scores_col.append(opponent_score)

        # offense_score/defense_score just mirror whichever side is on
        # offense for this row, using the running totals above.
        if offense == team:
            offense_scores_col.append(team_score)
            defense_scores_col.append(opponent_score)
        elif offense == opponent:
            offense_scores_col.append(opponent_score)
            defense_scores_col.append(team_score)
        else:
            offense_scores_col.append(np.nan)
            defense_scores_col.append(np.nan)

    df["team_score"] = team_scores_col
    df["opponent_score"] = opponent_scores_col
    df["offense_score"] = offense_scores_col
    df["defense_score"] = defense_scores_col

    return df


def calculate_ep_epa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add YARDLINE_100, ep, and epa columns to a dataframe of high school
    football plays.

    Expected columns on input (run add_score_columns first to get
    offense_score/defense_score):
        PLAY #         - play order within the game
        QTR            - quarter, used for the Q4 blowout dampener
        ODK            - "O", "D", or "K" (offense / defense / kicking play)
        PLAY TYPE      - e.g. "Run", "Pass", "Punt", "FG", "KO", ...
        DN             - down (1-4)
        DIST           - yards to go for a first down
        YARD LN        - raw yard line, positive on own side / negative on
                         opponent's side, per your data convention
        offense_score  - current offense's running score (see add_score_columns)
        defense_score  - current defense's running score (see add_score_columns)
        offense        - which team is on offense this play (used to
                         detect possession changes for epa)
        RESULT         - play result text (used for scoring detection
                         and the interception/fumble/turnover flags)

    Plays with no real down/distance situation get fixed ep values
    instead of the formula:
      - Extra points (incl. "Extra Pt. Block"): ep = XP_EP (0.9)
      - Two-point tries (incl. "2 Pt. Block" / "2 Pt. Defend"): ep = TWO_PT_EP (1.0)
      - Kickoffs ("KO" / "KO Rec"): 1st & 10 at the -25 (yardline_100 = 75),
        the receiving team's typical starting field position.
    Punts, fake punts, field goals, and blocked field goals still use the
    formula, since those are the "what happens on 4th down" plays.

    A Q4/blowout dampener is applied: if it's the 4th quarter and the
    score margin is more than 21, ep is scaled down (garbage-time plays
    swing the scoreboard less than the formula would otherwise suggest).

    The -2 floor is skipped for punts specifically (PLAY TYPE == "Punt"
    or "Punt Rec"), so a punt's pre-kick ep can go below -2 rather than
    clipping every bad 4th-and-long situation to the same value — this
    keeps punt epa reflecting the actual field-position swing instead of
    the clip.

    epa is calculated one of two ways, in PLAY # order:
      - If the play ends in a score or is a conversion try, epa = the
        point value of that play minus this play's ep. Point values:
          * Touchdown: 6 + the ep of the try that follows it (0.9 for an
            extra point, 1.0 for a two-point try; 0.9 if no try row
            follows). This way a TD's epa plus its try's epa always adds
            up to the points actually scored minus the pre-snap ep,
            instead of double-counting the try.
          * Defensive TD: the negative of the same value (the defense
            scores and then gets the try).
          * Extra point: 1 if Good, else 0  -> epa = +0.1 or -0.9
          * Two-point try: 2 if Good, else 0 -> epa = +1.0 or -1.0
          * A try with RESULT "Penalty" that's followed by another try is
            treated as a non-scoring play (next ep - this ep), so the
            re-try carries the make/miss.
          * Safety: -2. Made field goal / FG block: 3.
      - Otherwise, epa = the next play's ep minus this play's ep, with
        the next play's ep NEGATED first if possession changed hands
        (offense differs between this row and the next row) — covering
        interceptions, lost fumbles, turnovers on downs, and punts alike,
        without needing to check RESULT for each case individually.
    epa is left as NaN when it's the last play in the dataframe or when
    the next play's ep is NaN (e.g. end of half/game).

    interception / fumble / turnover are binary (1/0) flags read off the
    RESULT column: interception = 1 if RESULT contains "Interception",
    fumble = 1 if RESULT contains "Fumble", and turnover = 1 if either
    of those is 1 (e.g. a fumble that's recovered by the offense still
    sets fumble = 1 and turnover = 1, since it doesn't distinguish a lost
    fumble from one the offense kept — narrow the RESULT match if you
    need to exclude recovered fumbles).
    """
    df = df.copy()

    # --- YARDLINE_100: distance from the opponent's goal line, 1-99 ---
    df["YARDLINE_100"] = np.where(
        df["YARD LN"] > 0,
        df["YARD LN"],
        100 + df["YARD LN"],
    )

    # --- interception / fumble / turnover flags, from RESULT ---
    result_text = df["RESULT"].fillna("")
    df["interception"] = result_text.str.contains("Interception").astype(int)
    df["fumble"] = result_text.str.contains("Fumble").astype(int)
    df["turnover"] = ((df["interception"] == 1) | (df["fumble"] == 1)).astype(int)

    # --- EP formula pieces, calibrated against cfbfastR's published
    # EP-by-field-position-and-down table (distance = 10, tied game).
    # Field curve fixed at a -2 floor / 7 ceiling so it stays interpretable.
    def ep_field(yardline_100):
        return -2 + 9 / (1 + np.exp(0.037 * (yardline_100 - 50)))

    def down_penalty(down, dist, yardline_100):
        # Extra cost for downs beyond 1st, fit per-down against cfbfastR's
        # actual gaps rather than assumed as one flat per-down constant.
        # 2nd down is close to a flat tax; 3rd/4th taper off near the goal
        # line since field-goal range still has value even on a bad down.
        if down == 1:
            base = 0.0
        elif down == 2:
            base = 0.62
        elif down == 3:
            base = 1.12 + 0.0184 * yardline_100 - 0.00016 * yardline_100 ** 2
        else:
            base = 1.62 + 0.0735 * yardline_100 - 0.00075 * yardline_100 ** 2

        # cfbfastR's public table only covers distance = 10, so this part
        # is still a rough estimate, not something fit to real data.
        extra_distance = 0.06 * (dist - 10)
        return base + extra_distance

    # Kickoff plays (both "KO" and "KO Rec") get a fixed ep equal to 1st &
    # 10 at the -25 (i.e. the receiving team's typical starting field
    # position), rather than NaN.
    KICKOFF_YARDLINE_100 = 75  # -25 under this dataframe's YARD LN convention

    def compute_ep(row):
        if row["PLAY TYPE"] in KICKOFF_TYPES:
            raw_ep = ep_field(KICKOFF_YARDLINE_100) - down_penalty(
                1, 10, KICKOFF_YARDLINE_100
            )
            if row["QTR"] == 4 and row["_pre_margin"] > 21:
                raw_ep *= 0.6
            return float(np.clip(raw_ep, -2, 7))

        if row["PLAY TYPE"] in XP_TYPES:
            return XP_EP
        if row["PLAY TYPE"] in TWO_PT_TYPES:
            return TWO_PT_EP

        raw_ep = ep_field(row["YARDLINE_100"]) - down_penalty(
            row["DN"], row["DIST"], row["YARDLINE_100"]
        )

        if row["QTR"] == 4 and row["_pre_margin"] > 21:
            raw_ep *= 0.6

        # Punts are usually attempted from already-bad situations (4th &
        # long), which would otherwise all get flattened to the same -2
        # floor. Skipping the floor here lets a punt's pre-kick ep still
        # distinguish a merely-bad spot from a truly awful one, so the
        # resulting epa reflects the actual punt rather than the clip.
        lower_bound = -np.inf if row["PLAY TYPE"] in PUNT_TYPES else -2
        return float(np.clip(raw_ep, lower_bound, 7))

    # offense_score/defense_score are the score AFTER each play, so the
    # blowout check uses the previous row's score (the score at the snap).
    # The absolute margin is the same from either team's perspective.
    df = df.sort_values("PLAY #").reset_index(drop=True)
    df["_pre_margin"] = (df["team_score"] - df["opponent_score"]).abs().shift(1, fill_value=0)
    df["ep"] = df.apply(compute_ep, axis=1)
    df = df.drop(columns="_pre_margin")
    df = df.sort_values("PLAY #").reset_index(drop=True)

    play_types = df["PLAY TYPE"].to_numpy()

    def try_ep_after(i):
        """ep of the conversion try right after a TD (XP_EP if none is logged)."""
        if i + 1 < len(df) and play_types[i + 1] in (XP_TYPES | TWO_PT_TYPES):
            return XP_EP if play_types[i + 1] in XP_TYPES else TWO_PT_EP
        return XP_EP

    # --- score_value: point value of a play that ends the drive (or of a
    # conversion try), signed from that play's own offense's perspective.
    # None if the play doesn't score and isn't a try. ---
    def score_value(i):
        row = df.iloc[i]
        result = row["RESULT"] if pd.notna(row["RESULT"]) else ""
        play_type = row["PLAY TYPE"]

        # A try with a penalty that gets re-attempted isn't a make or a
        # miss yet: fall through to the next-play logic so it's scored by
        # the re-try (epa ~0 here, the re-try carries the result).
        is_try = play_type in (XP_TYPES | TWO_PT_TYPES)
        if is_try and result == "Penalty" and i + 1 < len(df) and play_types[i + 1] in (XP_TYPES | TWO_PT_TYPES):
            return None
        if play_type in XP_TYPES:
            return 1.0 if result == "Good" else 0.0
        if play_type in TWO_PT_TYPES:
            return 2.0 if result == "Good" else 0.0
        if "Def TD" in result:
            return -(TD_POINTS + try_ep_after(i))
        elif "TD" in result:
            return TD_POINTS + try_ep_after(i)
        elif "Safety" in result:
            return -2.0
        elif play_type in FG_TYPES and result == "Good":
            return 3.0
        return None

    # --- EPA ---
    # Scoring plays and conversion tries: epa = point value - ep (see
    # score_value above).
    # Non-scoring plays: epa = next play's ep - this play's ep, with the
    # next play's ep NEGATED if possession changed hands (interception,
    # lost fumble, turnover on downs, punt, etc.) so a turnover reads as
    # a loss of equity for the team that had the ball, not a gain.
    ep_vals = df["ep"].to_numpy()
    offense_vals = df["offense"].to_numpy()
    n = len(df)
    epa_vals = np.full(n, np.nan)

    for i in range(n):
        if pd.isna(ep_vals[i]):
            continue  # no ep for this play -> epa stays NaN

        sv = score_value(i)
        if sv is not None:
            epa_vals[i] = sv - ep_vals[i]
            continue

        if i == n - 1 or pd.isna(ep_vals[i + 1]):
            continue  # last play, or next play has no ep -> leave as NaN

        possession_changed = offense_vals[i] != offense_vals[i + 1]
        next_ep = -ep_vals[i + 1] if possession_changed else ep_vals[i + 1]
        epa_vals[i] = next_ep - ep_vals[i]

    df["epa"] = epa_vals

    return df


DEF_TD_MARK = "Def TD"  # RESULT text when the defense scores on the play ("Interception, Def TD")


def _try_after_defensive_td(df: pd.DataFrame) -> pd.Series:
    """The extra point / 2-point try that follows a defensive touchdown (the scoring team's try, even though
    Hudl often tags it with the same offense as the play before)."""
    res = df["RESULT"].astype("string").fillna("")
    is_try = df["PLAY TYPE"].isin(XP_TYPES | TWO_PT_TYPES)
    last_non_try = res.where(~is_try).ffill().fillna("")
    return is_try & last_non_try.str.contains(DEF_TD_MARK, regex=False)


def _drive_numbers(df: pd.DataFrame) -> pd.Series:
    """
    Running drive count for ONE game, rows in play order. A new drive starts when:
      * 'offense' changes from the previous play,
      * the half turns over (Q2 -> Q3), even if the same team has the ball on both sides of it, or
      * the play after a defensive touchdown and its try. After a pick-six or scoop-and-score the team that
        lost the ball gets it right back on the kickoff, so 'offense' never changes; without this, their next
        possession merged into the drive that ended in the score (one "drive" holding two or three).
    """
    halftime_boundary = (df["QTR"] == 3) & (df["QTR"].shift() == 2)
    res = df["RESULT"].astype("string").fillna("")
    is_try = df["PLAY TYPE"].isin(XP_TYPES | TWO_PT_TYPES).to_numpy()
    def_td = res.str.contains(DEF_TD_MARK, regex=False).to_numpy()
    after = np.zeros(len(df), dtype=bool)
    pending = False
    for i in range(len(df)):
        if pending and not is_try[i]:
            after[i], pending = True, False
        if def_td[i]:
            pending = True
    return ((df["offense"] != df["offense"].shift()) | halftime_boundary
            | pd.Series(after, index=df.index)).cumsum()


def assign_drives(game: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute drive, YDS_NET, and drive_result for ONE already-curated game, in play order. The app runs this
    when it loads curated-pbp.xlsx, so games curated before the defensive-touchdown fix are corrected without
    re-curating them. Needs the columns curation produces (offense, QTR, PLAY TYPE, RESULT, DN, success,
    turnover, GN/LS).
    """
    df = game.reset_index(drop=True).copy()
    df["drive"] = _drive_numbers(df)
    df["YDS_NET"] = df.groupby("drive")["GN/LS"].transform(lambda s: s.fillna(0).cumsum())
    is_kickoff = df["PLAY TYPE"].isin(KICKOFF_TYPES)
    n = len(df)

    def last_play_result(i):
        pt = str(df.at[i, "PLAY TYPE"])
        res = df.at[i, "RESULT"] if pd.notna(df.at[i, "RESULT"]) else ""
        if DEF_TD_MARK in res:
            return "Turnover, Def TD"
        if pt in XP_TYPES or pt in TWO_PT_TYPES or ("TD" in res and DEF_TD_MARK not in res):
            return "Touchdown"
        if pt in FG_TYPES:
            return "FG Attempt"
        if pt in PUNT_TYPES:
            return "Punt"
        if df.at[i, "turnover"] == 1:
            return "Turnover"
        if "Safety" in res:
            return "Safety"
        if df.at[i, "DN"] == 4 and df.at[i, "success"] == 0:
            return "Turnover on Downs"
        if df.at[i, "QTR"] == 2 and i + 1 < n and df.at[i + 1, "QTR"] == 3:
            return "End of Half"
        if df.at[i, "QTR"] == 4 and i == n - 1:
            return "End of Game"
        return "Error"

    plays = df[~is_kickoff & ~_try_after_defensive_td(df) & df["drive"].notna()]
    last_idx = plays.groupby("drive").tail(1).index
    df["drive_result"] = df["drive"].map({df.at[i, "drive"]: last_play_result(i) for i in last_idx})
    df.index = game.index
    return df


def add_play_detail_columns(df: pd.DataFrame, team: str, opponent: str, date: str, week) -> pd.DataFrame:
    """
    Add play-type flags, game_id, drive, and drive_result columns to a
    plays dataframe.

    Run this AFTER add_score_columns (needs offense_score/defense_score)
    and calculate_ep_epa (needs the turnover column). Also assumes a
    'success' column (used for THIRD_DOWN_CONVERTED, FOURTH_DOWN_CONVERTED,
    and the "Turnover on Downs" results) and a 'GN/LS' column (used for
    TACKLE_FOR_LOSS) already exist on the input.

    Columns added
    -------------
    KICKOFF, PUNT, XP_ATTEMPT, FG_ATTEMPT : 1/0, from PLAY TYPE
    KICK_MADE      : 1 if an XP/FG attempt and RESULT == "Good"
    TOUCHBACK      : 1 if RESULT == "Touchback"
    QB_SCRAMBLE    : 1 if RESULT == "Scramble"
    TACKLE_FOR_LOSS: 1 if RESULT == "Rush" and GN/LS < 0
    SACK           : 1 if RESULT contains "Sack"
    completion     : 1 if RESULT contains "Complete", 0 if it contains
                     "Incomplete" or "Interception", blank otherwise (so completion % =
                     sum / count of non-blank rows)
    PENALTY        : 1 if RESULT == "Penalty"
    SCORE_DIFFERENTIAL : offense_score - defense_score
    RUSH, PASS     : 1 if PLAY TYPE contains "Run" / "Pass"
    THIRD_DOWN_CONVERTED, FOURTH_DOWN_CONVERTED : 1 if DN == 3/4 and success == 1
    game_id        : f"{team}_{opponent}_{date}", same for every row
    WEEK           : the 'week' argument, same for every row
    drive          : running count, incrementing each time 'offense'
                     changes (and at halftime). Since kickoffs are
                     assigned to the receiving team, a kickoff row is the
                     first row of the receiving team's drive.
    YDS_NET        : running total of GN/LS within the current drive,
                     restarting at 0 on the first play of each new drive
    SERIES         : like drive, but also increments on every 1st down
                     where the offense is unchanged from the previous
                     play (including back-to-back 1st-and-10s, e.g. from
                     a penalty replaying the down), not just on a change
                     of possession. Kickoffs are NaN (not part of any
                     series), and PATs / 2-point tries carry over the same
                     SERIES value as the play before them.
    drive_result   : one of "Touchdown", "FG Attempt", "Punt", "Turnover",
                     "Safety", "Turnover on Downs", "End of Half",
                     "End of Game", or "Error" (if the last play of the
                     drive doesn't match any of those conditions) —
                     determined from the last NON-kickoff play of each
                     drive and applied to every play in that drive.
    SERIES_RESULT  : one of "Touchdown", "FG Made", "FG Miss", "Punt",
                     "Turnover", "Safety", "First Down",
                     "Turnover on Downs", "End of Half", "End of Game",
                     "Error" (if the last play of the series doesn't match
                     any of those conditions), or NaN (for kickoffs, which
                     have no SERIES) — determined from the last play of
                     each series and applied to every play in that series.
                     Scoring results are checked before "First Down", and
                     a following kickoff never counts as a first down.
    SERIES_SUCCESS : 1 if SERIES_RESULT is "Touchdown", "First Down", or
                     "FG Made", 0 if SERIES_RESULT is any other non-NaN
                     value (including "Error"), and NaN if SERIES_RESULT
                     itself is NaN (kickoffs)

    "Touchdown" (for both results) is triggered by a PAT / 2-point row, or
    by RESULT containing "TD" (but not "Def TD") in case no PAT row follows.
    """
    df = df.copy()
    df = df.sort_values("PLAY #").reset_index(drop=True)

    play_type = df["PLAY TYPE"]
    result = df["RESULT"].fillna("")

    df["KICKOFF"] = play_type.isin(KICKOFF_TYPES).astype(int)
    df["PUNT"] = play_type.isin(PUNT_TYPES).astype(int)
    df["XP_ATTEMPT"] = play_type.isin(XP_TYPES).astype(int)
    df["FG_ATTEMPT"] = play_type.isin(FG_TYPES).astype(int)

    df["KICK_MADE"] = (
        play_type.isin(XP_TYPES | FG_TYPES) & (df["RESULT"] == "Good")
    ).astype(int)

    df["TOUCHBACK"] = (df["RESULT"] == "Touchback").astype(int)
    df["QB_SCRAMBLE"] = (df["RESULT"] == "Scramble").astype(int)
    df["TACKLE_FOR_LOSS"] = (
        (df["RESULT"] == "Rush") & (df["GN/LS"] < 0)
    ).astype(int)
    df["SACK"] = result.str.contains("Sack").astype(int)
    # Interceptions count as incompletions. "Incomplete" is checked before
    # "Complete" because it also contains "Complete".
    df["completion"] = np.select(
        [result.str.contains("Incomplete|Interception"), result.str.contains("Complete")],
        [0, 1],
        default=np.nan,
    )
    df["PENALTY"] = (df["RESULT"] == "Penalty").astype(int)

    df["SCORE_DIFFERENTIAL"] = df["offense_score"] - df["defense_score"]

    df["RUSH"] = play_type.str.contains("Run", na=False).astype(int)
    df["PASS"] = play_type.str.contains("Pass", na=False).astype(int)

    df["THIRD_DOWN_CONVERTED"] = (
        (df["DN"] == 3) & (df["success"] == 1)
    ).astype(int)
    df["FOURTH_DOWN_CONVERTED"] = (
        (df["DN"] == 4) & (df["success"] == 1)
    ).astype(int)

    df["game_id"] = f"{team}_{opponent}_{date}"
    df["WEEK"] = week

    # --- drive: increments every time 'offense' changes from the previous
    # play, OR at halftime (Q2 -> Q3) even if offense happens to be the
    # same team on both sides of the gap (e.g. a deferred-receive team
    # also gets the ball right before half and right after it) — without
    # this, that gap wouldn't register as a new drive at all. ---
    # (also starts a new drive after a defensive touchdown — see assign_drives)
    df["drive"] = _drive_numbers(df)

    # --- YDS_NET: running total of GN/LS within each drive, restarting at
    # the start of every new drive. Missing GN/LS values are treated as 0
    # so they don't break the running total for the rest of the drive. ---
    df["YDS_NET"] = df.groupby("drive")["GN/LS"].transform(
        lambda s: s.fillna(0).cumsum()
    )

    # --- SERIES: like drive, but also increments on every 1st down where
    # the offense is unchanged from the previous play (including
    # back-to-back 1st-and-10s), and also forces a break at halftime for
    # the same reason 'drive' does above. Kickoffs, PATs, and 2-point
    # tries don't represent a real down/series, so they're excluded from
    # the increment logic below: kickoffs end up NaN, PATs / 2-point tries
    # just carry over whatever series the play before them was in.
    normal_play_mask = ~df["PLAY TYPE"].isin(NON_SCRIMMAGE_KICK_TYPES)
    normal_plays = df.loc[normal_play_mask]

    new_series = (
        (normal_plays["offense"] != normal_plays["offense"].shift())
        | (
            (normal_plays["DN"] == 1)
            & (normal_plays["offense"] == normal_plays["offense"].shift())
        )
        | ((normal_plays["QTR"] == 3) & (normal_plays["QTR"].shift() == 2))
    )
    series_for_normal_plays = new_series.cumsum()

    df["SERIES"] = np.nan
    df.loc[normal_play_mask, "SERIES"] = series_for_normal_plays.to_numpy()
    df["SERIES"] = df["SERIES"].ffill()  # carries into KO/PAT rows for now
    df.loc[df["PLAY TYPE"].isin(KICKOFF_TYPES), "SERIES"] = np.nan  # then null kickoffs back out
    # A try after a DEFENSIVE touchdown belongs to the other team, so it isn't
    # part of the series it would have inherited above (that series ended in
    # the turnover). Leave it out, the same as kickoffs.
    last_snap_offense = df["offense"].where(normal_play_mask).ffill()
    other_teams_try = (df["PLAY TYPE"].isin(XP_TYPES | TWO_PT_TYPES) & (df["offense"] != last_snap_offense)) \
        | _try_after_defensive_td(df)
    df.loc[other_teams_try, "SERIES"] = np.nan

    # --- drive_result / SERIES_RESULT: computed on the last NON-kickoff
    # play of each drive/series, then broadcast to every play in it. ---
    is_kickoff = df["PLAY TYPE"].isin(KICKOFF_TYPES)
    n = len(df)

    def _scored_td(i):
        pt = str(df.at[i, "PLAY TYPE"])
        res = df.at[i, "RESULT"] if pd.notna(df.at[i, "RESULT"]) else ""
        return pt in XP_TYPES or pt in TWO_PT_TYPES or ("TD" in res and "Def TD" not in res)

    def _end_of_period(i):
        if df.at[i, "QTR"] == 2 and i + 1 < n and df.at[i + 1, "QTR"] == 3:
            return "End of Half"
        if df.at[i, "QTR"] == 4 and i == n - 1:
            return "End of Game"
        return "Error"

    def last_play_series_result(i):
        pt = str(df.at[i, "PLAY TYPE"])
        res = df.at[i, "RESULT"] if pd.notna(df.at[i, "RESULT"]) else ""

        if _scored_td(i):
            return "Touchdown"
        if pt in FG_TYPES:
            return "FG Made" if res == "Good" else "FG Miss"
        if pt in PUNT_TYPES:
            return "Punt"
        if df.at[i, "turnover"] == 1:
            return "Turnover"
        if "Safety" in res:
            return "Safety"
        if (
            i + 1 < n
            and not is_kickoff.iat[i + 1]
            and df.at[i + 1, "offense"] == df.at[i, "offense"]
            and df.at[i + 1, "DN"] == 1
        ):
            return "First Down"
        if df.at[i, "DN"] == 4 and df.at[i, "success"] == 0:
            return "Turnover on Downs"
        return _end_of_period(i)

    def _broadcast(key, fn):
        # The try after a defensive TD is the scoring team's, so it can't be the last play of the drive that
        # ended in the turnover (it would read as that offense's "Touchdown").
        plays = df[~is_kickoff & ~_try_after_defensive_td(df) & df[key].notna()]
        last_idx = plays.groupby(key).tail(1).index
        return df[key].map({df.at[i, key]: fn(i) for i in last_idx})

    # Same rules as assign_drives (one copy, so curation and the app's load-time fix can't drift apart).
    df["drive_result"] = assign_drives(df)["drive_result"]
    df["SERIES_RESULT"] = _broadcast("SERIES", last_play_series_result)
    df["SERIES_SUCCESS"] = np.where(
        df["SERIES_RESULT"].isna(),
        np.nan,
        df["SERIES_RESULT"].isin(["Touchdown", "First Down", "FG Made"]).astype(float),
    )

    return df


def add_dowling_offense_plays(plays: pd.DataFrame, offense_plays: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of `plays` where, for every row with offense ==
    "Dowling Catholic", the OFF FORM and OFF PLAY values are replaced
    with the corresponding values from `offense_plays`, matched on
    PLAY #.

    `plays` and `offense_plays` are expected to have identical columns
    and rows, differing only in OFF FORM / OFF PLAY.
    """
    df = plays.copy()
    lookup = offense_plays.set_index("PLAY #")[["OFF FORM", "OFF PLAY"]]

    is_dowling = df["offense"] == "Dowling Catholic"
    play_nums = df.loc[is_dowling, "PLAY #"]

    df.loc[is_dowling, "OFF FORM"] = play_nums.map(lookup["OFF FORM"])
    df.loc[is_dowling, "OFF PLAY"] = play_nums.map(lookup["OFF PLAY"])

    schemes = [
        "HERKY", "STORM", "BULLDOGS", "PANTHERS", "CYCLONES", "HAWKEYES",
        "PATRIOTS", "DRAW", "SUPERSONICS", "MONEY", "SEATTLE", "JAYHAWKS",
        "IZZY", "MONSTER", "BLUNT"
    ]

    pattern = r"\b(" + "|".join(map(re.escape, schemes)) + r")\b"

    df["RUN SCHEME"] = (
        df["OFF PLAY"]
        .astype("string")
        .str.extract(pattern, flags=re.IGNORECASE, expand=False)
        .str.upper()
    )

    return df


def curate_play_by_play_data(plays: pd.DataFrame, team: str, opponent: str, date: str, week):
    plays = assign_offense_defense(team=team, opponent=opponent, plays=plays)
    plays = add_success_metrics(plays=plays)
    plays = add_score_columns(plays=plays, team=team, opponent=opponent)
    plays = calculate_ep_epa(df=plays)
    plays = add_play_detail_columns(df=plays, team=team, opponent=opponent, date=date, week=week)
    return plays


def curate_pbp_write_to_excel(
    plays: pd.DataFrame,
    team: str,
    opponent: str,
    date: str,
    week,
    dowling_film: bool = False,
    offense_plays: Optional[pd.DataFrame] = None,
):
    """
    Curate a game's play-by-play data and write it to curated-pbp.xlsx,
    appending to "Sheet1" if the file already exists, or creating it
    (with a header row) if it doesn't.
    """
    plays = curate_play_by_play_data(
        plays=plays, team=team, opponent=opponent, date=date, week=week
    )

    if dowling_film:
        offense_plays = curate_play_by_play_data(
            plays=offense_plays, team=team, opponent=opponent, date=date, week=week
        )
        plays = add_dowling_offense_plays(plays, offense_plays)

    file_path = "curated-pbp.xlsx"
    sheet_name = "Sheet1"

    if not os.path.exists(file_path):
        # File doesn't exist yet: create it fresh, with a header row.
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="w") as writer:
            plays.to_excel(writer, sheet_name=sheet_name, index=False, header=True)
        print("Data written successfully!")
        return

    # File exists: find where to append, and whether a header is needed
    # (only if the sheet is otherwise empty).
    wb = load_workbook(file_path)
    start_row = wb[sheet_name].max_row if sheet_name in wb.sheetnames else 0
    wb.close()

    header = start_row == 0

    with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
        plays.to_excel(
            writer, sheet_name=sheet_name, startrow=start_row, index=False, header=header
        )
    print("Data written successfully!")
