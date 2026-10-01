"""
Go for 2 Bot
=====================================================================
Extra point vs. two-point try, following the score-by-score guidance in
ESPN's game management cheat sheet (Brian Burke's win probability model):
https://www.espn.com/nfl/story/_/id/33059528/

Three possible outputs:
  - GO FOR 2
  - KICK EXTRA POINT
  - COACH'S CHOICE  (the cheat sheet calls it close / acceptable either way)

The recommendation depends on the score AND the time left, because the
cheat sheet's advice for most scores only kicks in at a certain point in the
game. Clock uses 12-minute quarters.

Scores the cheat sheet doesn't single out default to kicking the extra point.
All timing cutoffs live in the TIMING section below so they're easy to tune.
"""

import streamlit as st

import fourth_down_core as fd

# Runs as a page inside the scouting app; st.Page sets the title/icon.
st.set_page_config(layout="centered")

QUARTER_SECONDS = 12 * 60

# ---- TIMING (seconds left in the quarter, unless noted) --------------------
MID_Q3 = 6 * 60               # "roughly midway through the third quarter"
LATE_Q4 = 8 * 60 + 30         # "roughly 8-9 minutes or less in the fourth quarter"
VERY_LATE_Q4 = 5 * 60         # "very late in the game"
FINAL_SECONDS_GAME = 20       # "down to 15-20 seconds" (seconds left in the game)

st.title("🎯 Go for 2 Bot")
st.caption("Extra point vs. two-point try, based on ESPN's score-and-time cheat sheet.")

with st.sidebar:
    st.header("Game Situation")
    your_abbr = st.text_input("Your team", "DCHS", key="g2_your_abbr").upper()[:4]
    opp_abbr = st.text_input("Opponent", "VHS", key="g2_opp_abbr").upper()[:4]
    your_score = st.number_input(
        f"{your_abbr} score (BEFORE the try — after the TD's 6 points)", 0, 99, 20, key="g2_your_score"
    )
    opp_score = st.number_input(f"{opp_abbr} score", 0, 99, 25, key="g2_opp_score")
    quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=3, key="g2_quarter")
    minutes = st.number_input("Minutes remaining in quarter", 0, 12, 8, key="g2_minutes")
    seconds = st.number_input("Seconds", 0, 59, 0, key="g2_seconds")


@st.cache_data(show_spinner=False)
def dowling_pat_record(path: str = "curated-pbp.xlsx") -> tuple[int, int]:
    """(made, attempted) Dowling extra points this season. Penalties don't count as attempts."""
    import pandas as pd

    try:
        df = pd.read_excel(path, usecols=["offense", "PLAY TYPE", "RESULT"])
    except Exception:
        return 0, 0
    xp = df[(df["offense"] == "Dowling Catholic") & df["PLAY TYPE"].isin(["Extra Pt.", "Extra Pt. Block"])
            & (df["RESULT"] != "Penalty")]
    return int((xp["RESULT"] == "Good").sum()), len(xp)


_made, _att = dowling_pat_record()
# Default PAT rate: Dowling's season rate, shrunk toward a 90% prior (worth
# 10 kicks) so a perfect 4-for-4 start doesn't read as a 100% kicker.
_pat_default = int(round(100 * (_made + 9) / (_att + 10)))

with st.sidebar:
    st.header("Model check")
    g2_site = st.selectbox("Site", ["We're home", "We're away", "Neutral"], key="g2_site",
                           help="The win probability model includes home-field advantage.")
    pat_pct = st.slider("Extra point make rate", 50, 100, _pat_default, format="%d%%", key="g2_pat",
                        help=f"Default is Dowling's season rate ({_made}/{_att}) blended with a 90% "
                             f"prior so a few kicks don't swing it too far.")
    two_pct = st.slider("Two-point conversion rate", 20, 80, 45, format="%d%%", key="g2_two",
                        help="Around 40-50% is typical. Raise it if you have a 2-point play you trust.")

if minutes == 12:
    seconds = 0  # quarters are 12:00 max
qtr_left = min(minutes * 60 + seconds, QUARTER_SECONDS)
game_left = (4 - quarter) * QUARTER_SECONDS + qtr_left
margin = your_score - opp_score          # before the try; negative = trailing

second_half = quarter >= 3
from_mid_q3 = quarter == 4 or (quarter == 3 and qtr_left <= MID_Q3)
fourth_quarter = quarter == 4
late_q4 = fourth_quarter and qtr_left <= LATE_Q4
very_late_q4 = fourth_quarter and qtr_left <= VERY_LATE_Q4
final_seconds = game_left <= FINAL_SECONDS_GAME

GO, KICK, CHOICE = "Go for 2", "Kick extra point", "Coach's Choice"


def recommend():
    """Returns (decision, reason) from ESPN's cheat sheet for this score and time."""
    if margin < 0:
        d = -margin
        if d == 1:
            if final_seconds:
                return CHOICE, ("Down 1 with under ~20 seconds left: either choice is fine. "
                                "Go for 2 and the opponent has almost no time to answer with a field goal.")
            return KICK, ("Down 1: kick to tie. With time left, taking the lead by 1 invites the opponent "
                          "to play aggressively for a field goal. It becomes a coin flip in the final ~20 seconds.")
        if d == 2:
            return GO, "Down 2: a chance to be tied beats being certain you're still losing."
        if d == 4:
            if late_q4:
                return GO, ("Down 4 in the last ~8-9 minutes: make it and a field goal ties. "
                            "It's like finding out the result of overtime in advance.")
            return KICK, "Down 4: kick for now. Going for 2 becomes the call with about 8-9 minutes left in the 4th."
        if d == 5:
            if second_half:
                return GO, "Down 5 in the second half: go for 2 to be down only a field goal."
            return CHOICE, "Down 5 in the first half: either way is acceptable."
        if d == 8:
            if from_mid_q3:
                return GO, ("Down 8 from mid-3rd quarter on: make it and a TD + PAT wins. "
                            "Miss it and you can still go for 2 on the next TD to tie.")
            return KICK, "Down 8: kick for now. Going for 2 becomes the call starting around the middle of the 3rd."
        if d == 9:
            return CHOICE, ("Down 9: not clear-cut. Going for 2 tells you now whether you're down one score "
                            "or two, which helps your later decisions.")
        if d == 10:
            if late_q4:
                return GO, ("Down 10 in the last ~8-9 minutes: with few possessions left, being down 8 "
                            "(tie with a TD + 2) is clearly better than down 9.")
            return CHOICE, ("Down 10: can go either way for much of the game. A PAT lets a later FG + TD "
                            "take the lead.")
        if d == 11:
            if fourth_quarter:
                return GO, "Down 11 in the 4th quarter: go for 2 (the down-8 idea plus a field goal)."
            return KICK, "Down 11: kick for now. Going for 2 becomes the call around the start of the 4th."
        if d == 12:
            if very_late_q4:
                return CHOICE, "Down 12 very late: worth considering 2 (the down-9 idea plus a field goal)."
            return KICK, "Down 12: kick. Going for 2 only becomes worth considering very late."
        if d == 13:
            if late_q4:
                return CHOICE, ("Down 13 late: going for 2 can be advisable, keeping a FG + TD + 2 "
                                "path to a tie open.")
            return KICK, "Down 13: kick. Going for 2 only comes into play late in the game."
        if d == 15:
            if second_half:
                return GO, "Down 15 in the second half: go for 2 (the down-8 idea plus a touchdown)."
            return KICK, "Down 15: kick in the first half. Go for 2 in the second half."
    elif margin > 0:
        if margin == 5:
            if second_half:
                return GO, "Up 5 in the second half: go for 2 to be up a touchdown."
            return KICK, "Up 5: kick in the first half. Go for 2 in the second half."
        if margin == 7:
            return CHOICE, ("Up 7: the mirror image of down 9. Kicking keeps the opponent from knowing if "
                            "they're down one score or two, but the model sometimes favors going for 2.")
        if margin == 12:
            if fourth_quarter:
                return GO, "Up 12 late: go for 2 to try to go up two touchdowns."
            return KICK, "Up 12: kick for now. Going for 2 to go up two touchdowns is the call later in the game."
    return KICK, "The cheat sheet doesn't single out this score. Take the safer point."


def describe_margin(m):
    """Short, plain description of a resulting margin."""
    if m == 0:
        return "tied"
    if m > 0:
        if m < 3:
            return f"up {m} — a field goal beats you"
        if m == 3:
            return f"up {m} — a field goal only ties it"
        if m <= 8:
            return f"up {m} — takes a touchdown to catch you"
        return f"up {m} — takes two scores to catch you"
    d = -m
    if d < 3:
        return f"down {d} — a field goal wins it"
    if d == 3:
        return f"down {d} — a field goal ties it"
    if d <= 8:
        return f"down {d} — need a touchdown"
    return f"down {d} — need two scores"


decision, reason = recommend()
emoji = {GO: "👉", KICK: "🦵", CHOICE: "⚖️"}[decision]

# ---- Situation line ----
st.divider()
st.write(f"**{your_abbr} {your_score} — {opp_abbr} {opp_score}** (before the try)  ·  "
         f"Q{quarter} {minutes}:{seconds:02d}")

# ---- Big bold verdict ----
st.markdown(
    f"<h1 style='text-align:center; font-size:3.2rem; margin-bottom:0;'>{emoji} {decision.upper()}</h1>",
    unsafe_allow_html=True,
)

# ---- Rationale ----
st.divider()
st.write("#### Rationale")
st.write(f"Kick the PAT: {describe_margin(margin + 1)}.")
st.write(f"Go for 2: {describe_margin(margin + 2)}.")
st.write(reason)

# ---- Model check: the same EP/WP models as the 4th Down Bot ----
st.divider()
st.write("#### Model check")
_site = {"We're home": 1.0, "We're away": 0.0, "Neutral": 0.5}[g2_site]
_try = fd.evaluate_try(margin, fd.model_seconds_remaining(quarter, qtr_left), pat_pct / 100, two_pct / 100,
                       is_home_pos=_site)
_diff = (_try["wp_go"] - _try["wp_kick"]) * 100
if abs(_diff) < 0.5:
    _model_call = CHOICE
elif _diff > 0:
    _model_call = GO
else:
    _model_call = KICK
_be = _try["break_even_two"]

c1, c2, c3 = st.columns(3)
c1.metric("Win prob. if you kick", f"{_try['wp_kick']:.0%}")
c2.metric("Win prob. if you go for 2", f"{_try['wp_go']:.0%}", f"{_diff:+.1f} pts vs kicking")
c3.metric("Break-even 2-pt rate", "—" if _be != _be else f"{_be:.0%}",
          help="Going for 2 wins out if you convert at least this often.")

_agree = (_model_call == decision) or CHOICE in (_model_call, decision)
_be_text = "" if _be != _be else f" Going for 2 is the better call if you convert at least {_be:.0%} of the time."
if _agree:
    st.success(f"**The model agrees: {_model_call.lower()}.** With a {pat_pct}% kicker and a {two_pct}% "
               f"2-point play.{_be_text}", icon="✅")
else:
    st.warning(f"**The model says {_model_call.lower()}, the cheat sheet says {decision.lower()}.** The cheat sheet "
               f"assumes NFL rates (about 95% on extra points, 48% on 2-point tries); the model uses your "
               f"{pat_pct}% / {two_pct}%.{_be_text} The 12-minute clock also matters: the same minutes left "
               f"are a bigger share of a high school game.", icon="⚠️")

st.divider()
st.caption("The verdict at the top is ESPN's cheat sheet for a typical NFL game. The model check uses your EP/WP "
           "models with your make rates; when they disagree, the model is the one that knows about your kicker.")
with st.expander("Method & limitations"):
    st.markdown(f"""
- Source: ESPN's game management cheat sheet (Brian Burke's win probability model):
  https://www.espn.com/nfl/story/_/id/33059528/
- Scores with specific guidance: down 1, 2, 4, 5, 8, 9, 10, 11, 12, 13, 15 and up 5, 7, 12.
  Every other score defaults to kicking the extra point.
- **Coach's Choice** means the cheat sheet calls it close or acceptable either way.
- Timing cutoffs used here: "midway through the 3rd" = {MID_Q3 // 60}:{MID_Q3 % 60:02d} left in Q3,
  "late" = {LATE_Q4 // 60}:{LATE_Q4 % 60:02d} or less in Q4, "very late" = {VERY_LATE_Q4 // 60}:00 or less in Q4,
  "final seconds" = {FINAL_SECONDS_GAME} seconds or less in the game. All use 12-minute quarters.
- The cheat sheet is built on NFL conversion rates (about 48% on 2-point tries and near-automatic
  extra points). If your kicker is shaky or your 2-point offense is strong, lean toward going for 2
  in the close calls.
- Doesn't account for timeouts or team strength.
- **Model check:** prices each outcome as the other team's ball at their own 25 after the kickoff, with the new
  margin, using the same EP/WP models and 15/12 clock scaling as the 4th Down Bot. Timeouts are assumed 3 each.
""")
