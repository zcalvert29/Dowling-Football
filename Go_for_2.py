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

st.divider()
st.caption("Based on ESPN's cheat sheet for a typical game — not a live win probability calculation.")
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
- Doesn't account for timeouts or team strength. The model-driven 4th Down Bot page covers those.
""")
