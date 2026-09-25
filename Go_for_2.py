"""
Go for 2 Bot
=====================================================================
Pure score-based extra-point vs. two-point-conversion decision. NO win
probability, NO conversion-rate probabilities, NO trained models.

TWO RULES, in priority order:

1. SPECIAL CASE — trailing by exactly 8. Projecting one more (unconverted)
   touchdown forward, this is the one deficit where kicking both extra
   points guarantees, at best, a tie (forcing OT), while going for 2 now
   opens a path to an outright win. Checked every deficit 1-30 — D=8 is the
   only one where this happens. Matches the well-known "down 8" call from
   ESPN's cheat sheet: https://www.espn.com/nfl/story/_/id/33059528/

2. GENERAL RULE — exact sufficiency thresholds, checked asymmetrically:
   - WHEN TRAILING: all three of your own comeback values matter — field
     goal (3), TD+PAT (7), TD+2PT (8). Any of these flipping from "not
     enough" to "enough" is a real, actionable improvement.
   - WHEN LEADING, the three thresholds work differently:
       - Field goal (3): ANY change counts (stops a win, or stops a tie).
       - TD+PAT (7): only counts if it stops their STANDARD, reliable
         score from beating you outright.
       - TD+2PT (8): only counts if it closes off their HARDEST possible
         score entirely (their longshot no longer works at all).
     This is what correctly separates up 5 (stops their standard TD+PAT
     from beating you) and up 7 (closes off their TD+2PT longshot
     entirely) from up 6, which only hits the weaker half of each of
     those two thresholds and isn't a meaningful gain either way.
"""

import streamlit as st

# Runs as a page inside the scouting app; st.Page sets the title/icon.
st.set_page_config(layout="centered")

st.title("🎯 Go for 2 Bot")
st.caption("Extra point vs. two-point conversion — based purely on the score, no win probabilities.")

with st.sidebar:
    st.header("Game Situation")
    your_abbr = st.text_input("Your team", "DCHS", key="g2_your_abbr").upper()[:4]
    opp_abbr = st.text_input("Opponent", "VHS", key="g2_opp_abbr").upper()[:4]
    your_score = st.number_input(
        f"{your_abbr} score (BEFORE the try — after the TD's 6 points)", 0, 99, 20, key="g2_your_score"
    )
    opp_score = st.number_input(f"{opp_abbr} score", 0, 99, 25, key="g2_opp_score")


def category(margin, T):
    """0/1/2, monotonically increasing in margin, for whether a score of
    value |T| beats/ties/falls short."""
    if T > 0:
        if margin < T:
            return 0
        elif margin == T:
            return 1
        else:
            return 2
    else:
        value = -T
        deficit = -margin
        if deficit > value:
            return 0
        elif deficit == value:
            return 1
        else:
            return 2


def crosses_favorably(margin_pat, margin_2pt):
    """WHEN TRAILING: any improvement at any of your three comeback
    thresholds (field goal, TD+PAT, TD+2PT) is a real, actionable gain, so
    any category increase counts.

    WHEN LEADING, the three thresholds don't all work the same way:
      - Field goal (3): ANY change counts. It's common/reliable enough that
        both "stops them beating you" and "stops them even tying" matter.
      - TD+PAT (7): only counts if it stops their STANDARD, reliable score
        from beating you outright (category 0 -> higher). Going from
        "exactly ties" to "juuust short" isn't a meaningful gain.
      - TD+2PT (8): only counts if it closes off their HARDEST possible
        score entirely (-> category 2, insufficient even for their best
        shot). Going from "beats you" to "merely ties" at this level still
        just leaves it as their longshot parlay either way.
    This asymmetry is what correctly separates up 5 (fires, stops their
    standard TD+PAT from beating you) and up 7 (fires, closes off their
    hardest TD+2PT entirely) from up 6 (doesn't fire — it only catches the
    weaker half of each of those two thresholds, not the meaningful half
    of either)."""
    if margin_pat < 0:
        for T in (-3, -7, -8):
            if category(margin_2pt, T) > category(margin_pat, T):
                return T
        return None
    else:
        if category(margin_2pt, 3) > category(margin_pat, 3):
            return 3
        if category(margin_pat, 7) == 0 and category(margin_2pt, 7) > 0:
            return 7
        if category(margin_pat, 8) < 2 and category(margin_2pt, 8) == 2:
            return 8
        return None


def describe_margin(margin):
    """Short, plain description of a resulting margin."""
    if margin == 0:
        return "tied"
    if margin > 0:
        if margin < 3:
            return f"up {margin} — a field goal beats you"
        elif margin == 3:
            return f"up {margin} — a field goal only ties it"
        elif margin <= 8:
            return f"up {margin} — takes a touchdown to catch you"
        else:
            return f"up {margin} — takes two scores to catch you"
    else:
        deficit = -margin
        if deficit < 3:
            return f"down {deficit} — a field goal wins it"
        elif deficit == 3:
            return f"down {deficit} — a field goal ties it"
        elif deficit <= 8:
            return f"down {deficit} — need a touchdown"
        else:
            return f"down {deficit} — need two scores"


SCORE_LABELS = {3: "field goal", 7: "touchdown + extra point", 8: "touchdown + 2-point try"}


def crossing_text(margin_pat, margin_2pt, T):
    """Builds the specific 'from X to Y' phrase for whichever transition
    actually happened at threshold T, rather than one static sentence per
    threshold — T=3 can mean either 'win becomes a tie' (up 1) or 'tie
    becomes not enough' (up 2), and those need different wording."""
    label = SCORE_LABELS[abs(T)]
    c_from, c_to = category(margin_pat, T), category(margin_2pt, T)
    if T > 0:
        whose = f"their {label}"
        stages = {0: "a win", 1: "just a tie", 2: "not enough"}
    else:
        whose = f"your {label}"
        stages = {0: "not enough", 1: "a tie", 2: "the lead"}
    return f"turns {whose} from {stages[c_from]} into {stages[c_to]}"


your_score_pat_made = your_score + 1
your_score_2pt_made = your_score + 2
margin_pat = your_score_pat_made - opp_score
margin_2pt = your_score_2pt_made - opp_score
deficit_before_try = opp_score - your_score

# ---- Decision ----
down_8_special_case = (deficit_before_try == 8)
threshold_crossed = crosses_favorably(margin_pat, margin_2pt)
go_for_2 = down_8_special_case or (threshold_crossed is not None)

best_option = "Go for 2" if go_for_2 else "Kick extra point"
emoji = "👉" if go_for_2 else "🦵"

# ---- Situation line ----
st.divider()
st.write(f"**{your_abbr} {your_score} — {opp_abbr} {opp_score}** (before the try)")

# ---- Big bold verdict ----
st.markdown(
    f"<h1 style='text-align:center; font-size:3.2rem; margin-bottom:0;'>{emoji} {best_option.upper()}</h1>",
    unsafe_allow_html=True
)

# ---- Rationale ----
st.divider()
st.write("#### Rationale")

if down_8_special_case:
    st.write("You're down 8. Kick both extra points and the best you get is a tie — forces overtime.")
    st.write("Go for 2 now and convert, and a touchdown + PAT on your next score wins it outright. No OT.")
    st.write("This only works out this way at exactly down 8.")
else:
    st.write(f"Kick the PAT: {describe_margin(margin_pat)}.")
    st.write(f"Go for 2: {describe_margin(margin_2pt)}.")
    if go_for_2:
        st.write(f"Go for 2 — it {crossing_text(margin_pat, margin_2pt, threshold_crossed)}.")
    else:
        st.write("Same story either way. Take the safer point.")

st.divider()
st.caption("Based only on what happens if the try succeeds — no probabilities involved.")
with st.expander("Method & limitations"):
    st.markdown("""
- Inspired by ESPN's game-management cheat sheet (Brian Burke's model):
  https://www.espn.com/nfl/story/_/id/33059528/
- **When trailing**, all three comeback thresholds matter — field goal (3),
  touchdown + PAT (7), touchdown + 2-point try (8) — because each one is a
  real improvement to your own path back into the game.
- **When leading**, the three thresholds aren't treated the same. A field
  goal counts either way (stopping a win, or stopping a tie) since it's
  common and reliable. A touchdown + PAT only counts if it stops their
  standard, likely score from beating you outright. A touchdown + 2-point
  try only counts if it closes off their hardest possible score entirely —
  going from "beats you" to "merely ties" at that level still leaves it as
  their longshot either way, not worth the risk.
- **Down 8** is a one-off special case: projecting one more touchdown
  forward, it's the only deficit where kicking both extra points caps you
  at a tie while going for 2 opens a path to an outright win. Not
  generalized further — beyond this one case, "2pt looks better" in a
  two-score projection just because it's worth more points when it works,
  not because of any real change in what's needed to win.
- Doesn't account for time remaining, timeouts, or team strength — the
  model-driven 4th Down Bot page covers those.
""")
