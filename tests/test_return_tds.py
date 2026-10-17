"""Kicks returned for a touchdown: the returning team scores, the kicking team's drive still ends in a punt."""
import pandas as pd

import build_data as b
import insights as ins
import profiles
import special_teams as stm

AC_ANKENY = b.Game(file="Scout/ankeny-centennial-vs-ankeny-091826.xlsx", team="Ankeny Centennial", opponent="Ankeny",
                   season=2026)   # #78: Ankeny Centennial punts, Ankeny returns it for a TD (Ankeny kicks the PAT, #79)


def _row(df, n):
    return df[df["PLAY #"] == n].iloc[0]


def test_punt_return_td_goes_to_the_returning_team():
    df, _ = b.curate_game(AC_ANKENY)
    before, td = _row(df, 77), _row(df, 78)
    assert td["RESULT"] == "Return, Def TD"
    assert (td["team_score"], td["opponent_score"]) == (before["team_score"], before["opponent_score"] + 6)
    assert td["drive_result"] == "Punt" and td["turnover"] == 0         # a punt, not a turnover
    assert (df["team_score"].iat[-1], df["opponent_score"].iat[-1]) == (0, 59)


def test_return_tds_count_in_special_teams():
    df, _ = b.curate_game(AC_ANKENY)
    full = ins.add_game_state(df)
    ev = stm.kick_events(full)
    td = ev[ev["result"] == "Return, Def TD"].iloc[0]
    assert (td["kind"], td["kicker"], td["receiver"], td["start"]) == ("Punt", "Ankeny Centennial", "Ankeny", 100.0)
    kicks = profiles.standardize_hudl(full, "Ankeny Centennial")["kicks"]
    punt = kicks[(kicks["kind"] == "punt") & (kicks["recv_ytg"] == 0)].iloc[0]
    assert punt["receiver"] == "Ankeny" and punt["epa"] < -5            # about a touchdown and PAT given up


def test_kickoff_return_td(tmp_path):
    # Tag the opening kickoff (Ankeny Centennial receives, "KO Rec") as returned for a touchdown.
    raw = pd.read_excel(b._path(AC_ANKENY.file))
    first_ko = raw.index[raw["PLAY TYPE"] == "KO Rec"][0]
    raw.loc[first_ko, "RESULT"] = "Return, TD"
    f = tmp_path / AC_ANKENY.file.split("/")[-1]
    raw.to_excel(f, index=False)
    df, _ = b.curate_game(b.Game(file=str(f), team="Ankeny Centennial", opponent="Ankeny", season=2026))
    pos = df.index[df["PLAY #"] == raw.loc[first_ko, "PLAY #"]][0]
    ko, prev = df.loc[pos], df.loc[pos - 1]
    assert (ko["team_score"] - prev["team_score"], ko["opponent_score"] - prev["opponent_score"]) == (6, 0)
    ev = stm.kick_events(ins.add_game_state(df))
    td = ev[(ev["kind"] == "Kickoff") & (ev["result"] == "Return, TD")].iloc[0]
    assert (td["receiver"], td["kicker"], td["start"]) == ("Ankeny Centennial", "Ankeny", 100.0)
