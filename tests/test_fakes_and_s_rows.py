"""Fake punts / fake field goals count as the run or pass they were; ODK 'S' rows with a play tagged get flagged."""

import pandas as pd

import build_data as b

SEP = "Dowling/dowling-vs-sep-091826.xlsx"        # SEP's fake punt: #106, 4th & 8, Rush for 14
SEP_OFF = "Dowling/dowling-vs-sep-offense.xlsx"


def _sep_copy(tmp_path, change):
    """A copy of the SEP game (and its offense file) with `change(df)` applied to both."""
    for f in (SEP, SEP_OFF):
        df = change(pd.read_excel(b._path(f)))
        df.to_excel(tmp_path / f.split("/")[-1], index=False)
    return b.Game(file=str(tmp_path / SEP.split("/")[-1]), team="Dowling Catholic", opponent="SEP", season=2026)


def _row(df, n):
    return df[df["PLAY #"] == n].iloc[0]


def test_successful_fake_punt_is_a_converted_run():
    df, _ = b.curate_game(b.Game(file=SEP, team="Dowling Catholic", opponent="SEP", season=2026))
    r = _row(df, 106)
    assert (r["PLAY TYPE"], r["FAKE"], r["ODK"], r["offense"]) == ("Run", "Fake Punt", "D", "SEP")
    assert r["success"] == 1 and r["FOURTH_DOWN_CONVERTED"] == 1 and r["explosive_play"] == 1
    assert _row(df, 107)["drive"] == r["drive"]                       # SEP kept the ball: same drive
    assert df["FAKE"].isna().sum() == len(df) - 1                     # nothing else is marked a fake


def test_failed_fake_punt_is_a_turnover_on_downs(tmp_path):
    def fail(df):  # 3 yards on 4th & 8, then Dowling's ball (SEP's next possession plays removed)
        df.loc[df["PLAY #"] == 106, "GN/LS"] = 3
        return df[~df["PLAY #"].between(107, 114)]
    df, notes = b.curate_game(_sep_copy(tmp_path, fail))
    r = _row(df, 106)
    assert r["success"] == 0 and r["drive_result"] == "Turnover on Downs" and r["SERIES_RESULT"] == "Turnover on Downs"
    assert not any("goes from SEP" in n for n in notes)               # no "Error" drive


def test_fake_pass_and_fake_fg(tmp_path):
    def change(df):
        df.loc[df["PLAY #"] == 106, ["PLAY TYPE", "RESULT"]] = ["Fake FG", "Complete"]
        return df
    df, _ = b.curate_game(_sep_copy(tmp_path, change))
    r = _row(df, 106)
    assert (r["PLAY TYPE"], r["FAKE"], r["offense"]) == ("Pass", "Fake FG", "SEP")


def test_s_rows_with_a_play_tagged_are_flagged():
    _, notes = b.curate_game(b.Game(file="Scout/ankeny-centennial-vs-ankeny-091826.xlsx", team="Ankeny Centennial",
                                    opponent="Ankeny", season=2026))
    s = [n for n in notes if "tagged ODK 'S'" in n]
    assert len(s) == 1 and "#26 (Penalty)" in s[0] and "#32 (Penalty)" in s[0]
    assert "#51" not in s[0] and "#10 " not in s[0]                   # a timeout and a down-only row aren't listed


def test_blank_s_rows_are_not_flagged_but_tagged_ones_are_named(tmp_path):
    f = "Scout/waukee-vs-valley-091826.xlsx"                          # #97 is a blank S row
    g = b.Game(file=f, team="Waukee", opponent="Valley", season=2026)
    assert not any("tagged ODK 'S'" in n for n in b.curate_game(g)[1])
    raw = pd.read_excel(b._path(f))
    raw.loc[raw["PLAY #"] == 97, "PLAY TYPE"] = "Punt"                 # the missing punt, tagged S by mistake
    raw.to_excel(tmp_path / f.split("/")[-1], index=False)
    _, notes = b.curate_game(b.Game(file=str(tmp_path / f.split("/")[-1]), team="Waukee", opponent="Valley",
                                    season=2026))
    assert any("#97 (Punt)" in n for n in notes)
    assert any("#97 is an S row tagged Punt" in n for n in notes)


def test_sidebar_issues_leave_out_s_row_notes():
    _, problems = b.build()
    s_notes = [p for p in problems if b.is_s_row_note(p)]
    assert s_notes and all("tagged ODK 'S'" in p for p in s_notes)
    assert not any(b.is_s_row_note(p) for p in problems if "the ball goes from" in p)
