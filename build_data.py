"""
build_data.py — builds the season's play-by-play straight from the game files listed in games.csv.
The app calls load_games() when it starts, so there's no combined data file to keep up to date.

Adding a game:
    1. Save the Hudl export as  <team>-vs-<opponent>-MMDDYY.xlsx  in Dowling/ (our games) or Scout/ (scout film).
       For our games, save the offense film as  dowling-vs-<opponent>-offense.xlsx  next to it.
    2. Add a row to games.csv (team and opponent as they should appear in the app, and the season).
    3. Run  python build_data.py  to check it, then commit and push.

games.csv columns:
    file                path to the game export, e.g. Dowling/dowling-vs-valley-082826.xlsx (the date comes from
                        the MMDDYY at the end of the name)
    team, opponent      the filming team and who they played
    season              the season (e.g. 2026); the week comes from the game date (see week_of)
    drop_plays          optional: PLAY #s to remove from the game file (space-separated); the rest are renumbered
    drop_offense_plays  optional: the same for the offense file

Command line:
    python build_data.py           build every game and print a summary plus anything that needs a look
    python build_data.py --excel   also write curated-pbp.xlsx (for Tableau)
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache

import pandas as pd

import curate_pbp as cp

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "games.csv")
FILM_FOLDERS = ("Dowling", "Scout")
TEAM = "Dowling Catholic"
DATE_IN_NAME = re.compile(r"-(\d{2})(\d{2})(\d{2})\.xlsx$", re.IGNORECASE)

# Week 1 of each season, a Friday; each Friday after that is the next week, through LAST_WEEK. A season that isn't
# listed starts on the last Friday of August (2026's does too). Add a season here if its Week 1 is different.
SEASON_WEEK1 = {2026: date(2026, 8, 28)}
LAST_WEEK = 13


def week1(season: int) -> date:
    if season in SEASON_WEEK1:
        return SEASON_WEEK1[season]
    d = date(season, 8, 31)
    return d - timedelta(days=(d.weekday() - 4) % 7)  # last Friday of August


def week_of(game_day: date, season: int) -> int:
    """Week of the season for a game on game_day: Week 1 is the season's first Friday, Week 2 the next Friday, and
    so on through LAST_WEEK. A Thursday or Saturday game counts toward the nearest Friday."""
    first = week1(season)
    week = round((game_day - first).days / 7) + 1
    if not 1 <= week <= LAST_WEEK:
        last = first + timedelta(weeks=LAST_WEEK - 1)
        raise ValueError(f"{game_day:%m/%d/%Y} isn't in the {season} season (Week 1 is {first:%m/%d/%Y}, "
                         f"Week {LAST_WEEK} is {last:%m/%d/%Y}). Check the date and season.")
    return week


@dataclass(frozen=True)
class Game:
    file: str
    team: str
    opponent: str
    season: int
    drop_plays: tuple[int, ...] = ()
    drop_offense_plays: tuple[int, ...] = ()

    @property
    def game_day(self) -> date:
        """From the MMDDYY at the end of the file name."""
        m = DATE_IN_NAME.search(self.file)
        if not m:
            raise ValueError(f"{self.file}: the file name should end in -MMDDYY.xlsx (e.g. -082826.xlsx)")
        mm, dd, yy = m.groups()
        return date(2000 + int(yy), int(mm), int(dd))

    @property
    def date(self) -> str:
        """'2026_08_28', the form game_ids use."""
        return f"{self.game_day:%Y_%m_%d}"

    @property
    def week(self) -> int:
        return week_of(self.game_day, self.season)

    @property
    def offense_file(self) -> str | None:
        """Our games: the -offense file next to the game file, or the game file itself if there isn't one (its
        OFF FORM / OFF PLAY are used as-is). Scout games: None."""
        if self.team != TEAM:
            return None
        off = DATE_IN_NAME.sub("-offense.xlsx", self.file)
        return off if os.path.exists(_path(off)) else self.file


def _path(f: str) -> str:
    return f if os.path.isabs(f) else os.path.join(HERE, f)


def _play_list(x) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", str(x))) if pd.notna(x) else ()


def read_manifest(path: str = MANIFEST) -> list[Game]:
    m = pd.read_csv(path, dtype=str).fillna("")
    m.columns = m.columns.str.strip()
    return [Game(file=r["file"].strip(), team=r["team"].strip(), opponent=r["opponent"].strip(),
                 season=int(r["season"]), drop_plays=_play_list(r.get("drop_plays") or None),
                 drop_offense_plays=_play_list(r.get("drop_offense_plays") or None))
            for _, r in m.iterrows() if r["file"].strip()]


def _read_film(f: str, drop: tuple[int, ...]) -> pd.DataFrame:
    plays = pd.read_excel(_path(f))
    plays = plays[plays["ODK"] != "S"]  # curation drops these anyway; dropping first keeps renumbering in step
    if drop:
        plays = plays[~plays["PLAY #"].isin(drop)].sort_values("PLAY #").reset_index(drop=True)
        plays["PLAY #"] = range(1, len(plays) + 1)
    return plays


def _film_problems(f: str, plays: pd.DataFrame) -> list[str]:
    dup = plays.loc[plays["PLAY #"].duplicated(), "PLAY #"]
    if dup.empty:
        return []
    return [f"{f}: {len(dup)} PLAY # value(s) appear more than once (e.g. {', '.join(map(str, dup.head(3)))}). "
            f"Two games or two cut-ups in one file? Plays are put in PLAY # order, so this game's drives and "
            f"scores will be wrong until it's fixed."]


def _ordinal(n) -> str:
    n = int(n)
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _play_text(r: pd.Series, offense: bool = True) -> str:
    """'#48 (Q2, CR Kennedy 3rd & 13, Pass, Incomplete)'."""
    bits = [f"Q{int(r['QTR'])}" if pd.notna(r.get("QTR")) else None]
    who = r.get("offense") if offense and pd.notna(r.get("offense")) else None
    situation = f"{_ordinal(r['DN'])} & {int(r['DIST'])}" if pd.notna(r.get("DN")) and r["DN"] >= 1 and pd.notna(
        r.get("DIST")) else None
    bits.append(" ".join(x for x in (who, situation) if x) or None)
    bits += [r.get("PLAY TYPE"), r.get("RESULT")]
    return f"#{int(r['PLAY #'])} ({', '.join(str(b) for b in bits if b is not None and pd.notna(b))})"


def _skipped_text(raw: pd.DataFrame, after: int, before: int | None) -> str:
    """Rows of the export between two plays that curation drops (blank 'S' rows, timeouts)."""
    between = raw[(raw["PLAY #"] > after) & ((raw["PLAY #"] < before) if before is not None else True)]
    if between.empty:
        return ""
    kinds = [f"#{int(r['PLAY #'])} is {'a timeout' if r.get('RESULT') == 'Timeout' else 'a blank S row'}"
             for _, r in between.iterrows()]
    return f" In between, {', '.join(kinds)} (dropped when the game is built). Was that the missing play?"


def _error_drive_notes(f: str, df: pd.DataFrame, raw: pd.DataFrame) -> list[str]:
    """One plain-English note per drive whose result came out 'Error': what the last play was, and what the
    film shows next, so the play to fix in Hudl is easy to find."""
    plays = df.reset_index(drop=True)
    notes = []
    for _, d in plays[plays["drive_result"] == "Error"].groupby("drive", sort=False):
        scrimmage = d[~d["PLAY TYPE"].isin(cp.KICKOFF_TYPES)]
        last = (scrimmage if len(scrimmage) else d).iloc[-1]
        pos = int((scrimmage if len(scrimmage) else d).index[-1])
        nxt = plays.iloc[pos + 1] if pos + 1 < len(plays) else None
        skipped = _skipped_text(raw, last["PLAY #"], None if nxt is None else nxt["PLAY #"])
        if nxt is None:
            msg = (f"the film ends after {_play_text(last)}, in Q{int(last['QTR'])} rather than at the end of "
                   f"the game. Missing plays at the end?")
        elif nxt["offense"] != last["offense"]:
            msg = (f"the ball goes from {last['offense']} to {nxt['offense']} after {_play_text(last, False)} "
                   f"with no punt, turnover, or score tagged. Next play: {_play_text(nxt)}. A missing punt, or a "
                   f"fumble/interception not in RESULT?")
        else:
            msg = (f"{last['offense']}'s drive ends after {_play_text(last, False)} without a punt, turnover, or "
                   f"score tagged. Next play: {_play_text(nxt)}.")
        notes.append(f"{f}: {msg}{skipped} ({len(d)} play{'s' if len(d) != 1 else ''} show drive_result "
                     f"'Error'.)")
    return notes


def curate_game(g: Game) -> tuple[pd.DataFrame, list[str]]:
    """One game, curated exactly the way the old notebook did it. Returns (plays, problems)."""
    plays = _read_film(g.file, g.drop_plays)
    problems = _film_problems(g.file, plays)
    df = cp.curate_play_by_play_data(plays=plays, team=g.team, opponent=g.opponent, date=g.date, week=g.week)
    if g.offense_file:
        off_raw = _read_film(g.offense_file, g.drop_offense_plays)
        if g.offense_file != g.file:
            problems += _film_problems(g.offense_file, off_raw)
        off = cp.curate_play_by_play_data(plays=off_raw, team=g.team, opponent=g.opponent, date=g.date,
                                          week=g.week)
        df = cp.add_dowling_offense_plays(df, off)
    df["SEASON"] = g.season
    problems += _error_drive_notes(g.file, df, pd.read_excel(_path(g.file)))
    return df, problems


def unlisted_files(games: list[Game]) -> list[str]:
    """Game exports sitting in the film folders that games.csv doesn't list (offense files don't count)."""
    listed = {os.path.normpath(g.file) for g in games} | {os.path.normpath(g.offense_file) for g in games
                                                          if g.offense_file}
    out = []
    for folder in FILM_FOLDERS:
        d = _path(folder)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            rel = os.path.normpath(os.path.join(folder, name))
            if name.lower().endswith(".xlsx") and not name.startswith("~$") and rel not in listed:
                out.append(rel)
    return out


def build(manifest: str = MANIFEST) -> tuple[pd.DataFrame, list[str]]:
    """Every game in games.csv, curated and stacked. Returns (plays, problems): problems are plain-English notes
    about anything that needs a look. A game that can't be read is left out and listed as a problem."""
    games = read_manifest(manifest)
    frames, problems = [], []
    seen = set()
    for g in games:
        try:
            df, notes = curate_game(g)
        except FileNotFoundError:
            problems.append(f"{g.file}: listed in games.csv but the file isn't there. Left out.")
            continue
        except Exception as e:  # noqa: BLE001 — one bad file shouldn't take the whole app down
            problems.append(f"{g.file}: couldn't be processed ({type(e).__name__}: {e}). Left out.")
            continue
        gid = df["game_id"].iat[0]
        if gid in seen:
            problems.append(f"{g.file}: same team, opponent, and date as another game in games.csv. Left out.")
            continue
        seen.add(gid)
        frames.append(df)
        problems += notes
    problems += [f"{f}: in the film folder but not listed in games.csv, so it isn't in the app."
                 for f in unlisted_files(games)]
    plays = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return plays, problems


def fingerprint(manifest: str = MANIFEST) -> tuple:
    """Changes whenever games.csv, a film file, or the curation code changes, so cached data gets rebuilt."""
    files = [manifest, cp.__file__] + [os.path.join(_path(d), n) for d in FILM_FOLDERS if os.path.isdir(_path(d))
                                       for n in sorted(os.listdir(_path(d)))]
    return tuple((f, os.stat(f).st_mtime_ns, os.stat(f).st_size) for f in files if os.path.isfile(f))


@lru_cache(maxsize=2)
def _build_cached(manifest: str, stamp: tuple) -> tuple[pd.DataFrame, tuple[str, ...]]:
    plays, problems = build(manifest)
    return plays, tuple(problems)


def load_games(manifest: str = MANIFEST) -> tuple[pd.DataFrame, list[str]]:
    """build(), cached until a game file or games.csv changes. Returns a copy, so callers can change it freely."""
    plays, problems = _build_cached(manifest, fingerprint(manifest))
    return plays.copy(), list(problems)


def main(argv: list[str]) -> int:
    plays, problems = build()
    if plays.empty:
        print("No games built.")
    else:
        summary = plays.groupby("game_id", sort=False).agg(season=("SEASON", "first"), week=("WEEK", "first"), plays=("PLAY #", "size"))
        print(summary.to_string())
        print(f"\n{len(summary)} games, {len(plays):,} plays.")
    if problems:
        print(f"\n{len(problems)} thing(s) need a look:")
        for p in problems:
            print(f"  - {p}")
    if "--excel" in argv and not plays.empty:
        out = os.path.join(HERE, "curated-pbp.xlsx")
        plays.to_excel(out, sheet_name="Sheet1", index=False)
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
