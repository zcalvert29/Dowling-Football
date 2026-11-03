"""
teams.py — team names, colors, and Dowling's schedule from GoBound.

    teams.csv       one row per team: its name in the app, its name on GoBound, and its primary color (hex, for
                    the Rankings page). Edit colors there or on the Rankings page.
    app_name()      turns a GoBound name into the app's name: "Cedar Rapids Kennedy" -> "CR Kennedy",
                    "Iowa City High" -> "IC High", "Dallas Center-Grimes" -> "DCG", "Waukee Northwest" -> "Waukee NW".
    GoBound         the Add a Game dropdowns list every team on the IHSAA football standings page, and the app's
                    default opponent is the next game on Dowling's schedule. Both pages are read live (the app
                    caches them for a few hours) and fall back to teams.csv / next_opponent in the secrets when
                    GoBound can't be reached.

Command line:
    python teams.py             print the next opponent and the team list as read from GoBound right now
    python teams.py --refresh   also add any new GoBound teams to teams.csv (colors are kept)
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from html.parser import HTMLParser

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HERE, "teams.csv")
TEAM = "Dowling Catholic"
GOBOUND = "https://www.gobound.com"
SCHOOL_SLUG = "dowling"   # Dowling's part of its GoBound URLs


def season_slug(today: date) -> str:
    """GoBound's season in its URLs: '2026-27' for the fall 2026 season (Jan-Jul still belongs to last fall)."""
    y = today.year if today.month >= 8 else today.year - 1
    return f"{y}-{str(y + 1)[-2:]}"


def schedule_url(today: date) -> str:
    return f"{GOBOUND}/ia/ihsaa/football/{season_slug(today)}/{SCHOOL_SLUG}/v/schedule"


def standings_url(today: date) -> str:
    return f"{GOBOUND}/ia/ihsaa/football/{season_slug(today)}/standings"


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
# Whole-name aliases, compared ignoring case, dashes, and periods. Anything not here keeps its GoBound name
# after the prefix rules below.
ALIASES = {
    "dallas center grimes": "DCG",
    "waukee northwest": "Waukee NW",
    "iowa city high": "IC High",
    "iowa city city high": "IC High",
    "city high": "IC High",
    "ic city high": "IC High",
    "southeast polk": "SEP",
    "west des moines valley": "Valley",
    "wdm valley": "Valley",
    "west des moines dowling catholic": TEAM,
    "wdm dowling catholic": TEAM,
    "dowling": TEAM,
}
PREFIXES = (("Cedar Rapids ", "CR "), ("Iowa City ", "IC "))


def _key(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[-.]", " ", name)).strip().casefold()


def app_name(name: str) -> str:
    """The app's name for a team, from its GoBound name (an app name comes back unchanged)."""
    n = re.sub(r"\s+", " ", str(name)).strip()
    if _key(n) in ALIASES:
        return ALIASES[_key(n)]
    for long, short in PREFIXES:
        if n.casefold().startswith(long.casefold()):
            return short + n[len(long):]
    return n


def load(path: str = FILE) -> pd.DataFrame:
    """teams.csv: team, gobound, color."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["team", "gobound", "color"])
    t = pd.read_csv(path, dtype=str).fillna("")
    for c in ("team", "gobound", "color"):
        t[c] = t.get(c, "").astype(str).str.strip() if c in t else ""
    return t[t["team"] != ""].reset_index(drop=True)


def colors(path: str = FILE) -> dict[str, str]:
    """{team: '#RRGGBB'} for teams with a color set."""
    t = load(path)
    ok = t["color"].str.fullmatch(r"#[0-9A-Fa-f]{6}")
    return dict(zip(t.loc[ok, "team"], t.loc[ok, "color"].str.upper()))


def to_csv(t: pd.DataFrame) -> bytes:
    t = t[["team", "gobound", "color"]].copy()
    t = t.drop_duplicates("team", keep="last")
    return t.sort_values("team", key=lambda s: s.str.casefold()).to_csv(index=False).encode()


def merge(t: pd.DataFrame, gobound_names) -> pd.DataFrame:
    """teams.csv with every GoBound team in it: new teams added, existing teams keep their colors."""
    t = t.copy()
    for g in gobound_names:
        name = app_name(g)
        hit = t["team"].str.casefold() == name.casefold()
        if hit.any():
            t.loc[hit & (t["gobound"] == ""), "gobound"] = g
        else:
            t = pd.concat([t, pd.DataFrame([{"team": name, "gobound": g, "color": ""}])], ignore_index=True)
    return t


def long_names(gobound_names=()) -> dict[str, str]:
    """{app name: GoBound name} wherever they differ, so a dropdown can be searched by either spelling."""
    out = {}
    t = load()
    for team, g in zip(t["team"], t["gobound"]):
        if g and g != team:
            out[team] = g
    for g in gobound_names:
        if app_name(g) != g:
            out[app_name(g)] = g
    for team in list(out) + list(t["team"]):
        for long, short in PREFIXES:
            if team.startswith(short) and team not in out:
                out[team] = long + team[len(short):]
    return out


def option_label(team: str, longs: dict[str, str]) -> str:
    """'CR Kennedy (Cedar Rapids Kennedy)' for a dropdown, or just the name."""
    return f"{team} ({longs[team]})" if team in longs else team


def team_options(*extra, gobound_names=()) -> list[str]:
    """Every team for the Add a Game dropdowns: GoBound's standings, teams.csv, and any names passed in (teams
    already in the data), in app spelling, sorted, Dowling first."""
    names = {app_name(n) for n in gobound_names} | set(load()["team"])
    for group in extra:
        names |= {str(n).strip() for n in group if str(n).strip()}
    rest = sorted((n for n in names if n != TEAM), key=str.casefold)
    return [TEAM] + rest


# ---------------------------------------------------------------------------
# Reading GoBound
#
# The pages are read as a flat run of text and team links, in page order, so this doesn't depend on whether
# GoBound lays a schedule out as a table, a list, or cards. A team link is any link to a team's page for this
# sport and season: /ia/ihsaa/football/2026-27/<school>/...
# ---------------------------------------------------------------------------
class _Flat(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.items: list[tuple[str, str, str]] = []   # ("text", text, "") or ("link", text, href)
        self._href: str | None = None
        self._buf: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip += 1
        elif tag == "a":
            self._flush()
            self._href = dict(attrs).get("href") or ""
        elif tag in ("tr", "li", "div", "p", "br", "td", "th", "span", "h1", "h2", "h3", "h4", "section"):
            self._flush()

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self._skip = max(0, self._skip - 1)
        elif tag == "a":
            text = " ".join("".join(self._buf).split())
            if text:
                self.items.append(("link", text, self._href or ""))
            self._buf, self._href = [], None
        elif tag in ("tr", "li", "div", "p", "td", "th", "span", "h1", "h2", "h3", "h4", "section"):
            self._flush()

    def handle_data(self, data):
        if not self._skip:
            self._buf.append(data)

    def _flush(self):
        if self._href is not None:
            return    # inside a link: its text is kept with the link
        text = " ".join("".join(self._buf).split())
        if text:
            self.items.append(("text", text, ""))
        self._buf = []

    def close(self):
        super().close()
        self._flush()


def _flatten(html: str) -> list[tuple[str, str, str]]:
    p = _Flat()
    p.feed(html)
    p.close()
    return p.items


def _team_slug(href: str) -> str | None:
    """'dowling' from a link to /ia/ihsaa/football/2026-27/dowling/v/..., None for any other link."""
    m = re.search(r"/football/\d{4}-\d{2}/([a-z0-9-]+)(?:/|$|\?)", href or "", re.IGNORECASE)
    if not m or m.group(1).lower() in {"standings", "schedule", "scores", "rankings", "stats", "teams"}:
        return None
    return m.group(1).lower()


MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                      "dec"], start=1)}
_DATE_NUM = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")
_DATE_TXT = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:,?\s+(\d{4}))?\b",
                       re.IGNORECASE)
_SCORE = re.compile(r"(?:^|\s)(?:W|L|T)?\s*\d{1,3}\s*[-–]\s*\d{1,3}(?:\s|$|\()")   # "W 35-7", "35 - 7"
_FINAL = re.compile(r"^(?:W|L|T|Final|F)$", re.IGNORECASE)


def _find_date(text: str, season_year: int) -> date | None:
    for rx, month_of in ((_DATE_TXT, lambda m: MONTHS[m.group(1)[:3].lower()]), (_DATE_NUM, lambda m: int(m.group(1)))):
        m = rx.search(text)
        if not m:
            continue
        try:
            month = month_of(m)
            day = int(m.group(2))
            if m.group(3):
                year = int(m.group(3)) + (2000 if len(m.group(3)) == 2 else 0)
            else:
                year = season_year if month >= 7 else season_year + 1
            return date(year, month, day)
        except ValueError:
            continue
    return None


@dataclass(frozen=True)
class ScheduledGame:
    day: date | None
    opponent: str        # app spelling
    gobound_name: str
    played: bool


def parse_schedule(html: str, today: date, own_slug: str = SCHOOL_SLUG) -> list[ScheduledGame]:
    """Dowling's games from its GoBound schedule page, in page order. A game counts as played if a score (or W/L)
    shows up between its opponent link and the next one."""
    season_year = int(season_slug(today)[:4])
    items = _flatten(html)
    games, last_date = [], None
    idx = [i for i, (kind, _, href) in enumerate(items)
           if kind == "link" and (s := _team_slug(href)) and s != own_slug]
    def before(n):   # text between the previous opponent and this one, nearest first
        return [t for k, t, _ in reversed(items[(idx[n - 1] if n else -1) + 1:idx[n]]) if k == "text"]

    def after(n):    # text between this opponent and the next one
        return [t for k, t, _ in items[idx[n] + 1:(idx[n + 1] if n + 1 < len(idx) else len(items))] if k == "text"]

    def first_date(texts):
        return next((d for t in texts if (d := _find_date(t, season_year))), None)

    # Dates sit on one side of the opponent's name for the whole page. If the first game has none before its
    # name but one after it, the page puts dates after names (else text before a name belongs to the game above).
    dates_after = bool(idx) and first_date(before(0)) is None and first_date(after(0)) is not None
    for n, i in enumerate(idx):
        day = first_date(after(n) if dates_after else before(n)) or last_date
        last_date = day
        played = any(_SCORE.search(f" {t} ") or _FINAL.match(t) for t in after(n))
        name =re.sub(r"^(?:vs\.?|@|at)\s+", "", items[i][1], flags=re.IGNORECASE)
        games.append(ScheduledGame(day, app_name(name), name, played))
    # The same opponent linked twice in a row (logo + name) is one game.
    out = []
    for g in games:
        if out and out[-1].gobound_name == g.gobound_name and out[-1].day == g.day:
            out[-1] = ScheduledGame(g.day, g.opponent, g.gobound_name, out[-1].played or g.played)
        else:
            out.append(g)
    return out


def next_game(schedule: list[ScheduledGame], today: date) -> ScheduledGame | None:
    """The first game on the schedule that hasn't been played and isn't in the past."""
    for g in schedule:
        if not g.played and (g.day is None or g.day >= today):
            return g
    return None


def parse_standings(html: str) -> list[str]:
    """Every team named on the standings page (GoBound spelling), in page order, once each."""
    seen, out = set(), []
    for kind, text, href in _flatten(html):
        if kind != "link" or not _team_slug(href):
            continue
        name = re.sub(r"\s*\(\d+-\d+(?:-\d+)?\)\s*$", "", text).strip()   # drop a trailing record "(8-1)"
        if name and not re.fullmatch(r"[\d\s\-–.%]+", name) and name.casefold() not in seen:
            seen.add(name.casefold())
            out.append(name)
    return out


def fetch(url: str, timeout: float = 10.0) -> str:
    import requests

    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (Dowling Football app)"})
    r.raise_for_status()
    return r.text


def gobound_next_opponent(today: date | None = None) -> ScheduledGame | None:
    """The next game on Dowling's GoBound schedule. Raises if the page can't be read."""
    today = today or datetime.now().date()
    return next_game(parse_schedule(fetch(schedule_url(today)), today), today)


def gobound_teams(today: date | None = None) -> list[str]:
    """Every team on GoBound's standings page (GoBound spelling). Raises if the page can't be read."""
    today = today or datetime.now().date()
    return parse_standings(fetch(standings_url(today)))


def main(argv: list[str]) -> int:
    today = datetime.now().date()
    print(f"Schedule: {schedule_url(today)}")
    schedule = parse_schedule(fetch(schedule_url(today)), today)
    for g in schedule:
        print(f"  {g.day or '?'}  {g.opponent:<22} {'played' if g.played else ''}"
              f"{'' if g.opponent == g.gobound_name else f'  (GoBound: {g.gobound_name})'}")
    nxt = next_game(schedule, today)
    print(f"Next opponent: {nxt.opponent if nxt else 'none found'}")
    print(f"\nStandings: {standings_url(today)}")
    names = parse_standings(fetch(standings_url(today)))
    print(f"  {len(names)} teams: " + ", ".join(f"{app_name(n)}" for n in names))
    if "--refresh" in argv:
        t = merge(load(), names)
        with open(FILE, "wb") as f:
            f.write(to_csv(t))
        print(f"\nWrote {FILE} ({len(t)} teams)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
