"""Checks for github_sync.py against a fake GitHub API (tests/fake_github.py)."""
import pytest

import github_sync as gs
import insights as ins
from fake_github import FakeGitHub

GAMES = b"file,team,opponent,week,drop_plays,drop_offense_plays\nDowling/a-082826.xlsx,Dowling Catholic,A,1,,\n"


@pytest.fixture
def gh():
    fake = FakeGitHub({"games.csv": GAMES, "Dowling/a-082826.xlsx": b"old film"})
    yield fake
    fake.close()


def _cfg(fake, token="good-token"):
    return gs.Config(token=token, repo=fake.repo, branch=fake.branch, api=fake.url)


def _add_game(read):
    manifest = ins.updated_manifest("Dowling/dowling-vs-b-090426.xlsx", "Dowling Catholic", "B", 2,
                                    read("games.csv"))
    return {"Dowling/dowling-vs-b-090426.xlsx": b"new film", "games.csv": manifest}


def test_one_commit_with_film_and_manifest(gh):
    before = gh.head
    url = gs.commit_files(_cfg(gh), _add_game, "Add game: Dowling Catholic vs B (Week 2)")
    assert url and gh.head in url
    assert gh.commits[gh.head]["parents"] == [before]                     # exactly one new commit
    files = gh.files()
    assert files["Dowling/dowling-vs-b-090426.xlsx"] == b"new film"
    assert files["Dowling/a-082826.xlsx"] == b"old film"                  # everything else untouched
    assert files["games.csv"].decode().splitlines()[-1] == "Dowling/dowling-vs-b-090426.xlsx,Dowling Catholic,B,2,,"


def test_manifest_edit_uses_the_repos_newest_copy(gh):
    # The repo's games.csv has a game the app's local copy doesn't: it must survive the save.
    gh.head = gh._commit(gh._tree({**{p: gh._blob(d) for p, d in gh.files().items()},
                                   "games.csv": gh._blob(GAMES + b"Scout/x-091126.xlsx,X,Y,3,,\n")}),
                         [gh.head], "someone else's upload")
    gs.commit_files(_cfg(gh), _add_game, "Add game")
    rows = gh.files()["games.csv"].decode().splitlines()
    assert "Scout/x-091126.xlsx,X,Y,3,," in rows and rows[-1].startswith("Dowling/dowling-vs-b-090426.xlsx")


def test_retries_when_someone_pushes_mid_save(gh):
    gh.pushes_before_next_update = 1
    gs.commit_files(_cfg(gh), _add_game, "Add game")
    assert gh.commits[gh.commits[gh.head]["parents"][0]]["message"] == "someone else"   # built on top of it
    assert b"dowling-vs-b-090426" in gh.files()["games.csv"]


def test_gives_up_if_the_repo_keeps_changing(gh):
    gh.pushes_before_next_update = 5
    with pytest.raises(gs.SaveError, match="kept changing"):
        gs.commit_files(_cfg(gh), _add_game, "Add game")


def test_nothing_changed(gh):
    assert gs.commit_files(_cfg(gh), lambda read: {"Dowling/a-082826.xlsx": b"old film"}, "Re-upload") is None


def test_bad_token_reads_like_english(gh):
    with pytest.raises(gs.SaveError, match="didn't accept the token"):
        gs.commit_files(_cfg(gh, token="expired"), _add_game, "Add game")
