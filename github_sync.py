"""
github_sync.py — saves files to the app's GitHub repo, so a game added on the Add a Game page is there for
everyone: Streamlit Cloud redeploys when the repo changes, and the app rebuilds its data from the film files.

Needs a [github] section in the app's secrets (see secrets.toml.example):
    token    a fine-grained personal access token for this repo only, with Contents: Read and write
    repo     "owner/name"
    branch   the branch the app deploys from (default "main")
    api_url  optional, for GitHub Enterprise (default https://api.github.com)
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Callable

import requests

API = "https://api.github.com"


class SaveError(Exception):
    """Something went wrong saving to GitHub; the message is written for a coach, not a developer."""


@dataclass(frozen=True)
class Config:
    token: str
    repo: str
    branch: str = "main"
    api: str = API


def config() -> Config | None:
    """The [github] secrets, or None if saving to GitHub isn't set up."""
    from qol import _secret

    gh = _secret("github")
    if not gh or not gh.get("token") or not gh.get("repo"):
        return None
    return Config(token=str(gh["token"]), repo=str(gh["repo"]).strip("/"), branch=str(gh.get("branch") or "main"),
                  api=str(gh.get("api_url") or API).rstrip("/"))


class _Client:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.http = requests.Session()
        self.http.headers.update({"Authorization": f"Bearer {cfg.token}", "Accept": "application/vnd.github+json",
                                  "X-GitHub-Api-Version": "2022-11-28"})

    def request(self, method: str, path: str, ok=(200, 201), **kw) -> requests.Response:
        try:
            r = self.http.request(method, f"{self.cfg.api}/repos/{self.cfg.repo}{path}", timeout=30, **kw)
        except requests.RequestException as e:
            raise SaveError(f"Couldn't reach GitHub ({e}).") from e
        if r.status_code in ok:
            return r
        if r.status_code == 401:
            raise SaveError("GitHub didn't accept the token in the app's secrets. It may have expired.")
        if r.status_code in (403, 404):
            raise SaveError(f"The token in the app's secrets can't write to {self.cfg.repo} (GitHub said "
                            f"{r.status_code}). It needs Contents: Read and write on that repo.")
        try:
            detail = r.json().get("message", "")
        except ValueError:
            detail = r.text[:200]
        raise SaveError(f"GitHub returned {r.status_code}: {detail}")

    def read_text(self, path: str, ref: str) -> str | None:
        r = self.request("GET", f"/contents/{path}", ok=(200, 404), params={"ref": ref})
        if r.status_code == 404:
            return None
        return base64.b64decode(r.json()["content"]).decode("utf-8")


def commit_files(cfg: Config, build_files: Callable[[Callable[[str], str | None]], dict[str, bytes]],
                 message: str, attempts: int = 3) -> str | None:
    """
    Commit files to cfg.branch in ONE commit and return the commit's web URL (None if nothing changed).

    build_files gets a reader for text files at the branch's current head and returns {path: bytes}, so a change
    to a shared file like games.csv is made to the newest copy in the repo, not the app's possibly older one. If
    someone else pushes in the middle, it starts over from the new head (up to `attempts` times).
    """
    gh = _Client(cfg)
    for _ in range(attempts):
        head = gh.request("GET", f"/git/ref/heads/{cfg.branch}").json()["object"]["sha"]
        base_tree = gh.request("GET", f"/git/commits/{head}").json()["tree"]["sha"]
        files = build_files(lambda p: gh.read_text(p, head))
        entries = []
        for path, data in files.items():
            blob = gh.request("POST", "/git/blobs", json={"content": base64.b64encode(data).decode(),
                                                          "encoding": "base64"}).json()["sha"]
            entries.append({"path": path, "mode": "100644", "type": "blob", "sha": blob})
        tree = gh.request("POST", "/git/trees", json={"base_tree": base_tree, "tree": entries}).json()["sha"]
        if tree == base_tree:
            return None
        commit = gh.request("POST", "/git/commits", json={"message": message, "tree": tree,
                                                          "parents": [head]}).json()
        moved = gh.request("PATCH", f"/git/refs/heads/{cfg.branch}", ok=(200, 422),
                           json={"sha": commit["sha"], "force": False})
        if moved.status_code == 200:
            return commit.get("html_url") or f"https://github.com/{cfg.repo}/commit/{commit['sha']}"
        # 422: the branch moved since we read it (someone else saved). Rebuild on top of the new head.
    raise SaveError("The repo kept changing while saving. Try again in a minute.")
