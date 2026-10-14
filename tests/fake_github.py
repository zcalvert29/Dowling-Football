"""A tiny in-memory stand-in for the GitHub API endpoints github_sync.py uses, served over HTTP for tests."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


def _sha(*parts) -> str:
    return hashlib.sha1(repr(parts).encode()).hexdigest()


class FakeGitHub:
    def __init__(self, files: dict[str, bytes], repo: str = "coach/football", branch: str = "main",
                 token: str = "good-token"):
        self.repo, self.branch, self.token = repo, branch, token
        self.blobs: dict[str, bytes] = {}
        self.trees: dict[str, dict[str, str]] = {}
        self.commits: dict[str, dict] = {}
        self.head = self._commit(self._tree({p: self._blob(d) for p, d in files.items()}), [], "initial")
        self.pushes_before_next_update = 0  # simulate someone else pushing mid-save
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def files(self, sha: str | None = None) -> dict[str, bytes]:
        tree = self.trees[self.commits[sha or self.head]["tree"]]
        return {p: self.blobs[b] for p, b in tree.items()}

    def close(self):
        self.server.shutdown()

    def _blob(self, data: bytes) -> str:
        sha = _sha("blob", data)
        self.blobs[sha] = data
        return sha

    def _tree(self, entries: dict[str, str]) -> str:
        sha = _sha("tree", sorted(entries.items()))
        self.trees[sha] = dict(entries)
        return sha

    def _commit(self, tree: str, parents: list[str], message: str) -> str:
        sha = _sha("commit", tree, tuple(parents), message, len(self.commits))
        self.commits[sha] = {"tree": tree, "parents": parents, "message": message}
        return sha

    def _route(self, method: str, path: str, query: dict, body: dict):
        m = re.fullmatch(rf"/repos/{re.escape(self.repo)}(/.*)", path)
        if not m:
            return 404, {"message": "Not Found"}
        p = m.group(1)
        if method == "GET" and p == f"/git/ref/heads/{self.branch}":
            return 200, {"object": {"sha": self.head}}
        if method == "GET" and p.startswith("/git/commits/"):
            return 200, {"sha": p.rsplit("/", 1)[1], "tree": {"sha": self.commits[p.rsplit("/", 1)[1]]["tree"]}}
        if method == "GET" and p.startswith("/contents/"):
            data = self.files(query.get("ref", [self.head])[0]).get(p[len("/contents/"):])
            if data is None:
                return 404, {"message": "Not Found"}
            return 200, {"content": base64.b64encode(data).decode(), "encoding": "base64"}
        if method == "POST" and p == "/git/blobs":
            return 201, {"sha": self._blob(base64.b64decode(body["content"]))}
        if method == "POST" and p == "/git/trees":
            entries = dict(self.trees[body["base_tree"]])
            entries.update({e["path"]: e["sha"] for e in body["tree"]})
            return 201, {"sha": self._tree(entries)}
        if method == "POST" and p == "/git/commits":
            sha = self._commit(body["tree"], body["parents"], body["message"])
            return 201, {"sha": sha, "html_url": f"https://github.com/{self.repo}/commit/{sha}"}
        if method == "PATCH" and p == f"/git/refs/heads/{self.branch}":
            if self.pushes_before_next_update:
                self.pushes_before_next_update -= 1
                self.head = self._commit(self.commits[self.head]["tree"], [self.head], "someone else")
            if self.commits[body["sha"]]["parents"] != [self.head]:
                return 422, {"message": "Update is not a fast forward"}
            self.head = body["sha"]
            return 200, {"object": {"sha": self.head}}
        return 404, {"message": "Not Found"}

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _go(self, method):
                if self.headers.get("Authorization") != f"Bearer {fake.token}":
                    status, out = 401, {"message": "Bad credentials"}
                else:
                    u = urlparse(self.path)
                    n = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(n)) if n else {}
                    status, out = fake._route(method, u.path, parse_qs(u.query), body)
                data = json.dumps(out).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._go("GET")

            def do_POST(self):
                self._go("POST")

            def do_PATCH(self):
                self._go("PATCH")

        return Handler
