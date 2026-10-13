"""The paid fallback wired through `hyperresearch fetch` — offline.

Guards the wiring: when the fallback fires, what the note looks like, and that
with the fallback off (or any guard failing) every block behaves exactly as it
did before the fallback existed.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import ClassVar

import pytest
from typer.testing import CliRunner

from hyperresearch.cli import app
from hyperresearch.web.base import WebResult

runner = CliRunner()

TAG = "widgets-a1b2c3"
URL = "https://example.org/report"
GOOD = "A real article body, sentence after sentence. " * 60


class _Blocked:
    """Primary provider returning a configurable block."""

    name = "fake"

    def __init__(self, mode: str, status: int | None = None):
        self.mode, self.status = mode, status

    def fetch(self, url):
        meta = {"status_code": self.status}
        if self.mode == "raise":
            raise RuntimeError("net::ERR_TIMED_OUT")
        if self.mode == "login":
            return WebResult(url=url, title="Sign in to continue", content="Please sign in.", metadata=meta)
        if self.mode == "bot":
            return WebResult(url=url, title="Just a moment...", content="Checking your browser " * 30, metadata=meta)
        if self.mode == "empty":
            return WebResult(url=url, title="Report", content="", metadata=meta)
        if self.mode == "error":
            return WebResult(url=url, title="404 Not Found", content="404 not found " * 40, metadata=meta)
        raise AssertionError(self.mode)

    def fetch_many(self, urls):
        return [self.fetch(u) for u in urls]


class _FakeFirecrawl:
    calls: ClassVar[list[str]] = []
    result: ClassVar[WebResult | None] = None

    def __init__(self, key, **kw):
        assert key == "fc-test"

    def fetch(self, url):
        _FakeFirecrawl.calls.append(url)
        return _FakeFirecrawl.result


def _good(**meta) -> WebResult:
    return WebResult(
        url=URL, title="The Report", content=GOOD,
        metadata={"status_code": 200, "credits_used": 1, "cache_state": "miss", **meta},
    )


@pytest.fixture
def vault(tmp_path: Path, monkeypatch) -> Path:
    assert runner.invoke(app, ["init", str(tmp_path / "kb"), "--name", "FC"]).exit_code == 0
    root = tmp_path / "kb"
    cfg = root / ".hyperresearch" / "config.toml"
    text = cfg.read_text(encoding="utf-8")
    assert "[firecrawl]\nenabled = false" in text  # init writes the section, off by default
    cfg.write_text(text.replace("[firecrawl]\nenabled = false", "[firecrawl]\nenabled = true"), encoding="utf-8")
    (root / "research" / "runs" / TAG).mkdir(parents=True)

    from hyperresearch.core import oa
    from hyperresearch.web import firecrawl_provider

    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
    monkeypatch.setattr(oa.socket, "getaddrinfo", lambda h, p: [(2, 1, 6, "", ("93.184.216.34", 0))])
    monkeypatch.setattr(firecrawl_provider, "FirecrawlProvider", _FakeFirecrawl)
    _FakeFirecrawl.calls = []
    _FakeFirecrawl.result = _good()
    os.chdir(root)
    return root


def _primary(monkeypatch, mode, status=None):
    monkeypatch.setattr("hyperresearch.web.base.get_provider", lambda *a, **k: _Blocked(mode, status))


def _fetch(*extra):
    res = runner.invoke(app, ["fetch", URL, "--tag", TAG, "--json", *extra])
    return res.exit_code, json.loads(res.output)


def _rows(root: Path, sql: str):
    with sqlite3.connect(root / ".hyperresearch" / "hyperresearch.db") as conn:
        try:
            return conn.execute(sql).fetchall()
        except sqlite3.OperationalError:
            return []


def _note(root: Path, note_id: str):
    from hyperresearch.core.frontmatter import parse_frontmatter

    path = root / "research" / "notes" / f"{note_id}.md"
    return parse_frontmatter(path.read_text(encoding="utf-8-sig"))


# --- when it fires -----------------------------------------------------------

@pytest.mark.parametrize(
    "mode,status",
    [("raise", None), ("login", 200), ("bot", 403), ("bot", 503), ("empty", 200), ("empty", None)],
)
def test_fires_for_blocks(vault, monkeypatch, mode, status):
    _primary(monkeypatch, mode, status)
    code, out = _fetch()
    assert code == 0, out
    assert _FakeFirecrawl.calls == [URL]
    assert out["data"]["provider"] == "firecrawl"


@pytest.mark.parametrize(
    "mode,status,expected",
    [
        ("empty", 404, "JUNK_CONTENT"),
        ("empty", 503, "JUNK_CONTENT"),
        ("error", 410, "JUNK_CONTENT"),
        ("error", 200, "JUNK_CONTENT"),
        ("bot", 500, "JUNK_ESCALATED"),
    ],
)
def test_never_fires_for_error_pages(vault, monkeypatch, mode, status, expected):
    _primary(monkeypatch, mode, status)
    code, out = _fetch()
    assert code == 1
    assert out["error_code"] == expected
    assert _FakeFirecrawl.calls == []


def test_utility_below_threshold_is_not_attempted(vault, monkeypatch):
    _primary(monkeypatch, "bot", 403)
    code, _ = _fetch("--utility-score", "7")
    assert code == 1 and _FakeFirecrawl.calls == []


def test_no_run_identity_is_not_attempted(vault, monkeypatch):
    _primary(monkeypatch, "bot", 403)
    res = runner.invoke(app, ["fetch", URL, "--tag", "topic", "--json"])
    assert res.exit_code == 1 and _FakeFirecrawl.calls == []


# --- parity with the fallback off --------------------------------------------

@pytest.mark.parametrize(
    "mode,status,code_expected",
    [
        ("raise", None, "FETCH_ERROR"),
        ("login", 200, "AUTH_REQUIRED_ESCALATED"),
        ("bot", 403, "JUNK_ESCALATED"),
        ("empty", 200, "JUNK_CONTENT"),
    ],
)
@pytest.mark.parametrize("switch", ["disabled", "no-key"])
def test_off_means_todays_behaviour(vault, monkeypatch, mode, status, code_expected, switch):
    if switch == "disabled":
        cfg = vault / ".hyperresearch" / "config.toml"
        cfg.write_text(cfg.read_text().replace("[firecrawl]\nenabled = true", "[firecrawl]\nenabled = false"))
    else:
        monkeypatch.delenv("FIRECRAWL_API_KEY")
    _primary(monkeypatch, mode, status)

    code, out = _fetch()

    assert code == 1
    assert out["error_code"] == code_expected
    assert _FakeFirecrawl.calls == []
    assert _rows(vault, "SELECT COUNT(*) FROM fallback_fetches") in ([], [(0,)])


# --- outcome -----------------------------------------------------------------

def test_rescue_writes_a_disclosed_note_and_no_queue_row(vault, monkeypatch):
    _primary(monkeypatch, "bot", 403)
    code, out = _fetch()
    assert code == 0
    data = out["data"]
    assert data["fallback"]["blocked_reason"] == "bot_block"
    assert data["fallback"]["orig_status"] == 403
    assert data["fallback"]["credits"] == 1

    meta, body = _note(vault, data["note_id"])
    assert meta.fetch_provider == "firecrawl"
    assert body.startswith("> [!] **Retrieved via Firecrawl.**")
    assert "a bot-detection page, HTTP 403" in body
    assert "Firecrawl fetched it live." in body
    assert GOOD[:40] in body

    # the provenance survives a re-parse and an update through the note model
    shown = runner.invoke(app, ["note", "show", data["note_id"], "--json"])
    assert shown.exit_code == 0
    updated = runner.invoke(app, ["note", "update", data["note_id"], "--status", "review", "--json"])
    assert updated.exit_code == 0
    meta, body = _note(vault, data["note_id"])
    assert meta.status == "review"
    assert meta.fetch_provider == "firecrawl"
    assert body.startswith("> [!] **Retrieved via Firecrawl.**")

    assert _rows(vault, "SELECT provider FROM sources") == [("firecrawl",)]
    assert _rows(vault, "SELECT outcome, note_id, credits FROM fallback_fetches") == [
        ("rescued", data["note_id"], 1)
    ]
    assert _rows(vault, "SELECT COUNT(*) FROM escalations") in ([], [(0,)])


@pytest.mark.parametrize(
    "bad",
    [
        _good(status_code=403),
        WebResult(url=URL, title="T", content="too short", metadata={"status_code": 200}),
        WebResult(url=URL, title="Just a moment...", content="Checking your browser " * 30,
                  metadata={"status_code": 200}),
        None,
    ],
    ids=["non-2xx", "thin", "still-blocked", "error"],
)
def test_rejected_firecrawl_result_falls_through_to_todays_outcome(vault, monkeypatch, bad):
    _FakeFirecrawl.result = bad
    _primary(monkeypatch, "bot", 403)
    code, out = _fetch()
    assert code == 1
    assert out["error_code"] == "JUNK_ESCALATED"
    outcomes = [r[0] for r in _rows(vault, "SELECT outcome FROM fallback_fetches")]
    assert outcomes == ["error" if bad is None else "rejected"]


def test_mixed_pdf_rescue_names_the_unread_pages(vault, monkeypatch):
    _FakeFirecrawl.result = WebResult(
        url=URL, title="Annual report", content=GOOD,
        metadata={"status_code": 200, "page_count": 2, "pages_with_text": 1, "cache_state": "hit"},
        raw_bytes=b"%PDF-1.7 fake", raw_content_type="application/pdf",
    )
    _primary(monkeypatch, "empty", 200)
    code, out = _fetch()
    assert code == 0
    _, body = _note(vault, out["data"]["note_id"])
    assert "1 of 2 pages had no text layer and were not read." in body
    assert "cached copy of unknown age" in body


def test_budget_is_spent_once_per_attempt(vault, monkeypatch):
    cfg = vault / ".hyperresearch" / "config.toml"
    cfg.write_text(cfg.read_text().replace("max_attempts_per_run = 25", "max_attempts_per_run = 1"))
    _FakeFirecrawl.result = None
    _primary(monkeypatch, "bot", 403)
    _fetch()
    runner.invoke(app, ["fetch", URL + "/2", "--tag", TAG, "--json"])
    assert _FakeFirecrawl.calls == [URL]
