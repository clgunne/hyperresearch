"""The paid fallback's guards: eligibility, run identity, budget, egress, key.

Every guard fails closed, so most tests assert that nothing was attempted.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

import pytest

from hyperresearch.core import fallback as fb
from hyperresearch.core import oa

# --- eligibility ------------------------------------------------------------

@pytest.mark.parametrize("status", [None, 200, 204, 401, 403, 429])
def test_block_shaped_statuses_are_eligible(status):
    assert fb.status_eligible(status, fb.KIND_EMPTY)


@pytest.mark.parametrize("status", [404, 410, 400, 418, 500, 502, 504])
def test_error_statuses_are_never_eligible(status):
    for kind in (fb.KIND_BOT, fb.KIND_EMPTY, fb.KIND_LOGIN_WALL, fb.KIND_BOILERPLATE):
        assert not fb.status_eligible(status, kind)


def test_503_needs_positive_block_evidence():
    assert fb.status_eligible(503, fb.KIND_BOT)
    assert not fb.status_eligible(503, fb.KIND_EMPTY)
    assert not fb.status_eligible(503, fb.KIND_BOILERPLATE)


@pytest.mark.parametrize(
    "reason,kind",
    [
        ("Bot detection page: Just a moment...", fb.KIND_BOT),
        ("Empty or near-empty content", fb.KIND_EMPTY),
        ("Cookie/boilerplate page", fb.KIND_BOILERPLATE),
        ("Error page: 404 Not Found", None),
        ("Search results page: Google", None),
        ("Binary PDF garbage in content", None),
        ("High ratio of binary/non-printable content", None),
    ],
)
def test_junk_reasons_map_to_kinds(reason, kind):
    assert fb.junk_kind(reason) == kind


# --- run identity -----------------------------------------------------------

def test_run_identity_needs_exactly_one_minted_tag_with_a_run_dir(tmp_path: Path):
    runs = tmp_path / "runs"
    (runs / "widgets-a1b2c3").mkdir(parents=True)
    (runs / "gadgets-d4e5f6").mkdir(parents=True)

    assert fb.run_identity(runs, ["widgets-a1b2c3"]) == "widgets-a1b2c3"
    assert fb.run_identity(runs, ["topic", "widgets-a1b2c3"]) == "widgets-a1b2c3"
    assert fb.run_identity(runs, ["widgets-a1b2c3", "topic"]) == "widgets-a1b2c3"  # order-free
    assert fb.run_identity(runs, []) is None
    assert fb.run_identity(runs, [""]) is None
    assert fb.run_identity(runs, ["widgets"]) is None  # not minted
    assert fb.run_identity(runs, ["nowhere-0a0b0c"]) is None  # no run dir
    assert fb.run_identity(runs, ["widgets-a1b2c3", "gadgets-d4e5f6"]) is None  # ambiguous


# --- reservation ------------------------------------------------------------

def _count(db: Path, tag: str) -> int:
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT COUNT(*) FROM fallback_fetches WHERE vault_tag = ?", (tag,)).fetchone()[0]


def test_reservation_stops_at_the_cap(tmp_path: Path):
    db = tmp_path / "v.db"
    ids = [fb.reserve(db, "run-aaaaaa", f"https://e.org/{i}", "empty", None, cap=3) for i in range(5)]
    assert sum(1 for i in ids if i) == 3
    assert ids[3] is None and ids[4] is None


def test_cap_is_per_run(tmp_path: Path):
    db = tmp_path / "v.db"
    assert fb.reserve(db, "one-aaaaaa", "https://e.org/1", "empty", None, cap=1)
    assert fb.reserve(db, "one-aaaaaa", "https://e.org/2", "empty", None, cap=1) is None
    assert fb.reserve(db, "two-bbbbbb", "https://e.org/3", "empty", None, cap=1)


def test_concurrent_fetchers_on_independent_connections_never_overshoot(tmp_path: Path):
    db = tmp_path / "v.db"
    fb.reserve(db, "warm-000000", "https://e.org/warm", "empty", None, cap=1)  # create the table
    results: list[str | None] = []
    lock = threading.Lock()
    start = threading.Barrier(12)

    def worker(i):
        start.wait()
        rid = fb.reserve(db, "race-cccccc", f"https://e.org/{i}", "bot_block", 403, cap=5)
        with lock:
            results.append(rid)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(1 for r in results if r) == 5
    assert _count(db, "race-cccccc") == 5


def test_a_reservation_left_by_a_crash_still_counts(tmp_path: Path):
    db = tmp_path / "v.db"
    rid = fb.reserve(db, "crash-dddddd", "https://e.org/1", "empty", None, cap=1)
    assert rid  # ...and the process dies here, before finish()
    assert fb.reserve(db, "crash-dddddd", "https://e.org/2", "empty", None, cap=1) is None
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT outcome FROM fallback_fetches WHERE id = ?", (rid,)).fetchone()[0] == "reserved"


def test_a_lock_timeout_reserves_nothing(tmp_path: Path, monkeypatch):
    db = tmp_path / "v.db"
    fb.reserve(db, "warm-000000", "https://e.org/warm", "empty", None, cap=5)
    monkeypatch.setattr(fb, "_BUSY_MS", 100)
    holder = sqlite3.connect(db, isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")
    try:
        assert fb.reserve(db, "lock-eeeeee", "https://e.org/1", "empty", None, cap=5) is None
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert _count(db, "lock-eeeeee") == 0


def test_finish_records_outcome_and_fields(tmp_path: Path):
    db = tmp_path / "v.db"
    rid = fb.reserve(db, "done-ffffff", "https://e.org/1", "empty", None, cap=5)
    fb.finish(db, rid, "rescued", credits=1, cache_state="miss", note_id="n1", ignored="x")
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT outcome, credits, cache_state, note_id FROM fallback_fetches WHERE id = ?", (rid,)
        ).fetchone()
    assert row == ("rescued", 1, "miss", "n1")


# --- egress -----------------------------------------------------------------

@pytest.fixture
def public_dns(monkeypatch):
    monkeypatch.setattr(oa.socket, "getaddrinfo", lambda h, p: [(2, 1, 6, "", ("93.184.216.34", 0))])


DENY = fb.FirecrawlSettings().deny_query_params


def test_plain_public_url_may_be_sent(public_dns):
    assert fb.egress_ok("https://example.org/report?page=2&lang=en", DENY) == (True, "")


@pytest.mark.parametrize(
    "url",
    [
        "https://bucket.s3.amazonaws.com/f.pdf?X-Amz-Algorithm=AWS4&X-Amz-Signature=abc",
        "https://storage.googleapis.com/b/f.pdf?X-Goog-Algorithm=GOOG4&X-Goog-Signature=abc",
        "https://storage.googleapis.com/b/f.pdf?GoogleAccessId=x&Expires=1&Signature=abc",
        "https://d111.cloudfront.net/f.pdf?Expires=1&Signature=abc&Key-Pair-Id=K1",
        "https://acct.blob.core.windows.net/c/f.pdf?sv=2020&sig=abc",
        "https://example.org/doc?access_token=abc",
        "https://example.org/doc?apikey=abc",
        "https://example.org/doc?auth=abc",
    ],
    ids=["aws", "gcs-v4", "gcs-v2", "cloudfront", "azure-sas", "token", "apikey", "auth"],
)
def test_signed_or_token_bearing_urls_are_never_sent(public_dns, url):
    ok, why = fb.egress_ok(url, DENY)
    assert not ok and "credential" in why


def test_embedded_credentials_are_never_sent(public_dns):
    ok, _ = fb.egress_ok("https://user:pass@example.org/doc", DENY)
    assert not ok


def test_bare_and_private_hosts_are_never_sent(monkeypatch):
    assert not fb.egress_ok("http://intranet/doc", DENY)[0]
    monkeypatch.setattr(oa.socket, "getaddrinfo", lambda h, p: [(2, 1, 6, "", ("10.0.0.5", 0))])
    assert not fb.egress_ok("https://wiki.corp.example/doc", DENY)[0]


# --- key --------------------------------------------------------------------

def test_key_from_environment_wins(tmp_path: Path, monkeypatch):
    (tmp_path / ".env").write_text("FIRECRAWL_API_KEY=fc-file\n")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-env")
    assert fb.resolve_key(tmp_path) == "fc-env"


def test_key_from_vault_env_file(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("# comment\nOTHER=1\nFIRECRAWL_API_KEY='fc-file'\n")
    os.chmod(env, 0o600)
    assert fb.resolve_key(tmp_path) == "fc-file"


def test_no_key_anywhere(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    assert fb.resolve_key(tmp_path) is None


# --- rescue size ---------------------------------------------------------------

def test_prose_words_reduce_links_to_their_text():
    assert fb.prose_words("Read [the full report](https://example.org/a/b/c) today") == 5
    assert fb.prose_words("![Logo](https://example.org/l.png)") == 1
    assert fb.prose_words("") == 0


def test_prose_words_stays_linear_on_adversarial_input():
    import time

    hostile = "[" * 1_000_000 + "](" + "x" * 100_000
    started = time.monotonic()
    fb.prose_words(hostile, stop_at=400)
    assert time.monotonic() - started < 2.0
