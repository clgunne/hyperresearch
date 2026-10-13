"""Paid fetch fallback for blocked sources (Firecrawl).

`hyperresearch fetch` calls `try_fallback` after the open-access rescue fails
and before it queues the URL for the Chrome lane. Every guard here fails
closed: no key, no opt-in, no run identity, an ineligible block, an unsafe URL,
a spent budget or a reservation that cannot be written all mean "behave
exactly as before".

Spend is bounded per run. A run is its vault tag (`<slug>-<6-hex>`, minted by
`hyperresearch vault-tag`, with a directory under `research/runs/`), and each
attempt is reserved in the `fallback_fetches` table before the paid call, so
parallel fetchers cannot overshoot the cap and a crash after the call still
counts against it.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlparse

from hyperresearch.core.config import FirecrawlSettings, JunkGates

RUN_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,59}-[0-9a-f]{6}$")
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")

# Block kinds the fallback may retry. Anything else (error pages, search
# results, binary garbage) is not a block: a 404 through Firecrawl is still a 404.
KIND_FETCH_ERROR = "fetch_error"
KIND_LOGIN_WALL = "login_wall"
KIND_BOT = "bot_block"
KIND_EMPTY = "empty"
KIND_BOILERPLATE = "boilerplate"

_KIND_TEXT = {
    KIND_FETCH_ERROR: "the fetch failed",
    KIND_LOGIN_WALL: "a login wall",
    KIND_BOT: "a bot-detection page",
    KIND_EMPTY: "an empty page",
    KIND_BOILERPLATE: "a cookie or JavaScript wall",
}

_SCHEMA = """CREATE TABLE IF NOT EXISTS fallback_fetches (
    id          TEXT PRIMARY KEY,
    vault_tag   TEXT NOT NULL,
    url         TEXT NOT NULL,
    provider    TEXT NOT NULL,
    reason      TEXT NOT NULL,
    orig_status INTEGER,
    outcome     TEXT NOT NULL,
    credits     INTEGER,
    cache_state TEXT,
    cached_at   TEXT,
    note_id     TEXT,
    at          TEXT NOT NULL
)"""
_BUSY_MS = 10_000


def _log() -> logging.Logger:
    return logging.getLogger("hyperresearch.fallback")


def _host(url: str) -> str:
    return urlparse(url).hostname or "?"


def junk_kind(junk_reason: str) -> str | None:
    """Map a `looks_like_junk` reason to a retryable block kind, or None."""
    if junk_reason.startswith("Bot detection"):
        return KIND_BOT
    if junk_reason.startswith("Empty or near-empty"):
        return KIND_EMPTY
    if junk_reason.startswith("Cookie/boilerplate"):
        return KIND_BOILERPLATE
    return None


def status_eligible(status: int | None, kind: str) -> bool:
    """Is the original HTTP status one a paid retry could fix?

    Unknown (exceptions, providers that report none), 2xx (a page that loaded
    but was a wall), 401, 403 and 429 are. A 503 is only with positive block
    evidence, since Cloudflare challenges answer 503 but so do real outages.
    Every other status is a genuine error page.
    """
    if status is None or 200 <= status < 300 or status in (401, 403, 429):
        return True
    if status == 503:
        return kind == KIND_BOT
    return False


def run_identity(runs_dir: Path, tags: list[str]) -> str | None:
    """The single minted vault tag among `tags` that has a run directory."""
    found = {t for t in tags if RUN_TAG_RE.match(t) and (runs_dir / t).is_dir()}
    return found.pop() if len(found) == 1 else None


def egress_ok(url: str, deny_query_params: tuple[str, ...]) -> tuple[bool, str]:
    """May this URL be disclosed to a third-party fetcher?"""
    from hyperresearch.core.oa import check_oa_url

    ok, why = check_oa_url(url)
    if not ok:
        return False, why
    patterns = [p.lower() for p in deny_query_params]
    for name, _ in parse_qsl(urlparse(url).query, keep_blank_values=True):
        lname = name.lower()
        if any(fnmatch.fnmatchcase(lname, p) for p in patterns):
            return False, f"query parameter {name!r} looks like a credential"
    return True, ""


def resolve_key(hyperresearch_dir: Path) -> str | None:
    """FIRECRAWL_API_KEY from the environment, else `.hyperresearch/.env`."""
    key = os.environ.get("FIRECRAWL_API_KEY", "").strip()
    if key:
        return key
    env_file = hyperresearch_dir / ".env"
    if not env_file.is_file():
        return None
    try:
        if env_file.stat().st_mode & 0o077:
            _log().warning("%s is readable by other users; chmod 600 it", env_file)
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        name, sep, value = line.strip().partition("=")
        if sep and name.strip() == "FIRECRAWL_API_KEY":
            return value.strip().strip("'\"") or None
    return None


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=_BUSY_MS / 1000, isolation_level=None)
    conn.execute(f"PRAGMA busy_timeout = {_BUSY_MS}")
    conn.execute(_SCHEMA)
    return conn


def reserve(db_path: Path, vault_tag: str, url: str, reason: str,
            orig_status: int | None, cap: int) -> str | None:
    """Reserve one attempt for this run. Returns its id, or None for no attempt.

    The count and the insert share one IMMEDIATE transaction, committed before
    any HTTP call, so the write lock is never held across a scrape.
    """
    conn = None
    try:
        conn = _connect(db_path)
        conn.execute("BEGIN IMMEDIATE")
        used = conn.execute(
            "SELECT COUNT(*) FROM fallback_fetches WHERE vault_tag = ?", (vault_tag,)
        ).fetchone()[0]
        if used >= cap:
            conn.execute("ROLLBACK")
            return None
        rid = uuid.uuid4().hex
        conn.execute(
            """INSERT INTO fallback_fetches (id, vault_tag, url, provider, reason, orig_status, outcome, at)
               VALUES (?, ?, ?, 'firecrawl', ?, ?, 'reserved', ?)""",
            (rid, vault_tag, url, reason, orig_status, datetime.now(UTC).isoformat()),
        )
        conn.execute("COMMIT")
        return rid
    except sqlite3.Error as e:
        _log().warning("could not reserve a fallback attempt: %s", e)
        if conn is not None:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
        return None
    finally:
        if conn is not None:
            conn.close()


def finish(db_path: Path, rid: str, outcome: str, **fields) -> None:
    """Record an attempt's outcome. Never raises."""
    allowed = {"credits", "cache_state", "cached_at", "note_id"}
    cols = {k: v for k, v in fields.items() if k in allowed}
    sets = ", ".join(["outcome = ?"] + [f"{k} = ?" for k in cols])
    try:
        conn = _connect(db_path)
        try:
            conn.execute(f"UPDATE fallback_fetches SET {sets} WHERE id = ?", (outcome, *cols.values(), rid))
        finally:
            conn.close()
    except sqlite3.Error as e:
        _log().warning("could not record a fallback outcome: %s", e)


def prose_words(text: str) -> int:
    """Words left once markdown links and images are reduced to their text."""
    return len(_MD_LINK.sub(r"\1", text or "").split())


def _accepted(result, gates: JunkGates, min_words: int) -> bool:
    status = result.metadata.get("status_code")
    if status is not None and not (200 <= status < 300):
        return False
    if len(result.content or "") < gates.min_content_chars:
        return False
    if prose_words(result.content) < min_words:
        return False
    if result.looks_like_login_wall(result.url, gates):
        return False
    return result.looks_like_junk(gates) is None


def try_fallback(
    *,
    vault,
    url: str,
    tags: list[str],
    utility_score: float | None,
    kind: str,
    orig_status: int | None,
):
    """Retry a blocked fetch through Firecrawl. Returns (WebResult, ticket) or None."""
    settings: FirecrawlSettings = vault.config.firecrawl
    if not settings.enabled:
        return None
    if utility_score is not None and utility_score < settings.utility_threshold:
        return None
    if not status_eligible(orig_status, kind):
        return None
    vault_tag = run_identity(vault.runs_dir, tags)
    if vault_tag is None:
        return None
    key = resolve_key(vault.hyperresearch_dir)
    if not key:
        return None
    ok, why = egress_ok(url, settings.deny_query_params)
    if not ok:
        _log().info("not sending %s to Firecrawl: %s", _host(url), why)
        return None

    rid = reserve(vault.db_path, vault_tag, url, kind, orig_status, settings.max_attempts_per_run)
    if rid is None:
        return None

    from hyperresearch.web.firecrawl_provider import FirecrawlProvider

    provider = FirecrawlProvider(key, max_age_ms=settings.max_age_ms, timeout_ms=settings.timeout_ms)
    try:
        result = provider.fetch(url)
    except Exception as e:  # the provider promises not to raise; belt and braces
        _log().warning("Firecrawl fetch for %s raised %s", _host(url), type(e).__name__)
        result = None

    if result is None:
        finish(vault.db_path, rid, "error")
        return None
    meta = result.metadata
    recorded = {
        "credits": meta.get("credits_used") if isinstance(meta.get("credits_used"), int) else None,
        "cache_state": meta.get("cache_state"),
        "cached_at": meta.get("cached_at"),
    }
    if not _accepted(result, vault.config.junk, settings.min_rescue_words):
        finish(vault.db_path, rid, "rejected", **recorded)
        return None
    finish(vault.db_path, rid, "rescued", **recorded)
    ticket = {"id": rid, "kind": kind, "orig_status": orig_status, "db_path": vault.db_path, **recorded}
    return result, ticket


def link_note(ticket: dict, note_id: str) -> None:
    finish(ticket["db_path"], ticket["id"], "rescued", note_id=note_id)


def fallback_notice(ticket: dict, result) -> str:
    """The banner at the top of a rescued note's body."""
    status = ticket.get("orig_status")
    why = _KIND_TEXT.get(ticket["kind"], ticket["kind"])
    if status is not None:
        why += f", HTTP {status}"
    cache_state = ticket.get("cache_state")
    if cache_state == "hit":
        cached_at = ticket.get("cached_at")
        cache = f"Firecrawl served a cached copy dated {cached_at}." if cached_at else \
            "Firecrawl served a cached copy of unknown age."
    elif cache_state == "miss":
        cache = "Firecrawl fetched it live."
    else:
        cache = "Cache age unknown."
    lines = [
        f"> [!] **Retrieved via Firecrawl.** hyperresearch could not read this page directly ({why}). {cache}"
    ]
    page_count = result.metadata.get("page_count")
    with_text = result.metadata.get("pages_with_text")
    if isinstance(page_count, int) and isinstance(with_text, int) and with_text < page_count:
        lines.append(
            f"> {page_count - with_text} of {page_count} pages had no text layer and were not read."
        )
    return "\n".join(lines) + "\n"
