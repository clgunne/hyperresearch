"""Firecrawl provider: the paid fallback for pages crawl4ai cannot read.

Used only as a fallback inside `hyperresearch fetch` (see
`hyperresearch.core.fallback`), never as a vault's primary provider: it bills
per request, so it fires only after the primary crawl and the open-access
rescue have both failed.

Firecrawl only fetches. Every request sends `parsers: []`, so a PDF comes back
raw (base64, a flat 1 credit) and is parsed locally in full by the shared
PyMuPDF path, instead of by Firecrawl's per-page-billed PDF parser.

Configuration:
    export FIRECRAWL_API_KEY="fc-..."    # or FIRECRAWL_API_KEY=... in .hyperresearch/.env

    # in .hyperresearch/config.toml
    [firecrawl]
    enabled = true
"""

from __future__ import annotations

import base64
import binascii
import logging
from urllib.parse import urlparse

import httpx

from hyperresearch.web.base import WebResult
from hyperresearch.web.pdf_text import pdf_bytes_to_result

API_URL = "https://api.firecrawl.dev/v2/scrape"
_PDF_B64_PREFIX = "JVBERi0"  # base64 of "%PDF-"


def _log() -> logging.Logger:
    return logging.getLogger("hyperresearch.firecrawl")


def _host(url: str) -> str:
    # Diagnostics name the host only: a full URL can carry tokens.
    return urlparse(url).hostname or "?"


class FirecrawlProvider:
    name = "firecrawl"

    def __init__(
        self,
        api_key: str,
        *,
        max_age_ms: int = 172_800_000,
        timeout_ms: int = 60_000,
        client: httpx.Client | None = None,
    ):
        self._key = api_key
        self._max_age_ms = max_age_ms
        self._timeout_ms = timeout_ms
        self._client = client

    def _request_body(self, url: str) -> dict:
        return {
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": True,
            "proxy": "auto",
            "parsers": [],
            "storeInCache": False,
            "maxAge": self._max_age_ms,
            "timeout": self._timeout_ms,
        }

    def fetch(self, url: str) -> WebResult | None:
        """One scrape. Returns None on any failure; never raises."""
        headers = {"Authorization": f"Bearer {self._key}"}
        # Client-side budget a little above Firecrawl's own server-side timeout.
        timeout = self._timeout_ms / 1000 + 15
        post = self._client.post if self._client is not None else httpx.post
        try:
            resp = post(API_URL, json=self._request_body(url), headers=headers, timeout=timeout)
        except httpx.HTTPError as e:
            _log().warning("Firecrawl request for %s failed: %s", _host(url), type(e).__name__)
            return None

        if resp.status_code != 200:
            _log().warning("Firecrawl returned HTTP %s for %s", resp.status_code, _host(url))
            return None
        try:
            payload = resp.json()
        except ValueError:
            _log().warning("Firecrawl returned a non-JSON body for %s", _host(url))
            return None
        data = payload.get("data") if isinstance(payload, dict) and payload.get("success") else None
        if not isinstance(data, dict):
            _log().warning("Firecrawl response for %s had no data", _host(url))
            return None
        meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        content = data.get("markdown")
        if not isinstance(content, str):
            return None

        status = meta.get("statusCode")
        title = meta.get("title")
        if isinstance(title, list):
            title = title[0] if title else ""
        final_url = meta.get("url") if isinstance(meta.get("url"), str) else url
        extra = {
            "status_code": status if isinstance(status, int) else None,
            "credits_used": meta.get("creditsUsed"),
            "cache_state": meta.get("cacheState"),
            "cached_at": meta.get("cachedAt"),
        }

        content_type = str(meta.get("contentType") or "").lower()
        if "pdf" in content_type or content.startswith(_PDF_B64_PREFIX):
            return self._pdf_result(final_url, content, extra)

        return WebResult(
            url=final_url,
            title=title if isinstance(title, str) else "",
            content=content,
            metadata={"title": title, **extra},
        )

    def _pdf_result(self, url: str, b64: str, extra: dict) -> WebResult | None:
        try:
            pdf_bytes = base64.b64decode(b64, validate=False)
        except (binascii.Error, ValueError):
            _log().warning("Firecrawl PDF for %s was not valid base64", _host(url))
            return None
        if not pdf_bytes.startswith(b"%PDF-"):
            _log().warning("Firecrawl PDF for %s did not decode to PDF data", _host(url))
            return None
        try:
            import pymupdf
        except ImportError:
            _log().warning("pymupdf is unavailable, so the Firecrawl PDF for %s cannot be read", _host(url))
            return None
        try:
            result = pdf_bytes_to_result(url, pdf_bytes, pymupdf)
        except Exception as e:
            _log().warning("PDF extraction failed for %s: %s", _host(url), type(e).__name__)
            return None
        if result is None:
            return None
        result.metadata.update(extra)
        return result
