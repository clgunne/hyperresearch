"""FirecrawlProvider: one bounded scrape, mapped to a WebResult or None.

Offline: every test drives the provider through an httpx.MockTransport.
"""

from __future__ import annotations

import base64
import json

import httpx
import pymupdf
import pytest

from hyperresearch.web.firecrawl_provider import API_URL, FirecrawlProvider


def _pdf_bytes(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _provider(handler, **kw) -> tuple[FirecrawlProvider, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def _record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(_record))
    return FirecrawlProvider("fc-test", client=client, **kw), seen


def _ok(markdown: str, **meta) -> httpx.Response:
    metadata = {"statusCode": 200, "url": "https://example.org/final", "title": "A page", **meta}
    return httpx.Response(200, json={"success": True, "data": {"markdown": markdown, "metadata": metadata}})


def test_request_body_is_fixed_on_every_call():
    prov, seen = _provider(lambda r: _ok("x" * 500), max_age_ms=1234, timeout_ms=5000)
    prov.fetch("https://example.org/page")

    req = seen[0]
    assert str(req.url) == API_URL
    assert req.headers["Authorization"] == "Bearer fc-test"
    body = json.loads(req.content)
    assert body["url"] == "https://example.org/page"
    assert body["parsers"] == []  # a PDF must come back raw: 1 credit, parsed locally
    assert body["storeInCache"] is False
    assert body["proxy"] == "auto"
    assert body["onlyMainContent"] is True
    assert body["maxAge"] == 1234
    assert body["timeout"] == 5000


def test_html_maps_to_web_result_with_status_and_billing():
    prov, _ = _provider(lambda r: _ok("Body " * 100, creditsUsed=1, cacheState="miss"))
    result = prov.fetch("https://example.org/page")

    assert result is not None
    assert result.url == "https://example.org/final"
    assert result.title == "A page"
    assert result.content.startswith("Body ")
    assert result.metadata["status_code"] == 200
    assert result.metadata["credits_used"] == 1
    assert result.metadata["cache_state"] == "miss"


def test_list_title_is_flattened():
    prov, _ = _provider(lambda r: _ok("Body " * 100, title=["First", "Second"]))
    assert prov.fetch("https://example.org/page").title == "First"


def test_pdf_from_extensionless_url_is_decoded_and_parsed_in_full():
    pdf = _pdf_bytes(["Page one of the report, with enough words.", "Page two continues here."])
    b64 = base64.b64encode(pdf).decode()
    prov, _ = _provider(lambda r: _ok(b64, contentType="application/pdf", creditsUsed=1))

    result = prov.fetch("https://example.org/download?id=42")

    assert result is not None
    assert "Page one" in result.content and "Page two" in result.content
    assert result.metadata["page_count"] == 2
    assert result.metadata["pages_with_text"] == 2
    assert result.metadata["credits_used"] == 1
    assert result.raw_content_type == "application/pdf"


def test_pdf_detected_by_magic_when_content_type_missing():
    b64 = base64.b64encode(_pdf_bytes(["Only page, with plenty of text on it."])).decode()
    prov, _ = _provider(lambda r: _ok(b64))
    result = prov.fetch("https://example.org/x")
    assert result is not None and "Only page" in result.content


def test_mixed_pdf_reports_pages_without_text():
    b64 = base64.b64encode(_pdf_bytes(["Searchable first page with words.", ""])).decode()
    prov, _ = _provider(lambda r: _ok(b64, contentType="application/pdf"))
    result = prov.fetch("https://example.org/mixed.pdf")
    assert result.metadata["page_count"] == 2
    assert result.metadata["pages_with_text"] == 1


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(402, json={"success": False, "error": "Payment required"}),
        httpx.Response(429, json={"success": False}),
        httpx.Response(500, text="oops"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"success": False}),
        httpx.Response(200, json={"success": True, "data": {"metadata": {}}}),
        httpx.Response(200, json={"success": True, "data": "nope"}),
    ],
    ids=["402", "429", "500", "non-json", "unsuccessful", "no-markdown", "bad-data"],
)
def test_failures_return_none_without_raising(response):
    prov, _ = _provider(lambda r: response)
    assert prov.fetch("https://example.org/page") is None


def test_timeout_returns_none():
    def boom(request):
        raise httpx.ReadTimeout("slow", request=request)

    prov, _ = _provider(boom)
    assert prov.fetch("https://example.org/page") is None


def test_garbled_pdf_returns_none():
    prov, _ = _provider(lambda r: _ok("JVBERi0" + "!!!not base64!!!", contentType="application/pdf"))
    assert prov.fetch("https://example.org/a.pdf") is None


def test_logs_never_carry_the_full_url(caplog):
    prov, _ = _provider(lambda r: httpx.Response(500))
    with caplog.at_level("WARNING", logger="hyperresearch.firecrawl"):
        prov.fetch("https://example.org/doc?token=secret123")
    assert "example.org" in caplog.text
    assert "secret123" not in caplog.text
