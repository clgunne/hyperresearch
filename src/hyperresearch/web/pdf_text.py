"""PDF bytes to a WebResult, shared by every path that obtains raw PDF bytes.

The crawl4ai provider downloads PDFs itself; the Firecrawl fallback receives
them base64-encoded (`parsers: []`). Both hand the bytes here, so a PDF is read
the same way, in full, whichever path fetched it.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from hyperresearch.web.base import WebResult


def _pdf_log() -> logging.Logger:
    return logging.getLogger("hyperresearch.pdf")


def pdf_bytes_to_result(url: str, pdf_bytes: bytes, pymupdf) -> WebResult | None:
    """Extract every page's text. Returns None when no page has a text layer.

    `metadata` reports coverage as well as the text: `page_count` is the whole
    document and `pages_with_text` the pages that yielded anything, so a mixed
    PDF whose scanned pages were skipped is never mistaken for the complete
    document. `pages` keeps its old meaning (pages with text).
    """
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    pages = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text)

    page_count = doc.page_count
    doc.close()

    if not pages:
        _pdf_log().warning(
            "PDF at %s has %d page(s) but no extractable text layer — "
            "likely a scanned document requiring OCR.", url, page_count,
        )
        return None

    full_text = "\n\n---\n\n".join(pages)
    title = ""
    # Try to get title from first page (first non-empty line)
    for line in pages[0].split("\n"):
        line = line.strip()
        if len(line) > 10:
            title = line
            break

    return WebResult(
        url=url,
        title=title or f"PDF: {url.split('/')[-1]}",
        content=full_text,
        fetched_at=datetime.now(UTC),
        metadata={
            "content_type": "application/pdf",
            "pages": len(pages),
            "page_count": page_count,
            "pages_with_text": len(pages),
        },
        raw_bytes=pdf_bytes,
        raw_content_type="application/pdf",
    )
