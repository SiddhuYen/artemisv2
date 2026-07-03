"""Convert raw HTML into clean visible text for extraction."""
from __future__ import annotations

from bs4 import BeautifulSoup

from .. import config


def html_to_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "head", "nav", "footer",
                     "svg", "header", "aside", "form", "button", "template"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    # collapse runaway whitespace
    text = " ".join(text.split())
    return text[: config.MAX_PAGE_CHARS]
