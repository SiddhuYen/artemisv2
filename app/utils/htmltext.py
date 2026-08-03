"""Convert raw HTML into clean visible text for extraction."""
from __future__ import annotations

import json
import re
from typing import List

from bs4 import BeautifulSoup
from bs4.exceptions import ParserRejectedMarkup

from .. import config

_STRIP_TAGS = ["script", "style", "noscript", "head", "nav", "footer",
               "svg", "header", "aside", "form", "button", "template"]


def soup_of(html: str) -> BeautifulSoup:
    """Parse `html`, degrading to an EMPTY document on markup the parser
    refuses.

    html.parser raises ParserRejectedMarkup on genuinely malformed pages, and
    this is the chokepoint every consumer goes through (html_to_text,
    text_blocks, jsonld_names, firms._page_title). Letting it propagate means
    one bad page out of ~35 aborts a whole person's expansion — a real YC
    cache build lost Sam Altman's entire hop-0 that way. An unreadable page
    asserts nothing, which is exactly what an empty soup yields, so the node
    keeps the other 34 pages' evidence instead of losing all of it."""
    try:
        return BeautifulSoup(html or "", "html.parser")
    except ParserRejectedMarkup:
        return BeautifulSoup("", "html.parser")


# Inline <script>/<style> bodies, which routinely occupy the first several
# hundred KB of a modern page before any visible content. Stripped at FETCH
# time (see providers.base.fetch_page) so the truncation cap is spent on
# markup that can actually contain a roster.
#
# The negative lookahead on ld+json is load-bearing: schema.org Person data
# lives in <script type="application/ld+json">, jsonld_names() reads it, and
# both roster scrapers treat those names as authoritative precisely because
# they survive when visible text does not. Stripping every <script> would
# have deleted the best source in the page while trying to make room for it.
_INLINE_NOISE = re.compile(
    r"<script(?![^>]*ld\+json)\b[^>]*>.*?</script\s*>"
    r"|<style\b[^>]*>.*?</style\s*>"
    r"|<!--.*?-->",
    re.IGNORECASE | re.DOTALL,
)


def strip_inline_noise(html: str) -> str:
    """Drop inline script/style bodies and comments, keeping JSON-LD.

    Not a parser pass: this runs on multi-hundred-KB raw HTML before any
    truncation, where BeautifulSoup would be far more expensive and is not
    needed to delete two well-delimited element types.
    """
    if not html:
        return ""
    return _INLINE_NOISE.sub(" ", html)


def html_to_text(html: str) -> str:
    if not html:
        return ""
    soup = soup_of(html)
    for tag in soup(_STRIP_TAGS):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    # collapse runaway whitespace
    text = " ".join(text.split())
    return text[: config.MAX_PAGE_CHARS]


def text_blocks(html: str, max_chars: int = 80) -> List[str]:
    """Visible text of each DOM element, as SEPARATE strings.

    A roster page puts each person's name in its own element. Flattening the
    page to one string glues neighbouring elements together — a team grid's
    `<div>Email</div><div>Ryan Floyd</div>` would read as "Email Ryan Floyd"
    and downstream name-shape filtering invents a person out of the seam.
    Keeping the element boundaries is what makes a roster scrape trustworthy.

    Blocks longer than `max_chars` are prose, not roster cells, and dropped.
    """
    if not html:
        return []
    soup = soup_of(html)
    for tag in soup(_STRIP_TAGS):
        tag.decompose()

    seen, blocks = set(), []
    for line in soup.get_text("\n", strip=True).splitlines():
        line = " ".join(line.split())
        if not line or len(line) > max_chars or line in seen:
            continue
        seen.add(line)
        blocks.append(line)
    return blocks


def _walk_jsonld(node, want_type: str, out: list) -> None:
    """Collect the `name` of every object whose @type includes `want_type`."""
    if isinstance(node, dict):
        types = node.get("@type", "")
        types = types if isinstance(types, list) else [types]
        if want_type in types and isinstance(node.get("name"), str):
            out.append(node["name"].strip())
        for value in node.values():
            _walk_jsonld(value, want_type, out)
    elif isinstance(node, list):
        for item in node:
            _walk_jsonld(item, want_type, out)


def jsonld_names(html: str, schema_type: str = "Person") -> List[str]:
    """Names from schema.org `<script type="application/ld+json">` blocks.

    Some site builders embed structured Person data — the cleanest possible
    roster source, and a first-class structural assertion: the page DECLARES
    "this person is on our team". It also survives JS rendering intact, where
    visible-text scraping fails on sites whose team names live only in a
    JSON-LD graph, never in a text node.
    """
    if not html or "ld+json" not in html:
        return []
    soup = soup_of(html)
    names: List[str] = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text() or ""
        try:
            data = json.loads(raw)
        except Exception:
            continue
        _walk_jsonld(data, schema_type, names)
    seen, out = set(), []
    for n in names:
        if n and n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out
