"""Funding announcements -> org<->org co-investment facts (tier 3).

A round announcement is the one free source that *names the investors in a
round*. "Splitero secures $11.7M in Series A funding led by Fiat Ventures, with
participation from Gemini Ventures, Joint Effects and PBJ Capital" asserts that
those firms invested together. That is a structural assertion about the FIRMS.

The article is prose, though, and prose is exactly where co-occurrence hides.
So an investor is extracted ONLY from the span governed by an investor cue
("led by", "with participation from", ...), never from a capitalised name found
anywhere on the page. Three further guards, each from a real failure:

  * "Fiat Ventures, and its team, led by managing partner, Marcos Fernandez"
    puts a PERSON after "led by". Candidates must be org-shaped, so a person's
    name is rejected.
  * "Fiat Ventures, with $25M for first fund" announces a FUND, not a round.
    A cue must be present; a fund story has none.
  * Aggregators (ZoomInfo, Tracxn, LinkedIn) paraphrase rounds they were not
    party to. Only the announcing outlet or the company's own post is read.

IMPORTANT: this module only PARSES rounds into {company, investors[]} — it
never touches the graph. Callers (graph.builder.record_coinvestment) persist
the result as an ORGANIZATION-level fact only, never chained into a
person-to-person edge. See builder.record_coinvestment's docstring.

Ported from DrewGloverDemo's providers/funding.py.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .. import config
from ..utils.names import (
    ORG_SUFFIXES,
    is_noise_name,
    looks_like_org_name,
    looks_like_person_name,
    normalize,
    org_norm_key,
)
from ..utils.htmltext import html_to_text, soup_of
from . import cache
from .base import fetch_page

# Phrases that GOVERN an investor list. Text outside these spans is prose.
_CUES = (
    "with participation from", "with additional participation from",
    "participation from", "with support from",
    "co-led by", "led by", "backed by", "investors include",
    "investment from", "joined by",
)
_CUE_RE = re.compile("|".join(re.escape(c) for c in sorted(_CUES, key=len,
                                                           reverse=True)), re.I)

# A span ends at a sentence boundary, a clause that changes subject, or THE NEXT
# CUE. Without the last, "led by" swallows "Fiat Ventures with participation from
# Gemini Ventures" as one 7-token candidate, and both investors are lost.
_SPAN_END = re.compile(
    rf"[.;]|\bto\s|\bwhich\s|\bthat\s|\bwill\s|\bsaid\s|(?:{_CUE_RE.pattern})",
    re.I)
_SPLIT = re.compile(r"\s*(?:,|\band\b|&|\bas well as\b)\s*", re.I)

# The lead investor often PRECEDES the verb: "Fiat Ventures led the investment
# round with additional participation from ...". A cue-only scan misses it, and
# then the firm's own round fails the self-guard and is discarded.
_LED_PREFIX_RE = re.compile(
    r"\b([A-Z][\w&.'-]*(?:\s+[A-Z][\w&.'-]*){0,3})\s+"
    r"(?:co-)?led\s+(?:the|this|a|an)\s+"
    r"(?:\w+\s+){0,2}?(?:round|investment|financing|raise|seed|series)\b")

# "Splitero secures $11.7M in Series A funding led by ..." — the verb is
# title-cased in headlines, so match it case-insensitively without letting the
# company group match lowercase prose.
_COMPANY_RE = re.compile(
    r"^\s*([A-Z][\w.'&-]*(?:\s+[A-Z][\w.'&-]*){0,3})\s+"
    r"(?i:raises|raised|secures|secured|announces|announced|closes|closed"
    r"|lands|nets)\b",
    re.M)
_MONEY_RE = re.compile(r"\$\s?\d[\d.,]*\s?(?:million|billion|m|bn|k)?\b", re.I)

# Aggregators and socials restate rounds; they are not the announcement.
_BLOCKED = ("zoominfo.com", "tracxn.com", "linkedin.com", "instagram.com",
            "x.com", "twitter.com", "facebook.com", "crunchbase.com",
            "pitchbook.com", "dealroom.co", "cbinsights.com", "owler.com",
            "youtube.com", "reddit.com", "wikipedia.org", "signal.nfx.com",
            "f6s.com", "golden.com")

# Words that mark a candidate as a role or a leftover clause, not a firm.
_ROLE_WORDS = {"managing", "general", "partner", "founder", "cofounder",
               "ceo", "cto", "president", "chairman", "existing", "other",
               "several", "various", "angel", "angels", "investors", "team"}

_MAX_INVESTORS_PER_ROUND = 12
_MAX_SPAN_CHARS = 220
# Cue mentions further apart than this belong to DIFFERENT rounds.
_CLUSTER_GAP_CHARS = 400
# A page describing this many separate rounds is a newsletter roundup, not an
# announcement. Its investors have nothing to do with one another.
_MAX_ROUNDS_PER_PAGE = 2


def _is_blocked(url: str) -> bool:
    return any(bad in url.lower() for bad in _BLOCKED)


# A candidate may still carry a cue, possibly behind a verb: splitting
# "led by Northzone and was joined by Accel Partners" on "and" leaves the second
# piece as "was joined by Accel Partners". Drop everything up to the last cue.
_EMBEDDED_CUE_RE = re.compile(rf"^.*?(?:{_CUE_RE.pattern})\s*", re.I)


def _truncate_at_org_suffix(name: str) -> str:
    """Cut a candidate after its last organisational suffix.

    "Co-Led by Fiat Ventures Today Odynn, the AI-powered platform..." splits into
    the run-on "Fiat Ventures Today Odynn". The firm ends at "Ventures".
    """
    words = name.split()
    last = -1
    for i, word in enumerate(words):
        if normalize(word) in ORG_SUFFIXES:
            last = i
    return " ".join(words[: last + 1]) if last >= 0 else name


def _clean_candidate(raw: str) -> str:
    text = " ".join((raw or "").split())
    # A sentence boundary inside the match is a remnant of the previous
    # sentence: "…Series A Funding. Fiat Ventures led…" -> "Fiat Ventures".
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    text = _EMBEDDED_CUE_RE.sub("", text)
    text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.I)
    text = text.strip(" .,:;'\"()[]")
    # Drop a trailing possessive/qualifier: "Fiat Ventures' existing investors"
    text = re.sub(r"['’]s?\b.*$", "", text).strip()
    return _truncate_at_org_suffix(text)


def looks_like_investor(name: str) -> bool:
    """An investor in a round is an ORGANISATION here.

    Rejects "Marcos Fernandez" (a person who happens to follow "led by") and
    "managing partner" (a role). Angel individuals are real investors but are
    deliberately out of scope: we cannot tell them from a quoted executive.
    """
    if not name or len(name) > 48 or is_noise_name(name):
        return False
    tokens = normalize(name).split()
    if not (1 <= len(tokens) <= 5):
        return False
    if any(tok in _ROLE_WORDS for tok in tokens):
        return False
    if looks_like_person_name(name):
        return False
    return looks_like_org_name(name)


def _investors_in_span(span: str) -> List[str]:
    out, seen = [], set()
    for piece in _SPLIT.split(span):
        candidate = _clean_candidate(piece)
        if not looks_like_investor(candidate):
            continue
        key = org_norm_key(candidate)
        if key and key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def _mentions(text: str) -> List[tuple]:
    """Every investor mention as (start, end, [names]), in document order."""
    found: List[tuple] = []
    for match in _LED_PREFIX_RE.finditer(text):
        names = _investors_in_span(match.group(1))
        if names:
            found.append((match.start(), match.end(), names))
    for match in _CUE_RE.finditer(text):
        tail = text[match.end(): match.end() + _MAX_SPAN_CHARS]
        stop = _SPAN_END.search(tail)
        span = tail[: stop.start()] if stop else tail
        names = _investors_in_span(span)
        if names:
            found.append((match.start(), match.end() + len(span), names))
    return sorted(found)


def _cluster_rounds(text: str) -> List[Dict[str, object]]:
    """Group investor mentions into ROUNDS by proximity.

    A page holds one round when its cues sit together ("led by X, with
    participation from Y and Z"). A newsletter roundup holds many, hundreds of
    characters apart. Merging them fuses investors who never met.
    """
    rounds: List[Dict[str, object]] = []
    for start, end, names in _mentions(text):
        if rounds and start - rounds[-1]["end"] <= _CLUSTER_GAP_CHARS:
            current = rounds[-1]
            current["end"] = max(current["end"], end)
            for name in names:
                key = org_norm_key(name)
                if key not in current["keys"]:
                    current["keys"].add(key)
                    current["investors"].append(name)
        else:
            rounds.append({"start": start, "end": end, "investors": list(names),
                           "keys": {org_norm_key(n) for n in names}})
    return rounds


def parse_round(text: str, target_key: str = "") -> Dict[str, object]:
    """Extract {company, amount, investors[], evidence} for ONE round.

    `target_key` selects the round that names that firm. Investors are never
    pooled across rounds: a page with more than `_MAX_ROUNDS_PER_PAGE` of them
    is a roundup and yields nothing at all.

    Returns investors=[] when no cue is present — a fund launch or a profile
    page mentions firms without asserting that they invested together.
    """
    result: Dict[str, object] = {"company": "", "amount": "", "investors": [],
                                 "evidence": ""}
    if not text:
        return result

    rounds = _cluster_rounds(text)
    if not rounds or len(rounds) > _MAX_ROUNDS_PER_PAGE:
        return result  # no cue, or a multi-round digest

    chosen = None
    if target_key:
        chosen = next((r for r in rounds if target_key in r["keys"]), None)
    else:
        chosen = rounds[0]
    if chosen is None:
        return result

    window = text[max(0, chosen["start"] - 120): chosen["end"]]
    company = _COMPANY_RE.search(text)
    money = _MONEY_RE.search(window) or _MONEY_RE.search(text)
    result.update({
        "company": company.group(1).strip() if company else "",
        "amount": money.group(0).strip() if money else "",
        "investors": chosen["investors"][:_MAX_INVESTORS_PER_ROUND],
        "evidence": " ".join(window.split())[:300],
    })
    return result


class FundingProvider:
    """Find rounds a firm participated in, and who else was in them."""

    name = "funding"

    def __init__(self, search_provider=None) -> None:
        self._search = search_provider

    def _available(self) -> bool:
        return bool(config.FUNDING_ENABLED) and self._search is not None \
            and self._search.available()

    def rounds_for_firm(self, firm_name: str,
                        max_rounds: int = 0) -> List[Dict[str, object]]:
        """Rounds whose announcement names `firm_name` AS AN INVESTOR.

        Guard: the firm must appear in the parsed investor list. An article
        that merely mentions the firm asserts nothing about its participation.
        """
        max_rounds = max_rounds or config.MAX_ROUNDS_PER_FIRM
        if not firm_name or not self._available():
            return []

        target = org_norm_key(firm_name)
        key = cache.make_key(self.name, "rounds", target)
        cached = cache.get(key)
        if cached is not None:
            return cached.get("rounds", [])

        urls: List[str] = []
        for query in (f'"{firm_name}" "participation from"',
                      f'"{firm_name}" "led by" funding round raised',
                      f'"{firm_name}" "co-led by"',
                      f'"{firm_name}" seed round announcement investors'):
            for result in self._search.search(query):
                if not _is_blocked(result.url) and result.url not in urls:
                    urls.append(result.url)

        rounds: List[Dict[str, object]] = []
        for url in urls[: max_rounds * 3]:
            page = fetch_page(url)
            if page.status_code != 200 or not page.content:
                continue
            text = html_to_text(page.content)
            parsed = parse_round(text, target_key=target)
            if parsed["investors"] and not parsed["company"]:
                # The flattened body rarely starts with "<Company> raises …";
                # the headline does. Without this every round reads "the round".
                title = soup_of(page.content).title
                if title is not None:
                    headline = _COMPANY_RE.search(title.get_text(" ", strip=True))
                    if headline:
                        parsed["company"] = headline.group(1).strip()
            names = {org_norm_key(n) for n in parsed["investors"]}
            if target not in names or len(names) < 2:
                continue  # not named as an investor, or no co-investor to link
            parsed["source_url"] = url
            rounds.append(parsed)
            if len(rounds) >= max_rounds:
                break

        cache.set(key, "rounds", {"rounds": rounds}, config.CACHE_TTL_WIKI)
        return rounds

    def round_for_company(self, company_name: str,
                          target_firm_key: str = "") -> Optional[Dict[str, object]]:
        """The funding round of ONE company, and who invested in it.

        Firm-name search misses rounds whose announcement leads with the
        company ("Splitero secures $11.7M …") rather than the investor.
        """
        if not company_name or not self._available():
            return None
        key = cache.make_key(self.name, "companyround", org_norm_key(company_name))
        cached = cache.get(key)
        if cached is not None:
            return cached.get("round")

        urls: List[str] = []
        for query in (f'"{company_name}" raises funding round investors',
                      f'"{company_name}" "led by" "participation from"',
                      f'"{company_name}" seed series funding announcement'):
            for result in self._search.search(query):
                if not _is_blocked(result.url) and result.url not in urls:
                    urls.append(result.url)

        best = None
        for url in urls[:6]:
            page = fetch_page(url)
            if page.status_code != 200 or not page.content:
                continue
            parsed = parse_round(html_to_text(page.content),
                                 target_key=target_firm_key)
            if len(parsed["investors"]) < 2:
                continue
            parsed["source_url"] = url
            keys = {org_norm_key(n) for n in parsed["investors"]}
            if not target_firm_key or target_firm_key in keys:
                best = parsed          # a round naming the anchor — take it
                break
            best = best or parsed
        cache.set(key, "companyround", {"round": best}, config.CACHE_TTL_WIKI)
        return best
