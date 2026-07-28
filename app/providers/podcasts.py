"""Podcast RSS -> host<->guest relationships (Tier-1 structural data).

An episode structurally asserts one thing: *this host interviewed this guest*.
That is a demonstrated, on-the-record relationship.

It asserts NOTHING about two guests of the same show — they have typically
never met. Minting guest<->guest edges from a shared episode list is the
co-occurrence fallacy in its purest form, so this module never does it: every
edge it can produce is host<->guest, never guest<->guest.

Consequence: a feed whose host is not a NAMED HUMAN produces no edges at all.
"Bree Hanson & Vikram Lakhwara" yields two hosts; "Amplitude" yields none, and
that feed contributes nothing rather than contributing junk.

Ported from DrewGloverDemo's providers/podcasts.py, adapted to this project's
cache/base/name-shape utilities.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

from .. import config
from ..utils.names import (
    _DIMINUTIVES,
    looks_like_person_name,
    normalize,
    person_norm_key,
    strip_middle_initials,
)
from . import cache
from .base import request_with_retry

_ITUNES = "http://www.itunes.com/dtds/podcast-1.0.dtd"
_ITUNES_SEARCH = "https://itunes.apple.com/search"

# Episode-number markers and show tags that surround the guest's name in a title.
_EPISODE_MARKER = re.compile(
    r"^\s*(ep|eps|episode|pt|part|s\d+e\d+|season)\b[\s.:#-]*\d*\s*$", re.I)
_BARE_NUMBER = re.compile(r"^\s*[\d\s.#-]+\s*$")
_PARENTHETICAL = re.compile(r"\(([^)]*)\)")
# The ONLY phrase that says "this person was on this episode".
_WITH_GUEST = re.compile(r"\b(?:with|feat\.?|ft\.?|featuring)\s+(.+)$", re.I)
# An episode-number marker anywhere in the title ("| Ep. 37", "Episode 38 |").
_HAS_EP_MARKER = re.compile(r"\b(?:ep|eps|episode)\b\.?:?\s*\d+", re.I)
# How many template-shaped titles before we trust a feed's title convention.
_MIN_TEMPLATE_HITS = 3
_MIN_TEMPLATE_RATIO = 0.3

# Org words with no identifying power when corroborating a guest's identity.
_GENERIC_ORG_TOKENS = {"ventures", "capital", "partners", "fund", "group",
                       "holdings", "company", "corporation", "incorporated",
                       "limited", "management", "technologies", "labs"}

# formal-first-name -> nicknames, built once from utils.names' nickname map
# (nickname -> formal) so a search can try BOTH directions: a structured
# source often carries the formal name ("Joseph Gebbia") while podcasts use
# the nickname ("Joe Gebbia").
_DIMINUTIVES_REVERSE: Dict[str, List[str]] = {}
for _nick, _formal in _DIMINUTIVES.items():
    _DIMINUTIVES_REVERSE.setdefault(_formal, []).append(_nick)


def search_name_variants(name: str, limit: int = 4) -> List[str]:
    """First-name spellings to try when searching an external source for a
    person: expands the FIRST token both formal->nickname and nickname->formal,
    leaving the surname untouched. Returns the original first, deduped."""
    raw = strip_middle_initials((name or "").strip())
    parts = raw.split()
    if len(parts) < 2:
        return [raw] if raw else []

    first = parts[0]
    key = normalize(first)
    alts = []
    formal = _DIMINUTIVES.get(key)
    if formal and formal != key:
        alts.append(formal)                        # joe -> joseph
    alts.extend(_DIMINUTIVES_REVERSE.get(key, []))  # joseph -> joe, joey

    surname = " ".join(parts[1:])
    out = [raw]
    seen = {normalize(raw)}
    for alt in alts:
        variant = f"{alt.capitalize()} {surname}"
        if normalize(variant) not in seen:
            seen.add(normalize(variant))
            out.append(variant)
    return out[:limit]


def _looks_like_person_only(name: str) -> bool:
    """Grammar gate for a title-derived candidate: every token must look like
    part of a personal name and the whole string must not resemble a place or
    org (best-effort without full NER — see spacy_extractor for the stronger,
    NER-based version used elsewhere in extraction)."""
    return bool(name) and looks_like_person_name(name)


def _author_is_the_shows_brand(author: str, show: str) -> bool:
    """True when the 'author' is just the show's own name.

    "Riding Unicorns" and "Startup Insider" are shows, yet they pass every
    name-shape test — both are two proper nouns. Grammar cannot separate a
    brand from a person, but structure can: a show titled "Riding Unicorns:
    Venture Capital..." begins with its own brand, whereas "Billion Dollar
    Moves with Sarah Chen-Spellings" does not begin with hers.
    """
    a, s = normalize(author), normalize(show)
    return bool(a and s and s.startswith(a))


def _hosts_from_author(author: str, show: str = "") -> List[str]:
    """Split an itunes:author into named humans. Returns [] for a company.

    Empty means the feed asserts no personal relationship and must yield NO
    edges — a company cannot be one end of one, and guest<->guest is
    co-occurrence.
    """
    if not author or _author_is_the_shows_brand(author, show):
        return []
    hosts, seen = [], set()
    for part in re.split(r"\s*(?:&|,| and )\s*", author):
        part = part.strip()
        if not _looks_like_person_only(part):
            continue
        key = person_norm_key(part)
        if key and key not in seen:      # "Espree Devora, hosted by Espree Devora"
            seen.add(key)
            hosts.append(part)
    return hosts


def _org_and_bodies(title: str):
    """Return (org, stripped, flat).

    A parenthetical is a FIRM in "Tae Hea Nahm (Storm Ventures)" but a GUEST
    CLAUSE in "10 Years of Acquired (with Michael Lewis)". The two readings
    need different views of the title:

      stripped — parentheses and contents removed, so the template's name slot
                 is "Tae Hea Nahm" and not "Tae Hea Nahm Storm Ventures".
      flat     — brackets removed, contents kept, so the marker search can
                 still see a "with ..." clause that was hiding inside them.
    """
    org = ""
    for inner in _PARENTHETICAL.findall(title):
        inner = inner.strip()
        if (not inner or _WITH_GUEST.search(inner)
                or _looks_like_person_only(inner) or len(inner.split()) > 4):
            continue
        org = inner
    stripped = _PARENTHETICAL.sub(" ", title)
    flat = " ".join(title.replace("(", " ").replace(")", " ").split())
    return org, stripped, flat


def guest_by_marker(title: str) -> Optional[Dict[str, str]]:
    """Guest named by an explicit clause: "... with Michael Lewis".

    This is the only phrase that *asserts* someone was on the episode. Without
    it, a name in a title is a topic: "20VC: How We Got Fred Wilson ... to
    Invest $94M" is about Fred Wilson, who was never a guest.
    """
    if not title:
        return None
    org, _stripped, flat = _org_and_bodies(title.strip())
    match = _WITH_GUEST.search(flat)
    if not match:
        return None
    # "April Underwood from Adverb Ventures" / "Jane Doe, VP of Product"
    tail = re.split(r"\s*(?:,| from | at | of | on )\s*", match.group(1))[0]
    tail = tail.strip(" .,-–—|")
    return {"guest": tail, "org": org} if _looks_like_person_only(tail) else None


def guest_by_template(title: str) -> Optional[Dict[str, str]]:
    """Guest occupying a feed's fixed title slot: "DWAVC: <Name> | Ep. 37".

    Only trusted when the FEED consistently uses this shape (see
    `_feed_uses_guest_template`).
    """
    if not title or not _HAS_EP_MARKER.search(title):
        return None
    org, stripped, _flat = _org_and_bodies(title.strip())
    for seg in re.split(r"[:|]", stripped):
        seg = seg.strip(" .,-–—")
        if (not seg or _EPISODE_MARKER.match(seg) or _BARE_NUMBER.match(seg)):
            continue
        if _looks_like_person_only(seg):
            return {"guest": seg, "org": org}
    return None


# A guest's episode INTRODUCES them: "Sam Altman is the CEO of OpenAI...".
# A news episode's description is a running order, not a bio, so the bio must
# LEAD the description — a person merely discussed later is never mistaken
# for a guest.
_BIO_VERB = r"(?:is|was|serves\s+as|joins|co-?founded|founded|leads|runs)"
_BIO_LEAD_CHARS = 240
_BIO_NAME_MAX_OFFSET = 40
_HTML_TAG = re.compile(r"<[^>]+>")


def _plain(text: str) -> str:
    return " ".join(_HTML_TAG.sub(" ", text or "").split())


_AUDIO_EXT = re.compile(r"\.(?:mp3|m4a|mp4|aac|ogg|wav)(?:\?.*)?$", re.I)


def episode_url(item, fallback: str = "") -> str:
    """The canonical page for one episode.

    Many feeds omit <link>; the enclosure carries the episode path, so
    dropping the audio extension recovers it.
    """
    link = (item.findtext("link") or "").strip()
    if link:
        return link
    enclosure = item.find("enclosure")
    url = (enclosure.get("url") or "").strip() if enclosure is not None else ""
    if url and _AUDIO_EXT.search(url):
        return _AUDIO_EXT.sub("", url)
    return fallback


def guest_by_bio(title: str, description: str) -> Optional[Dict[str, str]]:
    """Guest named in the title and INTRODUCED by the description's opening
    bio. "<Name> on <topic>" by itself asserts nothing; the description
    settles it — "Sam Altman is the CEO of OpenAI" introduces a guest,
    "AGENDA: 05:00 ..." does not.
    """
    if not title or not description:
        return None
    lead = _plain(description)[:_BIO_LEAD_CHARS]
    if not lead:
        return None

    org, stripped, _flat = _org_and_bodies(title.strip())
    for segment in re.split(r"[:|]", stripped):
        name = re.split(r"\s+\bon\b\s+", segment.strip(), maxsplit=1)[0]
        name = name.strip(" .,-–—")
        if not name or not _looks_like_person_only(name):
            continue
        bio = re.search(rf"\b{re.escape(name)}\b\s+{_BIO_VERB}\b", lead, re.I)
        if bio and bio.start() <= _BIO_NAME_MAX_OFFSET:
            return {"guest": name, "org": org}

    opening = re.match(rf"\s*(?:\S+\s+){{0,3}}?"
                       rf"([A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){{1,3}})\s+{_BIO_VERB}\b",
                       lead)
    if opening:
        name = opening.group(1).strip()
        if (_looks_like_person_only(name)
                and normalize(name) in normalize(_org_and_bodies(title)[2])):
            return {"guest": name, "org": org}
    return None


def _feed_uses_guest_template(titles: List[str]) -> bool:
    hits = sum(1 for t in titles if guest_by_template(t))
    return hits >= _MIN_TEMPLATE_HITS and hits >= _MIN_TEMPLATE_RATIO * len(titles)


def _guest_from_title(title: str, allow_template: bool = True,
                      description: str = "") -> Optional[Dict[str, str]]:
    """Three assertions, strongest first: an explicit `with` clause, the
    description's opening bio, or the feed's own title template."""
    return (guest_by_marker(title)
            or guest_by_bio(title, description)
            or (guest_by_template(title) if allow_template else None))


class PodcastProvider:
    name = "podcasts"

    def available(self) -> bool:
        return bool(config.PODCASTS_ENABLED)

    def episodes(self, feed: dict) -> dict:
        """Parse one feed into {show, hosts[], guests[{guest, org, episode_title}]}.

        Returns hosts=[] when the show has no named human host, which callers
        MUST treat as "this feed yields no edges".
        """
        rss = feed.get("rss") or ""
        if not rss:
            return {"show": feed.get("show", ""), "hosts": [], "guests": [],
                    "page": feed.get("page", "")}

        key = cache.make_key(self.name, "feed", rss)
        cached = cache.get(key)
        if cached is not None:
            return cached

        resp = request_with_retry("GET", rss, provider=self.name)
        out = {"show": feed.get("show", ""), "hosts": [], "guests": [],
               "page": feed.get("page", "")}
        if resp is None or resp.status_code != 200:
            return out

        try:
            channel = ET.fromstring(resp.content).find("channel")
        except ET.ParseError:
            return out
        if channel is None:
            return out

        out["show"] = channel.findtext("title") or out["show"]
        out["page"] = channel.findtext("link") or out["page"]
        author = (channel.findtext(f"{{{_ITUNES}}}author") or ""
                  or feed.get("author", ""))
        out["hosts"] = _hosts_from_author(author, out["show"])

        if not out["hosts"]:
            cache.set(key, "feed", out, config.CACHE_TTL_WIKI)
            return out

        items = channel.findall("item")[: config.PODCAST_MAX_EPISODES]
        titles = [(it.findtext("title") or "") for it in items]
        allow_template = _feed_uses_guest_template(titles)

        seen = set()
        candidates = []
        for item in items:
            title = item.findtext("title") or ""
            description = (item.findtext("description")
                           or item.findtext(f"{{{_ITUNES}}}summary") or "")
            parsed = _guest_from_title(title, allow_template=allow_template,
                                       description=description)
            if not parsed:
                continue
            key_ = person_norm_key(parsed["guest"])
            if not key_ or key_ in seen:
                continue
            seen.add(key_)
            candidates.append({
                "guest": parsed["guest"],
                "org": parsed["org"],
                "episode_title": title.strip(),
                "episode_url": episode_url(item, out["page"] or ""),
            })

        out["guests"] = candidates
        cache.set(key, "feed", out, config.CACHE_TTL_WIKI)
        return out

    # --- person -> podcast appearances (the per-person silo) ---------------
    def _feed_hosts_and_items(self, rss: str) -> tuple:
        """(hosts, {normalised episode title: description}) for one feed."""
        key = cache.make_key(self.name, "hostitems", rss)
        cached = cache.get(key)
        if cached is not None:
            return cached.get("hosts", []), cached.get("items", {})

        resp = request_with_retry("GET", rss, provider=self.name)
        hosts, items = [], {}
        if resp is not None and resp.status_code == 200:
            try:
                channel = ET.fromstring(resp.content).find("channel")
            except ET.ParseError:
                channel = None
            if channel is not None:
                show = channel.findtext("title") or ""
                author = channel.findtext(f"{{{_ITUNES}}}author") or ""
                hosts = _hosts_from_author(author, show)
                for item in channel.findall("item")[: config.PODCAST_MAX_EPISODES]:
                    title = (item.findtext("title") or "").strip()
                    if title:
                        items[title] = {
                            "description": (item.findtext("description")
                                            or item.findtext(f"{{{_ITUNES}}}summary")
                                            or ""),
                            "link": episode_url(item),
                            "show": show,
                        }
        cache.set(key, "hostitems", {"hosts": hosts, "items": items},
                  config.CACHE_TTL_WIKI)
        return hosts, items

    def appearances(self, person_name: str, limit: int = 0,
                    known_orgs: Optional[List[str]] = None,
                    hint: str = "") -> List[dict]:
        """Episodes where `person_name` was the GUEST, with the show's hosts.

        Two guards:
        * Apple's episode search matches a name anywhere in the metadata, so
          the episode must ASSERT them as guest (a `with` clause or an opening
          bio), and the show must have a named human host.
        * Names are not identities: when we already know the person's
          organisations, the episode must corroborate one of them, or a
          homonym would silently merge into their node.
        """
        if not self.available():
            return []
        limit = limit or config.MAX_PODCAST_APPEARANCES
        if not person_name:
            return []
        target = person_norm_key(person_name)
        if not target:
            return []

        org_keys = sorted({tok for org in (known_orgs or [])
                           for tok in normalize(org).split()
                           if len(tok) > 3 and tok not in _GENERIC_ORG_TOKENS})
        fingerprint = f"{target}::{'|'.join(org_keys)}"
        key = cache.make_key(self.name, "appearances", fingerprint)
        cached = cache.get(key)
        if cached is not None:
            return cached.get("appearances", [])

        results = []
        seen_track_ids = set()
        for term in search_name_variants(person_name):
            resp = request_with_retry(
                "GET", _ITUNES_SEARCH, provider=self.name,
                params={"term": term, "entity": "podcastEpisode",
                        "limit": config.PODCAST_EPISODE_SEARCH_LIMIT,
                        "country": "US"},
            )
            if resp is None or resp.status_code != 200:
                continue
            try:
                for item in resp.json().get("results", []):
                    tid = item.get("trackId") or item.get("trackViewUrl") or id(item)
                    if tid not in seen_track_ids:
                        seen_track_ids.add(tid)
                        results.append(item)
            except Exception:
                continue

        out, seen = [], set()
        for item in results:
            rss = item.get("feedUrl") or ""
            episode_title = (item.get("trackName") or "").strip()
            if not rss or not episode_title or len(out) >= limit:
                continue
            if (rss, episode_title) in seen:
                continue
            seen.add((rss, episode_title))

            hosts, feed_items = self._feed_hosts_and_items(rss)
            if not hosts:
                continue  # a company-hosted show asserts no personal tie
            if any(person_norm_key(h) == target for h in hosts):
                continue  # they host it; that is not a guest appearance

            entry = feed_items.get(episode_title, {})
            description = entry.get("description", "") or item.get("description", "")
            parsed = _guest_from_title(episode_title, allow_template=False,
                                       description=description)
            if not parsed or person_norm_key(parsed["guest"]) != target:
                continue  # the episode is about them, not with them

            if org_keys:
                haystack = normalize(f"{episode_title} {_plain(description)}")
                if not any(org in haystack for org in org_keys):
                    continue  # same name, no tie to any org we know them by

            out.append({
                "show": entry.get("show") or item.get("collectionName", ""),
                "hosts": hosts,
                "rss": rss,
                "episode_title": episode_title,
                "episode_url": (entry.get("link") or item.get("trackViewUrl", "")),
                "org": parsed.get("org", ""),
            })

        cache.set(key, "appearances", {"appearances": out}, config.CACHE_TTL_WIKI)
        return out
