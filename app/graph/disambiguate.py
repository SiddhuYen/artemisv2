"""Deterministic homonym-disambiguation backstop.

builder.get_or_create_person() dedups people by normalized name. When a QID is
being ADOPTED onto an existing same-named node (see its "case 2" — a name
match with no wikidata_qid yet), that adoption is otherwise unconditional: it
would fuse a non-notable person already in the graph with whatever stranger a
name-matched Wikidata/Wikipedia lookup happens to resolve to. This module
supplies the one check standing between "same name" and "same person": do the
node's own already-accumulated evidence and the candidate identity's text
plainly anchor in different professional worlds (a venture capitalist vs. a
test-prep educator)?

Conservative by design: it reports a conflict ONLY when each side clearly and
separately anchors in a disjoint domain, and stays silent (no conflict)
whenever the two overlap or either side is unclear. The caller (see
builder._homonym_conflict) can only SEPARATE nodes, never merge them, so a
false negative merely preserves today's fuse-by-name behavior and a false
positive costs at most a second, QID-suffixed node for one name — never a
wrong bridge.

Known ceiling: a keyword-bucket lexicon cannot separate two different people
IN the same broad field (two same-named venture capitalists at two different,
unrelated funds look identical to it) — see builder._existing_evidence_signal
for how the candidate signal is broadened with concrete affiliations to
mitigate this a little; the rest is out of scope for a deterministic check.
"""
from __future__ import annotations

import re

# Professional-domain lexicon. Each cluster is a set of lowercase keywords that,
# appearing in a short bio/evidence snippet, anchor a person in that world. Kept
# deliberately generic — this is a backstop, not an ontology.
_DOMAINS = {
    "venture": {"venture", "vc", "investor", "investing", "investment",
                "capital", "fund", "general partner", "limited partner",
                "angel investor", "portfolio", "financier"},
    "education": {"education", "educator", "teacher", "tutor", "tutoring",
                  "test prep", "prep", "admissions", "academy", "curriculum",
                  "edtech", "professor", "lecturer", "principal of"},
    "sports": {"footballer", "cricketer", "athlete", "coach", "olympic",
               "boxer", "wrestler", "sprinter", "basketball", "baseball",
               "quarterback", "midfielder", "batsman", "bowler"},
    "music": {"singer", "musician", "composer", "rapper", "songwriter",
              "guitarist", "drummer", "pianist", "vocalist", "band"},
    "film": {"actor", "actress", "filmmaker", "screenwriter", "comedian",
             "cinematographer", "voice actor"},
    "politics": {"politician", "senator", "governor", "minister", "congressman",
                 "congresswoman", "mayor", "diplomat", "legislator", "councillor"},
    "science": {"scientist", "researcher", "physicist", "chemist", "biologist",
                "mathematician", "astronomer", "academic", "engineer"},
    "medicine": {"physician", "surgeon", "cardiologist", "psychiatrist",
                 "dentist", "doctor of medicine"},
    "law": {"lawyer", "attorney", "barrister", "solicitor", "jurist", "judge"},
    # "general" alone is deliberately excluded: it collides with "general
    # partner" (venture), "attorney general" (law), "general manager"
    # (business), etc. — too polysemous for a bare-word match.
    "military": {"colonel", "brigadier", "admiral", "soldier",
                 "army officer", "naval officer", "air force",
                 "four-star general", "major general", "lieutenant general"},
    "religion": {"priest", "pastor", "imam", "rabbi", "monk", "bishop",
                 "cleric", "theologian"},
    "arts": {"author", "novelist", "poet", "painter", "sculptor", "journalist",
             "cartoonist", "playwright"},
    # Corporate/sales/consulting track -- deliberately compound phrases, not
    # bare "director"/"manager"/"executive" (too generic, collides with
    # nonprofit board roles, film production titles, academic administration).
    # Added because this bucket's absence was a real, live gap: a Trinamix
    # "Vice President Sales & Strategy" and an ISRO "researcher"/"engineer"
    # share a name with nothing here to tell them apart -- domains_of() on
    # the business evidence returned empty, so domain_conflict silently
    # never fired even though the two evidently don't overlap.
    "business": {"vice president", "chief executive", "chief financial officer",
                 "chief operating officer", "chief revenue officer",
                 "chief technology officer", "managing director",
                 "business development", "management consulting",
                 "account executive", "corporate strategy", "sales strategy",
                 "vp sales", "executive vice president", "chairman of the board",
                 "co-founder", "cofounder"},
}

# Compile one boundary-anchored pattern per domain. `\b` bounds even the short
# tokens (vc) and lets multi-word keywords ("test prep") match as phrases.
_DOMAIN_RES = {
    domain: re.compile(
        r"\b(?:%s)\b" % "|".join(re.escape(kw) for kw in sorted(kws)),
        re.IGNORECASE,
    )
    for domain, kws in _DOMAINS.items()
}


def domains_of(text: str) -> set:
    """The professional domains a short text snippet anchors in (possibly empty)."""
    if not text:
        return set()
    return {d for d, rx in _DOMAIN_RES.items() if rx.search(text)}


def domain_conflict(signal: str, candidate: str) -> bool:
    """True when `signal` (evidence already attached to the graph node in
    question) and `candidate` (text describing the identity about to be
    adopted onto it) anchor in different, non-overlapping professional
    domains.

    Silent (False) when either side is unanchored or the two share any
    domain, so it only fires on a clear cross-domain mismatch.
    """
    s = domains_of(signal)
    c = domains_of(candidate)
    if not s or not c:
        return False
    return s.isdisjoint(c)
