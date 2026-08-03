"""Per-connection silo weights — how much of each silo a contact is worth.

Expansion currently runs the SAME 9 silos over every person: 36 queries whether
the subject is a US senator or a college sophomore. Most of those queries are
structurally hopeless. Asking the government silo about a high-schooler, or the
publications silo about a sales rep, spends real money to retrieve pages about
other people with the same name — which is also where a large share of the junk
edges come from (the live run pruned 73-85 junk nodes per contact).

A CSV row already says a lot about which silos can possibly pay off. A title of
"Professor" makes publications and education the high-value silos and government
near-worthless; "Founder" makes company and board the ones that matter. This
module turns those signals into a weight per silo, computed once when the run is
planned and stored on the task so it can be inspected and later tuned.

Weights govern QUERY ALLOCATION only — which silos run and how many of their
queries. They deliberately do NOT scale confidence: how much to believe a piece
of evidence is a property of the evidence, not of who we guessed the subject
was, and folding a prior into it would let a bad guess quietly manufacture
strong edges.

`initial_weights` is the "initial" in the name: a prior from the export alone,
with no observation of what a contact actually yielded. Nothing here learns yet.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

from .. import config
from ..silos import SILOS
from ..utils.names import normalize

# --- signal vocabularies ----------------------------------------------------
# Matched with word boundaries against the normalized title/company text.
_SENIOR = ("founder", "co founder", "cofounder", "ceo", "chief executive",
           "president", "managing partner", "general partner", "managing director",
           "chairman", "chairwoman", "partner", "vp", "vice president", "svp",
           "evp", "head of", "director", "chief", "cto", "cfo", "coo", "owner")
_ACADEMIC = ("professor", "prof", "lecturer", "researcher", "research fellow",
             "postdoc", "post doc", "phd", "dphil", "dean", "faculty",
             "scientist", "principal investigator")
_MEDIA = ("journalist", "editor", "reporter", "writer", "author",
          "correspondent", "columnist", "producer")
_POLITICAL = ("senator", "congressman", "congresswoman", "representative",
              "mayor", "governor", "commissioner", "minister", "ambassador",
              "secretary of", "council member", "councilmember", "legislator")
_EDU_ORG = ("university", "college", "school", "institute", "academy",
            "polytechnic", "universidad", "universite")
_NONPROFIT_ORG = ("foundation", "nonprofit", "non profit", "charity", "trust",
                  "ngo", "association", "society", "institute")
_GOV_ORG = ("department of", "ministry", "agency", "bureau", "city of",
            "county of", "state of", "senate", "congress", "parliament",
            "administration", "commission", "council")

# A silo below this contributes nothing worth its queries and is skipped.
_MIN = "ENRICH_SILO_MIN_WEIGHT"


def _has(text: str, words: Iterable[str]) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)


def initial_weights(titles: Optional[List[str]] = None,
                    companies: Optional[List[str]] = None,
                    schools: Optional[List[str]] = None,
                    email: str = "") -> Dict[str, float]:
    """Silo key -> weight in roughly 0..1.3, from a contact's export row alone.

    The baselines are deliberately low for everything except `company`: an
    employer is the one relationship an uploaded row actually asserts, so it is
    the only silo guaranteed to be about the right person. Every other silo has
    to be earned by a signal.
    """
    title_text = " ".join(normalize(t) for t in (titles or []) if t)
    org_text = " ".join(normalize(c) for c in (companies or []) if c)
    has_school = bool([s for s in (schools or []) if s])
    domain = (email or "").rsplit("@", 1)[-1].lower() if "@" in (email or "") else ""

    senior = _has(title_text, _SENIOR)
    academic = _has(title_text, _ACADEMIC)
    media = _has(title_text, _MEDIA)
    political = _has(title_text, _POLITICAL)
    edu_org = _has(org_text, _EDU_ORG) or domain.endswith(".edu")
    nonprofit_org = _has(org_text, _NONPROFIT_ORG)
    gov_org = _has(org_text, _GOV_ORG) or domain.endswith(".gov")

    w = {
        # The employer IS the assertion the row makes — always worth full budget.
        "company": 1.0 + (0.3 if senior else 0.0),
        # Press coverage tracks seniority and public role more than anything
        # else; a junior IC is almost never written about under their own name.
        "news": 0.3 + (0.5 if senior else 0.0) + (0.3 if political else 0.0)
                + (0.1 if academic else 0.0),
        # Board seats are a senior-and-above phenomenon, or a nonprofit one.
        "board_nonprofit": 0.1 + (0.6 if senior else 0.0)
                           + (0.5 if nonprofit_org else 0.0),
        "education": 0.1 + (0.5 if has_school else 0.0) + (0.5 if edu_org else 0.0)
                     + (0.4 if academic else 0.0),
        "publications": 0.1 + (0.7 if academic else 0.0) + (0.6 if media else 0.0),
        "events": 0.1 + (0.3 if senior else 0.0) + (0.3 if academic else 0.0),
        "government": 0.05 + (0.9 if political else 0.0) + (0.6 if gov_org else 0.0),
        # Family and friends stay near zero for everyone. Expansion already
        # down-weights family nodes when ranking a frontier (see
        # config.DOWNWEIGHT_FAMILY) because paths run through colleagues, not
        # relatives — spending a professional network's query budget on
        # genealogy is the same mistake one step earlier.
        "family": 0.05,
        "friends": 0.05,
    }
    # Only silos that actually exist, so a renamed/removed silo can't leave a
    # dangling weight that silently allocates nothing.
    keys = {s.key for s in SILOS}
    return {k: round(v, 3) for k, v in w.items() if k in keys}


def query_budget(weights: Optional[Dict[str, float]]) -> Dict[str, int]:
    """Silo key -> how many of its queries to run.

    A weight at or above 1.0 gets the silo's full allowance
    (config.MAX_QUERIES_PER_SILO); below the floor it gets none. `None` means
    "no weighting" and every silo gets its full allowance, which is exactly
    today's behavior — so an unweighted caller is unaffected.
    """
    full = config.MAX_QUERIES_PER_SILO
    if not weights:
        return {s.key: full for s in SILOS}
    floor = getattr(config, _MIN)
    out: Dict[str, int] = {}
    for silo in SILOS:
        weight = weights.get(silo.key, 0.0)
        if weight < floor:
            continue
        out[silo.key] = max(1, min(full, round(weight * full)))
    return out


# --- coverage: what a node's expansion ACTUALLY asked ------------------------
# `Person.processed` is a boolean, but an expansion is not one thing: it is a
# per-silo query allocation (above), and which silos ran depends entirely on
# the weights the CALLER supplied. A contact expanded by initial enrichment
# under founder-shaped weights was never asked the publications question; a
# later /connect toward an academic target needs exactly that question. Under
# a boolean gate the answer is "already processed", forever -- the node's
# neighbors get replayed from the DB and the new question is never asked.
#
# So a node records the budget it was expanded under, and reuse becomes a
# comparison rather than a flag: covered silos replay, uncovered ones are
# searched. That is what makes enrichment re-initializable per connect
# instead of frozen at whatever the first run happened to want.


def merge_coverage(covered: Optional[Dict[str, int]],
                   budget: Dict[str, int]) -> Dict[str, int]:
    """Fold a just-executed budget into a node's recorded coverage.

    Max per silo, never sum: running the news silo's 4 queries twice still
    only ever asked 4 distinct questions (`render_queries` is deterministic,
    so run N's queries are a prefix-identical set), and summing would let a
    repeat inflate coverage into claiming questions that were never asked.
    """
    out = dict(covered or {})
    for key, count in budget.items():
        if count > out.get(key, 0):
            out[key] = int(count)
    return out


def uncovered_budget(wanted: Dict[str, int],
                     covered: Optional[Dict[str, int]]) -> Dict[str, int]:
    """The part of `wanted` a node's recorded coverage does not already answer.

    A silo appears at its FULL wanted allowance, not the difference: queries
    are rendered as a prefix (`render_queries(...)[:allowance]`), so asking for
    4 when 2 were covered must re-render all 4 to reach queries 3 and 4. The
    2 repeats cost nothing -- the provider cache serves them (see
    config.CACHE_TTL_SEARCH), which is what makes widening a node's coverage
    cheap enough to do per connect.
    """
    covered = covered or {}
    return {key: count for key, count in wanted.items()
            if count > covered.get(key, 0)}
