"""Pre-build a warm graph of Y Combinator partners and their 2-hop network.

Same shape as DrewGloverDemo's scripts/precrawl.py + build_famous_backbone.py:
scrape the firm's roster once to get a trustworthy partner list, materialize the
colleague clique that the roster page structurally asserts, then walk each
partner outward into the SHARED persistent graph so later /connect queries hit a
warm cache instead of paying for on-demand enrichment per request.

The graph (artemis.db) is the cache. Nothing here is throwaway: expansion marks
each processed person, so a re-run reuses everything already discovered and only
pays for newly-reached people (see expansion._reuse_existing_neighbors). Killing
it partway and re-running is safe.

    ./.venv/bin/python scripts/build_yc_cache.py --depth 2
    ./.venv/bin/python scripts/build_yc_cache.py --depth 2 --budget 3600

To build into a SEPARATE cache file instead of the main graph:

    ARTEMIS_DB_URL=sqlite:///./yc_cache.db ./.venv/bin/python scripts/build_yc_cache.py

Costs Serper/Brave calls per newly-enriched person — depth 2 over ~40 partners
is a long run. Bounded by --budget wall-clock seconds; it stops cleanly at the
next hop boundary and reports what it covered.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select                          # noqa: E402

from app import config                                 # noqa: E402
from app.db import SessionLocal, init_db               # noqa: E402
from app.extraction.schemas import EdgeSignals, ExtractedEdge  # noqa: E402
from app.graph import builder                          # noqa: E402
from app.graph.connect import _adjacency                # noqa: E402
from app.graph.expansion import ORCH, expand_graph      # noqa: E402
from app.models import Person                           # noqa: E402
from app.providers import STATS                         # noqa: E402
from app.providers.base import SearchResult             # noqa: E402
from app.utils.names import person_norm_key             # noqa: E402

FIRM = "Y Combinator"
ROSTER_URL = "https://www.ycombinator.com/people"

# Fallback roster. YC's people page is JS-rendered and the scrape returns nothing
# when the headless browser is unavailable, so the run must not depend on it.
# Merged with whatever the live scrape finds — the scrape is authoritative for
# who is CURRENTLY a partner, this list only guarantees the run has a seed set.
YC_PARTNERS = [
    "Garry Tan", "Michael Seibel", "Jared Friedman", "Diana Hu", "Harj Taggar",
    "Dalton Caldwell", "Tom Blomfield", "Gustaf Alstromer", "Pete Koomen",
    "Brad Flora", "Nicolas Dessaigne", "David Lieb", "Aaron Epstein",
    "Surbhi Sarna", "Wayne Crosby", "Divya Bhat", "Emmett Shear",
    "Paul Buchheit", "Jessica Livingston", "Paul Graham", "Robert Morris",
    "Trevor Blackwell", "Sam Altman", "Kirsty Nathoo", "Kevin Hale",
    "Adora Cheung", "Geoff Ralston", "Anu Hariharan", "Ali Rowghani",
    "Carolynn Levy", "Jon Levy", "Tyler Bosmeny", "YK Sugishita",
]

# A roster this size or larger is a directory, not a team — materializing the
# clique would assert tens of thousands of false colleague ties. Mirrors the
# Rule 1 cap precrawl.py applies to oversized firm rosters.
MAX_CLIQUE_MEMBERS = 60


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _scraped_partners() -> list:
    """Live YC roster, or [] when the page can't be read."""
    roster = ORCH.firms.roster(ROSTER_URL, FIRM)
    return roster.get("members") or []


def _partner_list() -> list:
    """The full roster. --limit trims who gets EXPANDED, never who is on the
    team: the colleague clique has to reflect the whole roster or a partial run
    would assert a partial, wrong team."""
    scraped = _scraped_partners()
    print(f"roster scrape: {len(scraped)} names from {ROSTER_URL}")
    names, seen = [], set()
    for name in scraped + YC_PARTNERS:
        key = person_norm_key(name)
        if key and key not in seen:
            seen.add(key)
            names.append(name)
    return names


def _materialize_clique(db, names: list) -> int:
    """Every partner is a colleague of every other, asserted by the roster page.

    Structural (the page lists them as one team), so the edges are `trusted` and
    skip the Claude entity filter — same treatment expansion gives roster
    colleagues in its phase-0c.
    """
    if len(names) > MAX_CLIQUE_MEMBERS:
        print(f"  roster of {len(names)} exceeds the clique cap "
              f"({MAX_CLIQUE_MEMBERS}) — recording membership only, no edges")
        return 0

    org = builder.get_or_create_org(db, FIRM, org_type="firm")
    source = builder.save_source(
        db, SearchResult(f"{FIRM} people", ROSTER_URL, "firms", "firms"),
        "enrich:firms")
    people = [p for p in (builder.get_or_create_person(db, n) for n in names)
              if p is not None]

    made = 0
    for subject in people:
        for other in people:
            if subject.id == other.id:
                continue
            edge = ExtractedEdge(
                person_a=subject.canonical_name, person_b=other.canonical_name,
                other_kind="person", relationship_type="coworker",
                method="firm roster", source_url=ROSTER_URL,
                evidence_snippet=(f"{subject.canonical_name} and "
                                  f"{other.canonical_name} are both listed on "
                                  f"the {FIRM} team page."),
                confidence_base=0.8, confidence_adjusted=0.8,
                signals=EdgeSignals(trusted=True, explicit_keyword_match=True),
            )
            builder.add_edge_from_extraction(db, subject, edge, 0, source, other)
            made += 1
        # each partner is also an employee of the firm itself
        org_edge = ExtractedEdge(
            person_a=subject.canonical_name, organization=FIRM,
            other_kind="organization", org_type="firm",
            relationship_type="employee", method="firm roster",
            source_url=ROSTER_URL,
            evidence_snippet=f"{subject.canonical_name} is listed on the {FIRM} team page.",
            confidence_base=0.85, confidence_adjusted=0.85,
            signals=EdgeSignals(trusted=True, explicit_keyword_match=True),
        )
        builder.add_edge_from_extraction(db, subject, org_edge, 0, source, org)
    db.commit()
    return made


def _reach(db, root_ids: set, hops: int) -> set:
    """People within `hops` of any root, over the persisted person graph."""
    adj, _by, _s, _deg = _adjacency(db)
    seen = set(root_ids)
    frontier = deque((pid, 0) for pid in root_ids)
    while frontier:
        pid, d = frontier.popleft()
        if d >= hops:
            continue
        for nb, _e in adj.get(pid, []):
            if nb not in seen:
                seen.add(nb)
                frontier.append((nb, d + 1))
    return seen


def _partner_ids(db, names: list) -> set:
    norms = {person_norm_key(n) for n in names}
    return {p.id for p in db.execute(
        select(Person).where(Person.norm_name.in_(norms))).scalars()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="build a cached 2-hop graph of Y Combinator partners")
    parser.add_argument("--depth", type=int, default=2,
                        help="hops to walk out from each partner (default 2)")
    parser.add_argument("--limit", type=int, default=0,
                        help="expand only the first N partners (default: all). "
                             "The roster clique always covers the whole team.")
    parser.add_argument("--budget", type=float, default=7200.0,
                        help="wall-clock seconds before stopping. Checked at hop "
                             "boundaries, so an in-flight node can overrun it by "
                             "minutes; re-run to resume where this left off.")
    parser.add_argument("--fanout", type=int, default=config.EXPAND_TOP_STRONG,
                        help="frontier nodes expanded per hop, per partner")
    parser.add_argument("--strongest", action="store_true",
                        help="expand the best-documented neighbours instead of "
                             "the least-famous ones (default is reachability "
                             "mode, which walks toward normal people)")
    args = parser.parse_args(argv)

    depth = max(1, min(args.depth, 3))
    config.EXPAND_TOP_STRONG = args.fanout
    if args.strongest:
        config.EXPAND_PREFER_REACHABLE = False

    init_db()
    STATS.reset()
    deadline = time.monotonic() + args.budget

    db = SessionLocal()
    try:
        partners = _partner_list()
        to_expand = partners[: args.limit] if args.limit > 0 else partners
        print(f"\n{len(partners)} {FIRM} partners ({len(to_expand)} to expand)  ·  "
              f"depth {depth}  ·  fanout {config.EXPAND_TOP_STRONG}  ·  "
              f"budget {args.budget:.0f}s\n")

        edges = _materialize_clique(db, partners)
        print(f"  roster clique: {edges} colleague edges\n")

        protected = {person_norm_key(n) for n in partners}
        roots = _partner_ids(db, partners)
        before = len(_reach(db, roots, depth))
        expanded = 0

        for i, name in enumerate(to_expand, 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print(f"\nbudget exhausted after {expanded}/{len(to_expand)} partners")
                break
            print(f"[{i}/{len(to_expand)}] {name}  ({remaining:.0f}s left)")
            try:
                expand_graph(db, name, depth, progress=_log,
                             protected_norms=protected,
                             should_stop=lambda _s: time.monotonic() >= deadline)
                expanded += 1
            except Exception as exc:
                db.rollback()
                print(f"  ⚠ {name} failed ({exc.__class__.__name__}: {exc}) — skipped")

        roots = _partner_ids(db, partners)
        after = len(_reach(db, roots, depth))
        total_people = db.query(Person).count()

        print(f"\ndone — {expanded}/{len(to_expand)} partners expanded")
        print(f"  {depth}-hop reach: {before} -> {after} people")
        print(f"  people in the cached graph: {total_people}")
        print(f"  cache lives in {config.DB_URL} — re-run to extend it")
    finally:
        db.close()

    s = STATS.snapshot()
    print(f"\nprovider calls: serper={s['serper_searches']} brave={s['brave_searches']} "
          f"pages={s['page_fetches']}  cache hits/misses="
          f"{s['cache_hits']}/{s['cache_misses']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
