"""Terminal interface for Artemis V2.

Usage:
    python -m app.cli "Tim Cook"
    python -m app.cli "Tim Cook" --depth 2
    python -m app.cli                # prompts for a name interactively

Give a name, it discovers public relationships and prints the graph. No web
server, no frontend. Each run starts from a clean graph unless --keep is given.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from urllib.parse import urlparse

from .db import SessionLocal, init_db
from .extraction import claude_available
from .graph import builder
from .graph.expansion import expand_graph
from .graph.snapshot import save_graph_snapshot
from .providers import STATS
from .providers import cache as provider_cache
from .serializers import build_summary, serialize_edges, serialize_nodes
from .utils.names import normalize, person_norm_key


# most-specific / strongest relationship first — used to pick one label per pair
_REL_PRIORITY = [
    "linkedin_1st", "podcast_guest",
    "cofounder", "board_member", "investor", "employee", "faculty", "student",
    "advisor", "appointee", "coauthor", "author", "speaker", "coworker",
    "interview", "family_social", "unknown",
]
_REL_RANK = {r: i for i, r in enumerate(_REL_PRIORITY)}


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc or url
    except Exception:
        return url or "?"


def _reset_db(force: bool = False) -> None:
    """Clear the public graph but KEEP the uploaded local network."""
    init_db()
    db = SessionLocal()
    try:
        builder.reset_public_graph(db, force=force)
    finally:
        db.close()


def _start_clean(force_reset: bool) -> bool:
    """Whether this run should wipe the graph first.

    A private SQLite file keeps the original behaviour: each run starts fresh
    unless --keep. A SHARED database does not -- there, "start fresh" means
    deleting every collaborator's graph, and the command that does it is the
    most ordinary one there is (`python -m app.cli "Some Name"`, since --keep
    is opt-in). So on a shared graph this accumulates by default and says so,
    and wiping requires an explicit --force-reset.
    """
    if not builder.graph_is_shared():
        return True
    if force_reset:
        _progress("⚠ --force-reset: wiping the SHARED graph for everyone.")
        return True
    _progress("ℹ shared database — accumulating into the existing graph "
              "(pass --force-reset to wipe it for everyone).")
    return False


MAX_CONNS_SHOWN = 30


def _print_summary(summary: dict, stats: dict) -> None:
    t = summary["tiers"]
    print("\nSUMMARY")
    print(f"  nodes: {summary['nodes']}   edges: {summary['edges']}")
    print(f"  strong: {t['strong']}   candidate: {t['candidate']}   weak: {t['weak']}")
    dist = summary["confidence_distribution"]
    print("  confidence: " + "  ".join(f"{k}={v}" for k, v in dist.items()))
    depths = stats.get("nodes_processed_per_depth", [])
    if depths:
        print("  nodes processed per depth: "
              + ", ".join(f"hop{i}={n}" for i, n in enumerate(depths)))

    if summary["strongest_edges"]:
        print("\nTOP EDGES")
        for e in summary["strongest_edges"]:
            print(f"  - {e['from']} → {e['to']}  ({e['type']}, {e['confidence']:.2f})")

    if summary["top_people"]:
        print("\nTOP PEOPLE")
        for p in summary["top_people"][:8]:
            print(f"  - {p['label']}  (degree {p['degree']}, strength {p['strength']})")

    if summary["top_organizations"]:
        print("\nTOP ORGANIZATIONS")
        for o in summary["top_organizations"][:8]:
            print(f"  - {o['label']}  (degree {o['degree']}, strength {o['strength']})")


def _print_provider_stats() -> None:
    s = STATS.snapshot()
    print("\nPROVIDER STATS")
    print(f"  Serper searches:     {s['serper_searches']}")
    print(f"  Brave searches:      {s['brave_searches']}")
    print(f"  Wikipedia calls:     {s['wikipedia_calls']}")
    print(f"  Wikidata calls:      {s['wikidata_calls']}")
    print(f"  OpenAlex calls:      {s['openalex_calls']}")
    print(f"  OpenCorporates calls:{s['opencorporates_calls']}")
    print(f"  SEC EDGAR calls:     {s['edgar_calls']}")
    print(f"  ProPublica calls:    {s['propublica_calls']}")
    print(f"  DuckDuckGo searches: {s['duckduckgo_searches']}")
    print(f"  Page fetches:        {s['page_fetches']}")
    print(f"  Cache hits / misses: {s['cache_hits']} / {s['cache_misses']}")
    print(f"  Deduped queries:     {s['dedup_removed']}")
    print(f"  Circuit-breaker trips: {s['breaker_trips']}")
    print(f"  Avg request latency: {s['avg_latency_s']}s")
    print(f"  Total search time:   {s['total_search_time_s']}s")


def run(target_name: str, depth: int, keep: bool, show_all: bool = False,
        context: str = "", seed_is_person: bool = True,
        force_reset: bool = False) -> None:
    if keep or not _start_clean(force_reset):
        init_db()
    else:
        _reset_db(force=force_reset)

    STATS.reset()
    provider_cache.purge_expired()
    from .extraction.entity_filter import is_filtering_active
    from .extraction import spacy_available
    from . import config as _cfg
    if _cfg.CLAUDE_EXTRACT and claude_available():
        extractor = "claude"
    elif spacy_available():
        extractor = "spacy-ner"
    else:
        extractor = "heuristic"
    filt = "on" if is_filtering_active() else "off (set ANTHROPIC_API_KEY to enable)"
    print(f"\n🔎  Building relationship graph for: {target_name}")
    print(f"    depth={depth}  ·  extractor={extractor}  ·  claude node-filter={filt}\n",
          flush=True)

    db = SessionLocal()
    try:
        stats = expand_graph(db, target_name, depth, progress=_progress,
                             seed_context=context, seed_is_person=seed_is_person)
        nodes = serialize_nodes(db)
        edges = serialize_edges(db)
    finally:
        db.close()

    label = {n.id: n.label for n in nodes}
    kind = {n.id: n.kind for n in nodes}

    # group edges by source person, collapsing each counterpart to ONE line:
    # the strongest relationship type (by priority, then confidence) wins, with
    # any other meaningful types listed as "also". Sources are unioned.
    grouped = defaultdict(dict)  # from_id -> {to_id: edge_info}
    for e in edges:
        bucket = grouped[e.from_]
        cur = bucket.get(e.to)
        if cur is None:
            bucket[e.to] = {
                "to": e.to, "type": e.type, "confidence": e.confidence,
                "status": e.status, "depth": e.depth, "evidence": e.evidence,
                "types": {e.type},
                "sources": {e.source_url} if e.source_url else set(),
            }
            continue
        cur["types"].add(e.type)
        if e.source_url:
            cur["sources"].add(e.source_url)
        # promote the displayed type if this one is stronger
        better = _REL_RANK.get(e.type, 99) < _REL_RANK.get(cur["type"], 99)
        if better or (e.type == cur["type"] and e.confidence > cur["confidence"]):
            cur["type"] = e.type
            cur["confidence"] = max(e.confidence, cur["confidence"])
            cur["status"] = e.status
            cur["depth"] = e.depth
            cur["evidence"] = e.evidence or cur["evidence"]
        else:
            cur["confidence"] = max(cur["confidence"], e.confidence)

    summary = build_summary(nodes, edges)

    # save a snapshot of this build for safekeeping (never overwrites prior runs)
    snapshot_path = None
    if nodes:
        snapshot_path = save_graph_snapshot(target_name, depth, nodes, edges, stats, summary)

    print("\n" + "=" * 70)
    print(f"TARGET: {target_name}")
    print(
        f"people={stats['people_found']}  organizations={stats['organizations_found']}"
        f"  edges={stats['edges_found']}  sources={stats['sources_fetched']}"
    )
    print("=" * 70)

    if not edges:
        print("\nNo public relationships found (search may be rate-limited or the")
        print("name returned no usable results). Try a more specific name or --depth 2.\n")
        if snapshot_path:
            print(f"snapshot saved: {snapshot_path}")
        _print_provider_stats()
        return

    _print_summary(summary, stats)

    target_norm = normalize(target_name)
    # print the target first, then everyone else
    order = sorted(
        grouped.keys(),
        key=lambda fid: (normalize(label.get(fid, "")) != target_norm, label.get(fid, "")),
    )

    hidden_total = 0
    for from_id in order:
        conns = sorted(
            grouped[from_id].values(),
            key=lambda c: (_REL_RANK.get(c["type"], 99), -c["confidence"]),
        )
        if not show_all:
            # 'unknown'-typed edges are the un-signalled heuristic noise
            conns = [c for c in conns if c["type"] != "unknown"]
        if not conns:
            continue
        shown, extra = conns[:MAX_CONNS_SHOWN], conns[MAX_CONNS_SHOWN:]
        hidden_total += len(extra)
        marker = "🎯" if normalize(label.get(from_id, "")) == target_norm else "•"
        print(f"\n{marker} {label.get(from_id, from_id)}")
        for c in shown:
            to_label = label.get(c["to"], c["to"])
            to_kind = kind.get(c["to"], "person")
            tag = "🏢" if to_kind == "organization" else "👤"
            srcs = sorted(_domain(u) for u in c["sources"])
            src_str = ", ".join(srcs[:3]) + (" …" if len(srcs) > 3 else "")
            also = sorted(
                t for t in c["types"]
                if t not in (c["type"], "unknown", "family_social")
            )
            also_str = f"  (also: {', '.join(also)})" if also else ""
            print(
                f"    └─ {tag} {to_label}"
                f"  [{c['type']}, conf={c['confidence']:.2f}, {c['status']}, d{c['depth']}]"
                f"{also_str}"
            )
            if src_str:
                print(f"         src: {src_str}  ({len(c['sources'])} source(s))")
            if c["evidence"]:
                ev = c["evidence"].strip().replace("\n", " ")
                print(f"         “{ev[:140]}”")
        if extra:
            print(f"    … (+{len(extra)} more connections; use --all)")

    if not show_all:
        print("\n(low-signal 'unknown' relationships hidden — re-run with --all to see"
              " every extracted entity)")
    _print_provider_stats()
    if snapshot_path:
        print(f"\n💾 graph snapshot saved: {snapshot_path}")
    print()


# ---------------------------------------------------------------------------
# Local-network subcommands
# ---------------------------------------------------------------------------
def _find_target(db, name: str):
    from .models import Person
    from sqlalchemy import select
    norm = person_norm_key(name)
    return db.execute(select(Person).where(Person.norm_name == norm)).scalar_one_or_none()


def cmd_upload(argv) -> None:
    from .network.ingest import ingest_csv
    from .models import LocalProfile
    p = argparse.ArgumentParser(prog="upload-network")
    p.add_argument("path", help="path to a LinkedIn-style connections CSV")
    p.add_argument("--owner", default="",
                   help='whose contacts these are, e.g. "Jane Doe" -- when '
                        'given, each contact also becomes a real linkedin_1st '
                        'edge in the SHARED public graph (routable by anyone\'s '
                        '/connect), anchored to a Person node for this name. '
                        'Without it, ingestion stays in the private '
                        'LocalProfile/LocalEdge tables only.')
    if not argv:
        print("usage: python -m app.cli upload-network <path.csv> "
              '[--owner "Your Name"]', file=sys.stderr)
        sys.exit(1)
    ns = p.parse_args(argv)
    try:
        with open(ns.path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()
    except OSError as exc:
        print(f"cannot read {ns.path}: {exc}", file=sys.stderr)
        sys.exit(1)
    init_db()
    db = SessionLocal()
    try:
        stats = ingest_csv(db, content, owner_name=ns.owner)
        total = db.query(LocalProfile).count()
    finally:
        db.close()
    print(f"Ingested network from {ns.path}")
    print(f"  created={stats['created']} updated={stats['updated']} "
          f"edges={stats['edges']} skipped={stats['skipped']}")
    if ns.owner:
        print(f"  shared-graph linkedin_1st edges: {stats['graph_edges']} "
              f"(anchored to \"{ns.owner}\")")
    print(f"  total local profiles: {total}")


class _OrgSearchAdapter:
    """Adapts SearchOrchestrator to the plain search-provider interface
    (`.available()` / `.search(query)`) that FundingProvider expects."""

    def __init__(self, orch) -> None:
        self._orch = orch

    def available(self) -> bool:
        return True  # ORCH degrades to [] internally when nothing is configured

    def search(self, query: str):
        return self._orch.search(query, is_person=False)


def cmd_funding_rounds(argv) -> None:
    """Look up a firm's funding rounds and record co-investor firms as an
    ORG-level fact only (Organization.meta["co_investments"]) — never a
    person-to-person edge. See graph.builder.record_coinvestment."""
    from .graph.expansion import ORCH
    from .graph import builder
    from .providers.funding import FundingProvider
    from .utils.names import org_norm_key

    p = argparse.ArgumentParser(prog="funding-rounds")
    p.add_argument("name", nargs="*", help="firm name, e.g. \"Fiat Ventures\"")
    ns = p.parse_args(argv)
    firm_name = " ".join(ns.name).strip()
    if not firm_name:
        print('usage: python -m app.cli funding-rounds "Firm Name"', file=sys.stderr)
        sys.exit(1)

    init_db()
    STATS.reset()
    funding = FundingProvider(_OrgSearchAdapter(ORCH))
    print(f"\n💰  Looking up funding rounds naming: {firm_name}\n", flush=True)

    db = SessionLocal()
    try:
        anchor = builder.get_or_create_org(db, firm_name, org_type="firm")
        target = org_norm_key(firm_name)
        total_pairs = 0
        rounds = funding.rounds_for_firm(firm_name)
        for round_ in rounds:
            others = [n for n in round_["investors"]
                     if org_norm_key(n) != target]
            if not others:
                continue
            orgs = [anchor] + [builder.get_or_create_org(db, n, org_type="firm")
                               for n in others]
            recorded = builder.record_coinvestment(
                db, orgs, round_.get("company") or "the round",
                source_url=round_.get("source_url", ""))
            total_pairs += recorded
            names = ", ".join(o.name for o in orgs if o is not None)
            print(f"  {round_.get('company') or 'a round'}: {names}"
                  f"  ({recorded} new co-investor facts)")
        db.commit()
    finally:
        db.close()

    print(f"\nrounds found: {len(rounds)}   new co-investor facts recorded: {total_pairs}")
    if not rounds:
        print("  ⚠ No round announcements found naming this firm as an investor.")
    _print_provider_stats()


def cmd_connect(argv) -> None:
    """Find a path between two people by expanding both graphs and meeting in the middle."""
    from .graph.connect import connect_people
    p = argparse.ArgumentParser(prog="connect")
    p.add_argument("names", nargs="*", help='two names: "Person A" "Person B"')
    p.add_argument("--depth", type=int, default=2,  # depth 2 keeps Brave cost sane
                   help="hops per side (default 2; 3 is ~2x the Brave calls)")
    p.add_argument("--context-a", default="", help="disambiguation hint for person A")
    p.add_argument("--context-b", default="", help="disambiguation hint for person B")
    ns = p.parse_args(argv)
    if len(ns.names) != 2:
        print('usage: python -m app.cli connect "Person A" "Person B" [--depth N] '
              '[--context-a "hint"] [--context-b "hint"]', file=sys.stderr)
        sys.exit(1)
    a, b = ns.names
    depth = max(1, min(ns.depth, 3))  # mirrors the HTTP API's cap (main.py) --
    # unbounded depth is exponential in EXPAND_NODE_CONCURRENCY x SEARCH_WORKERS
    init_db()
    STATS.reset()
    ctx = (f"  (A: {ns.context_a})" if ns.context_a else "") + \
          (f"  (B: {ns.context_b})" if ns.context_b else "")
    print(f"\n🔗  Connecting: {a}  ⇄  {b}  (depth {depth} each){ctx}", flush=True)
    db = SessionLocal()
    try:
        result = connect_people(db, a, b, depth, progress=_progress,
                                context_a=ns.context_a, context_b=ns.context_b)
    finally:
        db.close()
    print("\n" + "=" * 70)
    if not result.get("connected"):
        print(f"NO PATH: {result.get('reason')}")
    else:
        routes = result.get("paths", [result])
        print(f"{len(routes)} ROUTE(S): {result['person_a']} → {result['person_b']}")
        print("=" * 70)
        for ri, route in enumerate(routes, 1):
            print(f"\n── Route {ri}  ({route['hops']} hops, score {route['score']}) ──")
            for i, node in enumerate(route["path"]):
                arrow = "" if i == 0 else \
                    f"  ──[{node.get('relationship_from_previous','?')}, " \
                    f"{node.get('confidence',0):.2f}]──▶ "
                print(f"{arrow}{node['label']}")
                if node.get("evidence"):
                    print(f"        “{node['evidence'][:120]}”")
                if node.get("source_url"):
                    print(f"        src: {node['source_url']}")
        print("\n⚠ " + " · ".join(result.get("warnings", [])))
    _print_provider_stats()


def cmd_add_org(argv) -> None:
    """Discover people affiliated with an org and add them to the local network."""
    from .network.org_discovery import discover_org_network
    from .providers import STATS
    p = argparse.ArgumentParser(prog="add-org-network")
    p.add_argument("name", nargs="*")
    p.add_argument("--depth", type=int, default=1)
    ns = p.parse_args(argv)
    depth = max(1, min(ns.depth, 3))  # mirrors the HTTP API's cap (main.py)
    org = " ".join(ns.name).strip()
    if not org:
        print('usage: python -m app.cli add-org-network "Org Name" [--depth N]',
              file=sys.stderr)
        sys.exit(1)
    init_db()
    STATS.reset()
    print(f"\n🔎  Discovering people affiliated with: {org}  (depth={depth})\n", flush=True)
    db = SessionLocal()
    try:
        result = discover_org_network(db, org, depth, source_tag="org_discovery",
                                      progress=_progress)
        from .models import LocalProfile
        total = db.query(LocalProfile).count()
    finally:
        db.close()
    print(f"\n  related people found: {result['discovered']}")
    print(f"  added to your network: {result['promoted']}")
    print(f"  existing profiles tagged with org: {result['updated']}")
    print(f"  total local profiles now: {total}")
    if result["discovered"] == 0:
        print("\n  ⚠ Nothing found. This org isn't on Wikipedia, so it needs Brave"
              " (set BRAVE_API_KEY in .env) — DuckDuckGo is currently throttled.")
    _print_provider_stats()


def cmd_match(argv) -> None:
    from .network.matching import run_matching
    from .network.paths import generate_paths_for_target
    name = " ".join(argv).strip()
    if not name:
        print("usage: python -m app.cli match \"Target Name\"", file=sys.stderr)
        sys.exit(1)
    init_db()
    db = SessionLocal()
    try:
        target = _find_target(db, name)
        if target is None:
            print(f"Target '{name}' not in the public graph. Build it first:\n"
                  f"  python -m app.cli \"{name}\"", file=sys.stderr)
            sys.exit(1)
        matches = run_matching(db)
        paths = generate_paths_for_target(db, target.id)
        by_type: dict = {}
        for m in matches:
            by_type[m.match_type] = by_type.get(m.match_type, 0) + 1
        print(f"\nMatched local network against public graph for: {target.canonical_name}")
        print(f"  graph matches: {len(matches)}  " +
              "  ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
        print(f"  candidate paths: {len(paths)}  (all UNVERIFIED)")
        for c in paths[:10]:
            hops = [n["label"] for n in c.path_json["path"]]
            print(f"   - score {c.score:.2f}: " + " → ".join(hops))
        print("\nNote: paths are unverified. Run 'paths' for full JSON. "
              "Claude verification is NOT part of this stage.")
    finally:
        db.close()


def cmd_paths(argv) -> None:
    import json as _json
    from sqlalchemy import select
    from .models import CandidatePath
    name = " ".join(argv).strip()
    if not name:
        print("usage: python -m app.cli paths \"Target Name\"", file=sys.stderr)
        sys.exit(1)
    init_db()
    db = SessionLocal()
    try:
        target = _find_target(db, name)
        if target is None:
            print(f"Target '{name}' not in the public graph.", file=sys.stderr)
            sys.exit(1)
        rows = list(db.execute(
            select(CandidatePath)
            .where(CandidatePath.target_person_id == target.id)
            .order_by(CandidatePath.score.desc())
        ).scalars())
        if not rows:
            print(f"No candidate paths for '{target.canonical_name}'. "
                  f"Run: python -m app.cli match \"{name}\"")
            return
        print(_json.dumps([c.path_json for c in rows], indent=2, ensure_ascii=False))
    finally:
        db.close()


def _run_search(argv) -> None:
    parser = argparse.ArgumentParser(
        prog="artemis",
        description="Discover public relationships for a person and print the graph.",
    )
    parser.add_argument("name", nargs="*", help="target person's name")
    parser.add_argument("-d", "--depth", type=int, default=1,
                        help="levels to explore: 1=target's direct relationships "
                             "(default, fast), 2=also expand top connections (slow)")
    parser.add_argument("--keep", action="store_true",
                        help="accumulate into existing graph instead of starting fresh "
                             "(always the default on a shared database)")
    parser.add_argument("--force-reset", action="store_true", dest="force_reset",
                        help="wipe a SHARED graph database before building — "
                             "deletes every collaborator's data, not just yours")
    parser.add_argument("--all", action="store_true", dest="show_all",
                        help="show every extracted entity, including low-signal "
                             "'unknown' relationships (noisy)")
    parser.add_argument("--context", default="",
                        help="disambiguation hint for a non-notable person "
                             "(e.g. 'Pearson Connections Academy finance')")
    parser.add_argument("--org", action="store_true",
                        help="seed is an organization (route via web search, not "
                             "the Wikipedia-person path)")
    args = parser.parse_args(argv)

    target = " ".join(args.name).strip()
    if not target:
        try:
            target = input("Enter a person's name: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
    if not target:
        print("No name given.", file=sys.stderr)
        sys.exit(1)
    run(target, max(1, min(args.depth, 3)), args.keep, args.show_all, context=args.context,
        seed_is_person=not args.org, force_reset=args.force_reset)


def cmd_directory(argv) -> None:
    """List the contacts an organization's own directory page names.

    Read-only and graph-free: this scrapes and prints, it writes no nodes or
    edges. Deciding that any of these people are connected to a *subject* is
    expansion phase 4f's job, and it needs a subject to do it -- see
    providers/directory.py. Here you are just looking at who a company lists.
    """
    from . import config
    from .providers import SearchOrchestrator

    p = argparse.ArgumentParser(prog="directory")
    p.add_argument("org", nargs="*", help='organization name, e.g. "Uncork Capital"')
    p.add_argument("--industry", default="",
                   help="sector hint; picks the query pack (see config.DIRECTORY_PACKS)")
    p.add_argument("--size", default="small",
                   choices=["small", "mid", "large", "unknown"],
                   help="small/mid scrape the full directory; large/unknown are "
                        "restricted to a leadership page (default: small)")
    ns = p.parse_args(argv)
    org = " ".join(ns.org).strip()
    if not org:
        print('usage: python -m app.cli directory "Org Name" [--industry X] [--size small|mid|large]',
              file=sys.stderr)
        sys.exit(1)

    orch = SearchOrchestrator()
    print(f"\n▤  Directory: {org}   (size={ns.size}"
          f"{', industry=' + ns.industry if ns.industry else ''})", flush=True)
    found = orch.directory_enrichment(org, industry=ns.industry, size_tier=ns.size)

    status, url, members = found.get("status"), found.get("url"), found.get("members") or []
    if not members:
        # Say WHY, never just "nothing found" -- an empty result that doesn't
        # explain itself is indistinguishable from "this org has no directory".
        reason = {
            "no_verified_page": "no page passed the roster-shape check and org-identity guard",
            "js_shell": "the page is a JavaScript shell with no readable text "
                        "(a headless browser would be needed; see providers/browser.py)",
            "identity_mismatch": "the page found does not belong to this organization",
            "no_names_found": "the page was readable but held no person-shaped names",
            "disabled": "directory lookups are disabled (ARTEMIS_DIRECTORY_ENABLED)",
        }.get(status, f"status={status}")
        if str(status).startswith("fetch_"):
            reason = f"the page refused the fetch ({status.split('_', 1)[1]}) — bot-blocked"
        print(f"\n  no contacts found — {reason}")
        if url:
            print(f"  page: {url}")
        return

    print(f"\n  source: {url}")
    if found.get("overflow"):
        print(f"  (page lists more than {config.DIRECTORY_MAX_MEMBERS}; showing the cap)")
    print(f"\n  {len(members)} contact(s):\n")
    for i, name in enumerate(members, 1):
        print(f"    {i:>3}. {name}")
    print()


_SUBCOMMANDS = {
    "upload-network": cmd_upload,
    "add-org-network": cmd_add_org,
    "connect": cmd_connect,
    "directory": cmd_directory,
    "match": cmd_match,
    "paths": cmd_paths,
    "funding-rounds": cmd_funding_rounds,
}


def main() -> None:
    argv = sys.argv[1:]
    try:
        if argv and argv[0] in _SUBCOMMANDS:
            _SUBCOMMANDS[argv[0]](argv[1:])
        else:
            _run_search(argv)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
