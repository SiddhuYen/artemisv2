"""Expansion engine — the BFS that grows the graph outward (hardened).

expand_graph(db, target, max_depth):
  hop 0 : process the target through ALL silos
  hop k : process only the TOP strong people discovered at hop k-1

Per-node processing:
  - network phase (searches + page fetches) runs concurrently,
  - extraction + dedup runs sequentially,
  - edges for the node are deduped, then capped: if a node yields more than
    MAX_EDGES_PER_NODE candidate edges, only the top EDGE_SAMPLE_LIMIT by
    confidence are persisted (anti-explosion),
  - new nodes are not created past MAX_TOTAL_NODES.

Ranking for expansion favours nodes with strong, explicit, source-diverse
relationships. Only the top strong nodes (or those with a >STRONG_MIN edge) are
expanded. No Claude / external-network matching here.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from .. import config
from . import disambiguate
from ..extraction import extract, tier
from ..extraction.entity_filter import is_filtering_active
from ..extraction.entity_filter import validate as filter_entities
from ..extraction.schemas import EdgeSignals, ExtractedEdge
from ..models import Organization, Person, RelationshipEdge, Source
from ..providers import SearchOrchestrator, SearchResult
from ..network.silo_weights import query_budget as silo_query_budget
from ..silos import COLLEAGUE_SILO, SILOS, STRUCTURED_SILO
from ..utils.htmltext import html_to_text
from ..utils.names import (
    is_noise_name,
    looks_like_person_name,
    org_norm_key,
    person_norm_key,
)
from . import builder

# search orchestrator (Brave primary -> Wikipedia/Wikidata -> DuckDuckGo fallback)
ORCH = SearchOrchestrator()

# phase-0 sources whose names are clean structured labels (no Claude filtering)
_CLEAN_STRUCTURED = {"wikidata", "wikidata-colleagues", "propublica-board"}


def _mark_trusted(edges, trusted: bool) -> None:
    if trusted:
        for e in edges:
            e.signals.trusted = True


@dataclass
class _Candidate:
    """Accumulated evidence about a discovered person, for expansion ranking."""
    name: str
    sources: Set[str] = field(default_factory=set)
    confidences: List[float] = field(default_factory=list)
    strong_edges: int = 0
    explicit_edges: int = 0
    max_conf: float = 0.0
    professional_edges: int = 0   # coworker/board/cofounder/investor/political/…
    family_edges: int = 0         # family_social (spouse/child/parent/sibling/friend)
    trusted: bool = False         # came from a structured source (skip Claude filter)

    def avg_conf(self) -> float:
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0

    def family_only(self) -> bool:
        return self.family_edges > 0 and self.professional_edges == 0

    def demote_family(self, downweight: bool) -> bool:
        """Whether to push this node to the back of the expansion frontier.

        Pure-family nodes are down-weighted (paths usually run through
        colleagues, not relatives) — UNLESS backed by explicit personal-tie
        evidence (a named spouse/sibling/friend keyword). A well-evidenced
        friend or relative is a legitimate warm-intro bridge worth expanding,
        so it is NOT demoted."""
        return downweight and self.family_only() and self.explicit_edges == 0

    def score(self) -> float:
        # strong edges + average confidence + source diversity + explicit ties
        base = (
            self.strong_edges * 3.0
            + self.avg_conf() * 2.0
            + len(self.sources) * 1.0
            + self.explicit_edges * 1.0
        )
        if config.DOWNWEIGHT_FAMILY:
            # prefer professional connections over genealogy: a path between two
            # people almost always runs through colleagues/boards, not relatives.
            base += (self.professional_edges * config.PROFESSIONAL_BONUS
                     - self.family_edges * config.FAMILY_PENALTY)
        return base

    def is_expandable(self) -> bool:
        # Strong (>0.6) edges qualify; so do nodes backed by an explicit-keyword
        # relationship (evidence-grounded), so depth still works under the
        # low-confidence heuristic extractor. Pure co-occurrence never qualifies.
        return (
            self.strong_edges > 0
            or self.max_conf > config.STRONG_MIN
            or self.explicit_edges > 0
        )


def _record(disc: Dict[str, _Candidate], edge: ExtractedEdge) -> None:
    if edge.other_kind != "person":
        return
    norm = person_norm_key(edge.person_b)
    if not norm:
        return
    cand = disc.get(norm)
    if cand is None:
        cand = _Candidate(name=edge.person_b)
        disc[norm] = cand
    if edge.source_url:
        cand.sources.add(edge.source_url)
    cand.confidences.append(edge.confidence_adjusted)
    cand.max_conf = max(cand.max_conf, edge.confidence_adjusted)
    if tier(edge.confidence_adjusted) == "strong":
        cand.strong_edges += 1
    if edge.signals.explicit_keyword_match:
        cand.explicit_edges += 1
    if edge.relationship_type == "family_social":
        cand.family_edges += 1
    elif edge.relationship_type != "unknown":
        cand.professional_edges += 1  # 'unknown' counts as neither
    if edge.signals.trusted:
        cand.trusted = True


def _merge_disc(into: Dict[str, _Candidate], other: Dict[str, _Candidate]) -> None:
    """Combine one worker's discovered-candidate tallies into the hop's shared
    `disc` map. Frontier nodes now process concurrently (see expand_graph),
    each accumulating into its OWN local dict via _record/_reuse_existing_
    neighbors; this is where those local dicts recombine, on the controlling
    thread as ex.map() yields results back -- never inside a worker -- so
    `into` needs no lock. Two different nodes can discover the SAME candidate
    (e.g. two coworkers both mention the same third person), so this replays
    _record's per-field accumulation rather than a plain dict update, which
    would silently drop one side's evidence."""
    for norm, c in other.items():
        existing = into.get(norm)
        if existing is None:
            into[norm] = c
            continue
        existing.sources |= c.sources
        existing.confidences.extend(c.confidences)
        existing.max_conf = max(existing.max_conf, c.max_conf)
        existing.strong_edges += c.strong_edges
        existing.explicit_edges += c.explicit_edges
        existing.professional_edges += c.professional_edges
        existing.family_edges += c.family_edges
        existing.trusted = existing.trusted or c.trusted


def _reuse_existing_neighbors(db: Session, subject: Person,
                              disc: Dict[str, _Candidate], progress=None) -> None:
    """Populate `disc` from a node's ALREADY-persisted person edges (from any
    prior run, including other teammates') so the next frontier can be ranked and
    expanded WITHOUT re-running the node's searches. Mirrors _record's tallies."""
    rows = list(db.execute(
        select(RelationshipEdge).where(
            RelationshipEdge.person_a_id == subject.id,
            RelationshipEdge.person_b_id.isnot(None),
        )
    ).scalars())
    if not rows:
        return
    b_ids = {e.person_b_id for e in rows}
    people = {p.id: p for p in db.execute(
        select(Person).where(Person.id.in_(b_ids))).scalars()}
    src_ids = {e.source_id for e in rows if e.source_id}
    src_url = {s.id: s.url for s in db.execute(
        select(Source).where(Source.id.in_(src_ids))).scalars()} if src_ids else {}

    for e in rows:
        b = people.get(e.person_b_id)
        if b is None:
            continue
        cand = disc.get(b.norm_name)
        if cand is None:
            cand = _Candidate(name=b.canonical_name)
            disc[b.norm_name] = cand
        url = src_url.get(e.source_id)
        if url:
            cand.sources.add(url)
        conf = e.confidence_raw or 0.0
        cand.confidences.append(conf)
        cand.max_conf = max(cand.max_conf, conf)
        if tier(conf) == "strong":
            cand.strong_edges += 1
        sig = e.signals or {}
        if sig.get("explicit_keyword_match"):
            cand.explicit_edges += 1
        if e.relationship_type == "family_social":
            cand.family_edges += 1
        elif e.relationship_type != "unknown":
            cand.professional_edges += 1
        if sig.get("trusted"):
            cand.trusted = True
    if progress:
        progress(f"  ♻ reuse {subject.canonical_name}: {len(disc)} known neighbors "
                 f"(skipped re-searching)")


def _dedup_and_cap(edges: List[ExtractedEdge]) -> List[ExtractedEdge]:
    """Dedup by (counterpart, type, source_url); cap/sample per node."""
    seen = {}
    for e in edges:
        key = (e.other_kind, person_norm_key(e.person_b) if e.other_kind == "person"
               else org_norm_key(e.organization), e.relationship_type, e.source_url)
        prev = seen.get(key)
        if prev is None or e.confidence_adjusted > prev.confidence_adjusted:
            seen[key] = e
    unique = sorted(seen.values(), key=lambda e: e.confidence_adjusted, reverse=True)
    if len(unique) > config.MAX_EDGES_PER_NODE:
        return unique[: config.EDGE_SAMPLE_LIMIT]
    return unique


def _process_person(db: Session, subject_name: str, hop: int, disc: Dict[str, _Candidate],
                    progress=None, is_person: bool = True, context: str = "",
                    cancel_checker: Optional[Callable[[], None]] = None,
                    silo_weights: Optional[Dict[str, float]] = None) -> None:
    def check_cancel() -> None:
        if cancel_checker:
            cancel_checker()

    check_cancel()
    subject = builder.get_or_create_person(db, subject_name)
    if subject is None:
        return

    # Disambiguation: when a context hint is given (e.g. "Indiana Pacers owner"),
    # the subject is NOT the famous bare-name Wikipedia entity — so skip wiki/
    # wikidata enrichment and route via web search with the context appended.
    effective_is_person = is_person and not context

    source_by_url: Dict[str, Source] = {}
    candidate_edges: List[ExtractedEdge] = []

    # --- phase 0: structured enrichment (Tier-1 high recall) ---------------
    # Wikipedia full article + Wikidata facts + colleagues, plus the page's
    # person-links added directly as clean contacts.
    check_cancel()
    enrichment = ORCH.enrich_person(subject_name) if effective_is_person else None
    if enrichment:
        # anchor the subject's identity to its Wikidata QID (homonym disambiguation):
        # two different notable same-name people have distinct QIDs and stay separate.
        if enrichment.get("qid"):
            # candidate signal for builder's homonym guard: the Wikipedia lead +
            # Wikidata facts text describing whoever this QID actually is. Already
            # fetched above (enrich_person), so this adds no extra cost.
            identity_text = " ".join(
                t for t in (enrichment.get("summary"), enrichment.get("wikidata_text")) if t
            )[:1500]
            resolved = builder.get_or_create_person(
                db, subject_name, qid=enrichment["qid"], identity_text=identity_text)
            if resolved is not None:
                subject = resolved
        wiki_url = "https://en.wikipedia.org/wiki/" + enrichment["title"].replace(" ", "_")
        # (label, text, silo) — direct facts/prose use STRUCTURED; shared-affiliation
        # colleagues use the lower-confidence COLLEAGUE silo.
        for label, text, silo in (
            ("wikipedia-article", enrichment.get("article", ""), STRUCTURED_SILO),
            ("wikipedia-summary", enrichment["summary"], STRUCTURED_SILO),
            ("wikidata", enrichment["wikidata_text"], STRUCTURED_SILO),
            ("wikidata-colleagues", enrichment.get("colleagues_text", ""), COLLEAGUE_SILO),
            ("propublica-board", enrichment.get("nonprofit_text", ""), COLLEAGUE_SILO),
        ):
            check_cancel()
            if not text:
                continue
            url = f"{wiki_url}#{label}"  # distinct source per label (preserve provenance)
            res = SearchResult(enrichment["title"], url, text[:200], label)
            source = builder.save_source(db, res, f"enrich:{label}", text)
            source_by_url[res.url] = source
            out = extract(subject_name, text, silo, text[:200], res.url)
            # clean structured facts -> trusted (skip Claude entity filter); the
            # full article/summary are prose and still need filtering.
            _mark_trusted(out.edges, label in _CLEAN_STRUCTURED)
            candidate_edges.extend(out.edges)

    # --- phase 0b: shared-affiliation colleague sources (lower confidence) ---
    # OpenAlex is handled separately, LATER (see phase 4b below) -- it needs
    # an identity check against evidence this same call gathers, so it can't
    # run this early. OpenCorporates/EDGAR still run here, ungated: unlike
    # OpenAlex's bare coauthor-name list, no per-provider identity signal was
    # available to build a comparable check for these two without deeper
    # provider changes -- a known, scoped-out gap, not an oversight.
    check_cancel()
    if effective_is_person:
        for src_name, url, query, text in (
            ("opencorporates", "https://opencorporates.com/", "enrich:opencorporates",
             ORCH.officer_enrichment(subject_name)["officers_text"]),
            ("edgar", "https://www.sec.gov/cgi-bin/browse-edgar", "enrich:edgar",
             ORCH.edgar_enrichment(subject_name)["edgar_text"]),
        ):
            check_cancel()
            if not text:
                continue
            res = SearchResult(subject_name, url, src_name, src_name)
            source = builder.save_source(db, res, query, text)
            source_by_url[res.url] = source
            out = extract(subject_name, text, COLLEAGUE_SILO, src_name, res.url)
            _mark_trusted(out.edges, True)  # clean structured names (skip entity filter)
            candidate_edges.extend(out.edges)

    # --- phase 0c: firm team-roster colleagues (own source URL per roster) ---
    # Unlike the enrichments above, each colleague carries the actual roster
    # page it was scraped from — a stronger, citable structural assertion —
    # so each distinct roster gets its own Source rather than one shared URL.
    check_cancel()
    if effective_is_person:
        firm_cols = ORCH.firm_enrichment(subject_name)["firms"]
        by_url: Dict[str, List[dict]] = {}
        for c in firm_cols:
            check_cancel()
            by_url.setdefault(c.get("url") or "", []).append(c)
        for url, cols in by_url.items():
            check_cancel()
            if not url:
                continue
            text = ORCH.firms.colleagues_text(subject_name, cols)
            if not text:
                continue
            res = SearchResult(subject_name, url, "firms", "firms")
            source = builder.save_source(db, res, "enrich:firms", text)
            source_by_url[res.url] = source
            out = extract(subject_name, text, COLLEAGUE_SILO, "firms", res.url)
            _mark_trusted(out.edges, True)  # scraped roster names (skip entity filter)
            candidate_edges.extend(out.edges)

    # --- phase 0d: podcast host<->guest (structural, tier-1) ---------------
    # An episode asserts one thing: this host interviewed this subject. Direct
    # edge creation (not the text/silo pipeline below) because 'podcast_guest'
    # is its own relationship_type, not one a silo's keyword map produces.
    # Never a guest<->guest edge — see providers/podcasts.py.
    check_cancel()
    if effective_is_person and config.PODCASTS_ENABLED:
        known_orgs = [
            o.name for o in db.execute(
                select(Organization)
                .join(RelationshipEdge, RelationshipEdge.organization_id == Organization.id)
                .where(RelationshipEdge.person_a_id == subject.id)
            ).scalars()
        ]
        appearances = ORCH.podcasts.appearances(subject_name, known_orgs=known_orgs)
        for appearance in appearances:
            check_cancel()
            url = appearance.get("episode_url") or ""
            if not url:
                continue
            res = SearchResult(appearance.get("episode_title", ""), url,
                               "podcast_rss", "podcasts")
            source = builder.save_source(db, res, "enrich:podcasts")
            source_by_url[res.url] = source
            for host_name in appearance.get("hosts", []):
                if person_norm_key(host_name) == person_norm_key(subject_name):
                    continue
                edge = ExtractedEdge(
                    person_a=subject_name, person_b=host_name,
                    other_kind="person", relationship_type="podcast_guest",
                    method="podcast RSS feed", source_url=url,
                    evidence_snippet=(
                        f"{host_name} interviewed {subject_name} on "
                        f"{appearance.get('show', 'a podcast')} "
                        f"(“{appearance.get('episode_title', '')}”)."),
                    confidence_base=0.85, confidence_adjusted=0.85,
                    signals=EdgeSignals(trusted=True, explicit_keyword_match=True),
                )
                candidate_edges.append(edge)

    # --- phase 1: build (silo, query) pairs, then DEDUP across silos -------
    check_cancel()
    # Per-silo query allowance. Without weights every silo gets the full
    # MAX_QUERIES_PER_SILO, i.e. unchanged behavior; with them, silos that
    # cannot pay off for this particular subject are dropped and the rest are
    # scaled — see network/silo_weights.query_budget.
    budget = silo_query_budget(silo_weights)
    pairs = []
    for silo in SILOS:
        allowance = budget.get(silo.key, 0)
        if allowance <= 0:
            continue
        for query in silo.render_queries(subject_name)[:allowance]:
            pairs.append((silo, f"{query} {context}".strip() if context else query))
    unique_queries, query_to_silos = ORCH.dedup(pairs)

    if progress:
        ctx = f" [context: {context}]" if context else ""
        weighted = (f"  ·  {len(budget)}/{len(SILOS)} silos"
                    if silo_weights else "")
        progress(f"  [hop {hop}] {subject_name}{ctx}  ·  {len(unique_queries)} unique queries "
                 f"(deduped from {len(pairs)}){weighted}…")

    # --- phase 2: routed search, concurrent (cache-first) ------------------
    def _do_search(query):
        try:
            check_cancel()
            return query, ORCH.search(query, is_person=effective_is_person)
        except Exception:
            check_cancel()
            return query, []

    with ThreadPoolExecutor(max_workers=config.SEARCH_WORKERS) as ex:
        searched = list(ex.map(_do_search, unique_queries))

    # --- phase 3: fetch result pages concurrently (cache-first, deduped) ---
    check_cancel()
    to_scrape: Set[str] = set()
    for _query, results in searched:
        for rank, res in enumerate(results):
            if res.provider != "wikipedia" and rank < config.SCRAPE_TOP_N:
                to_scrape.add(res.url)

    page_text: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=config.SEARCH_WORKERS) as ex:
        def _do_fetch(url):
            check_cancel()
            return url, ORCH.fetch(url)

        for url, page in ex.map(_do_fetch, to_scrape):
            check_cancel()
            page_text[url] = html_to_text(page.content) if page.content else ""

    # --- phase 4: extraction per (result × originating silo) --------------
    check_cancel()
    for query, results in searched:
        check_cancel()
        silos = query_to_silos.get(query, set())
        for rank, res in enumerate(results):
            check_cancel()
            if res.provider == "wikipedia":
                full_text = ORCH.wikipedia.summary(res.title) or None
            elif rank < config.SCRAPE_TOP_N:
                full_text = page_text.get(res.url) or None
            else:
                full_text = None

            source = builder.save_source(db, res, query, full_text)
            source_by_url[res.url] = source
            text = full_text or f"{res.title}. {res.snippet}"

            for silo in silos:
                check_cancel()
                out = extract(subject_name, text, silo, res.snippet, res.url)
                candidate_edges.extend(out.edges)

    # --- phase 4b: OpenAlex coauthors, identity-gated -----------------------
    # A bare-name lookup (this one included) can't tell two same-named people
    # apart on its own -- OpenAlex's own match guard (works_count + name
    # similarity, see providers/openalex.py) narrows candidates, it doesn't
    # confirm identity. This runs HERE, after the main search above (not
    # back in phase 0b with the other enrichments), specifically so there's
    # real evidence about the subject already gathered to check the resolved
    # author's own affiliation against -- catching exactly the failure mode
    # that shipped live: an OpenAlex "Prantik Chakraborty" resolving to an
    # unrelated ISRO researcher while the actual subject (per real web
    # results already in candidate_edges, or per user-given `context`) is a
    # sales executive at a chemicals company. Gated on `is_person`, not
    # `effective_is_person`: a context hint no longer means "skip this
    # source," it means "here's a strong signal to verify it against."
    check_cancel()
    if is_person:
        oa = ORCH.coauthors_enrichment(subject_name)
        oa_text = oa["coauthors_text"]
        if oa_text:
            signal = " ".join(filter(None, [
                context,
                " ".join(e.evidence_snippet for e in candidate_edges if e.evidence_snippet),
            ]))
            if disambiguate.domain_conflict(signal, oa["identity_text"]):
                # Advisory record only (mirrors builder._homonym_conflict's
                # homonym_rejected note) -- doesn't block a later, better-
                # evidenced acceptance, just explains why this pass skipped it.
                meta = dict(subject.meta or {})
                meta["openalex_rejected"] = {"identity_text": oa["identity_text"][:300]}
                subject.meta = meta
                if progress:
                    progress(f"  ⚠ OpenAlex coauthors for {subject_name} rejected — "
                             f"resolved identity doesn't match: {oa['identity_text'][:80]}")
            else:
                res = SearchResult(subject_name, "https://openalex.org/", "openalex", "openalex")
                source = builder.save_source(db, res, "enrich:openalex", oa_text)
                source_by_url[res.url] = source
                out = extract(subject_name, oa_text, COLLEAGUE_SILO, "openalex", res.url)
                _mark_trusted(out.edges, True)  # verified above, or nothing to conflict with
                candidate_edges.extend(out.edges)

    # --- phase 5: dedup + per-node cap, then persist ----------------------
    check_cancel()
    final_edges = _dedup_and_cap(candidate_edges)
    if progress and len(candidate_edges) > len(final_edges):
        progress(f"  [hop {hop}] {subject_name}  ·  capped "
                 f"{len(candidate_edges)} → {len(final_edges)} edges (anti-explosion)")

    # Snapshot the cap once for this node's whole edge batch instead of
    # re-querying COUNT(*) on every candidate edge (up to MAX_EDGES_PER_NODE
    # per node) -- the cap is already a soft/best-effort bound (concurrent
    # hop workers can each read a stale count too), so reusing one snapshot
    # for a single node's edges costs nothing beyond what's already true.
    at_cap = builder.at_node_cap(db)
    for edge in final_edges:
        check_cancel()
        if edge.other_kind == "person":
            counterpart = builder.get_or_create_person(
                db, edge.person_b, allow_create=not at_cap
            )
            if counterpart is None:
                continue
            builder.add_edge_from_extraction(
                db, subject, edge, hop, source_by_url.get(edge.source_url), counterpart
            )
            _record(disc, edge)
        else:
            counterpart = builder.get_or_create_org(
                db, edge.organization, edge.org_type, allow_create=not at_cap
            )
            if counterpart is None:
                continue
            builder.add_edge_from_extraction(
                db, subject, edge, hop, source_by_url.get(edge.source_url), counterpart
            )

    # mark expanded: a later/deeper run will REUSE these neighbors instead of
    # re-searching this node (see _reuse_existing_neighbors).
    #
    # commit_with_retry, not a bare db.commit(): every write earlier in this
    # node's processing (save_source, add_edge_from_extraction) already went
    # through its own SAVEPOINT retry, so by the time we get here this
    # connection has either already secured SQLite's write lock for the rest
    # of this transaction (this commit can't newly contend) or nothing at all
    # was written for this node (this commit is the transaction's first write
    # attempt, and a lock retry here is safe because there's nothing else
    # pending to lose). Either way, re-applying just `processed = 1` on retry
    # is correct and cheap.
    builder.commit_with_retry(db, lambda: setattr(subject, "processed", 1))


def _ranked_expandable(disc: Dict[str, _Candidate], visited: Set[str],
                       progress=None, prefer_reachable: Optional[bool] = None) -> List[str]:
    """Choose the next hop's frontier.

    Two modes:
      - strongest (legacy): expand the highest-scoring, best-documented people.
      - reachable (default): expand the LEAST-famous real connections — people
        with no Wikipedia page and few sources — to walk DOWN the fame gradient
        toward a normal person's network (warm-intro pathfinding).

    `prefer_reachable` selects between them PER CALL. It used to be read
    straight off the config global, which meant connect_people had to flip that
    global for the duration of its build and restore it afterwards — process
    global state that forced every build in the API to run one at a time. As a
    parameter it is just an argument two concurrent builds can disagree about.
    None keeps the configured default.

    Claude filtering (when active) removes junk nodes from the frontier first.
    """
    if prefer_reachable is None:
        prefer_reachable = config.EXPAND_PREFER_REACHABLE
    if prefer_reachable:
        # real people with at least a candidate-tier edge (not just explicit/strong),
        # since the bridge people toward a normal network are weakly-linked by design.
        eligible = [c for norm, c in disc.items()
                    if norm not in visited and c.max_conf >= config.WEAK_MAX]
    else:
        eligible = [c for norm, c in disc.items()
                    if norm not in visited and c.is_expandable()]
    if not eligible:
        return []

    # pre-rank to bound the expensive checks (Claude + Wikipedia notability).
    # family-only nodes go LAST (hard) when down-weighting, so a few high-source
    # relatives can't crowd out genuine professional connections.
    fam = config.DOWNWEIGHT_FAMILY
    if prefer_reachable:
        eligible.sort(key=lambda c: (c.demote_family(fam), len(c.sources), -c.avg_conf()))
    else:
        eligible.sort(key=lambda c: (c.demote_family(fam), -c.score()))
    shortlist = eligible[: max(config.EXPAND_TOP_STRONG * 3, 30)]

    if is_filtering_active():
        # trusted (structured-source) candidates skip the Claude check entirely
        to_check = [c for c in shortlist if not c.trusted]
        valid = filter_entities([c.name for c in to_check], "person")
        dropped = [c.name for c in to_check if c.name not in valid]
        shortlist = [c for c in shortlist if c.trusted or c.name in valid]
        if progress and dropped:
            progress(f"  ⊘ Claude filter skipped {len(dropped)} non-person frontier "
                     f"nodes (e.g. {', '.join(dropped[:3])})")

    if prefer_reachable and shortlist:
        # fame signal: has a Wikidata-backed Wikipedia page -> famous -> deprioritize
        notable = ORCH.notable_set([c.name for c in shortlist])
        # least-famous first, but family-only nodes last (prefer professional ties),
        # then fewest sources, then solid edge
        fam = config.DOWNWEIGHT_FAMILY
        shortlist.sort(key=lambda c: (c.name in notable,
                                      c.demote_family(fam),
                                      len(c.sources), -c.avg_conf()))
        chosen = shortlist[: config.EXPAND_TOP_STRONG]
        if progress:
            famous = [c.name for c in chosen if c.name in notable]
            progress(f"  ↧ reachability: expanding {len(chosen)} least-famous nodes "
                     f"({len(chosen) - len(famous)} with no Wikipedia page)")
        return [c.name for c in chosen]

    return [c.name for c in shortlist[: config.EXPAND_TOP_STRONG]]


def expand_graph(db: Session, target_name: str, max_depth: int, progress=None,
                 seed_is_person: bool = True, seed_context: str = "",
                 protected_norms: Optional[Set[str]] = None,
                 on_step: Optional[Callable[[dict], None]] = None,
                 cancel_checker: Optional[Callable[[], None]] = None,
                 should_stop: Optional[Callable[[Session], bool]] = None,
                 prefer_reachable: Optional[bool] = None,
                 silo_weights: Optional[Dict[str, float]] = None) -> dict:
    """`protected_norms` are exempt from the final noise-shape prune in addition
    to this call's own seed. connect_people needs this: it runs expand_graph
    TWICE (once per endpoint) into the same shared graph, and without it the
    second call's prune sees the first call's seed as just another node —
    nothing marks it as an endpoint the caller still needs. Defaults to only
    this call's own seed, i.e. today's single-seed behavior, for every other
    caller (CLI, /expand, org_discovery).

    `on_step`, unlike `progress` (a free-text log line), reports STRUCTURED
    hop/node counters — {"phase": "hop_start"|"node_done", "hop", "total",
    "done"} — so a caller (the HTTP API) can turn them into an actual percent
    complete instead of an indeterminate spinner.

    `should_stop`, when provided, is checked before each hop and after each
    processed node. connect_people uses it to stop discovery as soon as the two
    endpoint graphs have met, instead of always exhausting the requested depth.

    `prefer_reachable` overrides config.EXPAND_PREFER_REACHABLE for THIS call
    only (see _ranked_expandable). connect_people passes False; leaving it None
    keeps the configured default. Per-call rather than global so two builds can
    run concurrently without fighting over one another's strategy."""
    visited: Set[str] = set()
    frontier: List[str] = [target_name]
    per_depth: List[int] = []  # nodes processed per hop

    # Frontier nodes within one hop are independent of each other -- nothing
    # about processing candidate #3 needs candidate #2 done first -- so they
    # run concurrently, each on its OWN Session (bound to the same engine as
    # `db`; a Session isn't thread-safe to share). Hops themselves stay
    # sequential: _ranked_expandable needs every node in a hop finished before
    # it can rank the next one, a real data dependency, not an oversight.
    engine = db.get_bind()
    WorkerSession = sessionmaker(bind=engine, autoflush=False,
                                 expire_on_commit=False, future=True)

    def check_cancel() -> None:
        if cancel_checker:
            cancel_checker()

    def stop_requested(session: Session) -> bool:
        check_cancel()
        return bool(should_stop and should_stop(session))

    def _process_one(name: str, hop: int) -> Dict[str, "_Candidate"]:
        local_disc: Dict[str, _Candidate] = {}
        worker_db = WorkerSession()
        try:
            if stop_requested(worker_db):
                return local_disc
            check_cancel()
            # If this node was already expanded (this run, a prior run, or by
            # another teammate in the shared map), REUSE its persisted
            # neighbors to rank the next frontier instead of re-searching —
            # so we keep the shallow work and just continue deeper
            # (incremental deepening).
            existing = builder.get_or_create_person(worker_db, name, allow_create=False)
            if existing is not None and existing.processed:
                _reuse_existing_neighbors(worker_db, existing, local_disc, progress)
            else:
                # only the seed at hop 0 may be an org; discovered nodes are people.
                # the disambiguation context applies only to the seed (hop 0).
                kwargs = {
                    "progress": progress,
                    "is_person": (seed_is_person or hop > 0),
                    "context": (seed_context if hop == 0 else ""),
                    # Weights describe THIS contact, derived from their own
                    # export row — they say nothing about the strangers found
                    # at hop 1+, so they apply to the seed only. Guessing that
                    # a founder's neighbours are also founders is exactly the
                    # unfounded prior this feature exists to remove.
                    "silo_weights": (silo_weights if hop == 0 else None),
                }
                if cancel_checker:
                    kwargs["cancel_checker"] = cancel_checker
                _process_person(worker_db, name, hop, local_disc, **kwargs)
        except Exception as exc:
            worker_db.rollback()
            try:
                check_cancel()
            except Exception:
                raise
            if progress:
                progress(f"  ⚠ {name!r} at hop {hop} failed "
                         f"({exc.__class__.__name__}) — skipped")
        finally:
            worker_db.close()
        return local_disc

    for hop in range(0, max_depth):
        if stop_requested(db):
            if progress:
                progress("  ✓ stop condition met; stopping expansion early")
            break
        disc: Dict[str, _Candidate] = {}
        to_process: List[str] = []
        for name in frontier:
            norm = person_norm_key(name)
            if norm in visited:
                continue
            visited.add(norm)
            to_process.append(name)

        if on_step:
            check_cancel()
            on_step({"phase": "hop_start", "hop": hop, "max_depth": max_depth,
                     "total": len(to_process)})

        if len(to_process) > 1:
            workers = min(config.EXPAND_NODE_CONCURRENCY, len(to_process))
            stop_after_node = False
            with ThreadPoolExecutor(max_workers=workers) as ex:
                # Drain futures as they COMPLETE (not input order) so a caller
                # that can stop early -- e.g. connect_people once a route exists
                # -- observes the first enriched node that creates the route.
                # _merge_disc still runs only on this controlling thread, so
                # `disc` and on_step counters need no lock.
                futures = [ex.submit(_process_one, name, hop) for name in to_process]
                done = 0
                for future in as_completed(futures):
                    if future.cancelled():
                        continue
                    local_disc = future.result()
                    _merge_disc(disc, local_disc)
                    done += 1
                    if on_step:
                        on_step({"phase": "node_done", "hop": hop,
                                 "max_depth": max_depth, "done": done,
                                 "total": len(to_process)})
                    if stop_requested(db):
                        for pending in futures:
                            pending.cancel()
                        stop_after_node = True
                        if progress:
                            progress("  ✓ stop condition met; stopping expansion early")
                        break
            if stop_after_node:
                per_depth.append(done)
                break
        else:
            stop_after_node = False
            done = 0
            for i, name in enumerate(to_process, 1):
                _merge_disc(disc, _process_one(name, hop))
                done = i
                if on_step:
                    on_step({"phase": "node_done", "hop": hop,
                             "max_depth": max_depth, "done": i,
                             "total": len(to_process)})
                if stop_requested(db):
                    stop_after_node = True
                    if progress:
                        progress("  ✓ stop condition met; stopping expansion early")
                    break
            if stop_after_node:
                per_depth.append(done)
                break
        per_depth.append(len(to_process))

        if hop == max_depth - 1:
            break
        if builder.at_node_cap(db):
            if progress:
                progress(f"  → node cap ({config.MAX_TOTAL_NODES}) reached; stopping expansion")
            break

        check_cancel()
        frontier = _ranked_expandable(disc, visited, progress=progress,
                                      prefer_reachable=prefer_reachable)
        if progress and frontier:
            progress(f"  → expanding top {len(frontier)} strong nodes to hop {hop + 1}: "
                     + ", ".join(frontier[:5]) + (" …" if len(frontier) > 5 else ""))
        if not frontier:
            if progress:
                progress("  → no strong nodes to expand; stopping")
            break

    protected = {person_norm_key(target_name)} | (protected_norms or set())
    check_cancel()
    _prune_invalid_nodes(db, protected, progress=progress)
    check_cancel()
    _retype_unknown_edges(db, progress=progress)
    return _stats(db, per_depth)


def _retype_unknown_edges(db: Session, progress=None) -> int:
    """Re-type 'unknown' edges via the Claude relationship classifier, using each
    edge's evidence sentence. Turns 'unknown 0.40' into e.g. 'coworker 0.8'."""
    from ..extraction import relation_classifier
    if not relation_classifier.is_active():
        return 0
    rows = list(db.execute(
        select(RelationshipEdge).where(RelationshipEdge.relationship_type == "unknown")
    ).scalars())
    if not rows:
        return 0
    people = {p.id: p.canonical_name for p in db.execute(select(Person)).scalars()}
    orgs = {o.id: o.name for o in db.execute(select(Organization)).scalars()}

    # Only re-type edges whose evidence sentence actually contains BOTH endpoints.
    # Otherwise the snippet is about a third party (the page wasn't about either
    # of them), and the classifier would confidently mislabel a co-occurrence
    # artifact (e.g. Heintz↔Clinton from a sentence about Eric Liu).
    items, eligible = [], []
    skipped_mismatch = 0
    for e in rows:
        a = people.get(e.person_a_id, "")
        b = people.get(e.person_b_id) or orgs.get(e.organization_id) or ""
        ev = (e.evidence_snippet or "")
        if a and b and a.lower() in ev.lower() and b.lower() in ev.lower():
            items.append({"a": a, "b": b, "evidence": ev})
            eligible.append(e)
        else:
            skipped_mismatch += 1

    verdicts = relation_classifier.classify(items)  # the expensive part -- never redone on retry

    # commit_with_retry, not a bare db.commit(): each edge here is an
    # ALREADY-persistent row being mutated, not a new insert -- a rollback()
    # forced by a failed commit reverts a pending attribute change back to
    # its last-committed value (confirmed empirically, see
    # tests/test_commit_retry.py), so a bare commit() retry would silently
    # commit none of these retypes. Redoing the (cheap, in-memory) mutation
    # loop on each attempt is what makes the retry actually retype the edges
    # instead of quietly doing nothing.
    def _apply() -> int:
        updated = 0
        for e, v in zip(eligible, verdicts):
            rtype, conf = v.get("type", "unknown"), v.get("confidence", 0.0)
            if rtype != "unknown" and conf >= config.CLAUDE_CLASSIFY_MIN_CONF:
                new_conf = round(min(conf, config.RELATION_CONF_CEILING), 3)
                e.relationship_type = rtype
                e.confidence_raw = max(e.confidence_raw or 0.0, new_conf)
                e.status = builder.derive_status(rtype, e.confidence_raw)
                sig = dict(e.signals or {})
                sig["relationship_classified_by"] = "claude"
                e.signals = sig
                updated += 1
        return updated

    updated = builder.commit_with_retry(db, _apply) or 0
    if progress and (updated or skipped_mismatch):
        progress(f"  ✎ Claude typed {updated} edges "
                 f"(skipped {skipped_mismatch} with mismatched evidence)")
    return updated


def _prune_invalid_nodes(db: Session, protected_norms: Set[str], progress=None) -> int:
    """Final pass: remove nodes that aren't real named people/orgs (with edges).

    PEOPLE are pruned by the DETERMINISTIC name-shape filter, NOT Claude. The LLM
    entity filter proved unreliable on names: it false-DELETED real connections
    (named co-founders) while false-KEEPING page-title junk like "Drew Glover -
    LinkedIn" — which carries strong explicit edges indistinguishable from a real
    node's. In a relationship graph a false-delete loses the answer while a
    false-keep is cheap noise, so a well-formed personal name is authoritative and
    the LLM never gets to delete a plausible person. This also means people get
    cleaned even where no Claude key is configured.

    ORGS still use the Claude filter when active — org names are far messier and a
    wrong drop is much less costly than losing a person. Nodes reached via a
    TRUSTED structured source are clean by construction and never pruned.

    `protected_norms` may hold more than this call's own seed (see expand_graph)
    — every one of them is exempt.

    Flushes first: this runs after edges were added to `db` in the same
    transaction, and the bulk deletes below are raw SQL against
    `RelationshipEdge` — with autoflush off, they're blind to any edge still
    only pending in the session, and that edge's endpoint can vanish here while
    the edge itself survives the commit, an orphaned reference to a person that
    no longer exists."""
    db.flush()
    trusted_pids, trusted_oids = set(), set()
    for e in db.execute(select(RelationshipEdge)).scalars():
        if (e.signals or {}).get("trusted"):
            if e.person_b_id:
                trusted_pids.add(e.person_b_id)
            if e.organization_id:
                trusted_oids.add(e.organization_id)

    removed = 0
    # --- people: deterministic shape filter (LLM-independent, safe) ---------
    junk_people = [
        p for p in db.execute(select(Person)).scalars()
        if p.norm_name not in protected_norms and p.id not in trusted_pids
        and (is_noise_name(p.canonical_name) or not looks_like_person_name(p.canonical_name))
    ]
    if junk_people:
        pids = [p.id for p in junk_people]
        # One batched delete instead of one DELETE per node -- edges are the
        # only thing that needs a manual cascade (RelationshipEdge has no FK
        # cascade), and IN(...) does it in a single round trip either way.
        # Retries on a SQLite lock timeout instead of failing the whole job
        # outright -- a large batch (hundreds of ids, e.g. after researching
        # a very public figure) is exactly the kind of slow write likely to
        # still be contending with the OTHER concurrent /connect side when
        # busy_timeout's own wait runs out (see builder._is_locked).
        builder.delete_relationship_edges_with_retry(
            db, (RelationshipEdge.person_a_id.in_(pids))
                | (RelationshipEdge.person_b_id.in_(pids)))
        for p in junk_people:
            db.delete(p)
        removed += len(junk_people)

    # --- orgs: Claude entity filter (only when configured) -----------------
    junk_orgs: list = []
    if is_filtering_active():
        orgs = [o for o in db.execute(select(Organization)).scalars()
                if o.id not in trusted_oids]
        valid_orgs = filter_entities([o.name for o in orgs], "organization")
        junk_orgs = [o for o in orgs if o.name not in valid_orgs]
        if junk_orgs:
            oids = [o.id for o in junk_orgs]
            builder.delete_relationship_edges_with_retry(
                db, RelationshipEdge.organization_id.in_(oids))
            removed += len(junk_orgs)

    # commit_with_retry, not a bare db.commit(): db.delete() is itself a
    # pending mutation like any other -- a rollback() forced by a failed
    # commit reverts a persistent object's pending "deleted" state right back
    # to normal, same as it reverts a pending attribute change (confirmed
    # empirically for attribute changes, see tests/test_commit_retry.py; the
    # delete-flag hazard is the same mechanism). Re-issuing both delete loops
    # on retry is what actually makes a retry redo the deletion instead of
    # silently committing a no-op.
    def _apply_deletes() -> None:
        for p in junk_people:
            db.delete(p)
        for o in junk_orgs:
            db.delete(o)

    builder.commit_with_retry(db, _apply_deletes)
    if progress and removed:
        progress(f"  ✓ pruned {removed} junk nodes from the final graph")
    return removed


def _stats(db: Session, per_depth: List[int]) -> dict:
    return {
        "people_found": db.scalar(select(func.count()).select_from(Person)) or 0,
        "organizations_found": db.scalar(select(func.count()).select_from(Organization)) or 0,
        "edges_found": db.scalar(select(func.count()).select_from(RelationshipEdge)) or 0,
        "sources_fetched": db.scalar(select(func.count()).select_from(Source)) or 0,
        "nodes_processed_per_depth": per_depth,
    }
