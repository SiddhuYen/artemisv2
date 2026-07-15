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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Set

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import config
from ..extraction import extract, tier
from ..extraction.entity_filter import is_filtering_active
from ..extraction.entity_filter import validate as filter_entities
from ..extraction.schemas import ExtractedEdge
from ..models import Organization, Person, RelationshipEdge, Source
from ..providers import SearchOrchestrator, SearchResult
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

# phase-0 sources whose names are clean structured labels (no Ollama filtering)
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
    trusted: bool = False         # came from a structured source (skip Ollama filter)

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
                    progress=None, is_person: bool = True, context: str = "") -> None:
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
    enrichment = ORCH.enrich_person(subject_name) if effective_is_person else None
    if enrichment:
        # anchor the subject's identity to its Wikidata QID (homonym disambiguation):
        # two different notable same-name people have distinct QIDs and stay separate.
        if enrichment.get("qid"):
            resolved = builder.get_or_create_person(db, subject_name, qid=enrichment["qid"])
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
            if not text:
                continue
            url = f"{wiki_url}#{label}"  # distinct source per label (preserve provenance)
            res = SearchResult(enrichment["title"], url, text[:200], label)
            source = builder.save_source(db, res, f"enrich:{label}", text)
            source_by_url[res.url] = source
            out = extract(subject_name, text, silo, text[:200], res.url)
            # clean structured facts -> trusted (skip Ollama entity filter); the
            # full article/summary are prose and still need filtering.
            _mark_trusted(out.edges, label in _CLEAN_STRUCTURED)
            candidate_edges.extend(out.edges)

    # --- phase 0b: shared-affiliation colleague sources (lower confidence) ---
    if effective_is_person:
        for src_name, url, query, text in (
            ("openalex", "https://openalex.org/", "enrich:openalex",
             ORCH.coauthors_enrichment(subject_name)["coauthors_text"]),
            ("opencorporates", "https://opencorporates.com/", "enrich:opencorporates",
             ORCH.officer_enrichment(subject_name)["officers_text"]),
            ("edgar", "https://www.sec.gov/cgi-bin/browse-edgar", "enrich:edgar",
             ORCH.edgar_enrichment(subject_name)["edgar_text"]),
        ):
            if not text:
                continue
            res = SearchResult(subject_name, url, src_name, src_name)
            source = builder.save_source(db, res, query, text)
            source_by_url[res.url] = source
            out = extract(subject_name, text, COLLEAGUE_SILO, src_name, res.url)
            _mark_trusted(out.edges, True)  # clean structured names (skip entity filter)
            candidate_edges.extend(out.edges)

    # --- phase 1: build (silo, query) pairs, then DEDUP across silos -------
    pairs = []
    for silo in SILOS:
        for query in silo.render_queries(subject_name)[: config.MAX_QUERIES_PER_SILO]:
            pairs.append((silo, f"{query} {context}".strip() if context else query))
    unique_queries, query_to_silos = ORCH.dedup(pairs)

    if progress:
        ctx = f" [context: {context}]" if context else ""
        progress(f"  [hop {hop}] {subject_name}{ctx}  ·  {len(unique_queries)} unique queries "
                 f"(deduped from {len(pairs)})…")

    # --- phase 2: routed search, concurrent (cache-first) ------------------
    def _do_search(query):
        try:
            return query, ORCH.search(query, is_person=effective_is_person)
        except Exception:
            return query, []

    with ThreadPoolExecutor(max_workers=config.SEARCH_WORKERS) as ex:
        searched = list(ex.map(_do_search, unique_queries))

    # --- phase 3: fetch result pages concurrently (cache-first, deduped) ---
    to_scrape: Set[str] = set()
    for _query, results in searched:
        for rank, res in enumerate(results):
            if res.provider != "wikipedia" and rank < config.SCRAPE_TOP_N:
                to_scrape.add(res.url)

    page_text: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=config.SEARCH_WORKERS) as ex:
        for url, page in ex.map(lambda u: (u, ORCH.fetch(u)), to_scrape):
            page_text[url] = html_to_text(page.content) if page.content else ""

    # --- phase 4: extraction per (result × originating silo) --------------
    for query, results in searched:
        silos = query_to_silos.get(query, set())
        for rank, res in enumerate(results):
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
                out = extract(subject_name, text, silo, res.snippet, res.url)
                candidate_edges.extend(out.edges)

    # --- phase 5: dedup + per-node cap, then persist ----------------------
    final_edges = _dedup_and_cap(candidate_edges)
    if progress and len(candidate_edges) > len(final_edges):
        progress(f"  [hop {hop}] {subject_name}  ·  capped "
                 f"{len(candidate_edges)} → {len(final_edges)} edges (anti-explosion)")

    for edge in final_edges:
        if edge.other_kind == "person":
            counterpart = builder.get_or_create_person(
                db, edge.person_b, allow_create=not builder.at_node_cap(db)
            )
            if counterpart is None:
                continue
            builder.add_edge_from_extraction(
                db, subject, edge, hop, source_by_url.get(edge.source_url), counterpart
            )
            _record(disc, edge)
        else:
            counterpart = builder.get_or_create_org(
                db, edge.organization, edge.org_type, allow_create=not builder.at_node_cap(db)
            )
            if counterpart is None:
                continue
            builder.add_edge_from_extraction(
                db, subject, edge, hop, source_by_url.get(edge.source_url), counterpart
            )

    # mark expanded: a later/deeper run will REUSE these neighbors instead of
    # re-searching this node (see _reuse_existing_neighbors).
    subject.processed = 1
    db.commit()


def _ranked_expandable(disc: Dict[str, _Candidate], visited: Set[str],
                       progress=None) -> List[str]:
    """Choose the next hop's frontier.

    Two modes:
      - strongest (legacy): expand the highest-scoring, best-documented people.
      - reachable (default): expand the LEAST-famous real connections — people
        with no Wikipedia page and few sources — to walk DOWN the fame gradient
        toward a normal person's network (warm-intro pathfinding).

    Ollama filtering (when active) removes junk nodes from the frontier first.
    """
    if config.EXPAND_PREFER_REACHABLE:
        # real people with at least a candidate-tier edge (not just explicit/strong),
        # since the bridge people toward a normal network are weakly-linked by design.
        eligible = [c for norm, c in disc.items()
                    if norm not in visited and c.max_conf >= config.WEAK_MAX]
    else:
        eligible = [c for norm, c in disc.items()
                    if norm not in visited and c.is_expandable()]
    if not eligible:
        return []

    # pre-rank to bound the expensive checks (Ollama + Wikipedia notability).
    # family-only nodes go LAST (hard) when down-weighting, so a few high-source
    # relatives can't crowd out genuine professional connections.
    fam = config.DOWNWEIGHT_FAMILY
    if config.EXPAND_PREFER_REACHABLE:
        eligible.sort(key=lambda c: (c.demote_family(fam), len(c.sources), -c.avg_conf()))
    else:
        eligible.sort(key=lambda c: (c.demote_family(fam), -c.score()))
    shortlist = eligible[: max(config.EXPAND_TOP_STRONG * 3, 30)]

    if is_filtering_active():
        # trusted (structured-source) candidates skip the Ollama check entirely
        to_check = [c for c in shortlist if not c.trusted]
        valid = filter_entities([c.name for c in to_check], "person")
        dropped = [c.name for c in to_check if c.name not in valid]
        shortlist = [c for c in shortlist if c.trusted or c.name in valid]
        if progress and dropped:
            progress(f"  ⊘ Ollama filter skipped {len(dropped)} non-person frontier "
                     f"nodes (e.g. {', '.join(dropped[:3])})")

    if config.EXPAND_PREFER_REACHABLE and shortlist:
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
                 seed_is_person: bool = True, seed_context: str = "") -> dict:
    visited: Set[str] = set()
    frontier: List[str] = [target_name]
    per_depth: List[int] = []  # nodes processed per hop

    for hop in range(0, max_depth):
        disc: Dict[str, _Candidate] = {}
        processed = 0
        for name in frontier:
            norm = person_norm_key(name)
            if norm in visited:
                continue
            visited.add(norm)
            # If this node was already expanded (this run, a prior run, or by
            # another teammate in the shared map), REUSE its persisted neighbors
            # to rank the next frontier instead of re-searching — so we keep the
            # shallow work and just continue deeper (incremental deepening).
            existing = builder.get_or_create_person(db, name, allow_create=False)
            if existing is not None and existing.processed:
                _reuse_existing_neighbors(db, existing, disc, progress)
            else:
                # only the seed at hop 0 may be an org; discovered nodes are people.
                # the disambiguation context applies only to the seed (hop 0).
                _process_person(db, name, hop, disc, progress=progress,
                                is_person=(seed_is_person or hop > 0),
                                context=(seed_context if hop == 0 else ""))
            processed += 1
        per_depth.append(processed)

        if hop == max_depth - 1:
            break
        if builder.at_node_cap(db):
            if progress:
                progress(f"  → node cap ({config.MAX_TOTAL_NODES}) reached; stopping expansion")
            break

        frontier = _ranked_expandable(disc, visited, progress=progress)
        if progress and frontier:
            progress(f"  → expanding top {len(frontier)} strong nodes to hop {hop + 1}: "
                     + ", ".join(frontier[:5]) + (" …" if len(frontier) > 5 else ""))
        if not frontier:
            if progress:
                progress("  → no strong nodes to expand; stopping")
            break

    _prune_invalid_nodes(db, person_norm_key(target_name), progress=progress)
    _retype_unknown_edges(db, progress=progress)
    return _stats(db, per_depth)


def _retype_unknown_edges(db: Session, progress=None) -> int:
    """Re-type 'unknown' edges via the Ollama relationship classifier, using each
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

    verdicts = relation_classifier.classify(items)
    updated = 0
    for e, v in zip(eligible, verdicts):
        rtype, conf = v.get("type", "unknown"), v.get("confidence", 0.0)
        if rtype != "unknown" and conf >= config.OLLAMA_CLASSIFY_MIN_CONF:
            new_conf = round(min(conf, config.RELATION_CONF_CEILING), 3)
            e.relationship_type = rtype
            e.confidence_raw = max(e.confidence_raw or 0.0, new_conf)
            e.status = builder.derive_status(rtype, e.confidence_raw)
            sig = dict(e.signals or {})
            sig["relationship_classified_by"] = "ollama"
            e.signals = sig
            updated += 1
    db.commit()
    if progress and (updated or skipped_mismatch):
        progress(f"  ✎ Ollama typed {updated} edges "
                 f"(skipped {skipped_mismatch} with mismatched evidence)")
    return updated


def _prune_invalid_nodes(db: Session, seed_norm: str, progress=None) -> int:
    """Final pass: remove nodes that aren't real named people/orgs (with edges).

    PEOPLE are pruned by the DETERMINISTIC name-shape filter, NOT Ollama. The LLM
    entity filter proved unreliable on names: it false-DELETED real connections
    (named co-founders) while false-KEEPING page-title junk like "Drew Glover -
    LinkedIn" — which carries strong explicit edges indistinguishable from a real
    node's. In a relationship graph a false-delete loses the answer while a
    false-keep is cheap noise, so a well-formed personal name is authoritative and
    the LLM never gets to delete a plausible person. This also means people get
    cleaned even where Ollama is absent (e.g. the hosted instance).

    ORGS still use the Ollama filter when active — org names are far messier and a
    wrong drop is much less costly than losing a person. Nodes reached via a
    TRUSTED structured source are clean by construction and never pruned."""
    trusted_pids, trusted_oids = set(), set()
    for e in db.execute(select(RelationshipEdge)).scalars():
        if (e.signals or {}).get("trusted"):
            if e.person_b_id:
                trusted_pids.add(e.person_b_id)
            if e.organization_id:
                trusted_oids.add(e.organization_id)

    removed = 0
    # --- people: deterministic shape filter (LLM-independent, safe) ---------
    for p in db.execute(select(Person)).scalars():
        if p.norm_name == seed_norm or p.id in trusted_pids:
            continue
        if is_noise_name(p.canonical_name) or not looks_like_person_name(p.canonical_name):
            db.query(RelationshipEdge).filter(
                (RelationshipEdge.person_a_id == p.id)
                | (RelationshipEdge.person_b_id == p.id)
            ).delete(synchronize_session=False)
            db.delete(p)
            removed += 1

    # --- orgs: Ollama entity filter (only when reachable) ------------------
    if is_filtering_active():
        orgs = [o for o in db.execute(select(Organization)).scalars()
                if o.id not in trusted_oids]
        valid_orgs = filter_entities([o.name for o in orgs], "organization")
        for o in orgs:
            if o.name not in valid_orgs:
                db.query(RelationshipEdge).filter(
                    RelationshipEdge.organization_id == o.id
                ).delete(synchronize_session=False)
                db.delete(o)
                removed += 1

    db.commit()
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
