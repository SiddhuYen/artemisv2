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
from typing import Callable, Dict, List, Optional, Set, Tuple

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.orm.exc import ObjectDeletedError

from .. import config
from . import disambiguate
from ..extraction import extract, tier
from ..extraction import coauthor_plausibility, node_profiler, search_strategy
from ..extraction.entity_filter import is_filtering_active
from ..extraction.entity_filter import validate as filter_entities
from ..extraction.schemas import EdgeSignals, ExtractedEdge
from ..models import Organization, Person, RelationshipEdge, Source
from ..providers import SearchOrchestrator, SearchResult
from ..network.silo_weights import merge_coverage, uncovered_budget
from ..network.silo_weights import query_budget as silo_query_budget
from ..silos import COLLEAGUE_SILO, PROFESSIONAL_SILOS, SILO_BY_KEY, SILOS, STRUCTURED_SILO
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

# Person.meta key holding {"context": str, "silos": {silo_key: n_queries}} --
# the record of what a node's expansion actually asked, written next to
# `processed` and read by the reuse gate. In metadata rather than its own
# column so it needs no migration on a live graph; absent on every node
# expanded before this existed, which _residual_weights reads as "covered",
# leaving warm graphs behaving exactly as they did.
_COVERAGE_KEY = "silo_coverage"


def _residual_weights(existing: Person, context: str,
                      wanted: Optional[Dict[str, float]],
                      professional_only: bool) -> Optional[Dict[str, float]]:
    """Which silos this call wants that `existing` has never been asked.

    Returns None when the node's recorded coverage already answers everything
    wanted (reuse its neighbors and search nothing), otherwise a weights dict
    naming only the uncovered silos -- ready to hand straight back to
    _process_person as `silo_weights`.

    Two deliberate conservative readings, both of which preserve today's
    behavior on an existing graph:

      - A node with NO coverage record (expanded before this existed, or by an
        older deploy) counts as fully covered. The alternative -- treating the
        whole warm shared graph as unexpanded -- would re-search every node any
        teammate ever touched, on the first walk after deploy.
      - Coverage is only consulted within one disambiguation context; a
        different context means a different search, so nothing is covered.
    """
    if not config.EXPAND_COVERAGE_REUSE:
        return None
    record = (existing.meta or {}).get(_COVERAGE_KEY)
    if not isinstance(record, dict):
        return None
    if (record.get("context") or "") != (context or ""):
        wanted_budget = silo_query_budget(wanted)
        covered: Dict[str, int] = {}
    else:
        wanted_budget = silo_query_budget(wanted)
        covered = record.get("silos") or {}
    # professional_only drops these silos from the call entirely, so wanting
    # them is not a reason to re-search -- they would not be issued anyway.
    if professional_only:
        allowed = {s.key for s in PROFESSIONAL_SILOS}
        wanted_budget = {k: v for k, v in wanted_budget.items() if k in allowed}
    missing = uncovered_budget(wanted_budget, covered)
    if not missing:
        return None
    # query_budget maps weight -> count as round(weight * MAX_QUERIES_PER_SILO)
    # and floors at ENRICH_SILO_MIN_WEIGHT; inverting keeps the residual call
    # asking for the same per-silo counts the caller originally wanted.
    full = max(1, config.MAX_QUERIES_PER_SILO)
    return {key: max(count / full, config.ENRICH_SILO_MIN_WEIGHT)
            for key, count in missing.items()}


def _mark_trusted(edges, trusted: bool) -> None:
    if trusted:
        for e in edges:
            e.signals.trusted = True


# OpenAlex-sourced edges all share this SearchResult.url (see phase 4b and
# _resolve_expansion_depths' coauthors_enrichment call) -- a cheap, already-
# existing way to tell "this candidate came from a bare coauthor-name list"
# apart from every other extraction source, with no new field needed.
_OPENALEX_SOURCE_URL = "https://openalex.org/"


def _counterpart_identity_text(edge: ExtractedEdge) -> Optional[str]:
    """A signal describing WHO this specific edge's counterpart is, for
    builder.get_or_create_person's homonym guard on a plain-name merge (see
    that function's docstring). Without this, counterpart resolution merges
    onto ANY existing same-named node with zero identity check -- confirmed
    live: an OpenAlex coauthor named "Donald Trump" (a real academic,
    discovered as one of Jaya Sharma's real coauthors) merged straight onto
    the sitting-president "Donald Trump" node already in the graph, silently
    bridging two unrelated real people through one shared name.

    OpenAlex-sourced edges get an explicit "academic author" tag -- the same
    wording-gap fix already applied to _resolve_author's SUBJECT-side
    identity_text this session: coauthors_text's raw sentence ("X coauthor
    of Y.") has no profession keyword in it at all, so domains_of() would
    never anchor it in "science" without this. Every other edge falls back
    to its own evidence sentence, which may or may not carry enough signal
    to matter -- the guard stays silent (no false separation) when it doesn't.
    """
    text = edge.evidence_snippet or ""
    if edge.source_url == _OPENALEX_SOURCE_URL:
        text = f"{text} (an academic author, from a research coauthorship)".strip()
    return text or None


def _identity_signal(context: str, candidate_edges: List[ExtractedEdge]) -> str:
    """User-given `context` plus a SMALL, confidence-ranked sample of this
    node's own already-discovered evidence -- the signal an enrichment
    match (see phase 4b in _process_person) gets checked against before
    it's trusted.

    Deliberately NOT every candidate edge found so far: confirmed live that
    concatenating everything breaks this. A well-searched node can rack up
    1000+ raw candidate edges before this runs, and a text blob that large
    touches nearly every professional-domain bucket by sheer volume (a
    stray "engineer"/"researcher" surfacing ANYWHERE in a thousand
    snippets) -- domains_of(signal) becomes a near-superset that overlaps
    with almost any candidate identity, so domain_conflict silently never
    fires. Not because the identity check is wrong -- because "signal"
    stopped meaning anything once it was everything. Same fix
    builder._existing_evidence_signal already applies to the identical
    problem, for the identical reason: a small, confidence-ranked sample,
    not the whole pile.
    """
    top_evidence = [
        e.evidence_snippet for e in
        sorted(candidate_edges, key=lambda e: e.confidence_adjusted, reverse=True)
        if e.evidence_snippet
    ][:config.IDENTITY_SIGNAL_MAX_SNIPPETS]
    return " ".join(filter(None, [context, " ".join(top_evidence)]))


def _repeat_candidates(candidate_edges: List[ExtractedEdge]) -> List[str]:
    """Person-kind names that keep coming up across independently-found
    candidate edges, but haven't earned 'strong' confidence yet -- the
    targeted-recheck phase's beam (see its own comment in _process_person).

    Ranked by how often the name repeats, since that's the actual "keeps
    coming up" signal being acted on -- not by whatever confidence the
    generic search happened to land on, which is exactly the unreliable
    signal this phase exists to work around.
    """
    counts: Dict[str, int] = {}
    best_conf: Dict[str, float] = {}
    display: Dict[str, str] = {}
    for e in candidate_edges:
        if e.other_kind != "person" or not e.person_b:
            continue
        norm = person_norm_key(e.person_b)
        if not norm:
            continue
        counts[norm] = counts.get(norm, 0) + 1
        best_conf[norm] = max(best_conf.get(norm, 0.0), e.confidence_adjusted)
        display.setdefault(norm, e.person_b)

    eligible = [
        norm for norm, count in counts.items()
        if count >= config.ENHANCED_SEARCH_MIN_MENTIONS
        and best_conf[norm] < config.STRONG_MIN
    ]
    eligible.sort(key=lambda norm: -counts[norm])
    return [display[norm] for norm in eligible[: config.ENHANCED_SEARCH_MAX_CANDIDATES]]


_AFFILIATION_TYPES = {"employee", "cofounder", "board_member", "faculty"}

# Structural leadership types, for _Candidate.score()'s seniority bonus
# (Alpha step 7) -- narrower than _AFFILIATION_TYPES: "employee"/"faculty"
# say nothing about seniority on their own, cofounder/board_member do.
_SENIORITY_TYPES = {"cofounder", "board_member"}


def _best_org_affiliation_edge(candidate_edges: List[ExtractedEdge]) -> Optional[ExtractedEdge]:
    best: Optional[ExtractedEdge] = None
    for e in candidate_edges:
        if e.other_kind != "organization" or e.relationship_type not in _AFFILIATION_TYPES:
            continue
        if best is None or e.confidence_adjusted > best.confidence_adjusted:
            best = e
    return best


def _best_known_org(candidate_edges: List[ExtractedEdge]) -> Optional[str]:
    """The subject's own highest-confidence org affiliation found so far, if
    any -- used by the targeted-recheck phase to search "{candidate} {org}"
    in addition to "{subject} {candidate}". Confirmed live this second query
    matters: a colleague's own bio/leadership page ("Molly Chakraborty,
    Cofounder and President, Trinamix") doesn't necessarily co-mention the
    subject by name at all, so a dual-name-only search never re-finds it --
    while a name+company search reliably does.
    """
    best = _best_org_affiliation_edge(candidate_edges)
    return best.organization if best else None


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
    seniority_edges: int = 0      # cofounder/board_member, or business-domain language
    trusted: bool = False         # came from a structured source (skip Claude filter)

    def avg_conf(self) -> float:
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0

    def absorb(self, other: "_Candidate") -> None:
        """Fold another accumulation of the SAME person into this one.

        The single owner of _Candidate field merging. Tallies are built by
        two independent paths (_record, from fresh extraction, and
        _reuse_existing_neighbors, from persisted edges) and hop workers each
        accumulate into their OWN dict, so one person can be discovered twice
        and has to recombine without either side's evidence being dropped.

        That already went wrong once: `seniority_edges` was added to the
        dataclass and to _record but not to the merge, so Alpha's ranking
        signal silently vanished for exactly the cross-worker duplicates it
        exists to find -- a candidate two different coworkers both name is
        the strongest signal available, and it was the one being discarded.
        `name` is identity, not evidence, so it is deliberately not merged.
        test_candidate_absorb_covers_every_accumulated_field pins every other
        field to this method, so the next one added fails loudly, not quietly.
        """
        self.sources |= other.sources
        self.confidences.extend(other.confidences)
        self.max_conf = max(self.max_conf, other.max_conf)
        self.strong_edges += other.strong_edges
        self.explicit_edges += other.explicit_edges
        self.professional_edges += other.professional_edges
        self.family_edges += other.family_edges
        self.seniority_edges += other.seniority_edges
        self.trusted = self.trusted or other.trusted

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
        # Alpha step 7 ("pick the strongest, most high up and well connected
        # people"): a bonus for candidates whose own edges carry leadership
        # signal (cofounder/board_member typing, or business-domain language
        # like "Vice President"/"Chief Executive" in the evidence sentence --
        # see disambiguate.py's "business" bucket). This is a GENERAL
        # seniority signal, not "well-connected specifically toward THIS
        # target" -- reasoning about a specific target's world is search_
        # strategy's job (phase 4e); this only ranks who's worth spending the
        # next hop's search budget on among candidates already found.
        base += self.seniority_edges * config.SENIORITY_BONUS
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
    if (edge.relationship_type in _SENIORITY_TYPES
            or "business" in disambiguate.domains_of(edge.evidence_snippet)):
        cand.seniority_edges += 1
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
    would silently drop one side's evidence -- see _Candidate.absorb, which
    owns that accumulation so it can't drift out of sync with the dataclass
    again."""
    for norm, c in other.items():
        existing = into.get(norm)
        if existing is None:
            into[norm] = c
            continue
        existing.absorb(c)


def _reuse_existing_neighbors(db: Session, subject: Person,
                              disc: Dict[str, _Candidate], progress=None) -> None:
    """Populate `disc` from a node's ALREADY-persisted person edges (from any
    prior run, including other teammates') so the next frontier can be ranked and
    expanded WITHOUT re-running the node's searches. Mirrors _record's tallies.

    Edges aren't direction-normalized (person_a/person_b just reflect which
    side was extracted first), so a subject stored as person_b on every one
    of its edges must still be matched -- checking person_a_id alone silently
    treated such a node as having zero neighbors."""
    rows = list(db.execute(
        select(RelationshipEdge).where(
            RelationshipEdge.person_b_id.isnot(None),
            (RelationshipEdge.person_a_id == subject.id)
            | (RelationshipEdge.person_b_id == subject.id),
        )
    ).scalars())
    if not rows:
        return
    other_id = lambda e: e.person_b_id if e.person_a_id == subject.id else e.person_a_id
    b_ids = {other_id(e) for e in rows}
    people = {p.id: p for p in db.execute(
        select(Person).where(Person.id.in_(b_ids))).scalars()}
    src_ids = {e.source_id for e in rows if e.source_id}
    src_url = {s.id: s.url for s in db.execute(
        select(Source).where(Source.id.in_(src_ids))).scalars()} if src_ids else {}

    for e in rows:
        b = people.get(other_id(e))
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
        # Same seniority tally _record computes from a freshly extracted edge
        # (Alpha step 7). Omitting it here meant the bonus was ALWAYS zero on
        # an already-processed node -- i.e. on most nodes in a warm graph,
        # since "the graph is the cache" routes every re-run through here --
        # so the ranking signal was effectively off in exactly the shared-DB
        # case it was built for.
        if (e.relationship_type in _SENIORITY_TYPES
                or "business" in disambiguate.domains_of(e.evidence_snippet or "")):
            cand.seniority_edges += 1
        if sig.get("trusted"):
            cand.trusted = True
    if progress:
        progress(f"  ♻ reuse {subject.canonical_name}: {len(disc)} known neighbors "
                 f"(skipped re-searching)")


def _record_directory_membership(db: Session, members: List[str], org_row: Organization,
                                 source: Optional[Source], hop: int) -> int:
    """Persist `member works at org` edges — the weak, honest fallback when a
    directory does not list the subject (or lists too many people to treat as
    one another's colleagues).

    Unlike every other edge this module writes, person_a here is NOT the
    subject: each edge belongs to the member it describes. That is the point
    -- a directory the subject is absent from is evidence about the people on
    it, not about the subject's relationships, and writing these as subject->
    edges would smuggle back exactly the colleague claim phase 4f declined to
    make.

    Returns how many members were recorded.
    """
    if not members or org_row is None:
        return 0
    at_cap = builder.at_node_cap(db)
    written = 0
    for name in members:
        person = builder.get_or_create_person(db, name, allow_create=not at_cap)
        if person is None:
            continue
        edge = ExtractedEdge(
            person_a=name, person_b="", other_kind="organization",
            organization=org_row.name, relationship_type="employee",
            method="organization directory page",
            source_url=(source.url if source else ""),
            evidence_snippet=f"{name} is listed as an employee of {org_row.name}.",
            confidence_base=0.5, confidence_adjusted=0.5,
            signals=EdgeSignals(trusted=True, explicit_keyword_match=True),
        )
        builder.add_edge_from_extraction(db, person, edge, hop, source, org_row)
        written += 1
    return written


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
                    silo_weights: Optional[Dict[str, float]] = None,
                    enhanced_professional_search: bool = False,
                    professional_only: bool = False,
                    target_person_name: str = "",
                    target_context: str = "") -> None:
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
    # `professional_only` drops the personal-tie silos (family/friends) --
    # used for the shallow/famous side of an asymmetric /connect walk, where
    # _resolve_expansion_depths already concluded the OTHER side's best path
    # runs through a professional bridge: spending part of a 1-hop budget on
    # a public figure's spouse or close friends is a wasted hop toward that
    # specific goal, not just noise to filter out afterward.
    check_cancel()
    # Per-silo query allowance. Without weights every silo gets the full
    # MAX_QUERIES_PER_SILO, i.e. unchanged behavior; with them, silos that
    # cannot pay off for this particular subject are dropped and the rest are
    # scaled — see network/silo_weights.query_budget.
    budget = silo_query_budget(silo_weights)
    silo_set = PROFESSIONAL_SILOS if professional_only else SILOS
    pairs = []
    # What this call genuinely asks, silo -> query count. Distinct from
    # `budget`: professional_only drops the family/friends silos AFTER the
    # allowance is computed, so recording `budget` as coverage would claim a
    # node had been asked questions that were never issued -- and a later walk
    # that DOES want those silos would then skip them as already covered.
    executed: Dict[str, int] = {}
    for silo in silo_set:
        allowance = budget.get(silo.key, 0)
        if allowance <= 0:
            continue
        rendered = silo.render_queries(subject_name)[:allowance]
        if not rendered:
            continue
        executed[silo.key] = len(rendered)
        for query in rendered:
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

    # --- phase 4b: OpenAlex coauthors, plausibility- and identity-gated ----
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
    #
    # coauthor_plausibility.check() runs FIRST, before the OpenAlex call
    # even happens -- a cheaper, prior question using the SAME signal:
    # given what's already known about this subject, would they plausibly
    # have academic publications at all? Closes the homonym-collision risk
    # a layer earlier than the domain_conflict check below, which only ever
    # fires AFTER OpenAlex has already resolved a name and returned a
    # coauthor list to check.
    check_cancel()
    if is_person:
        signal = _identity_signal(context, candidate_edges)
        plausibility = coauthor_plausibility.check(subject_name, context, signal)
        if plausibility is not None and not plausibility["plausible"]:
            meta = dict(subject.meta or {})
            meta["openalex_skipped"] = {"why": plausibility["why"]}
            subject.meta = meta
            if progress:
                progress(f"  ⊘ skipping OpenAlex coauthors for {subject_name} — "
                         f"{plausibility['why']}")
        else:
            oa = ORCH.coauthors_enrichment(subject_name)
            oa_text = oa["coauthors_text"]
            if oa_text:
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

    # --- phase 4c: targeted re-query for names that keep coming up ---------
    # Only on the non-famous side of an asymmetric /connect walk (see
    # connect._expand_both_concurrently, which is the only caller that ever
    # passes True here) -- this is the fix for a specific, observed failure:
    # a real cofounder/close colleague, mentioned only in passing across
    # several LinkedIn posts with no sentence ever stating the relationship,
    # stays capped at "weak coworker" confidence forever under the generic
    # silo search alone (see extraction.confidence's evidence-ceiling rules
    # -- co-occurrence with no explicit keyword cannot exceed 0.39, no matter
    # how many times the same weak mention repeats). The fix isn't a
    # confidence-model change -- weak evidence should stay weak -- it's
    # asking a SHARPER question for names that already earned it: search the
    # subject and that specific candidate together, directly, the same way
    # connect._direct_pair_search checks the two /connect endpoints.
    #
    # Bounded to ENHANCED_SEARCH_MAX_CANDIDATES names (a beam, not full
    # recursion -- see config's comment on why this keeps cost predictable
    # across hops) and only for names that (a) repeated at least
    # ENHANCED_SEARCH_MIN_MENTIONS times, since a single passing mention
    # isn't the "keeps coming up" signal this acts on, and (b) haven't
    # already reached 'strong' -- nothing to gain re-querying a pair that's
    # already well-evidenced.
    # Claude reclassification of what phase 4c finds, when configured: the
    # deterministic spaCy/keyword confidence model (extraction.confidence)
    # starts every match at a modest base and multiplies up from there, so
    # even a clean, targeted, co-occurring hit ("X, Cofounder and President
    # of Y, has worked alongside the subject for a decade") can land short
    # of 'strong' on the arithmetic alone. Reusing the SAME batched-verdict
    # mechanism _retype_unknown_edges already applies to weak/unknown edges
    # elsewhere gives a targeted hit the decisive read it was worth going
    # and looking for in the first place, instead of leaving it exactly as
    # capped as the generic mention it was meant to replace.
    from ..extraction import relation_classifier

    check_cancel()
    if enhanced_professional_search and is_person:
        org_name = _best_known_org(candidate_edges)
        for candidate_name in _repeat_candidates(candidate_edges):
            check_cancel()
            # Two DIFFERENT questions, both worth asking: "is there a stated
            # relationship between them" (dual-name) and "what IS this
            # candidate, at the subject's own company" (candidate+org).
            # Confirmed live these find different things -- a colleague's own
            # bio/leadership page states their role without ever naming the
            # subject, so dual-name-only misses exactly the fact that
            # matters ("Molly Chakraborty, Cofounder and President,
            # Trinamix" never mentions "Prantik Chakraborty" at all).
            queries = [f'"{subject_name}" "{candidate_name}"']
            if org_name:
                queries.append(f'"{candidate_name}" "{org_name}"')

            candidate_norm = person_norm_key(candidate_name)
            found_edges: List[ExtractedEdge] = []
            seen_urls: Set[str] = set()
            for query in queries:
                check_cancel()
                try:
                    results = ORCH.search(query, is_person=True)
                except Exception:
                    continue
                for res in results[: config.SCRAPE_TOP_N]:
                    check_cancel()
                    if res.url in seen_urls:
                        continue
                    seen_urls.add(res.url)
                    page = ORCH.fetch(res.url)
                    text = html_to_text(page.content) if page.content else ""
                    text = text or f"{res.title}. {res.snippet}"
                    source = builder.save_source(db, res, query, text)
                    source_by_url[res.url] = source
                    out = extract(subject_name, text, SILO_BY_KEY["company"], res.snippet, res.url)
                    # Keep only edges actually about the candidate this query
                    # targeted -- a scraped page can mention plenty of other
                    # names, and those belong to whichever query's own pass
                    # would have found them, not this one's targeted intent.
                    found_edges.extend(
                        e for e in out.edges
                        if e.other_kind == "person" and person_norm_key(e.person_b) == candidate_norm
                    )

            if found_edges and relation_classifier.is_active():
                check_cancel()
                items = [{"a": subject_name, "b": candidate_name, "evidence": e.evidence_snippet}
                        for e in found_edges]
                verdicts = relation_classifier.classify(items)
                for e, v in zip(found_edges, verdicts):
                    rtype, conf = v.get("type", "unknown"), v.get("confidence", 0.0)
                    if rtype != "unknown" and conf >= config.CLAUDE_CLASSIFY_MIN_CONF:
                        e.relationship_type = rtype
                        e.confidence_adjusted = max(
                            e.confidence_adjusted, round(min(conf, config.RELATION_CONF_CEILING), 3))
                        e.signals.explicit_keyword_match = True

            candidate_edges.extend(found_edges)

    # --- phase 4d: node profiling (Alpha step 4/5 -- "understand current
    # node"): how big is the subject's own org, what industry is it in. Same
    # gating as phase 4c (non-famous side of an asymmetric walk only) -- the
    # famous side's own notability already came from the Wikidata check in
    # connect._resolve_expansion_depths, nothing to profile there. Cached on
    # the Organization row's own meta (no TTL, same convention as
    # openalex_rejected above): an org already profiled by one colleague
    # isn't re-profiled by the next, in this run or a future one.
    #
    # Deliberately fewer, more targeted queries than phase 1's silo search --
    # see config.NODE_PROFILE_QUERIES's comment: structured-source-first
    # (LinkedIn's employee-count badge, Crunchbase's headcount field) rather
    # than generic "about us" copy, which almost never states real numbers
    # and just invites node_profiler's model to infer instead of read.
    check_cancel()
    org_row = None
    org_name = None
    org_edge = None
    # Resolving the subject's org is NOT part of profiling -- phase 4e needs
    # the same row, and nesting this lookup inside node_profiler.is_active()
    # made ARTEMIS_NODE_PROFILE=0 silently disable the strategy stage too,
    # via `org_row is None`, rather than through 4e's own (real, semantic)
    # requirement that a GROUNDED profile exist. Two knobs that read as
    # independent should be independent; 4e still declines on its own terms
    # when no usable profile is present.
    if enhanced_professional_search and is_person:
        org_edge = _best_org_affiliation_edge(candidate_edges)
        org_name = org_edge.organization if org_edge else None
        if org_name:
            org_row = builder.get_or_create_org(db, org_name)
    if (org_row is not None and node_profiler.is_active()
            and not node_profiler.is_current((org_row.meta or {}).get("profile"))):
        snippets: List[str] = []
        seen_urls: Set[str] = set()
        for template in config.NODE_PROFILE_QUERIES:
            check_cancel()
            query = template.format(org=org_name)
            try:
                results = ORCH.search(query, is_person=False)
            except Exception:
                continue
            for res in results[:2]:
                check_cancel()
                if res.url in seen_urls:
                    continue
                seen_urls.add(res.url)
                page = ORCH.fetch(res.url)
                text = html_to_text(page.content) if page.content else ""
                text = text or f"{res.title}. {res.snippet}"
                source = builder.save_source(db, res, query, text)
                source_by_url[res.url] = source
                snippets.append(text)
        known_context = org_edge.evidence_snippet if org_edge else ""
        profile = node_profiler.profile_org(org_name, snippets, known_context or "")
        if profile is not None:
            meta = dict(org_row.meta or {})
            meta["profile"] = profile
            org_row.meta = meta
            if progress:
                progress(f"  ⓘ profiled {org_name}: size={profile['size_tier']} "
                         f"industry={profile['industry']} (grounded={profile['grounded']})")

    # --- phase 4e: search strategy (Alpha step 6 -- "run reasoning to
    # identify best type of search"). Only runs with a GROUNDED org profile
    # in hand (fresh from 4d just above, or already cached from an earlier
    # hop/run on the same org) and a known target -- deciding a strategy from
    # an ungrounded profile would just be reasoning on top of a guess, and
    # with no target there's nothing to reason TOWARD. The chosen angle maps
    # to a small, FIXED set of extra queries (config.STRATEGY_ANGLE_QUERIES)
    # -- the model picks which angle applies, it never writes query text
    # itself, so a wrong pick costs a couple of irrelevant queries, not an
    # ungrounded search direction.
    check_cancel()
    if (enhanced_professional_search and is_person and org_row is not None
            and target_person_name and search_strategy.is_active()):
        profile = (org_row.meta or {}).get("profile")
        # Grounded AND current: a profile from an older prompt/guard set is
        # no safer to reason on top of than an ungrounded one, and 4d above
        # only re-profiles when it is itself active -- with profiling turned
        # off, a stale cached profile is all there is, and it must not
        # silently become the basis for a strategy decision.
        if node_profiler.is_current(profile) and profile.get("grounded"):
            decision = search_strategy.decide_angle(
                subject_name, org_name, profile, target_person_name, target_context)
            if decision is not None:
                meta = dict(subject.meta or {})
                meta["strategy"] = decision
                subject.meta = meta
                if progress:
                    progress(f"  ➤ strategy: {decision['angle']} — {decision['why']}")
                templates = config.STRATEGY_ANGLE_QUERIES.get(decision["angle"], [])
                if templates:
                    industry = profile.get("industry", "")
                    seen_urls = {s for s in source_by_url}
                    for template in templates:
                        check_cancel()
                        query = template.format(subject=subject_name, org=org_name,
                                                industry=industry, target=target_person_name)
                        try:
                            results = ORCH.search(query, is_person=True)
                        except Exception:
                            continue
                        for res in results[:config.SCRAPE_TOP_N]:
                            check_cancel()
                            if res.url in seen_urls:
                                continue
                            seen_urls.add(res.url)
                            page = ORCH.fetch(res.url)
                            text = html_to_text(page.content) if page.content else ""
                            text = text or f"{res.title}. {res.snippet}"
                            source = builder.save_source(db, res, query, text)
                            source_by_url[res.url] = source
                            out = extract(subject_name, text, COLLEAGUE_SILO, res.snippet, res.url)
                            candidate_edges.extend(out.edges)

    # --- phase 4f: the subject's employer's own directory ------------------
    # Alpha's densest source of real professional connections: for someone
    # with no press coverage and no publications, the people listed on their
    # employer's own staff/leadership page ARE their professional network.
    #
    # Structural, not prose -- a directory page listing two people is an
    # assertion, and running it through the sentence-proximity extractor is
    # what made the current_employer_leadership strategy angle useless (see
    # providers/directory.py's docstring). So it sits here with the other
    # structural sources, and the strategy angle no longer issues prose
    # queries of its own (config.STRATEGY_ANGLE_QUERIES).
    #
    # The evidence rule, which is the whole safety story:
    #   subject IS listed  -> subject<->member coworker edges. The page
    #                         co-listing them is the assertion; nothing is
    #                         inferred.
    #   subject NOT listed -> member->org employment edges ONLY. A directory
    #     (or overflow)       the subject is absent from says those people
    #                         work there; it says NOTHING about whether they
    #                         know the subject, and asserting otherwise is
    #                         precisely the fabrication this replaces.
    #
    # NB those membership edges do not themselves create a /connect route --
    # connect._adjacency traverses person<->person edges only. They are
    # honest graph facts (and feed network.matching's org_overlap tier), not
    # a back door for the person-level claim we just declined to make.
    check_cancel()
    if (enhanced_professional_search and is_person and org_row is not None
            and org_name and config.DIRECTORY_ENABLED):
        profile = (org_row.meta or {}).get("profile") or {}
        if not node_profiler.is_current(profile):
            profile = {}
        found = ORCH.directory_enrichment(
            org_name,
            industry=profile.get("industry", ""),
            size_tier=profile.get("size_tier", ""),
        )
        members = found.get("members") or []
        url = found.get("url") or ""
        if members and url:
            subject_norm = person_norm_key(subject_name)
            listed = any(person_norm_key(m) == subject_norm for m in members)
            res = SearchResult(found.get("org") or org_name, url, "directory", "directory")
            source = builder.save_source(db, res, "enrich:directory")
            source_by_url[url] = source

            if listed and not found.get("overflow"):
                # Direct edge construction, NOT the text/silo pipeline -- the
                # same choice phase 0d makes, for a sharper version of the
                # same reason. Rendering the roster to prose and re-extracting
                # it puts spaCy NER between us and names we ALREADY have: the
                # members came from rosters.clean_roster_names, a deterministic
                # shape filter, and re-deriving personhood from a sentence is
                # strictly weaker evidence than the roster we scraped them
                # from. It is also biased -- en_core_web_sm tags "Dana
                # Whitfield" PERSON but "Molly Iyer" and "Prantik Chakraborty"
                # ORG, so round-tripping silently drops non-Anglo names from a
                # page that structurally asserted every one of them. For a
                # feature whose whole purpose is growing NON-famous
                # professional networks, that bias lands squarely on the
                # people it exists to find.
                for member in members:
                    if person_norm_key(member) == subject_norm:
                        continue
                    candidate_edges.append(ExtractedEdge(
                        person_a=subject_name, person_b=member, other_kind="person",
                        relationship_type="coworker",
                        method="organization directory page",
                        source_url=url,
                        evidence_snippet=(
                            f"{org_name}'s own directory page lists both "
                            f"{subject_name} and {member}."),
                        # Candidate tier by construction, never strong: being
                        # listed on one roster establishes a shared affiliation,
                        # not a working relationship of any particular closeness
                        # (same stance as COLLEAGUE_SILO's multiplier).
                        confidence_base=0.5, confidence_adjusted=0.5,
                        signals=EdgeSignals(trusted=True, explicit_keyword_match=True),
                    ))
                if progress:
                    progress(f"  ▤ directory {org_name}: {len(members)} listed, "
                             f"subject among them → colleague edges")
            else:
                _record_directory_membership(db, members, org_row, source, hop)
                if progress:
                    reason = "overflowed" if found.get("overflow") else "subject not listed"
                    progress(f"  ▤ directory {org_name}: {len(members)} listed, {reason} "
                             "→ employment edges only (no colleague claim)")

    # --- phase 5: dedup + per-node cap, then persist ----------------------
    check_cancel()
    final_edges = _dedup_and_cap(candidate_edges)
    if progress and len(candidate_edges) > len(final_edges):
        progress(f"  [hop {hop}] {subject_name}  ·  capped "
                 f"{len(candidate_edges)} → {len(final_edges)} edges (anti-explosion)")

    # Claude-validate counterpart names BEFORE they ever become a Person/Org
    # row -- the heuristic extractor's confidence score describes how sure it
    # is that a RELATIONSHIP exists, not whether the string it grabbed is
    # even a real entity, so "USA Key" (mis-tokenized from "...California,
    # USA / Key people: ...") sailed through at whatever confidence the
    # co-occurrence happened to score. Deterministic name-shape checks (see
    # is_noise_name/looks_like_person_name) only catch shapes someone already
    # anticipated; asking Claude "is this a real, specific person/org" is the
    # same judgment call a human skimming the evidence sentence would make,
    # and it generalizes to whatever malformed shape shows up next instead of
    # requiring a new pattern per failure mode. No-ops (keeps everyone) when
    # filtering is disabled or no key resolves -- see entity_filter.validate.
    #
    # Skips edges already marked signals.trusted -- those came from a
    # structured source (Wikidata/EDGAR/ProPublica/a scraped roster) with a
    # clean canonical name and were deliberately exempted from the Claude
    # filter by _mark_trusted at extraction time (see phase 0/0b above);
    # _ranked_expandable applies this same trusted-skips-filtering rule when
    # ranking the next frontier, so persistence now agrees with it instead of
    # spending a Claude call re-litigating a source extraction already
    # vouched for.
    check_cancel()
    valid_people: Set[str] = set()
    valid_orgs: Set[str] = set()
    if is_filtering_active():
        person_names = [e.person_b for e in final_edges
                        if e.other_kind == "person" and not e.signals.trusted]
        org_names = [e.organization for e in final_edges
                    if e.other_kind == "organization" and not e.signals.trusted]
        if person_names:
            valid_people = filter_entities(person_names, "person")
        if org_names:
            valid_orgs = filter_entities(org_names, "organization")

    # Snapshot the cap once for this node's whole edge batch instead of
    # re-querying COUNT(*) on every candidate edge (up to MAX_EDGES_PER_NODE
    # per node) -- the cap is already a soft/best-effort bound (concurrent
    # hop workers can each read a stale count too), so reusing one snapshot
    # for a single node's edges costs nothing beyond what's already true.
    at_cap = builder.at_node_cap(db)
    for edge in final_edges:
        check_cancel()
        if edge.other_kind == "person":
            if (is_filtering_active() and not edge.signals.trusted
                    and edge.person_b not in valid_people):
                if progress:
                    progress(f"  ✕ dropped {edge.person_b!r} — not a real person "
                             "(Claude entity filter)")
                continue
            counterpart = builder.get_or_create_person(
                db, edge.person_b, allow_create=not at_cap,
                identity_text=_counterpart_identity_text(edge),
            )
            if counterpart is None:
                continue
            builder.add_edge_from_extraction(
                db, subject, edge, hop, source_by_url.get(edge.source_url), counterpart
            )
            _record(disc, edge)
        else:
            if (is_filtering_active() and not edge.signals.trusted
                    and edge.organization not in valid_orgs):
                if progress:
                    progress(f"  ✕ dropped {edge.organization!r} — not a real organization "
                             "(Claude entity filter)")
                continue
            counterpart = builder.get_or_create_org(
                db, edge.organization, edge.org_type, allow_create=not at_cap
            )
            if counterpart is None:
                continue
            builder.add_edge_from_extraction(
                db, subject, edge, hop, source_by_url.get(edge.source_url), counterpart
            )

    # mark expanded, and record WHAT was asked: a later/deeper run reuses this
    # node's persisted neighbors instead of re-searching it, but only for the
    # silos this budget actually covered (see _coverage and the reuse gate in
    # _process_one_attempt). `processed` stays a boolean for every existing
    # reader; the coverage map rides along in metadata, so this needs no schema
    # change against a live graph.
    #
    # commit_with_retry, not a bare db.commit(): every write earlier in this
    # node's processing (save_source, add_edge_from_extraction) already went
    # through its own SAVEPOINT retry, so by the time we get here this
    # connection has either already secured SQLite's write lock for the rest
    # of this transaction (this commit can't newly contend) or nothing at all
    # was written for this node (this commit is the transaction's first write
    # attempt, and a lock retry here is safe because there's nothing else
    # pending to lose). Either way, re-applying these two fields on retry is
    # correct and cheap.
    def _mark_expanded() -> None:
        subject.processed = 1
        meta = dict(subject.meta or {})
        prior = meta.get(_COVERAGE_KEY) or {}
        # Coverage is only comparable within one disambiguation context: every
        # query above had `context` appended, so "Acme"-qualified questions
        # never answered the unqualified ones (nor another context's). On a
        # context switch the old counts describe a different search entirely
        # and are replaced rather than merged.
        same_context = (prior.get("context") or "") == (context or "")
        meta[_COVERAGE_KEY] = {
            "context": context or "",
            "silos": (merge_coverage(prior.get("silos"), executed)
                      if same_context else dict(executed)),
        }
        subject.meta = meta

    builder.commit_with_retry(db, _mark_expanded)


def _ranked_expandable(disc: Dict[str, _Candidate], visited: Set[str],
                       progress=None, prefer_reachable: Optional[bool] = None,
                       top_n: Optional[int] = None) -> List[str]:
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

    `top_n` overrides config.EXPAND_TOP_STRONG's final cap for THIS call only
    (Alpha step 7's "pick 5 of the strongest," narrower than the general
    15-wide beam) -- None keeps the configured default. Only narrows the
    FINAL selection; the pre-filter shortlist size (candidate pool size
    before ranking) is unaffected, so a smaller top_n still ranks over the
    same breadth of candidates, it just keeps fewer of them.

    Claude filtering (when active) removes junk nodes from the frontier first.
    """
    if prefer_reachable is None:
        prefer_reachable = config.EXPAND_PREFER_REACHABLE
    limit = top_n if top_n is not None else config.EXPAND_TOP_STRONG
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
        chosen = shortlist[:limit]
        if progress:
            famous = [c.name for c in chosen if c.name in notable]
            progress(f"  ↧ reachability: expanding {len(chosen)} least-famous nodes "
                     f"({len(chosen) - len(famous)} with no Wikipedia page)")
        return [c.name for c in chosen]

    return [c.name for c in shortlist[:limit]]


def expand_graph(db: Session, target_name: str, max_depth: int, progress=None,
                 seed_is_person: bool = True, seed_context: str = "",
                 protected_norms: Optional[Set[str]] = None,
                 on_step: Optional[Callable[[dict], None]] = None,
                 cancel_checker: Optional[Callable[[], None]] = None,
                 should_stop: Optional[Callable[[Session], bool]] = None,
                 prefer_reachable: Optional[bool] = None,
                 silo_weights: Optional[Dict[str, float]] = None,
                 enhanced_professional_search: bool = False,
                 professional_only: bool = False, target_person_name: str = "",
                 target_context: str = "",
                 on_frontier: Optional[Callable[[List[str]], None]] = None) -> dict:
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
    run concurrently without fighting over one another's strategy.

    `enhanced_professional_search`, when True, is threaded to EVERY node
    processed on this call (not just the seed) -- see _process_person's
    phase 4c. connect_people sets this for whichever side is NOT the shallow,
    famous one in an asymmetric walk (see _expand_both_concurrently): as that
    side's frontier walks outward hop by hop, each new node gets the same
    targeted-recheck treatment the seed did, which is what turns a single
    node's fix into the recursive "top candidates, searched properly, at
    every hop" behavior this was designed for.

    `on_frontier`, when provided, is handed each hop's ranked frontier BEFORE
    it is expanded. connect_people uses it to ask the question this walk cannot:
    does this specific node reach the far endpoint? Expanding a node costs ~35
    queries; asking that costs one, and for a famous endpoint -- capped at
    SHALLOW_FAMOUS_DEPTH precisely because their neighborhood is too large to
    walk -- it is the only affordable way to close the gap. A node that answers
    yes makes its own expansion unnecessary, which `should_stop` then notices.

    `professional_only`, the mirror image, goes to the OTHER side -- the
    shallow, famous one. connect_people sets it when the other side already
    concluded a professional bridge is the likeliest path (that's what
    triggered the asymmetric depth in the first place): the family/friends
    silos are dropped from every query this call renders, so a public
    figure's limited 1-hop budget goes toward colleagues and board seats,
    not a wasted hop on their spouse or close friends.

    `target_person_name`/`target_context` (Alpha step 6): who this walk is
    ultimately trying to reach, and any context on them -- NOT this call's
    own seed (`target_name` above is this expansion's own starting person,
    an unfortunately-overlapping name kept for backward compat with every
    other caller). connect._expand_both_concurrently passes the OTHER
    endpoint's name/context here, so _process_person's search-strategy phase
    can reason about who it's actually walking toward instead of picking a
    query angle with no destination in mind. Empty for every non-/connect
    caller (CLI, /expand, org_discovery) -- the strategy phase itself no-ops
    without a target name, so this is inert unless explicitly supplied."""
    visited: Set[str] = set()
    frontier: List[str] = [target_name]
    per_depth: List[int] = []  # nodes processed per hop
    visited_by_hop: Dict[int, List[str]] = {}  # hop -> node names selected for it
    # Declared here, not inside the hop loop below: should_stop can trip on
    # hop 0 before the loop body ever runs (a real race in connect_people's
    # concurrent two-sided expansion -- the other endpoint's search can find
    # the route first), and the post-loop boundary computation reads `disc`
    # unconditionally. Left unassigned by any hop, it's just the empty dict
    # the first hop would have started from anyway.
    disc: Dict[str, _Candidate] = {}
    # Alpha step 7 (per-candidate depth): a node selected for the Alpha
    # frontier that turns out to be independently notable/famous relative to
    # the target gets fully processed and persisted (its own "1 hop"), but
    # its OWN discoveries are excluded from seeding the NEXT hop -- don't
    # keep walking outward from someone already close to the target's own
    # world; that just re-explores a famous person's huge network instead of
    # continuing to hunt for a targeted bridge. Populated after each Alpha
    # frontier selection below; read (as a closure) inside _process_one.
    shallow_nodes: Set[str] = set()

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
        """Process one frontier node, retrying the WHOLE node on a transient
        DB error.

        The retry has to live here, not deeper. Every write helper in builder
        already retries the failing STATEMENT inside a SAVEPOINT, and that is
        enough for a SQLite lock -- the contending writer finishes and the
        next attempt goes through. It cannot fix a Postgres deadlock. There,
        two workers each hold locks the other needs (both are mid-transaction,
        having already inserted overlapping people/orgs in different orders),
        so retrying the same statement inside a transaction that STILL HOLDS
        the conflicting locks can never succeed. Confirmed against a real
        Postgres: every one of the six statement-level attempts deadlocked,
        then the node was dropped -- which is what lost a node from a live
        /connect walk.

        Breaking the cycle needs the whole transaction gone, so each attempt
        gets a FRESH session. Re-doing the node is cheap relative to losing
        it: the searches behind it are served from the provider cache, so a
        retry mostly re-runs extraction and the writes.
        """
        attempts = max(0, config.NODE_DB_RETRY_ATTEMPTS)
        for attempt in range(attempts + 1):
            result, retry = _process_one_attempt(name, hop, attempt, attempts)
            if not retry:
                return result
        return {}

    def _process_one_attempt(name: str, hop: int, attempt: int,
                             attempts: int) -> Tuple[Dict[str, "_Candidate"], bool]:
        """One attempt at a node. Returns (discoveries, should_retry)."""
        local_disc: Dict[str, _Candidate] = {}
        worker_db = WorkerSession()
        try:
            if stop_requested(worker_db):
                return local_disc, False
            check_cancel()
            # If this node was already expanded (this run, a prior run, or by
            # another teammate in the shared map), REUSE its persisted
            # neighbors to rank the next frontier instead of re-searching —
            # so we keep the shallow work and just continue deeper
            # (incremental deepening).
            #
            # Reuse is now conditional on COVERAGE, not on the `processed`
            # flag alone: a node expanded under one walk's silo weights was
            # only asked that walk's questions, and replaying its neighbors
            # for a walk asking different ones is how a node got frozen at
            # whatever the first run happened to want. When this call wants
            # silos the node has never been asked, do both — replay the
            # covered neighbors (they are still real, and still rank the next
            # frontier) AND search only the uncovered silos on top.
            node_context = seed_context if hop == 0 else ""
            node_weights = silo_weights if hop == 0 else None
            existing = builder.get_or_create_person(worker_db, name, allow_create=False)
            residual: Optional[Dict[str, float]] = None
            reuse = existing is not None and existing.processed
            if reuse:
                residual = _residual_weights(existing, node_context, node_weights,
                                             professional_only)
                _reuse_existing_neighbors(worker_db, existing, local_disc, progress)
                if residual is not None and progress:
                    progress(f"  ↻ {name} was expanded before, but not for "
                             f"{'/'.join(sorted(residual))} — widening")
            if not reuse or residual is not None:
                # only the seed at hop 0 may be an org; discovered nodes are people.
                # the disambiguation context applies only to the seed (hop 0).
                kwargs = {
                    "progress": progress,
                    "is_person": (seed_is_person or hop > 0),
                    "context": node_context,
                    # Weights describe THIS contact, derived from their own
                    # export row — they say nothing about the strangers found
                    # at hop 1+, so they apply to the seed only. Guessing that
                    # a founder's neighbours are also founders is exactly the
                    # unfounded prior this feature exists to remove.
                    #
                    # `residual` narrows them to just the silos this node has
                    # never been asked, so widening an already-expanded node
                    # pays for the new questions only, not the whole ~35 again.
                    "silo_weights": residual if residual is not None else node_weights,
                    "enhanced_professional_search": enhanced_professional_search,
                    "professional_only": professional_only,
                    "target_person_name": target_person_name,
                    "target_context": target_context,
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
            # A lock/deadlock is worth redoing from a clean transaction; a
            # genuine bug is not -- retrying that just burns the budget and
            # delays the same failure.
            #
            # ObjectDeletedError earns the same fresh-session retry, for the
            # same reason a deadlock does: it is a property of THIS session,
            # not of the node. The other /connect side ends its expand_graph
            # in _prune_invalid_nodes, which deletes junk nodes on its own
            # session -- a worker mid-hop here can still hold one of those
            # people in its identity map, and the next attribute access on
            # the stale instance raises. Nothing about the node is broken; a
            # fresh session re-selects live rows and cannot hit it. Confirmed
            # live on a Paul Graham -> Sam Altman walk: 'Public License' at
            # hop 1 lost its whole capped hop (~150 edges) to exactly this.
            # Not folded into is_transient_db_error: the statement-level
            # retry sites in builder share that classifier, and re-running a
            # statement inside the SAME session cannot clear a stale identity
            # map -- only this whole-node, fresh-session level can.
            retryable = (builder.is_transient_db_error(exc)
                         or isinstance(exc, ObjectDeletedError))
            if retryable and attempt < attempts:
                if progress:
                    progress(f"  ↻ {name!r} at hop {hop} hit a transient DB "
                             f"error ({exc.__class__.__name__}) — retrying "
                             f"({attempt + 1}/{attempts})")
                # let the contending worker commit and release its locks
                # before we take the same ones again
                builder._deadlock_backoff(attempt)
                return {}, True
            if progress:
                # str(exc), not just the class name: a dropped node is silent
                # data loss (it can turn a real /connect route into "NO PATH"),
                # and the class name alone is useless for telling a transient
                # DB lock apart from a genuine bug -- diagnosing one such drop
                # cost a full instrumented re-run purely because the message
                # was discarded here. Truncated: some DBAPI errors embed the
                # entire offending statement plus parameters.
                detail = " ".join(str(exc).split())[:300]
                progress(f"  ⚠ {name!r} at hop {hop} failed "
                         f"({exc.__class__.__name__}: {detail}) — skipped")
        finally:
            worker_db.close()
        if person_norm_key(name) in shallow_nodes:
            # Fully processed and persisted above -- only excluded from
            # feeding the NEXT hop's frontier selection (see shallow_nodes'
            # own comment above).
            return {}, False
        return local_disc, False

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
        # Who Artemis actually looked at, hop by hop -- kept so a caller
        # (connect_people) can show "what did Artemis explore" even when no
        # connecting path was ever found between the two sides. Captured up
        # front (the intended frontier for this hop), not narrowed to
        # whatever finished before a cancellation -- a node that was
        # SELECTED for this hop is still something the search "tried",
        # whether or not it got to finish.
        visited_by_hop[hop] = list(to_process)

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

        if progress:
            progress(f"  ✓ hop {hop + 1}/{max_depth} complete — {len(to_process)} node(s) expanded")

        if hop == max_depth - 1:
            break
        if builder.at_node_cap(db):
            if progress:
                progress(f"  → node cap ({config.MAX_TOTAL_NODES}) reached; stopping expansion")
            break

        check_cancel()
        # Alpha step 7: the non-famous/origin side of an asymmetric /connect
        # walk narrows to the top ALPHA_TOP_CANDIDATES (5), not the general
        # EXPAND_TOP_STRONG beam (15) -- a reasoning-selected angle (phase
        # 4e) already narrowed the field, so expanding as many candidates as
        # the generic case doesn't need is wasted search budget, not thoroughness.
        alpha_top_n = config.ALPHA_TOP_CANDIDATES if enhanced_professional_search else None
        frontier = _ranked_expandable(disc, visited, progress=progress,
                                      prefer_reachable=prefer_reachable,
                                      top_n=alpha_top_n)
        # Ask the far endpoint's question before paying to walk outward. Placed
        # here, after ranking and before expansion, so a node that turns out to
        # reach the target never has its own ~35 queries spent.
        if on_frontier and frontier:
            check_cancel()
            on_frontier(frontier)
            if should_stop and should_stop(db):
                if progress:
                    progress("  → a frontier node reaches the target; stopping expansion")
                break
        # Alpha step 7 (per-candidate depth): among the selected frontier,
        # any independently notable/famous candidate gets marked shallow --
        # see shallow_nodes' declaration above. Checked here (once per hop,
        # batched) rather than per-node during processing, since notability
        # is a property of the NAME alone and this is the one place the
        # whole hop's frontier is already assembled in one list.
        if enhanced_professional_search and frontier:
            check_cancel()
            try:
                famous = ORCH.notable_set(frontier)
            except Exception:
                famous = set()
            if famous:
                shallow_nodes.update(person_norm_key(n) for n in famous)
                if progress:
                    progress(f"  ⚑ {len(famous)} frontier node(s) independently notable — "
                             f"shallow (1 hop, not walked further): {', '.join(sorted(famous))}")
        if progress and frontier:
            progress(f"  → expanding top {len(frontier)} strong nodes to hop {hop + 1}: "
                     + ", ".join(frontier[:5]) + (" …" if len(frontier) > 5 else ""))
        if not frontier:
            if progress:
                progress("  → no strong nodes to expand; stopping")
            break

    # Whatever candidates the last processed hop turned up are real,
    # persisted data (see _process_one/_merge_disc) -- the loop just never
    # got to walk them as their own hop (max_depth reached, the node cap hit,
    # or a cancellation). Recomputing the ranking once more here doesn't
    # process/persist anything new; it only tells a caller what WOULD have
    # been expanded next, so the no-route visualization can show "found, not
    # walked" instead of silently implying Artemis found nothing further --
    # see connect_people's "explored" field. When the loop instead ended
    # because ranking already came back empty, this repeats that same
    # (cheap, eligible-list-empty) call and correctly yields nothing.
    alpha_top_n = config.ALPHA_TOP_CANDIDATES if enhanced_professional_search else None
    boundary = _ranked_expandable(disc, visited, prefer_reachable=prefer_reachable,
                                  top_n=alpha_top_n)

    protected = {person_norm_key(target_name)} | (protected_norms or set())
    check_cancel()
    _prune_invalid_nodes(db, protected, progress=progress)
    check_cancel()
    _retype_unknown_edges(db, progress=progress)
    stats = _stats(db, per_depth)
    stats["visited_by_hop"] = visited_by_hop
    stats["boundary"] = boundary
    return stats


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
        # Re-check existence on EVERY call, including the first: `eligible`
        # was loaded before the (slow, network-bound) Claude classify() call
        # above, so a concurrent writer -- another /connect build pruning a
        # junk node, most likely -- has had a real window to delete one of
        # these same rows by the time this runs. Skipping it here is what
        # keeps the commit's matched-row count honest; without this check
        # the flush below raises StaleDataError (SQLAlchemy detected the
        # mismatch itself), and retrying would just re-attempt the identical
        # mutation on the same now-gone row and fail the same way again.
        still_present = set(db.execute(
            select(RelationshipEdge.id).where(
                RelationshipEdge.id.in_([e.id for e in eligible]))
        ).scalars())
        updated = 0
        for e, v in zip(eligible, verdicts):
            if e.id not in still_present:
                continue
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

    # --- people: deterministic shape filter (LLM-independent, safe) ---------
    junk_people = [
        p for p in db.execute(select(Person)).scalars()
        if p.norm_name not in protected_norms and p.id not in trusted_pids
        and (is_noise_name(p.canonical_name) or not looks_like_person_name(p.canonical_name))
    ]

    # --- orgs: Claude entity filter (only when configured) -----------------
    junk_orgs: list = []
    if is_filtering_active():
        orgs = [o for o in db.execute(select(Organization)).scalars()
                if o.id not in trusted_oids]
        valid_orgs = filter_entities([o.name for o in orgs], "organization")
        junk_orgs = [o for o in orgs if o.name not in valid_orgs]

    # One node at a time (not a single batched statement covering every
    # candidate), each via builder.delete_node_with_retry -- deleting a
    # node's edges and the node itself atomically, in one savepoint per
    # attempt. The two /connect sides write into this same shared graph
    # concurrently; a fresh edge referencing one of these "junk" nodes can
    # land from the OTHER side between an edge-delete and a node-delete,
    # which Postgres's real FK constraint correctly rejects (SQLite never
    # enforces it at all -- see db.py -- so the same race used to silently
    # corrupt the graph instead of raising). Per-node retry recovers from
    # that; per-node isolation also means one contested node no longer takes
    # an entire otherwise-legitimate batch down with it -- confirmed live: a
    # whole /connect job died to a ForeignKeyViolation over ONE contested
    # organization out of a larger batch, when this was still one IN(...)
    # statement for the whole batch.
    #
    # commit_with_retry, not a bare db.commit(): a transient failure of the
    # FINAL commit still needs the whole loop redone, not just retried empty
    # (see commit_with_retry's own docstring) -- delete_node_with_retry's own
    # per-node deletes are themselves idempotent (skip/redo cleanly) either way.
    def _apply_deletes() -> int:
        removed = 0
        for p in junk_people:
            if builder.delete_node_with_retry(
                    db, p,
                    (RelationshipEdge.person_a_id == p.id)
                    | (RelationshipEdge.person_b_id == p.id)):
                removed += 1
        for o in junk_orgs:
            if builder.delete_node_with_retry(
                    db, o, RelationshipEdge.organization_id == o.id):
                removed += 1
        return removed

    removed = builder.commit_with_retry(db, _apply_deletes) or 0
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
