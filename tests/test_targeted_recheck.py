"""Targeted professional-network re-search (phase 4c) -- the fix for a
specific, observed failure: a real close colleague, mentioned only in
passing across several sources with no sentence ever stating the
relationship, stays capped at "weak coworker" confidence forever under
generic silo search alone (see extraction.confidence's evidence-ceiling
rules). Concrete motivating case: Molly Chakraborty, Trinamix's actual
cofounder and president, showing up only in a hashtag-heavy LinkedIn post
alongside Prantik Chakraborty with no stated relationship.

Two pieces:
  1. graph.expansion._repeat_candidates + the phase-4c re-query itself,
     gated on `enhanced_professional_search`.
  2. graph.connect._expand_both_concurrently computing which side gets that
     flag: whichever side is NOT the shallow, famous one in an asymmetric
     /connect walk (see _resolve_expansion_depths) -- the side actually
     walking TOWARD a famous target, not away from one.
"""
from app import config
from app.extraction.schemas import EdgeSignals, ExtractedEdge
from app.graph import builder, connect as C, expansion
from app.models import RelationshipEdge
from app.providers.base import SearchResult


def _edge(person_b: str, confidence: float, url: str = "http://x/1") -> ExtractedEdge:
    return ExtractedEdge(
        person_a="Prantik Chakraborty", person_b=person_b, other_kind="person",
        relationship_type="coworker", confidence_base=confidence,
        confidence_adjusted=confidence, source_url=url,
        evidence_snippet=f"{person_b} co-listed", signals=EdgeSignals(),
    )


# ---------------------------------------------------------------------------
# expansion._repeat_candidates
# ---------------------------------------------------------------------------
def test_repeat_candidates_picks_up_a_name_mentioned_multiple_times_weakly():
    edges = [_edge("Molly Chakraborty", 0.35, "http://x/1"),
             _edge("Molly Chakraborty", 0.39, "http://x/2"),
             _edge("Molly Chakraborty", 0.5, "http://x/3")]  # still < STRONG_MIN (0.6)
    assert expansion._repeat_candidates(edges) == ["Molly Chakraborty"]


def test_repeat_candidates_ignores_a_single_mention():
    edges = [_edge("One-Off Person", 0.39)]
    assert expansion._repeat_candidates(edges) == []


def test_repeat_candidates_ignores_a_name_already_strong():
    edges = [_edge("Already Strong", 0.85, "http://x/1"),
             _edge("Already Strong", 0.7, "http://x/2")]
    assert expansion._repeat_candidates(edges) == []


def test_repeat_candidates_ranks_by_mention_count_and_caps_at_the_configured_max(monkeypatch):
    monkeypatch.setattr(config, "ENHANCED_SEARCH_MAX_CANDIDATES", 2)
    edges = []
    # "Most Mentioned" appears 4x, "Second" 3x, "Third" 2x -- all weak
    for i in range(4):
        edges.append(_edge("Most Mentioned", 0.3, f"http://x/most/{i}"))
    for i in range(3):
        edges.append(_edge("Second", 0.3, f"http://x/second/{i}"))
    for i in range(2):
        edges.append(_edge("Third", 0.3, f"http://x/third/{i}"))
    assert expansion._repeat_candidates(edges) == ["Most Mentioned", "Second"]


def test_repeat_candidates_ignores_organizations():
    org_edge = ExtractedEdge(
        person_a="Prantik Chakraborty", organization="Some Org Mentioned Twice",
        other_kind="organization", relationship_type="employee",
        confidence_base=0.3, confidence_adjusted=0.3, source_url="http://x/1",
        signals=EdgeSignals(),
    )
    assert expansion._repeat_candidates([org_edge, org_edge]) == []


# ---------------------------------------------------------------------------
# connect._expand_both_concurrently: which side gets the flag
# ---------------------------------------------------------------------------
def test_enhanced_search_goes_to_the_full_depth_side_not_the_famous_one(db, monkeypatch):
    captured = {}

    def fake_expand_graph(worker_db, name, side_depth, **kwargs):
        captured[name] = kwargs.get("enhanced_professional_search")

    monkeypatch.setattr(C, "expand_graph", fake_expand_graph)

    C._expand_both_concurrently(
        db=db, name_a="Obscure Person", name_b="Famous Person",
        depth_a=3, depth_b=C.SHALLOW_FAMOUS_DEPTH,
        protected=set(), progress=None, context_a="", context_b="",
    )

    assert captured["Obscure Person"] is True, "the full-depth side walks TOWARD the famous target"
    assert captured["Famous Person"] is False, "the shallow famous side doesn't need it"


def test_enhanced_search_is_off_for_both_sides_when_symmetric(db, monkeypatch):
    captured = {}

    def fake_expand_graph(worker_db, name, side_depth, **kwargs):
        captured[name] = kwargs.get("enhanced_professional_search")

    monkeypatch.setattr(C, "expand_graph", fake_expand_graph)

    C._expand_both_concurrently(
        db=db, name_a="Alpha", name_b="Beta",
        depth_a=2, depth_b=2,  # symmetric -- no famous target identified either way
        protected=set(), progress=None, context_a="", context_b="",
    )

    assert captured["Alpha"] is False
    assert captured["Beta"] is False


# ---------------------------------------------------------------------------
# expansion._process_person's phase 4c, end to end
# ---------------------------------------------------------------------------
def _silence_everything_but_search(monkeypatch, search_results=None, fetched_text=""):
    """Stub every ORCH call EXCEPT .search/.fetch, so phase 4c's targeted
    re-query is the only thing that can add candidate_edges."""
    monkeypatch.setattr(expansion.ORCH, "enrich_person", lambda name: None)
    monkeypatch.setattr(expansion.ORCH, "officer_enrichment", lambda name: {"officers_text": ""})
    monkeypatch.setattr(expansion.ORCH, "edgar_enrichment", lambda name: {"edgar_text": ""})
    monkeypatch.setattr(expansion.ORCH, "firm_enrichment", lambda name: {"firms": []})
    monkeypatch.setattr(expansion.ORCH, "coauthors_enrichment", lambda name: {
        "coauthors": [], "coauthors_text": "", "identity_text": "",
    })
    monkeypatch.setattr(expansion.config, "PODCASTS_ENABLED", False)
    monkeypatch.setattr(expansion.ORCH, "dedup", lambda pairs: ([], {}))  # no generic silo queries

    class _Page:
        def __init__(self, content):
            self.content = content

    monkeypatch.setattr(expansion.ORCH, "search",
                        lambda query, is_person=True: search_results or [])
    monkeypatch.setattr(expansion.ORCH, "fetch", lambda url: _Page(fetched_text))


def _find_molly_edges(db, subject):
    edges = db.query(RelationshipEdge).all()
    return [e for e in edges if e.person_b_id and
            db.get(type(subject), e.person_b_id).canonical_name == "Molly Chakraborty"]


def test_phase_4c_finds_and_persists_an_edge_from_the_targeted_query(db, monkeypatch):
    """Even without Claude configured, the targeted re-query itself is the
    fix: a deliberate, both-names search finds a real, specific sentence
    instead of relying on an accidental co-mention."""
    _silence_everything_but_search(
        monkeypatch,
        search_results=[SearchResult(
            "Trinamix leadership", "https://example.com/leadership", "snippet", "serper")],
        fetched_text=("Prantik Chakraborty has worked alongside Molly Chakraborty, "
                      "Cofounder and President of Trinamix, for over a decade."),
    )
    monkeypatch.setattr(expansion, "_repeat_candidates", lambda edges: ["Molly Chakraborty"])

    from app.extraction import relation_classifier
    monkeypatch.setattr(relation_classifier, "is_active", lambda: False)

    subject = builder._new_person_or_existing(db, "Prantik Chakraborty",
                                              "prantik chakraborty", None)
    db.commit()

    expansion._process_person(db, "Prantik Chakraborty", 0, {},
                              enhanced_professional_search=True)

    molly_edges = _find_molly_edges(db, subject)
    assert molly_edges, "phase 4c should have persisted a real edge to Molly Chakraborty"
    assert molly_edges[0].relationship_type == "cofounder"


def test_phase_4c_uses_claude_to_reach_a_decisive_confidence_when_configured(db, monkeypatch):
    """The actual fix for the motivating case: the deterministic spaCy
    confidence model (base x silo_multiplier x strength_factor) can leave
    even a clean, targeted, co-occurring hit short of 'strong' on the
    arithmetic alone (confirmed: 0.35 x 1.2 x 1.15 = 0.483 for exactly this
    sentence). Claude classification -- the same batched-verdict mechanism
    _retype_unknown_edges already uses -- gives a targeted hit the decisive
    read it was worth going and looking for."""
    _silence_everything_but_search(
        monkeypatch,
        search_results=[SearchResult(
            "Trinamix leadership", "https://example.com/leadership", "snippet", "serper")],
        fetched_text=("Prantik Chakraborty has worked alongside Molly Chakraborty, "
                      "Cofounder and President of Trinamix, for over a decade."),
    )
    monkeypatch.setattr(expansion, "_repeat_candidates", lambda edges: ["Molly Chakraborty"])

    from app.extraction import relation_classifier
    monkeypatch.setattr(relation_classifier, "is_active", lambda: True)
    monkeypatch.setattr(relation_classifier, "classify",
                        lambda items: [{"type": "cofounder", "confidence": 0.9} for _ in items])

    subject = builder._new_person_or_existing(db, "Prantik Chakraborty",
                                              "prantik chakraborty", None)
    db.commit()

    expansion._process_person(db, "Prantik Chakraborty", 0, {},
                              enhanced_professional_search=True)

    molly_edges = _find_molly_edges(db, subject)
    assert molly_edges, "phase 4c should have persisted a real edge to Molly Chakraborty"
    best = max(e.confidence_raw for e in molly_edges)
    assert best > config.STRONG_MIN, (
        "Claude's confident verdict should land this at strong confidence, "
        "not stay capped the way the original weak mentions were")
    assert molly_edges[0].relationship_type == "cofounder"


def test_phase_4c_does_nothing_when_disabled(db, monkeypatch):
    search_called = {"n": 0}

    def fake_search(query, is_person=True):
        search_called["n"] += 1
        return []

    _silence_everything_but_search(monkeypatch)
    monkeypatch.setattr(expansion.ORCH, "search", fake_search)
    monkeypatch.setattr(expansion, "_repeat_candidates", lambda edges: ["Molly Chakraborty"])

    disc = {}
    expansion._process_person(db, "Prantik Chakraborty", 0, disc,
                              enhanced_professional_search=False)

    assert search_called["n"] == 0, "phase 4c must not run at all when the flag is off"
