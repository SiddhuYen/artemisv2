"""Alpha piece 3: target-aware top-5 candidate ranking.

Two changes to _ranked_expandable's existing scoring/selection, both scoped
to the non-famous/origin side of an asymmetric /connect walk
(enhanced_professional_search):
  - _Candidate.score() gains a seniority bonus for candidates whose own
    edges carry leadership signal (cofounder/board_member typing, or
    business-domain language in the evidence) -- Alpha step 7's "most high
    up and well connected," layered on top of the existing evidence-
    strength terms, not a replacement for them.
  - the final frontier narrows to ALPHA_TOP_CANDIDATES (5) instead of the
    general EXPAND_TOP_STRONG beam (15) -- a reasoning-selected angle
    (phase 4e) already narrowed the field, so expanding as many candidates
    as the untargeted case doesn't need is wasted search budget.
"""
from app import config
from app.extraction.schemas import EdgeSignals, ExtractedEdge
from app.graph import expansion


def _edge(person_b: str, relationship_type: str, confidence: float,
         evidence: str = "", explicit: bool = False) -> ExtractedEdge:
    return ExtractedEdge(
        person_a="Subject", person_b=person_b, other_kind="person",
        relationship_type=relationship_type, confidence_base=confidence,
        confidence_adjusted=confidence, evidence_snippet=evidence, source_url=f"http://x/{person_b}",
        signals=EdgeSignals(explicit_keyword_match=explicit),
    )


# ---------------------------------------------------------------------------
# _Candidate.score() -- seniority bonus
# ---------------------------------------------------------------------------
def test_cofounder_edge_scores_higher_than_an_equally_evidenced_coworker_edge():
    disc = {}
    expansion._record(disc, _edge("Cofounder Cate", "cofounder", 0.85, explicit=True))
    expansion._record(disc, _edge("Coworker Cody", "coworker", 0.85, explicit=True))
    assert disc["cofounder cate"].score() > disc["coworker cody"].score()


def test_business_domain_language_in_evidence_also_earns_the_bonus():
    """Not every leadership signal comes through as a typed cofounder/
    board_member edge -- a "Vice President Sales" mention in evidence text,
    even on a generically-typed edge, should count too."""
    disc = {}
    expansion._record(disc, _edge(
        "VP Vera", "coworker", 0.7,
        evidence="Vera is Vice President Sales & Strategy at the company."))
    expansion._record(disc, _edge("Coworker Cody", "coworker", 0.7))
    assert disc["vp vera"].seniority_edges == 1
    assert disc["coworker cody"].seniority_edges == 0
    assert disc["vp vera"].score() > disc["coworker cody"].score()


def test_seniority_bonus_is_additive_not_a_replacement():
    """A cofounder edge with weak overall evidence should NOT automatically
    outrank a coworker edge with much stronger evidence -- the bonus adds
    to score(), it doesn't override the existing evidence-strength terms."""
    disc = {}
    expansion._record(disc, _edge("Weak Cofounder", "cofounder", 0.31))
    for i in range(5):
        expansion._record(disc, ExtractedEdge(
            person_a="Subject", person_b="Strong Coworker", other_kind="person",
            relationship_type="coworker", confidence_base=0.85, confidence_adjusted=0.85,
            source_url=f"http://x/{i}", signals=EdgeSignals(explicit_keyword_match=True),
        ))
    assert disc["strong coworker"].score() > disc["weak cofounder"].score()


# ---------------------------------------------------------------------------
# _ranked_expandable's top_n override
# ---------------------------------------------------------------------------
def _candidate_pool(n: int, base_score_desc=True):
    disc = {}
    for i in range(n):
        conf = 0.9 - (i * 0.01) if base_score_desc else 0.9
        expansion._record(disc, _edge(f"Person {i}", "coworker", conf, explicit=True))
    return disc


def test_ranked_expandable_uses_configured_default_with_no_top_n(monkeypatch):
    monkeypatch.setattr(expansion, "is_filtering_active", lambda: False)
    disc = _candidate_pool(config.EXPAND_TOP_STRONG + 5)
    frontier = expansion._ranked_expandable(disc, visited=set(), prefer_reachable=False)
    assert len(frontier) == config.EXPAND_TOP_STRONG


def test_ranked_expandable_top_n_narrows_the_final_selection(monkeypatch):
    monkeypatch.setattr(expansion, "is_filtering_active", lambda: False)
    disc = _candidate_pool(config.EXPAND_TOP_STRONG + 5)
    frontier = expansion._ranked_expandable(disc, visited=set(), prefer_reachable=False, top_n=5)
    assert len(frontier) == 5
    # still the top 5 BY SCORE, not an arbitrary slice
    assert frontier == [f"Person {i}" for i in range(5)]


def test_ranked_expandable_top_n_also_applies_in_reachable_mode(monkeypatch):
    monkeypatch.setattr(expansion, "is_filtering_active", lambda: False)
    monkeypatch.setattr(expansion.ORCH, "notable_set", lambda names: set())
    disc = _candidate_pool(config.EXPAND_TOP_STRONG + 5)
    frontier = expansion._ranked_expandable(disc, visited=set(), prefer_reachable=True, top_n=5)
    assert len(frontier) == 5


# ---------------------------------------------------------------------------
# expand_graph wiring: alpha_top_n only applies on the enhanced_professional_
# search side
# ---------------------------------------------------------------------------
def test_expand_graph_passes_alpha_top_candidates_when_enhanced(db, monkeypatch):
    captured = {}

    def fake_ranked_expandable(disc, visited, progress=None, prefer_reachable=None, top_n=None):
        captured["top_n"] = top_n
        return []

    monkeypatch.setattr(expansion, "_ranked_expandable", fake_ranked_expandable)
    monkeypatch.setattr(expansion, "_process_person", lambda *a, **k: None)
    monkeypatch.setattr(expansion.builder, "get_or_create_person", lambda *a, **k: None)
    monkeypatch.setattr(expansion.builder, "at_node_cap", lambda db: False)

    expansion.expand_graph(db, "Seed", max_depth=2, enhanced_professional_search=True)

    assert captured["top_n"] == config.ALPHA_TOP_CANDIDATES


def test_expand_graph_passes_no_top_n_override_when_not_enhanced(db, monkeypatch):
    captured = {}

    def fake_ranked_expandable(disc, visited, progress=None, prefer_reachable=None, top_n=None):
        captured["top_n"] = top_n
        return []

    monkeypatch.setattr(expansion, "_ranked_expandable", fake_ranked_expandable)
    monkeypatch.setattr(expansion, "_process_person", lambda *a, **k: None)
    monkeypatch.setattr(expansion.builder, "get_or_create_person", lambda *a, **k: None)
    monkeypatch.setattr(expansion.builder, "at_node_cap", lambda db: False)

    expansion.expand_graph(db, "Seed", max_depth=2, enhanced_professional_search=False)

    assert captured["top_n"] is None
