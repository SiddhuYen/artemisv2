"""Wiring the no-QID homonym guard (see test_homonym_guard.py) into the ONE
choke point every discovered counterpart passes through: expansion.py's
phase-5 persist loop. builder.get_or_create_person's guard is inert without
a caller actually giving it something to check -- this is that caller.
"""
from app.extraction.schemas import EdgeSignals, ExtractedEdge
from app.graph import builder, expansion
from app.models import RelationshipEdge
from app.providers.base import SearchResult


def _edge(person_b: str, evidence: str, source_url: str = "") -> ExtractedEdge:
    return ExtractedEdge(
        person_a="Subject", person_b=person_b, other_kind="person",
        relationship_type="coauthor", confidence_base=0.33, confidence_adjusted=0.33,
        evidence_snippet=evidence, source_url=source_url, signals=EdgeSignals(),
    )


# ---------------------------------------------------------------------------
# _counterpart_identity_text
# ---------------------------------------------------------------------------
def test_openalex_sourced_edge_gets_the_academic_author_hint():
    edge = _edge("Donald Trump", "Jaya Sharma coauthor of Donald Trump.",
                source_url=expansion._OPENALEX_SOURCE_URL)
    text = expansion._counterpart_identity_text(edge)
    assert "academic author" in text
    assert "Jaya Sharma coauthor of Donald Trump." in text


def test_non_openalex_edge_falls_back_to_bare_evidence():
    edge = _edge("Some Person", "Some Person spoke at the conference.",
                source_url="https://example.com/article")
    text = expansion._counterpart_identity_text(edge)
    assert text == "Some Person spoke at the conference."
    assert "academic author" not in text


def test_no_evidence_and_not_openalex_returns_none():
    edge = _edge("Some Person", "", source_url="https://example.com/article")
    assert expansion._counterpart_identity_text(edge) is None


# ---------------------------------------------------------------------------
# End-to-end reproduction of the live bug via _process_person: an OpenAlex
# coauthor named "Donald Trump" must NOT merge onto an existing "Donald
# Trump" node that's clearly anchored as the US president.
# ---------------------------------------------------------------------------
def _silence_everything_but_openalex(monkeypatch, coauthors, identity_text=""):
    monkeypatch.setattr(expansion.ORCH, "enrich_person", lambda name: None)
    monkeypatch.setattr(expansion.ORCH, "officer_enrichment", lambda name: {"officers_text": ""})
    monkeypatch.setattr(expansion.ORCH, "edgar_enrichment", lambda name: {"edgar_text": ""})
    monkeypatch.setattr(expansion.ORCH, "firm_enrichment", lambda name: {"firms": []})
    # Phase 4f (org directory) -- stubbed off here so it can't issue searches
    # of its own; it has its own dedicated tests.
    monkeypatch.setattr(expansion.ORCH, "directory_enrichment",
                        lambda org, industry="", size_tier="": {
                            "org": org, "url": "", "members": [], "overflow": False})
    monkeypatch.setattr(expansion.config, "PODCASTS_ENABLED", False)
    monkeypatch.setattr(expansion.ORCH, "dedup", lambda pairs: ([], {}))
    # This test targets the counterpart-merge guard specifically, which only
    # matters once OpenAlex has actually run -- disable the new pre-gate
    # (fail-open) so it doesn't skip the call before getting there, and so
    # this test doesn't make a real network call.
    monkeypatch.setattr(expansion.coauthor_plausibility, "is_active", lambda: False)
    coauthors_text = " ".join(f"Jaya Sharma coauthor of {c}." for c in coauthors)
    monkeypatch.setattr(expansion.ORCH, "coauthors_enrichment", lambda name: {
        "coauthors": [{"name": c, "count": 1} for c in coauthors],
        "coauthors_text": coauthors_text,
        "identity_text": identity_text,
    })


def test_openalex_coauthor_does_not_merge_onto_an_unrelated_same_named_president(db, monkeypatch):
    # The president node, already in the graph with clearly political evidence --
    # simulating Larry Ellison's own hop-1 search having already discovered him.
    president = builder.get_or_create_person(db, "Donald Trump")
    db.add(RelationshipEdge(
        person_a_id=president.id, relationship_type="unknown",
        evidence_snippet=("U.S. President Donald Trump and CDC Director Robert R. Redfield "
                          "participate in the daily briefing on the coronavirus."),
    ))
    db.commit()

    # Jaya Sharma's identity_text is empty (subject-side OpenAlex identity gate,
    # phase 4b, is a SEPARATE check from this one -- irrelevant here since it's
    # about Jaya's own identity, not her coauthor's).
    _silence_everything_but_openalex(monkeypatch, coauthors=["Donald Trump"])

    expansion._process_person(db, "Jaya Sharma", 0, {}, context="")

    trump_edges = db.query(RelationshipEdge).filter(
        RelationshipEdge.person_a_id == builder.get_or_create_person(db, "Jaya Sharma").id
    ).all()
    assert trump_edges, "the coauthor edge should still have been created"
    counterpart_id = trump_edges[0].person_b_id
    assert counterpart_id != president.id, (
        "the academic coauthor 'Donald Trump' must land on a DIFFERENT node "
        "than the sitting president -- this is the exact live bridge bug "
        "(Amit Sharma -> ... -> Jaya Sharma -> Trump -> Larry Ellison)")
    assert president.meta.get("homonym_rejected") is not None
