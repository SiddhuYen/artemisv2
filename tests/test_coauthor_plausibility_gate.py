"""Wiring coauthor_plausibility.check() into _process_person's phase 4b,
ahead of the OpenAlex call itself.
"""
from app.extraction import coauthor_plausibility
from app.graph import builder, expansion
from app.models import RelationshipEdge


def _silence_everything_but_openalex(monkeypatch, coauthors_text="", identity_text=""):
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
    monkeypatch.setattr(expansion.ORCH, "coauthors_enrichment", lambda name: {
        "coauthors": [{"name": "Some Coauthor", "count": 1}] if coauthors_text else [],
        "coauthors_text": coauthors_text,
        "identity_text": identity_text,
    })


def test_implausible_verdict_skips_openalex_entirely(db, monkeypatch):
    calls = []
    _silence_everything_but_openalex(monkeypatch, coauthors_text="Jane Phillips coauthor of X.")
    monkeypatch.setattr(expansion.ORCH, "coauthors_enrichment",
                        lambda name: calls.append(name) or {
                            "coauthors": [], "coauthors_text": "", "identity_text": ""})
    monkeypatch.setattr(coauthor_plausibility, "is_active", lambda: True)
    monkeypatch.setattr(coauthor_plausibility, "call_json", lambda *a, **k: {
        "plausible": False, "why": "Senior SWE at a tech company, no academic signal.",
    })

    expansion._process_person(db, "Jane Phillips", 0, {},
                              context="Senior SWE at Microsoft")

    assert calls == [], "OpenAlex must not even be called when implausible"
    subject = builder.get_or_create_person(db, "Jane Phillips")
    assert (subject.meta or {}).get("openalex_skipped") == {
        "why": "Senior SWE at a tech company, no academic signal.",
    }


def test_plausible_verdict_still_calls_openalex(db, monkeypatch):
    calls = []
    _silence_everything_but_openalex(
        monkeypatch, coauthors_text="Jaya Sharma coauthor of Lynn Hlatky.", identity_text="")
    original = expansion.ORCH.coauthors_enrichment

    def tracking_coauthors_enrichment(name):
        calls.append(name)
        return {"coauthors": [{"name": "Lynn Hlatky", "count": 1}],
                "coauthors_text": "Jaya Sharma coauthor of Lynn Hlatky.", "identity_text": ""}

    monkeypatch.setattr(expansion.ORCH, "coauthors_enrichment", tracking_coauthors_enrichment)
    monkeypatch.setattr(coauthor_plausibility, "is_active", lambda: True)
    monkeypatch.setattr(coauthor_plausibility, "call_json", lambda *a, **k: {
        "plausible": True, "why": "Cancer researcher with published work.",
    })

    expansion._process_person(db, "Jaya Sharma", 0, {}, context="cancer researcher")

    assert calls == ["Jaya Sharma"]
    subject = builder.get_or_create_person(db, "Jaya Sharma")
    edges = db.query(RelationshipEdge).filter(RelationshipEdge.person_a_id == subject.id).all()
    assert edges, "the coauthor edge should still have been persisted"


def test_unresolved_verdict_fails_open_and_still_calls_openalex(db, monkeypatch):
    """None (inactive / no signal / failed call) must default to the OLD
    behavior -- attempt OpenAlex -- not silently skip it."""
    calls = []
    _silence_everything_but_openalex(monkeypatch, coauthors_text="")
    monkeypatch.setattr(expansion.ORCH, "coauthors_enrichment",
                        lambda name: calls.append(name) or {
                            "coauthors": [], "coauthors_text": "", "identity_text": ""})
    monkeypatch.setattr(coauthor_plausibility, "is_active", lambda: False)

    expansion._process_person(db, "Someone", 0, {}, context="")

    assert calls == ["Someone"]


def test_implausible_verdict_does_not_fire_for_non_person_subjects(db, monkeypatch):
    """is_person=False (org seed) has no coauthor concept at all -- the gate
    must not even be consulted."""
    calls = []
    _silence_everything_but_openalex(monkeypatch, coauthors_text="")
    monkeypatch.setattr(coauthor_plausibility, "check",
                        lambda *a, **k: calls.append(1) or None)

    expansion._process_person(db, "Some Org", 0, {}, is_person=False, context="")

    assert calls == []
