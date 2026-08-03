"""Wiring entity_filter.validate into _process_person's phase 5 (persist).

Regression coverage for the "USA Key" bug: a heuristic-extracted candidate
that LOOKS like a name (capitalized words, no noise token) but isn't a real
person/org must never reach the DB as a Person/Organization row, even though
the deterministic name-shape check alone would have let it through.
"""
from app.extraction.schemas import EdgeSignals, ExtractedEdge
from app.graph import builder, expansion
from app.models import Organization


def _silence_everything_but_phase0(monkeypatch):
    """Stub every OTHER candidate-producing stage so phase 0's single
    enrich_person() text (see below) is the only source of candidate_edges --
    dedup/search/fetch stay unused entirely, same as this file not caring
    about the real heuristic tokenizer, only about what phase 5 does with
    whatever candidate_edges ends up holding."""
    monkeypatch.setattr(expansion.ORCH, "officer_enrichment", lambda name: {"officers_text": ""})
    monkeypatch.setattr(expansion.ORCH, "edgar_enrichment", lambda name: {"edgar_text": ""})
    monkeypatch.setattr(expansion.ORCH, "firm_enrichment", lambda name: {"firms": []})
    monkeypatch.setattr(expansion.ORCH, "coauthors_enrichment", lambda name: {
        "coauthors": [], "coauthors_text": "", "identity_text": "",
    })
    monkeypatch.setattr(expansion.coauthor_plausibility, "is_active", lambda: False)
    monkeypatch.setattr(expansion.config, "PODCASTS_ENABLED", False)
    monkeypatch.setattr(expansion.ORCH, "dedup", lambda pairs: ([], {}))
    monkeypatch.setattr(expansion, "_repeat_candidates", lambda edges: [])
    monkeypatch.setattr(expansion.ORCH, "search", lambda query, is_person=True: [])

    class _Page:
        content = ""

    monkeypatch.setattr(expansion.ORCH, "fetch", lambda url: _Page())
    # Non-empty `summary` is phase 0's only populated field -- extract() gets
    # called exactly once, for the "wikipedia-summary" label (not in
    # _CLEAN_STRUCTURED, so _mark_trusted leaves signals.trusted as whatever
    # the stubbed extract() result below sets it to).
    monkeypatch.setattr(expansion.ORCH, "enrich_person", lambda name: {
        "title": name, "qid": None, "summary": "placeholder summary text",
        "wikidata_text": "", "colleagues_text": "", "nonprofit_text": "", "article": "",
    })


def _stub_single_extraction(monkeypatch, edge: ExtractedEdge):
    """Bypass the real heuristic tokenizer and hand phase 0 exactly one
    candidate edge -- what's under test is phase 5's persist-time filter,
    not the extractor's text parsing."""
    class _Result:
        edges = [edge]

    monkeypatch.setattr(expansion, "extract", lambda *a, **k: _Result())


def test_claude_rejected_person_is_never_persisted(db, monkeypatch):
    _silence_everything_but_phase0(monkeypatch)
    _stub_single_extraction(monkeypatch, ExtractedEdge(
        person_a="Fred Volinsky", person_b="USA Key", other_kind="person",
        relationship_type="cofounder", source_url="https://example.com/epiphany",
        signals=EdgeSignals(trusted=False),
    ))
    monkeypatch.setattr(expansion, "is_filtering_active", lambda: True)
    monkeypatch.setattr(expansion, "filter_entities", lambda names, kind: set())

    expansion._process_person(db, "Fred Volinsky", 0, {})

    assert builder.get_or_create_person(db, "USA Key", allow_create=False) is None


def test_claude_accepted_person_is_persisted(db, monkeypatch):
    _silence_everything_but_phase0(monkeypatch)
    _stub_single_extraction(monkeypatch, ExtractedEdge(
        person_a="Fred Volinsky", person_b="Jane Colleague", other_kind="person",
        relationship_type="coworker", source_url="https://example.com/page",
        signals=EdgeSignals(trusted=False),
    ))
    monkeypatch.setattr(expansion, "is_filtering_active", lambda: True)
    monkeypatch.setattr(expansion, "filter_entities",
                        lambda names, kind: set(names))

    expansion._process_person(db, "Fred Volinsky", 0, {})

    assert builder.get_or_create_person(db, "Jane Colleague", allow_create=False) is not None


def test_trusted_edges_skip_the_filter_even_when_claude_would_reject(db, monkeypatch):
    """A structured-source candidate (Wikidata/EDGAR/roster) is exempt from
    the Claude filter -- see _mark_trusted and _ranked_expandable's identical
    trusted-skips-filtering rule. Persistence must agree with that, not
    re-litigate a source extraction already vouched for."""
    _silence_everything_but_phase0(monkeypatch)
    _stub_single_extraction(monkeypatch, ExtractedEdge(
        person_a="Fred Volinsky", person_b="Trusted Person", other_kind="person",
        relationship_type="coworker", source_url="https://example.com/page",
        signals=EdgeSignals(trusted=True),
    ))
    monkeypatch.setattr(expansion, "is_filtering_active", lambda: True)
    # Claude would reject everyone -- must not matter for a trusted edge.
    monkeypatch.setattr(expansion, "filter_entities", lambda names, kind: set())

    expansion._process_person(db, "Fred Volinsky", 0, {})

    assert builder.get_or_create_person(db, "Trusted Person", allow_create=False) is not None


def test_claude_rejected_org_is_never_persisted(db, monkeypatch):
    _silence_everything_but_phase0(monkeypatch)
    _stub_single_extraction(monkeypatch, ExtractedEdge(
        person_a="Fred Volinsky", organization="Front-Cover Texts", other_kind="organization",
        relationship_type="employee", source_url="https://example.com/page",
        signals=EdgeSignals(trusted=False),
    ))
    monkeypatch.setattr(expansion, "is_filtering_active", lambda: True)
    monkeypatch.setattr(expansion, "filter_entities", lambda names, kind: set())

    expansion._process_person(db, "Fred Volinsky", 0, {})

    org = db.query(Organization).filter(Organization.name == "Front-Cover Texts").first()
    assert org is None


def test_filter_inactive_falls_back_to_old_behavior(db, monkeypatch):
    """Filtering disabled (no key, or ARTEMIS_CLAUDE_FILTER=0) must not
    change behavior at all -- same as every other Claude stage in this
    codebase degrading to a no-op."""
    _silence_everything_but_phase0(monkeypatch)
    _stub_single_extraction(monkeypatch, ExtractedEdge(
        person_a="Fred Volinsky", person_b="USA Key", other_kind="person",
        relationship_type="cofounder", source_url="https://example.com/page",
        signals=EdgeSignals(trusted=False),
    ))
    monkeypatch.setattr(expansion, "is_filtering_active", lambda: False)

    expansion._process_person(db, "Fred Volinsky", 0, {})

    assert builder.get_or_create_person(db, "USA Key", allow_create=False) is not None
