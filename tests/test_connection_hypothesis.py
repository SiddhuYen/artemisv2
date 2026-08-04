"""Hypothesis-first node search (expansion phase 0e).

Look at the node before searching it: from the structured enrichment already
in hand, ask which named people/orgs it is likely to be publicly documented
with, then spend the searches PROVING those specific connections instead of
firing the same nine generic silo templates at every subject alike.

What these tests pin is the containment, because that is where the risk is.
The model names entities and nothing else:

  - queries come from config.NODE_HYPOTHESIS_QUERIES, not from the model;
  - only edges about the entity that was hypothesised survive the page it was
    found on -- a bystander named on the same page is dropped;
  - the relationship the model PREDICTED never reaches an edge;
  - HYPOTHESIS_SILO's multiplier is 1.0, so predicting a connection buys its
    confirmation nothing;
  - the searches are paid for out of the node's generic silo allowance, and
    the coverage record honestly reports the queries that were given up.
"""
from sqlalchemy import select

from app import config
from app.extraction import connection_hypothesis
from app.extraction.schemas import EdgeSignals, ExtractedEdge, ExtractionOutput
from app.graph import expansion
from app.models import Person, RelationshipEdge
from app.network.silo_weights import query_budget, trim_budget
from app.providers import cache
from app.providers.base import SearchResult
from app.silos import HYPOTHESIS_SILO
from app.utils.names import person_norm_key


def _no_cache(monkeypatch):
    """propose() caches its verdict; most tests here are about the parsing."""
    monkeypatch.setattr(cache, "get", lambda key, track=True: None)
    monkeypatch.setattr(cache, "set", lambda key, kind, value, ttl: None)


def _payload(*candidates):
    return {"candidates": list(candidates)}


def _candidate(name, kind="person", relationship="coworker", why="documented together"):
    return {"name": name, "kind": kind, "relationship": relationship, "why": why}


# ---------------------------------------------------------------------------
# connection_hypothesis.propose -- the model names entities, nothing more
# ---------------------------------------------------------------------------
def test_propose_returns_nothing_when_inactive(monkeypatch):
    monkeypatch.setattr(connection_hypothesis, "is_active", lambda: False)
    assert connection_hypothesis.propose("Prantik Chakraborty") == []


def test_propose_returns_nothing_when_the_call_fails(monkeypatch):
    _no_cache(monkeypatch)
    monkeypatch.setattr(connection_hypothesis, "is_active", lambda: True)
    monkeypatch.setattr(connection_hypothesis, "call_json", lambda *a, **k: None)
    assert connection_hypothesis.propose("Prantik Chakraborty") == []


def test_propose_parses_a_candidate_and_normalizes_it(monkeypatch):
    _no_cache(monkeypatch)
    monkeypatch.setattr(connection_hypothesis, "is_active", lambda: True)
    monkeypatch.setattr(connection_hypothesis, "call_json", lambda *a, **k: _payload(
        _candidate("Molly Chakraborty", why="named as Trinamix cofounder")))

    out = connection_hypothesis.propose("Prantik Chakraborty")

    assert out == [{"name": "Molly Chakraborty", "kind": "person",
                    "norm": person_norm_key("Molly Chakraborty"),
                    "relationship": "coworker",
                    "why": "named as Trinamix cofounder"}]


def test_propose_drops_a_one_word_person_but_keeps_a_one_word_org(monkeypatch):
    """A single token is a first name or a fragment when it is meant to be a
    person, and searching it returns noise. An org is legitimately one word."""
    _no_cache(monkeypatch)
    monkeypatch.setattr(connection_hypothesis, "is_active", lambda: True)
    monkeypatch.setattr(connection_hypothesis, "call_json", lambda *a, **k: _payload(
        _candidate("Molly"), _candidate("Oracle", kind="org", relationship="employee")))

    names = [h["name"] for h in connection_hypothesis.propose("Prantik Chakraborty")]

    assert names == ["Oracle"]


def test_propose_drops_the_subject_named_back_at_us(monkeypatch):
    _no_cache(monkeypatch)
    monkeypatch.setattr(connection_hypothesis, "is_active", lambda: True)
    monkeypatch.setattr(connection_hypothesis, "call_json", lambda *a, **k: _payload(
        _candidate("Prantik Chakraborty"), _candidate("Molly Chakraborty")))

    names = [h["name"] for h in connection_hypothesis.propose("Prantik Chakraborty")]

    assert names == ["Molly Chakraborty"]


def test_propose_drops_a_counterpart_the_caller_has_already_settled(monkeypatch):
    """Nothing left to prove about an already-strong tie -- the same discipline
    phase 4c applies when it declines to re-query a 'strong' pair."""
    _no_cache(monkeypatch)
    monkeypatch.setattr(connection_hypothesis, "is_active", lambda: True)
    monkeypatch.setattr(connection_hypothesis, "call_json", lambda *a, **k: _payload(
        _candidate("Already Known"), _candidate("Worth Proving")))

    names = [h["name"] for h in connection_hypothesis.propose(
        "Prantik Chakraborty", exclude={person_norm_key("Already Known")})]

    assert names == ["Worth Proving"]


def test_propose_deduplicates_candidates(monkeypatch):
    _no_cache(monkeypatch)
    monkeypatch.setattr(connection_hypothesis, "is_active", lambda: True)
    monkeypatch.setattr(connection_hypothesis, "call_json", lambda *a, **k: _payload(
        _candidate("Molly Chakraborty"), _candidate("molly  chakraborty")))

    assert len(connection_hypothesis.propose("Prantik Chakraborty")) == 1


def test_propose_normalizes_an_unusable_relationship_to_unknown(monkeypatch):
    """The schema constrains this, but the value is an EXPECTATION that steers
    real spending -- so it is validated here rather than trusted to arrive."""
    _no_cache(monkeypatch)
    monkeypatch.setattr(connection_hypothesis, "is_active", lambda: True)
    monkeypatch.setattr(connection_hypothesis, "call_json", lambda *a, **k: _payload(
        _candidate("Molly Chakraborty", relationship="best_friend_forever")))

    assert connection_hypothesis.propose("P Q")[0]["relationship"] == "unknown"


def test_propose_caps_at_the_configured_maximum(monkeypatch):
    _no_cache(monkeypatch)
    monkeypatch.setattr(config, "NODE_HYPOTHESIS_MAX", 2)
    monkeypatch.setattr(connection_hypothesis, "is_active", lambda: True)
    monkeypatch.setattr(connection_hypothesis, "call_json", lambda *a, **k: _payload(
        *[_candidate(f"Person Number{i}") for i in range(6)]))

    assert len(connection_hypothesis.propose("Prantik Chakraborty")) == 2


def test_propose_grounds_the_prompt_in_the_facts_and_the_target(monkeypatch):
    captured = {}
    _no_cache(monkeypatch)
    monkeypatch.setattr(connection_hypothesis, "is_active", lambda: True)

    def fake_call_json(prompt, schema, model, max_tokens=0):
        captured["prompt"] = prompt
        return _payload()

    monkeypatch.setattr(connection_hypothesis, "call_json", fake_call_json)
    connection_hypothesis.propose(
        "Prantik Chakraborty", context="Trinamix",
        facts=["found this run: employee — Trinamix (organization)"],
        target_name="Larry Ellison", target_context="Oracle")

    prompt = captured["prompt"]
    assert "Trinamix" in prompt
    assert "found this run: employee" in prompt
    assert "Larry Ellison" in prompt
    assert "Oracle" in prompt


def test_propose_truncates_the_fact_list(monkeypatch):
    captured = {}
    _no_cache(monkeypatch)
    monkeypatch.setattr(config, "NODE_HYPOTHESIS_FACTS", 2)
    monkeypatch.setattr(connection_hypothesis, "is_active", lambda: True)

    def fake_call_json(prompt, schema, model, max_tokens=0):
        captured["prompt"] = prompt
        return _payload()

    monkeypatch.setattr(connection_hypothesis, "call_json", fake_call_json)
    connection_hypothesis.propose("P Q", facts=[f"fact-{i}" for i in range(5)])

    assert "fact-1" in captured["prompt"]
    assert "fact-2" not in captured["prompt"]


def test_propose_reuses_a_cached_verdict_for_the_same_evidence(monkeypatch):
    """...and asks again once the evidence changes: a node re-encountered with
    more known about it is a different question, which is the entire premise
    of asking from what the enrichment found."""
    calls = []
    store = {}
    monkeypatch.setattr(cache, "get", lambda key, track=True: store.get(key))
    monkeypatch.setattr(cache, "set",
                        lambda key, kind, value, ttl: store.__setitem__(key, value))
    monkeypatch.setattr(connection_hypothesis, "is_active", lambda: True)

    def fake_call_json(prompt, schema, model, max_tokens=0):
        calls.append(prompt)
        return _payload(_candidate("Molly Chakraborty"))

    monkeypatch.setattr(connection_hypothesis, "call_json", fake_call_json)

    connection_hypothesis.propose("Prantik Chakraborty", facts=["at Trinamix"])
    connection_hypothesis.propose("Prantik Chakraborty", facts=["at Trinamix"])
    assert len(calls) == 1, "same subject, same evidence -- one call"

    connection_hypothesis.propose("Prantik Chakraborty",
                                  facts=["at Trinamix", "board of Something"])
    assert len(calls) == 2, "new evidence is a new question"


# ---------------------------------------------------------------------------
# silo_weights.trim_budget -- how the searches are paid for
# ---------------------------------------------------------------------------
def test_trim_budget_gives_up_exactly_what_was_asked():
    full = query_budget(None)
    trimmed = trim_budget(full, 6)
    assert sum(trimmed.values()) == sum(full.values()) - 6


def test_trim_budget_never_zeroes_a_silo():
    """Breadth is the point of the allocation: nine differently-shaped
    questions. Zeroing one to protect another's fourth query trades a whole
    angle of enquiry for a marginal repeat."""
    trimmed = trim_budget(query_budget(None), 999)
    assert trimmed and all(count >= 1 for count in trimmed.values())
    assert set(trimmed) == set(query_budget(None))


def test_trim_budget_is_a_no_op_for_nothing_to_pay():
    full = query_budget(None)
    assert trim_budget(full, 0) == full
    assert trim_budget({}, 5) == {}


def test_trim_budget_is_deterministic():
    full = query_budget(None)
    assert trim_budget(full, 5) == trim_budget(dict(reversed(list(full.items()))), 5)


# ---------------------------------------------------------------------------
# expansion phase 0e, end to end
# ---------------------------------------------------------------------------
def _silence_everything_but_hypotheses(monkeypatch, searched=None, extracted=None):
    """Stub every other source, so the only searches issued and the only edges
    persisted are phase 0e's. Same shape as test_node_profiling.py's helper."""
    monkeypatch.setattr(expansion.ORCH, "enrich_person", lambda name: None)
    monkeypatch.setattr(expansion.ORCH, "officer_enrichment", lambda name: {"officers_text": ""})
    monkeypatch.setattr(expansion.ORCH, "edgar_enrichment", lambda name: {"edgar_text": ""})
    monkeypatch.setattr(expansion.ORCH, "firm_enrichment", lambda name: {"firms": []})
    monkeypatch.setattr(expansion.ORCH, "coauthors_enrichment", lambda name: {
        "coauthors": [], "coauthors_text": "", "identity_text": ""})
    monkeypatch.setattr(expansion.ORCH, "directory_enrichment",
                        lambda org, industry="", size_tier="": {
                            "org": org, "url": "", "members": [], "overflow": False})
    monkeypatch.setattr(expansion.coauthor_plausibility, "is_active", lambda: False)
    monkeypatch.setattr(expansion.relation_classifier, "is_active", lambda: False)
    monkeypatch.setattr(expansion.config, "PODCASTS_ENABLED", False)
    # Phase 1's own queries: nothing to search, so every recorded search below
    # belongs to phase 0e.
    monkeypatch.setattr(expansion.ORCH, "dedup", lambda pairs: ([], {}))
    monkeypatch.setattr(expansion, "_repeat_candidates", lambda edges: [])
    # No Claude entity filter: these are synthetic names and the filter would
    # need a live call to judge them.
    monkeypatch.setattr(expansion, "is_filtering_active", lambda: False)

    class _Page:
        content = "<html>a fetched page</html>"

    def _search(query, is_person=True):
        (searched if searched is not None else []).append(query)
        return [SearchResult("A page", "https://example.test/1", "snippet", "serper")]

    monkeypatch.setattr(expansion.ORCH, "search", _search)
    monkeypatch.setattr(expansion.ORCH, "fetch", lambda url: _Page())
    if extracted is not None:
        monkeypatch.setattr(
            expansion, "extract",
            lambda subject, text, silo, snippet, url, deep=False: extracted(silo, url))


def _edge(subject, counterpart, kind="person", relationship="unknown", conf=0.4):
    return ExtractedEdge(
        person_a=subject,
        person_b=counterpart if kind == "person" else "",
        organization=counterpart if kind != "person" else "",
        other_kind="person" if kind == "person" else "organization",
        relationship_type=relationship, confidence_base=conf, confidence_adjusted=conf,
        source_url="https://example.test/1",
        evidence_snippet=f"{subject} and {counterpart} appear together.",
        signals=EdgeSignals(sentence_cooccurrence=True),
    )


def _activate(monkeypatch, hypotheses):
    monkeypatch.setattr(expansion.connection_hypothesis, "is_active", lambda: True)
    monkeypatch.setattr(
        expansion.connection_hypothesis, "propose",
        lambda subject_name, **kwargs: [
            dict(h, norm=person_norm_key(h["name"]) if h["kind"] == "person"
                 else h["name"].lower()) for h in hypotheses])


def test_phase_0e_searches_for_the_hypothesised_pair(db, monkeypatch):
    """The query surface is config's, not the model's: it named an entity and
    the code wrote the query around it."""
    searched = []
    _silence_everything_but_hypotheses(monkeypatch, searched=searched,
                                       extracted=lambda silo, url: ExtractionOutput())
    _activate(monkeypatch, [_candidate("Molly Chakraborty")])

    expansion._process_person(db, "Prantik Chakraborty", 0, {})

    assert searched == ['"Prantik Chakraborty" "Molly Chakraborty"']


def test_phase_0e_keeps_only_edges_about_the_hypothesised_entity(db, monkeypatch):
    """A page found by '"X" "Y"' names plenty of other people. Harvesting them
    here would let one guess about Y seed the graph with everyone who happens
    to share Y's page -- they are found by whichever query genuinely surfaces
    them, with evidence of their own."""
    def extracted(silo, url):
        return ExtractionOutput(edges=[
            _edge("Prantik Chakraborty", "Molly Chakraborty"),
            _edge("Prantik Chakraborty", "Unrelated Bystander"),
            _edge("Prantik Chakraborty", "Some Sponsor Inc", kind="org"),
        ])

    _silence_everything_but_hypotheses(monkeypatch, extracted=extracted)
    _activate(monkeypatch, [_candidate("Molly Chakraborty")])

    expansion._process_person(db, "Prantik Chakraborty", 0, {})

    stored = {row.canonical_name for row in db.execute(select(Person)).scalars()}
    assert "Molly Chakraborty" in stored
    assert "Unrelated Bystander" not in stored
    assert "Some Sponsor Inc" not in stored


def test_phase_0e_extracts_under_the_neutral_hypothesis_silo(db, monkeypatch):
    """Predicting a connection must not make its confirmation cheaper to
    believe: the multiplier is 1.0, the same bar as an unpredicted edge."""
    silos = []

    def extracted(silo, url):
        silos.append(silo)
        return ExtractionOutput()

    _silence_everything_but_hypotheses(monkeypatch, extracted=extracted)
    _activate(monkeypatch, [_candidate("Molly Chakraborty")])

    expansion._process_person(db, "Prantik Chakraborty", 0, {})

    assert silos == [HYPOTHESIS_SILO]
    assert HYPOTHESIS_SILO.confidence_multiplier == 1.0


def test_phase_0e_does_not_write_the_predicted_relationship_onto_the_edge(db, monkeypatch):
    """The model expected a cofounder tie; the evidence says nothing of the
    sort. What lands is what the page supports."""
    _silence_everything_but_hypotheses(
        monkeypatch,
        extracted=lambda silo, url: ExtractionOutput(edges=[
            _edge("Prantik Chakraborty", "Molly Chakraborty", relationship="unknown")]))
    _activate(monkeypatch, [_candidate("Molly Chakraborty", relationship="cofounder")])

    expansion._process_person(db, "Prantik Chakraborty", 0, {})

    edge = db.execute(select(RelationshipEdge)).scalars().first()
    assert edge is not None
    assert edge.relationship_type == "unknown"


def test_phase_0e_records_what_was_guessed_and_what_came_of_it(db, monkeypatch):
    def extracted(silo, url):
        return ExtractionOutput(edges=[_edge("Prantik Chakraborty", "Molly Chakraborty")])

    _silence_everything_but_hypotheses(monkeypatch, extracted=extracted)
    _activate(monkeypatch, [_candidate("Molly Chakraborty"),
                            _candidate("Never Mentioned")])

    expansion._process_person(db, "Prantik Chakraborty", 0, {},
                              target_person_name="Larry Ellison")

    subject = db.execute(
        select(Person).where(Person.norm_name == person_norm_key("Prantik Chakraborty"))
    ).scalar_one()
    record = subject.meta["connection_hypotheses"]
    assert record["target"] == "Larry Ellison"
    outcomes = {item["name"]: item["outcome"] for item in record["items"]}
    assert outcomes == {"Molly Chakraborty": "confirmed",
                        "Never Mentioned": "unsupported"}


def test_phase_0e_pays_for_its_searches_out_of_the_generic_allowance(db, monkeypatch):
    """The stage SHIFTS spending rather than adding it, and the coverage record
    reports the queries that were given up -- so a later walk that wants the
    trimmed silo still runs it, instead of finding it claimed as covered."""
    _silence_everything_but_hypotheses(
        monkeypatch, extracted=lambda silo, url: ExtractionOutput())
    _activate(monkeypatch, [_candidate("A Candidate"), _candidate("B Candidate")])

    expansion._process_person(db, "Prantik Chakraborty", 0, {})

    subject = db.execute(
        select(Person).where(Person.norm_name == person_norm_key("Prantik Chakraborty"))
    ).scalar_one()
    covered = subject.meta[expansion._COVERAGE_KEY]["silos"]
    assert sum(covered.values()) == sum(query_budget(None).values()) - 2


def test_phase_0e_can_be_told_to_spend_on_top_instead(db, monkeypatch):
    _silence_everything_but_hypotheses(
        monkeypatch, extracted=lambda silo, url: ExtractionOutput())
    _activate(monkeypatch, [_candidate("A Candidate")])
    monkeypatch.setattr(config, "NODE_HYPOTHESIS_TRADE_BUDGET", False)

    expansion._process_person(db, "Prantik Chakraborty", 0, {})

    subject = db.execute(
        select(Person).where(Person.norm_name == person_norm_key("Prantik Chakraborty"))
    ).scalar_one()
    covered = subject.meta[expansion._COVERAGE_KEY]["silos"]
    assert sum(covered.values()) == sum(query_budget(None).values())


def test_phase_0e_is_inert_when_the_stage_is_off(db, monkeypatch):
    """No key, no knob, no searches, and the node's full generic allowance --
    exactly the behaviour that existed before this stage."""
    searched = []
    _silence_everything_but_hypotheses(monkeypatch, searched=searched,
                                       extracted=lambda silo, url: ExtractionOutput())
    monkeypatch.setattr(expansion.connection_hypothesis, "is_active", lambda: False)

    expansion._process_person(db, "Prantik Chakraborty", 0, {})

    assert searched == []
    subject = db.execute(
        select(Person).where(Person.norm_name == person_norm_key("Prantik Chakraborty"))
    ).scalar_one()
    assert "connection_hypotheses" not in (subject.meta or {})
    covered = subject.meta[expansion._COVERAGE_KEY]["silos"]
    assert sum(covered.values()) == sum(query_budget(None).values())


def test_phase_0e_is_skipped_for_an_organization_seed(db, monkeypatch):
    """_process_person also runs for org seeds (network/org_discovery). The
    prompt is person-shaped; nothing here applies to a company."""
    searched = []
    _silence_everything_but_hypotheses(monkeypatch, searched=searched,
                                       extracted=lambda silo, url: ExtractionOutput())
    _activate(monkeypatch, [_candidate("Somebody Else")])

    expansion._process_person(db, "Trinamix Inc", 0, {}, is_person=False)

    assert searched == []


# ---------------------------------------------------------------------------
# expansion._hypothesis_facts -- what the model is allowed to reason from
# ---------------------------------------------------------------------------
def test_hypothesis_facts_uses_this_runs_findings_before_the_stored_graph(db, monkeypatch):
    from app.graph import builder

    subject = builder.get_or_create_person(db, "Prantik Chakraborty")
    db.commit()
    facts = expansion._hypothesis_facts(
        db, subject, "Trinamix", {"summary": "A sales executive.", "wikidata_text": ""},
        [_edge("Prantik Chakraborty", "Molly Chakraborty", conf=0.9)])

    assert facts[0] == "context given by the operator: Trinamix"
    assert "Wikipedia: A sales executive." in facts
    assert any("Molly Chakraborty" in f for f in facts)


def test_hypothesis_facts_falls_back_to_what_earlier_runs_persisted(db, monkeypatch):
    """A node someone else expanded last week has no fresh enrichment but often
    the richest record in the graph -- and both edge orientations count, since
    which side is person_a is an accident of who was expanded first."""
    from app.graph import builder

    subject = builder.get_or_create_person(db, "Prantik Chakraborty")
    ahead = builder.get_or_create_person(db, "Earlier Neighbor")
    behind = builder.get_or_create_person(db, "Reverse Neighbor")
    db.add(RelationshipEdge(person_a_id=subject.id, person_b_id=ahead.id,
                            relationship_type="coworker", confidence_raw=0.7))
    db.add(RelationshipEdge(person_a_id=behind.id, person_b_id=subject.id,
                            relationship_type="board_member", confidence_raw=0.6))
    db.commit()

    facts = expansion._hypothesis_facts(db, subject, "", None, [])

    assert any("Earlier Neighbor" in f for f in facts)
    assert any("Reverse Neighbor" in f for f in facts)


def test_hypothesis_facts_respects_the_cap(db, monkeypatch):
    from app.graph import builder

    monkeypatch.setattr(config, "NODE_HYPOTHESIS_FACTS", 3)
    subject = builder.get_or_create_person(db, "Prantik Chakraborty")
    db.commit()
    edges = [_edge("Prantik Chakraborty", f"Person Number{i}") for i in range(10)]

    assert len(expansion._hypothesis_facts(db, subject, "", None, edges)) == 3
