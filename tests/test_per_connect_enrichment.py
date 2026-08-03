"""Enrichment initialized per connect, rather than once, globally.

The old shape was a single target-agnostic batch whose verdicts were permanent:
`Person.processed` froze a node after one expansion, and the contact ranking had
no notion of who the operator was trying to reach. So the first walk to touch a
node decided what that node would ever be asked, and a later /connect toward a
completely different target inherited that decision.

These tests pin the four things that changed:

  1. reuse is conditional on per-silo COVERAGE, not the processed flag;
  2. ranking conditions on a BridgeTarget;
  3. /connect expands the operator's own contacts as a third front;
  4. re-running any of it converges instead of duplicating.

Nothing here touches the network: expansion is faked, and the ranking/coverage
layers are pure by construction.
"""
from sqlalchemy import or_, select

from app import config
from app.extraction.schemas import EdgeSignals, ExtractedEdge
from app.graph import builder
from app.models import Person, RelationshipEdge
from app.models import RelationshipEdge
from app.network.ingest import ingest_rows
from app.network.ranking import BridgeTarget, score_contacts
from app.network.silo_weights import merge_coverage, uncovered_budget


def _rows(*contacts):
    return [{
        "Name": c.get("name", ""),
        "Company": c.get("company", ""),
        "Position": c.get("title", ""),
        "School": c.get("school", ""),
        "Email Address": c.get("email", ""),
        "Url": c.get("url", ""),
    } for c in contacts]


def _order(scored):
    return [c.display_name for c in scored if c.skip_reason is None]


# --- 1. coverage algebra ----------------------------------------------------

def test_coverage_merges_by_max_not_by_sum():
    """render_queries is deterministic, so running a silo twice asks the same
    questions twice — summing would claim coverage that was never bought."""
    assert merge_coverage({"news": 4}, {"news": 4}) == {"news": 4}
    assert merge_coverage({"news": 2}, {"news": 4}) == {"news": 4}
    assert merge_coverage({"news": 4}, {"news": 2}) == {"news": 4}


def test_coverage_records_silos_never_previously_asked():
    covered = merge_coverage({"company": 4}, {"news": 2})
    assert covered == {"company": 4, "news": 2}


def test_uncovered_reports_the_full_allowance_not_the_difference():
    """Queries are rendered as a prefix, so reaching query 4 means re-rendering
    1..4. The repeats are served from the provider cache."""
    assert uncovered_budget({"news": 4}, {"news": 2}) == {"news": 4}


def test_nothing_is_uncovered_when_the_budget_is_already_satisfied():
    assert uncovered_budget({"news": 2}, {"news": 4}) == {}
    assert uncovered_budget({"news": 2}, {"news": 2}) == {}


def test_a_node_with_no_coverage_record_is_treated_as_covered(db):
    """The compatibility rule that keeps a warm shared graph from being
    re-searched wholesale on the first walk after deploy."""
    from app.graph.expansion import _residual_weights

    person = builder.get_or_create_person(db, "Ada Lovelace")
    person.processed = 1
    db.commit()
    assert _residual_weights(person, "", {"news": 1.0}, False) is None


def test_a_node_is_re_searched_for_a_silo_it_was_never_asked(db):
    """The actual freeze bug: expanded under company-only weights, a later walk
    asking the publications question must be able to ask it."""
    from app.graph.expansion import _COVERAGE_KEY, _residual_weights

    person = builder.get_or_create_person(db, "Ada Lovelace")
    person.processed = 1
    person.meta = {_COVERAGE_KEY: {"context": "", "silos": {"company": 4}}}
    db.commit()

    residual = _residual_weights(person, "", {"publications": 1.0}, False)
    assert residual is not None
    assert set(residual) == {"publications"}   # only the NEW question is paid for


def test_a_different_disambiguation_context_covers_nothing(db):
    """Queries carry the context, so "Acme"-qualified questions never answered
    the unqualified ones — plausibly not even about the same person."""
    from app.graph.expansion import _COVERAGE_KEY, _residual_weights

    person = builder.get_or_create_person(db, "Ada Lovelace")
    person.processed = 1
    person.meta = {_COVERAGE_KEY: {"context": "Acme", "silos": {"company": 4}}}
    db.commit()

    residual = _residual_weights(person, "Bletchley", {"company": 1.0}, False)
    assert residual is not None and "company" in residual


def test_coverage_reuse_can_be_switched_off(db, monkeypatch):
    """The escape hatch back to freeze-on-first-expansion."""
    from app.graph.expansion import _COVERAGE_KEY, _residual_weights

    person = builder.get_or_create_person(db, "Ada Lovelace")
    person.processed = 1
    person.meta = {_COVERAGE_KEY: {"context": "", "silos": {"company": 4}}}
    db.commit()

    monkeypatch.setattr(config, "EXPAND_COVERAGE_REUSE", False)
    assert _residual_weights(person, "", {"publications": 1.0}, False) is None


def test_professional_only_does_not_force_a_re_search_for_dropped_silos(db):
    """Wanting the family silo is no reason to re-search a node on a walk that
    would not have issued family queries at all."""
    from app.graph.expansion import _COVERAGE_KEY, _residual_weights

    person = builder.get_or_create_person(db, "Ada Lovelace")
    person.processed = 1
    person.meta = {_COVERAGE_KEY: {"context": "", "silos": {"company": 4}}}
    db.commit()

    residual = _residual_weights(person, "", {"family": 1.0}, True)
    assert residual is None


# --- 2. target-conditioned ranking ------------------------------------------

def test_ranking_without_a_target_is_unchanged(db):
    """The cold-start batch still asks its own question."""
    ingest_rows(db, _rows(
        {"name": "Aa Junior", "company": "Smallco", "title": "Analyst"},
        {"name": "Bb Founder", "company": "Otherco", "title": "Founder"},
    ))
    # seniority is a footprint proxy, so the founder leads on an untargeted rank
    assert _order(score_contacts(db))[0] == "Bb Founder"


def test_a_contact_at_the_targets_employer_outranks_a_more_senior_stranger(db):
    """The core reversal: who bridges to THIS person, not who is most notable."""
    ingest_rows(db, _rows(
        {"name": "Aa Junior", "company": "Target Corp", "title": "Analyst"},
        {"name": "Bb Founder", "company": "Otherco", "title": "Founder"},
    ))
    target = BridgeTarget(name="Zz Stranger", companies=["Target Corp"])
    assert _order(score_contacts(db, target=target))[0] == "Aa Junior"


def test_the_bridge_reason_is_reported(db):
    ingest_rows(db, _rows({"name": "Aa Junior", "company": "Target Corp"}))
    scored = score_contacts(db, target=BridgeTarget(name="Zz",
                                                    companies=["Target Corp"]))
    assert "shared_employer" in scored[0].bridge_reasons


def test_a_shared_school_with_the_target_counts_but_less_than_an_employer(db):
    ingest_rows(db, _rows(
        {"name": "Aa Alum", "company": "Otherco", "school": "Target University"},
        {"name": "Bb Colleague", "company": "Target Corp"},
    ))
    target = BridgeTarget(name="Zz", companies=["Target Corp"],
                          schools=["Target University"])
    order = _order(score_contacts(db, target=target))
    assert order.index("Bb Colleague") < order.index("Aa Alum")


def test_the_targets_employer_is_exempt_from_the_coverage_decay(db):
    """The decay stops one company monopolising a breadth-seeking budget. Aimed
    at a person, that company IS the destination — a second and third route into
    it is the most valuable thing left to buy, so it must not be damped."""
    ingest_rows(db, _rows(
        {"name": "Aa Colleague", "company": "Target Corp"},
        {"name": "Bb Colleague", "company": "Target Corp"},
        {"name": "Cc Colleague", "company": "Target Corp"},
    ))
    target = BridgeTarget(name="Zz", companies=["Target Corp"])
    scores = [c.score for c in score_contacts(db, target=target)
              if c.skip_reason is None]
    assert len(scores) == 3
    assert len(set(scores)) == 1        # undamped: all three score identically

    # …while an unrelated employer still decays normally.
    undamped = [c.score for c in score_contacts(db) if c.skip_reason is None]
    assert undamped[0] > undamped[-1]


def test_a_generic_target_employer_does_not_bridge_everyone(db):
    """"Self-Employed" is a job status, not a building to meet someone in."""
    ingest_rows(db, _rows({"name": "Aa Contact", "company": "Self-Employed"}))
    target = BridgeTarget(name="Zz", companies=["Self-Employed"])
    scored = score_contacts(db, target=target)
    assert "shared_employer" not in scored[0].bridge_reasons


def test_an_empty_target_scores_exactly_like_no_target(db):
    """/connect toward someone the graph knows nothing about must not silently
    reorder the plan on the strength of no evidence."""
    ingest_rows(db, _rows(
        {"name": "Aa Junior", "company": "Smallco", "title": "Analyst"},
        {"name": "Bb Founder", "company": "Otherco", "title": "Founder"},
    ))
    assert (_order(score_contacts(db, target=BridgeTarget(name="Zz")))
            == _order(score_contacts(db)))


# --- 3. the /connect bridge front -------------------------------------------

def test_connect_expands_contacts_ranked_toward_the_target(db, monkeypatch):
    from app.graph import connect as C

    ingest_rows(db, _rows(
        {"name": "Aa Colleague", "company": "Target Corp"},
        {"name": "Bb Stranger", "company": "Unrelated Inc"},
    ))
    expanded = []

    def fake_expand_graph(worker_db, name, depth, **kwargs):
        expanded.append(name)
        return {}

    monkeypatch.setattr(C, "expand_graph", fake_expand_graph)
    monkeypatch.setattr(config, "CONNECT_BRIDGE_CONTACTS", 1)

    C._expand_both_concurrently(db, "Alpha", "Beta", 2, 2, set(), None,
                                "", "Target Corp")

    assert "Alpha" in expanded and "Beta" in expanded      # both endpoints
    assert "Aa Colleague" in expanded                      # the ranked bridge
    assert "Bb Stranger" not in expanded                   # not worth a slot


def test_bridge_contacts_get_the_alpha_treatment(db, monkeypatch):
    """A bridge contact is a normal person searched toward a known target —
    the case Alpha exists for. Phase 4e needs BOTH the target name and the
    enhanced flag, so passing only the target leaves the strategy step inert."""
    from app.graph import connect as C

    ingest_rows(db, _rows({"name": "Aa Colleague", "company": "Target Corp"}))
    seen = {}

    def fake_expand_graph(worker_db, name, depth, **kwargs):
        seen[name] = kwargs
        return {}

    monkeypatch.setattr(C, "expand_graph", fake_expand_graph)
    monkeypatch.setattr(config, "CONNECT_BRIDGE_CONTACTS", 1)

    C._expand_both_concurrently(db, "Alpha", "Beta", 2, 2, set(), None,
                                "", "Target Corp")

    kw = seen["Aa Colleague"]
    assert kw["enhanced_professional_search"] is True
    assert kw["target_person_name"] == "Beta"     # 4e needs both
    assert kw["target_context"] == "Target Corp"


def test_symmetric_endpoints_still_decide_alpha_by_notability(db, monkeypatch):
    """The bridge front's unconditional flag must not leak into the endpoint
    walk — with neither endpoint notable, neither side gets enhanced search.

    Stubs notable_set rather than trusting the names: this used to pass real
    strings through to a live Wikipedia lookup, and "Alpha" and "Beta" are both
    genuine articles, so the endpoints came back notable and the assertion
    below started failing for a reason that had nothing to do with the bridge
    front. See test_alpha_gate.py for the notability behavior itself.
    """
    from app.graph import connect as C

    seen = {}
    monkeypatch.setattr(C.ORCH, "notable_set", lambda names: set())
    monkeypatch.setattr(C, "expand_graph",
                        lambda wdb, name, depth, **kw: seen.setdefault(name, kw) or {})
    monkeypatch.setattr(config, "CONNECT_BRIDGE_CONTACTS", 0)

    C._expand_both_concurrently(db, "Alpha", "Beta", 2, 2, set(), None, "", "")
    assert seen["Alpha"]["enhanced_professional_search"] is False
    assert seen["Beta"]["enhanced_professional_search"] is False


def test_the_bridge_front_can_be_switched_off(db, monkeypatch):
    from app.graph import connect as C

    ingest_rows(db, _rows({"name": "Aa Colleague", "company": "Target Corp"}))
    expanded = []
    monkeypatch.setattr(C, "expand_graph",
                        lambda wdb, name, depth, **kw: expanded.append(name) or {})
    monkeypatch.setattr(config, "CONNECT_BRIDGE_CONTACTS", 0)

    C._expand_both_concurrently(db, "Alpha", "Beta", 2, 2, set(), None,
                                "", "Target Corp")
    assert expanded == ["Alpha", "Beta"] or expanded == ["Beta", "Alpha"]


def test_a_failing_bridge_contact_does_not_fail_the_connect(db, monkeypatch):
    """The front is speculative; an endpoint expansion failing is fatal, a
    contact's is not."""
    from app.graph import connect as C

    ingest_rows(db, _rows({"name": "Aa Colleague", "company": "Target Corp"}))

    def fake_expand_graph(worker_db, name, depth, **kwargs):
        if name == "Aa Colleague":
            raise RuntimeError("provider exploded")
        return {"visited_by_hop": {}}

    monkeypatch.setattr(C, "expand_graph", fake_expand_graph)
    monkeypatch.setattr(config, "CONNECT_BRIDGE_CONTACTS", 1)

    stats = C._expand_both_concurrently(db, "Alpha", "Beta", 2, 2, set(), None,
                                        "", "Target Corp")
    assert "a" in stats and "b" in stats     # the connect still produced a graph


def test_bridge_contacts_stop_once_a_route_is_found(db, monkeypatch):
    """They expand in rank order and are re-checked between contacts, so a
    route found early means the rest are never paid for."""
    from app.graph import connect as C

    ingest_rows(db, _rows(
        {"name": "Aa Colleague", "company": "Target Corp"},
        {"name": "Bb Colleague", "company": "Target Corp"},
        {"name": "Cc Colleague", "company": "Target Corp"},
    ))
    expanded = []
    monkeypatch.setattr(C, "expand_graph",
                        lambda wdb, name, depth, **kw: expanded.append(name) or {})

    class _Session:
        def __init__(self): pass
        def close(self): pass

    contacts = [c for c in score_contacts(
        db, target=BridgeTarget(name="Zz", companies=["Target Corp"]))
        if c.skip_reason is None]

    # "a route already exists" from the very first check
    C._expand_bridge_contacts(_Session, contacts, set(), None, "Beta", "",
                              should_stop=lambda _s: True)
    assert expanded == []


# --- 3b. step 1: the origin's own initial enrichment -------------------------

def test_the_origin_gets_its_first_degree_without_any_import_side_effect(db):
    """The foundation used to exist only as a side effect of importing a CSV.
    A connect whose operator imported elsewhere pathfound over a graph missing
    their own first degree."""
    from app.graph.connect import _ensure_origin_enriched

    # owner_name on import is what records WHOSE contacts these are; the
    # bridge only asserts first-degree ties for rows that say so (see
    # network.ingest.backfill_graph_edges).
    ingest_rows(db, _rows(
        {"name": "Aa Colleague", "company": "Origin Corp"},
        {"name": "Bb Colleague", "company": "Origin Corp"},
    ), owner_name="Oo Operator")
    counts = _ensure_origin_enriched(db, "Oo Operator", owner_name="Oo Operator")
    db.commit()

    assert counts["linkedin_1st_edges"] == 2
    assert counts["wave0"]["coworker_edges"] == 1     # the two colleagues, one pair

    origin = db.execute(select(Person).where(
        Person.norm_name == "oo operator")).scalar_one()
    first_degree = db.execute(select(RelationshipEdge).where(
        or_(RelationshipEdge.person_a_id == origin.id,
            RelationshipEdge.person_b_id == origin.id),
        RelationshipEdge.relationship_type == "linkedin_1st")).scalars().all()
    assert len(first_degree) == 2


def test_running_origin_enrichment_twice_adds_nothing(db):
    """It runs on EVERY connect, so convergence is the property that makes that
    affordable rather than a slow-growing duplication."""
    from app.graph.connect import _ensure_origin_enriched

    ingest_rows(db, _rows(
        {"name": "Aa Colleague", "company": "Origin Corp"},
        {"name": "Bb Colleague", "company": "Origin Corp"},
    ))
    _ensure_origin_enriched(db, "Oo Operator", owner_name="Oo Operator")
    db.commit()
    before = len(db.execute(select(RelationshipEdge)).scalars().all())

    _ensure_origin_enriched(db, "Oo Operator", owner_name="Oo Operator")
    db.commit()
    assert len(db.execute(select(RelationshipEdge)).scalars().all()) == before


def test_origin_enrichment_costs_no_searches(db, monkeypatch):
    """Step 1 runs before a cent is spent — if it ever starts searching, it
    stops being safe to run unconditionally."""
    from app.graph import expansion
    from app.graph.connect import _ensure_origin_enriched

    def _boom(*a, **k):
        raise AssertionError("step 1 must not touch the network")

    monkeypatch.setattr(expansion.ORCH, "search", _boom)
    monkeypatch.setattr(expansion.ORCH, "fetch", _boom)

    ingest_rows(db, _rows({"name": "Aa Colleague", "company": "Origin Corp"}))
    _ensure_origin_enriched(db, "Oo Operator")
    db.commit()


def test_origin_enrichment_can_answer_the_connect_for_free(db):
    """The payoff: after step 1 the target is already reachable from the
    origin, so the paid walk never starts."""
    from app.graph.connect import _ensure_origin_enriched, _route_exists

    ingest_rows(db, _rows(
        {"name": "Aa Colleague", "company": "Origin Corp"},
        {"name": "Zz Target", "company": "Origin Corp"},
    ), owner_name="Oo Operator")
    db.query(RelationshipEdge).delete()   # isolate step 1's own bridging
    db.commit()
    assert not _route_exists(db, "Oo Operator", "Zz Target", 5)

    _ensure_origin_enriched(db, "Oo Operator", owner_name="Oo Operator")
    db.commit()

    assert _route_exists(db, "Oo Operator", "Zz Target", 5)


def test_origin_enrichment_uses_the_operators_saved_profile(db):
    """The origin of a connect IS the operator, so their own employer should
    place them in that cluster — see owner.get_owner_by_name."""
    from app.graph.connect import _ensure_origin_enriched
    from app.network.owner import upsert_owner

    ingest_rows(db, _rows({"name": "Aa Colleague", "company": "Origin Corp"}),
                owner_name="Oo Operator")
    upsert_owner(db, "gid1", name="Oo Operator", company="Origin Corp")
    counts = _ensure_origin_enriched(db, "Oo Operator")
    db.commit()

    # operator + colleague now share the employer cluster, so a coworker tie
    # exists that contacts-only wave 0 could not have produced (one contact).
    assert counts["wave0"]["coworker_edges"] == 1


def test_origin_enrichment_can_be_switched_off(db, monkeypatch):
    from app.graph.connect import _ensure_origin_enriched

    ingest_rows(db, _rows({"name": "Aa Colleague", "company": "Origin Corp"}))
    monkeypatch.setattr(config, "CONNECT_ENRICH_ORIGIN", False)
    counts = _ensure_origin_enriched(db, "Oo Operator")
    assert counts["linkedin_1st_edges"] == 0
    assert db.execute(select(RelationshipEdge)).scalars().all() == []


def test_a_non_operator_origin_gets_no_first_degree_bridge(db):
    """Trace a route from a tagged stranger and your contacts are simply not
    their connections. Writing them anyway invents first-degree ties the
    pathfinder then walks as if real."""
    from app.graph.connect import _ensure_origin_enriched

    ingest_rows(db, _rows(
        {"name": "Aa Colleague", "company": "Origin Corp"},
        {"name": "Bb Colleague", "company": "Origin Corp"},
    ))
    counts = _ensure_origin_enriched(db, "Zz Stranger", owner_name="Oo Operator")
    db.commit()

    assert counts["linkedin_1st_edges"] == 0
    assert db.execute(select(RelationshipEdge).where(
        RelationshipEdge.relationship_type == "linkedin_1st")).scalars().all() == []
    # …but wave 0 still runs: it asserts nothing about the origin.
    assert counts["wave0"]["coworker_edges"] == 1


def test_an_unstated_caller_gets_no_first_degree_bridge(db):
    """No owner_name and no saved profile means "not stated", which must not be
    read as "yes, the origin is me"."""
    from app.graph.connect import _ensure_origin_enriched

    ingest_rows(db, _rows({"name": "Aa Colleague", "company": "Origin Corp"}))
    counts = _ensure_origin_enriched(db, "Oo Operator")
    db.commit()
    assert counts["linkedin_1st_edges"] == 0


def test_a_saved_owner_profile_also_proves_the_origin_is_the_operator(db):
    """The browser need not supply owner_name if the server already knows who
    that name belongs to."""
    from app.graph.connect import _ensure_origin_enriched
    from app.network.owner import upsert_owner

    ingest_rows(db, _rows({"name": "Aa Colleague", "company": "Origin Corp"}),
                owner_name="Oo Operator")
    upsert_owner(db, "gid1", name="Oo Operator", company="Origin Corp")
    counts = _ensure_origin_enriched(db, "Oo Operator")
    db.commit()
    assert counts["linkedin_1st_edges"] == 1


def test_the_identity_match_ignores_case_and_spacing(db):
    from app.graph.connect import _origin_is_operator
    assert _origin_is_operator(db, "Oo Operator", "  oo   operator ")
    assert not _origin_is_operator(db, "Oo Operator", "Someone Else")


def test_the_contact_bridge_can_be_switched_off_without_losing_wave_0(db, monkeypatch):
    """The two halves make different claims. Wave 0 derives ties BETWEEN
    contacts and asserts nothing about the origin, so it survives; the
    linkedin_1st bridge claims the origin knows all of them, so it doesn't."""
    from app.graph.connect import _ensure_origin_enriched

    ingest_rows(db, _rows(
        {"name": "Aa Colleague", "company": "Origin Corp"},
        {"name": "Bb Colleague", "company": "Origin Corp"},
    ))
    monkeypatch.setattr(config, "CONNECT_ORIGIN_BACKFILL", False)
    counts = _ensure_origin_enriched(db, "Oo Operator", owner_name="Oo Operator")
    db.commit()

    assert counts["linkedin_1st_edges"] == 0
    assert counts["wave0"]["coworker_edges"] == 1
    assert db.execute(select(RelationshipEdge).where(
        RelationshipEdge.relationship_type == "linkedin_1st")).scalars().all() == []


def test_a_nameless_origin_is_a_no_op(db):
    from app.graph.connect import _ensure_origin_enriched
    assert _ensure_origin_enriched(db, "   ")["linkedin_1st_edges"] == 0


def test_step_1_short_circuits_the_entire_paid_walk(db, monkeypatch):
    """The payoff, end to end: step 1 builds the answer, the free re-check
    finds it, and neither paid step is ever reached."""
    from app.graph import connect as C

    calls = []
    monkeypatch.setattr(C, "_direct_pair_search",
                        lambda *a, **k: calls.append("direct") or (False, False))
    monkeypatch.setattr(C, "_expand_both_concurrently",
                        lambda *a, **k: calls.append("expand") or {})

    ingest_rows(db, _rows({"name": "Aa Colleague", "company": "Origin Corp"},
                          {"name": "Zz Target", "company": "Origin Corp"}),
                owner_name="Oo Operator")
    db.commit()

    result = C.connect_people(db, "Oo Operator", "Zz Target", depth=2,
                              owner_name="Oo Operator")
    assert result["connected"] is True
    assert calls == []


def test_a_weak_direct_mention_does_not_cancel_the_expansion(db, monkeypatch):
    """A weak co-mention is not a route. Treating it as one cancelled both
    endpoint expansions and the bridge front, then reported "no path" seconds
    later having never expanded the far endpoint — confirmed live on a
    Charlie->Elon trace that left Elon Musk with processed=0."""
    from app.graph import connect as C

    calls = []
    monkeypatch.setattr(C, "_direct_pair_search",
                        lambda *a, **k: (True, False))        # found, NOT confident
    monkeypatch.setattr(C, "_expand_both_concurrently",
                        lambda *a, **k: calls.append("expand") or {})
    monkeypatch.setattr(C, "_adjacency", lambda db: ({}, {}, {}, {}))

    C.connect_people(db, "Oo Operator", "Zz Stranger", depth=2)
    assert calls == ["expand"]


def test_a_confident_direct_mention_still_short_circuits(db, monkeypatch):
    """The optimisation is kept where it was always sound: a typed, non-weak
    direct edge IS the route, so the neighborhood walk buys nothing."""
    from app.graph import connect as C

    calls = []

    def direct_hit(db_, name_a, name_b, *a, **k):
        """Persist the edge the verdict refers to, as the real
        _direct_pair_search does. Returning (True, True) while writing nothing
        asserts a route that does not exist -- exactly the false positive the
        short-circuit now re-checks for with the pathfinder's own rule."""
        from app.models import Person, RelationshipEdge
        from app.utils.names import person_norm_key
        pa = Person(canonical_name=name_a, norm_name=person_norm_key(name_a))
        pb = Person(canonical_name=name_b, norm_name=person_norm_key(name_b))
        db_.add_all([pa, pb])
        db_.flush()
        db_.add(RelationshipEdge(
            person_a_id=pa.id, person_b_id=pb.id, relationship_type="coworker",
            status="strong", confidence_raw=0.8,
            evidence_snippet="Oo and Zz worked together.",
            signals={"sentence_cooccurrence": True}))
        db_.commit()
        return (True, True)

    monkeypatch.setattr(C, "_direct_pair_search", direct_hit)  # found AND confident
    monkeypatch.setattr(C, "_expand_both_concurrently",
                        lambda *a, **k: calls.append("expand") or {})
    monkeypatch.setattr(C, "_adjacency", lambda db: ({}, {}, {}, {}))

    C.connect_people(db, "Oo Operator", "Zz Stranger", depth=2)
    assert calls == []


def test_the_cheap_search_precedes_origin_enrichment(db, monkeypatch):
    """Order matters, and it used to be the other way round.

    The old rationale was that origin enrichment is free, so running it first
    might save a search. It is free of SEARCHES and not of time:
    materialize_contact_cliques resolves one Person per contact in a Python
    loop, ~20 minutes on a 2,153-contact export over an 84ms link, and it ran
    even for an origin unrelated to those contacts. Putting that in front of a
    single search meant a pair the search answers in seconds waited on work
    irrelevant to it -- observed on Sanjay Ghemawat -> Larry Page, where nine
    results for one pair query named the intermediary on every one.

    The trade is deliberate and lopsided: a route through the operator's own
    contacts now pays one or two searches it did not strictly need, and every
    route the searches can answer skips the 20 minutes entirely.
    """
    from app.graph import connect as C

    calls = []
    monkeypatch.setattr(C, "_ensure_origin_enriched",
                        lambda *a, **k: calls.append("origin") or
                        {"linkedin_1st_edges": 0, "wave0": {}})
    monkeypatch.setattr(C, "_direct_pair_search",
                        lambda *a, **k: calls.append("direct") or (False, False))
    monkeypatch.setattr(C, "_expand_both_concurrently",
                        lambda *a, **k: calls.append("expand") or {})
    monkeypatch.setattr(C, "_adjacency", lambda db: ({}, {}, {}, {}))

    C.connect_people(db, "Oo Operator", "Zz Stranger", depth=2)
    # bridge_hypothesis is inactive without a credential (see conftest), so it
    # contributes no call here; its own placement is pinned in
    # tests/test_bridge_hypothesis.py.
    assert calls == ["direct", "origin", "expand"]


# --- 4. re-running converges ------------------------------------------------

def _coworker(db, a_name: str, b_name: str, url: str):
    """Persist "a coworker b" as if extracted from `url`, subject = a."""
    a = builder.get_or_create_person(db, a_name)
    b = builder.get_or_create_person(db, b_name)
    from app.providers.base import SearchResult
    source = builder.save_source(db, SearchResult("t", url, "s", "web"), "q")
    edge = ExtractedEdge(
        person_a=a_name, person_b=b_name, other_kind="person",
        relationship_type="coworker", method="test", evidence_snippet="ev",
        confidence_base=0.6, confidence_adjusted=0.6, source_url=url,
        signals=EdgeSignals(explicit_keyword_match=True),
    )
    return builder.add_edge_from_extraction(db, a, edge, 0, source, b)


def test_the_same_connection_found_from_the_other_side_is_not_a_new_edge(db):
    """The duplication re-enrichment would otherwise cause: expanding A finds
    "A coworker B"; later expanding B finds "B coworker A" off the SAME page.
    One connection, one row."""
    _coworker(db, "Ada Lovelace", "Grace Hopper", "https://example.com/team")
    _coworker(db, "Grace Hopper", "Ada Lovelace", "https://example.com/team")
    db.commit()

    edges = db.execute(select(RelationshipEdge).where(
        RelationshipEdge.relationship_type == "coworker")).scalars().all()
    assert len(edges) == 1


def test_re_running_the_identical_extraction_is_idempotent(db):
    _coworker(db, "Ada Lovelace", "Grace Hopper", "https://example.com/team")
    _coworker(db, "Ada Lovelace", "Grace Hopper", "https://example.com/team")
    db.commit()
    assert len(db.execute(select(RelationshipEdge)).scalars().all()) == 1


def test_a_directional_relationship_keeps_both_orientations(db):
    """"A interviewed B" and "B interviewed A" are different facts — collapsing
    them would destroy one, not dedupe it."""
    for a, b in (("Ada Lovelace", "Grace Hopper"), ("Grace Hopper", "Ada Lovelace")):
        subject = builder.get_or_create_person(db, a)
        other = builder.get_or_create_person(db, b)
        from app.providers.base import SearchResult
        source = builder.save_source(
            db, SearchResult("t", "https://example.com/i", "s", "web"), "q")
        builder.add_edge_from_extraction(db, subject, ExtractedEdge(
            person_a=a, person_b=b, other_kind="person",
            relationship_type="interview", method="m", evidence_snippet="e",
            confidence_base=0.6, confidence_adjusted=0.6,
            source_url="https://example.com/i", signals=EdgeSignals(),
        ), 0, source, other)
    db.commit()
    assert len(db.execute(select(RelationshipEdge)).scalars().all()) == 2


def test_a_different_source_is_a_separate_piece_of_evidence(db):
    """Dedup is per (pair, type, SOURCE): two pages independently asserting the
    same tie are two pieces of evidence, and the graph keeps both."""
    _coworker(db, "Ada Lovelace", "Grace Hopper", "https://example.com/one")
    _coworker(db, "Grace Hopper", "Ada Lovelace", "https://example.com/two")
    db.commit()
    assert len(db.execute(select(RelationshipEdge)).scalars().all()) == 2


def test_a_pre_existing_mirrored_pair_converges_instead_of_crashing(db):
    """Graphs written before symmetric dedup already hold both orientations.
    Writing that tie again must settle onto one of them, not raise."""
    a = builder.get_or_create_person(db, "Ada Lovelace")
    b = builder.get_or_create_person(db, "Grace Hopper")
    from app.providers.base import SearchResult
    source = builder.save_source(
        db, SearchResult("t", "https://example.com/team", "s", "web"), "q")
    for subject, other in ((a, b), (b, a)):
        db.add(RelationshipEdge(
            person_a_id=subject.id, person_b_id=other.id,
            relationship_type="coworker", source_id=source.id,
            confidence_base=0.5, confidence_raw=0.5, status="candidate"))
    db.commit()

    _coworker(db, "Ada Lovelace", "Grace Hopper", "https://example.com/team")
    db.commit()
    # still the two legacy rows — no third one piled on top
    assert len(db.execute(select(RelationshipEdge)).scalars().all()) == 2
