"""Alpha piece 2: search strategy (extraction/search_strategy.py +
expansion.py phase 4e + connect.py's target-name threading) -- "run
reasoning to identify best type of search."

Given node_profiler's (piece 1) grounded org profile and who the walk is
ultimately trying to reach, decide which of a FIXED set of query angles is
worth searching in addition to the normal broad silo search. The model picks
an angle from an enum; it never writes query text -- config.
STRATEGY_ANGLE_QUERIES is the actual, fully deterministic query surface, so
a wrong pick costs a couple of irrelevant queries, never an ungrounded
search direction. This mirrors node_profiler's own containment principle
one level up.
"""
from app import config
from app.extraction import node_profiler, search_strategy
from app.graph import connect as C
from app.graph import expansion
from app.models import Organization
from app.providers.base import SearchResult


# ---------------------------------------------------------------------------
# search_strategy.decide_angle -- unit tests, Claude call mocked
# ---------------------------------------------------------------------------
_PROFILE = {"size_tier": "large", "industry": "Oracle ERP consulting",
           "summary": "An Oracle partner.", "grounded": True}


def test_decide_angle_returns_none_when_inactive(monkeypatch):
    monkeypatch.setattr(search_strategy, "is_active", lambda: False)
    assert search_strategy.decide_angle("Prantik", "Trinamix", _PROFILE, "Larry Ellison") is None


def test_decide_angle_accepts_a_valid_pick(monkeypatch):
    monkeypatch.setattr(search_strategy, "is_active", lambda: True)
    monkeypatch.setattr(search_strategy, "call_json", lambda *a, **k: {
        "angle": "current_employer_leadership", "why": "Trinamix is an Oracle partner.",
    })
    decision = search_strategy.decide_angle("Prantik", "Trinamix", _PROFILE, "Larry Ellison")
    assert decision == {"angle": "current_employer_leadership",
                        "why": "Trinamix is an Oracle partner."}


def test_decide_angle_normalizes_an_out_of_vocabulary_angle(monkeypatch):
    monkeypatch.setattr(search_strategy, "is_active", lambda: True)
    monkeypatch.setattr(search_strategy, "call_json", lambda *a, **k: {
        "angle": "something_made_up", "why": "x",
    })
    decision = search_strategy.decide_angle("Prantik", "Trinamix", _PROFILE, "Larry Ellison")
    assert decision["angle"] == "generic"


def test_decide_angle_returns_none_when_call_fails(monkeypatch):
    monkeypatch.setattr(search_strategy, "is_active", lambda: True)
    monkeypatch.setattr(search_strategy, "call_json", lambda *a, **k: None)
    assert search_strategy.decide_angle("Prantik", "Trinamix", _PROFILE, "Larry Ellison") is None


def test_decide_angle_includes_target_context_when_given(monkeypatch):
    captured = {}

    def fake_call_json(prompt, schema, model, max_tokens=256):
        captured["prompt"] = prompt
        return {"angle": "generic", "why": "x"}

    monkeypatch.setattr(search_strategy, "is_active", lambda: True)
    monkeypatch.setattr(search_strategy, "call_json", fake_call_json)
    search_strategy.decide_angle("Prantik", "Trinamix", _PROFILE, "Larry Ellison", "Oracle")
    assert "Oracle" in captured["prompt"]


# ---------------------------------------------------------------------------
# connect._expand_both_concurrently -- threads the OTHER endpoint as target
# ---------------------------------------------------------------------------
def test_run_passes_the_other_endpoints_name_as_target(db, monkeypatch):
    captured = {}

    def fake_expand_graph(worker_db, name, side_depth, **kwargs):
        captured[name] = (kwargs.get("target_person_name"), kwargs.get("target_context"))

    monkeypatch.setattr(C, "expand_graph", fake_expand_graph)

    C._expand_both_concurrently(
        db=db, name_a="Obscure Person", name_b="Famous Person",
        depth_a=3, depth_b=C.SHALLOW_FAMOUS_DEPTH,
        protected=set(), progress=None,
        context_a="Some Co", context_b="Some Corp",
    )

    assert captured["Obscure Person"] == ("Famous Person", "Some Corp")
    assert captured["Famous Person"] == ("Obscure Person", "Some Co")


# ---------------------------------------------------------------------------
# expansion._process_person's phase 4e, end to end
# ---------------------------------------------------------------------------
def _silence_everything_but_strategy(monkeypatch, search_results=None, fetched_text=""):
    monkeypatch.setattr(expansion.ORCH, "enrich_person", lambda name: None)
    monkeypatch.setattr(expansion.ORCH, "officer_enrichment", lambda name: {"officers_text": ""})
    monkeypatch.setattr(expansion.ORCH, "edgar_enrichment", lambda name: {"edgar_text": ""})
    monkeypatch.setattr(expansion.ORCH, "firm_enrichment", lambda name: {"firms": []})
    # Phase 4f (org directory) -- stubbed off here so it can't issue searches
    # of its own; it has its own dedicated tests.
    monkeypatch.setattr(expansion.ORCH, "directory_enrichment",
                        lambda org, industry="", size_tier="": {
                            "org": org, "url": "", "members": [], "overflow": False})
    monkeypatch.setattr(expansion.ORCH, "coauthors_enrichment", lambda name: {
        "coauthors": [], "coauthors_text": "", "identity_text": "",
    })
    # Unrelated to what this file tests -- disable so it doesn't make a real
    # network call in a test.
    monkeypatch.setattr(expansion.coauthor_plausibility, "is_active", lambda: False)
    monkeypatch.setattr(expansion.config, "PODCASTS_ENABLED", False)
    monkeypatch.setattr(expansion.ORCH, "dedup", lambda pairs: ([], {}))
    monkeypatch.setattr(expansion, "_repeat_candidates", lambda edges: [])
    # node_profiler.is_active() must stay True -- it gates phase 4d's OUTER
    # lookup of org_row, which phase 4e depends on to even find the org.
    # Phase 4d's own search/fetch calls are silenced instead by each test
    # pre-seeding a cached, grounded profile on the Organization row, which
    # trips 4d's "already profiled" short-circuit before it searches anything.
    monkeypatch.setattr(node_profiler, "is_active", lambda: True)

    class _Page:
        def __init__(self, content):
            self.content = content

    monkeypatch.setattr(expansion.ORCH, "search",
                        lambda query, is_person=True: search_results or [])
    monkeypatch.setattr(expansion.ORCH, "fetch", lambda url: _Page(fetched_text))


def test_phase_4e_fires_the_chosen_angles_queries_and_records_the_decision(db, monkeypatch):
    org = Organization(name="Trinamix", norm_name="trinamix",
                       meta={"profile": {"v": config.NODE_PROFILE_VERSION, "size_tier": "large", "industry": "Oracle ERP consulting",
                                         "summary": "An Oracle partner.", "grounded": True}})
    db.add(org)
    db.commit()

    # board_or_advisory, not current_employer_leadership: the latter now maps
    # to NO queries on purpose -- a company roster is a structural assertion
    # and moved to expansion's phase 4f (providers/directory.py), because
    # feeding it to the prose extractor either dropped everything or wired
    # the whole exec roster to the subject. See test_company_directories.py.
    # This test is about the angle->queries mechanism, so it uses an angle
    # that still has queries.
    _silence_everything_but_strategy(
        monkeypatch,
        search_results=[SearchResult("Prantik Chakraborty board", "https://example.com/board",
                                     "snippet", "serper")],
        fetched_text="Prantik Chakraborty serves as a board member alongside Dana Whitfield.",
    )
    monkeypatch.setattr(expansion, "_best_org_affiliation_edge",
                        lambda edges: _fake_org_edge("Trinamix"))
    monkeypatch.setattr(search_strategy, "is_active", lambda: True)
    monkeypatch.setattr(search_strategy, "decide_angle", lambda *a, **k: {
        "angle": "board_or_advisory", "why": "His board seats are the better angle.",
    })

    from app.graph import builder
    from app.models import RelationshipEdge

    expansion._process_person(db, "Prantik Chakraborty", 0, {},
                              enhanced_professional_search=True,
                              target_person_name="Larry Ellison", target_context="Oracle")

    subject = builder.get_or_create_person(db, "Prantik Chakraborty")
    assert (subject.meta or {}).get("strategy") == {
        "angle": "board_or_advisory", "why": "His board seats are the better angle.",
    }
    # the fake search result's page text should have produced at least one
    # candidate edge that made it through extraction/persist
    assert db.query(RelationshipEdge).filter(RelationshipEdge.person_a_id == subject.id).count() > 0


def test_current_employer_leadership_maps_to_no_prose_queries():
    """The angle is still selectable -- it just no longer issues prose
    queries. The org roster it used to chase is handled structurally by
    phase 4f now; firing '"{org}" leadership team' into the prose extractor
    is the failure that motivated the whole change."""
    assert config.STRATEGY_ANGLE_QUERIES["current_employer_leadership"] == []
    assert config.STRATEGY_ANGLE_QUERIES["industry_peers"] == []
    # still a valid choice for the model, and still recorded on the subject
    assert "current_employer_leadership" in config.STRATEGY_ANGLE_QUERIES


def test_phase_4e_fires_no_extra_queries_for_the_generic_angle(db, monkeypatch):
    org = Organization(name="Trinamix", norm_name="trinamix",
                       meta={"profile": {"v": config.NODE_PROFILE_VERSION, "size_tier": "large", "industry": "Oracle ERP consulting",
                                         "summary": "An Oracle partner.", "grounded": True}})
    db.add(org)
    db.commit()

    search_calls = []
    _silence_everything_but_strategy(monkeypatch)
    monkeypatch.setattr(expansion.ORCH, "search",
                        lambda query, is_person=True: search_calls.append(query) or [])
    monkeypatch.setattr(expansion, "_best_org_affiliation_edge",
                        lambda edges: _fake_org_edge("Trinamix"))
    monkeypatch.setattr(search_strategy, "is_active", lambda: True)
    monkeypatch.setattr(search_strategy, "decide_angle", lambda *a, **k: {
        "angle": "generic", "why": "no clear angle",
    })

    expansion._process_person(db, "Prantik Chakraborty", 0, {},
                              enhanced_professional_search=True,
                              target_person_name="Larry Ellison", target_context="Oracle")

    assert search_calls == []


def test_phase_4e_skips_with_no_target(db, monkeypatch):
    """No target name means nothing to reason toward -- must not even call
    decide_angle, regardless of how good the org profile is."""
    org = Organization(name="Trinamix", norm_name="trinamix",
                       meta={"profile": {"v": config.NODE_PROFILE_VERSION, "size_tier": "large", "industry": "Oracle ERP consulting",
                                         "summary": "An Oracle partner.", "grounded": True}})
    db.add(org)
    db.commit()

    calls = []
    _silence_everything_but_strategy(monkeypatch)
    monkeypatch.setattr(expansion, "_best_org_affiliation_edge",
                        lambda edges: _fake_org_edge("Trinamix"))
    monkeypatch.setattr(search_strategy, "is_active", lambda: True)
    monkeypatch.setattr(search_strategy, "decide_angle",
                        lambda *a, **k: calls.append(1) or None)

    expansion._process_person(db, "Prantik Chakraborty", 0, {},
                              enhanced_professional_search=True,
                              target_person_name="", target_context="")

    assert calls == []


def test_phase_4e_skips_when_org_profile_is_not_grounded(db, monkeypatch):
    org = Organization(name="Trinamix", norm_name="trinamix",
                       meta={"profile": {"v": config.NODE_PROFILE_VERSION, "size_tier": "unknown", "industry": "unknown",
                                         "summary": "", "grounded": False}})
    db.add(org)
    db.commit()

    calls = []
    _silence_everything_but_strategy(monkeypatch)
    monkeypatch.setattr(expansion, "_best_org_affiliation_edge",
                        lambda edges: _fake_org_edge("Trinamix"))
    monkeypatch.setattr(search_strategy, "is_active", lambda: True)
    monkeypatch.setattr(search_strategy, "decide_angle",
                        lambda *a, **k: calls.append(1) or None)

    expansion._process_person(db, "Prantik Chakraborty", 0, {},
                              enhanced_professional_search=True,
                              target_person_name="Larry Ellison", target_context="")

    assert calls == []


def test_phase_4e_is_off_for_the_famous_shallow_side(db, monkeypatch):
    org = Organization(name="Oracle", norm_name="oracle",
                       meta={"profile": {"v": config.NODE_PROFILE_VERSION, "size_tier": "large", "industry": "enterprise software",
                                         "summary": "x", "grounded": True}})
    db.add(org)
    db.commit()

    calls = []
    _silence_everything_but_strategy(monkeypatch)
    monkeypatch.setattr(expansion, "_best_org_affiliation_edge",
                        lambda edges: _fake_org_edge("Oracle"))
    monkeypatch.setattr(search_strategy, "is_active", lambda: True)
    monkeypatch.setattr(search_strategy, "decide_angle",
                        lambda *a, **k: calls.append(1) or None)

    expansion._process_person(db, "Larry Ellison", 0, {},
                              enhanced_professional_search=False,
                              target_person_name="Prantik Chakraborty", target_context="")

    assert calls == []


def _fake_org_edge(org_name: str):
    from app.extraction.schemas import EdgeSignals, ExtractedEdge
    return ExtractedEdge(
        person_a="Subject", organization=org_name, other_kind="organization",
        relationship_type="employee", confidence_base=0.7, confidence_adjusted=0.7,
        evidence_snippet=f"works at {org_name}", signals=EdgeSignals(),
    )
