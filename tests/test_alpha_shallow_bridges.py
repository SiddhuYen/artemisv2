"""Alpha piece 4: per-candidate depth budget.

"depending on if its asymmetric power/fame or symmetric, respectively do
either 1 hop away or equal amount of hops as origin node" -- among the Alpha
frontier (piece 3's top-5), any candidate who turns out to be independently
notable/famous relative to the target gets fully processed and persisted
(their own "1 hop"), but their OWN discoveries are excluded from seeding the
NEXT hop -- don't keep walking outward from someone already close to the
target's own world. A non-famous candidate is unaffected and keeps
expanding for as many hops as max_depth allows, same as today.

Implemented without restructuring expand_graph's hop loop: _process_one
still fully processes and persists a shallow-marked node (real graph data
isn't lost), it just returns an EMPTY discovery dict so _ranked_expandable
never sees that node's neighbors as next-hop candidates.
"""
from app.graph import builder, expansion


def test_expand_graph_stops_recursing_past_an_independently_famous_bridge(db, monkeypatch):
    """3-hop walk: Seed -> Bridge (found at hop 1, turns out to be famous) ->
    would-be hop-2 discovery. The hop-2 discovery must NEVER get processed,
    since Bridge is marked shallow the moment it's selected for the frontier."""
    processed = []

    def fake_process_person(worker_db, name, hop, local_disc, progress=None,
                            is_person=True, context="",
                            enhanced_professional_search=False,
                            professional_only=False,
                            target_person_name="", target_context=""):
        processed.append(name)
        subject = builder.get_or_create_person(worker_db, name)
        subject.processed = 1
        if name == "Seed":
            expansion._record(local_disc, _edge("Seed", "Bridge"))
        elif name == "Bridge":
            # Bridge is famous, but still fully processed -- this discovery
            # must be thrown away by the shallow-node filter, not persisted
            # into the next hop's frontier.
            expansion._record(local_disc, _edge("Bridge", "Should Never Be Reached"))

    monkeypatch.setattr(expansion, "_process_person", fake_process_person)
    monkeypatch.setattr(expansion, "is_filtering_active", lambda: False)
    monkeypatch.setattr(expansion.ORCH, "notable_set", lambda names: (
        {"Bridge"} if "Bridge" in names else set()
    ))

    expansion.expand_graph(db, "Seed", max_depth=3, enhanced_professional_search=True)

    assert "Seed" in processed
    assert "Bridge" in processed
    assert "Should Never Be Reached" not in processed


def test_expand_graph_keeps_recursing_past_a_non_famous_bridge(db, monkeypatch):
    """Same shape, but Bridge is NOT independently notable -- the hop-2
    discovery must still get processed, same as today's behavior."""
    processed = []

    def fake_process_person(worker_db, name, hop, local_disc, progress=None,
                            is_person=True, context="",
                            enhanced_professional_search=False,
                            professional_only=False,
                            target_person_name="", target_context=""):
        processed.append(name)
        subject = builder.get_or_create_person(worker_db, name)
        subject.processed = 1
        if name == "Seed":
            expansion._record(local_disc, _edge("Seed", "Bridge"))
        elif name == "Bridge":
            expansion._record(local_disc, _edge("Bridge", "Reachable Next Hop"))

    monkeypatch.setattr(expansion, "_process_person", fake_process_person)
    monkeypatch.setattr(expansion, "is_filtering_active", lambda: False)
    monkeypatch.setattr(expansion.ORCH, "notable_set", lambda names: set())

    expansion.expand_graph(db, "Seed", max_depth=3, enhanced_professional_search=True)

    assert "Reachable Next Hop" in processed


def test_shallow_marking_is_off_when_not_enhanced_professional_search(db, monkeypatch):
    """The general /discover path (enhanced_professional_search=False) must
    never truncate a branch this way -- notable_set shouldn't even be
    consulted for this purpose outside an Alpha walk."""
    processed = []
    notable_set_calls = []

    def fake_process_person(worker_db, name, hop, local_disc, progress=None,
                            is_person=True, context="",
                            enhanced_professional_search=False,
                            professional_only=False,
                            target_person_name="", target_context=""):
        processed.append(name)
        subject = builder.get_or_create_person(worker_db, name)
        subject.processed = 1
        if name == "Seed":
            expansion._record(local_disc, _edge("Seed", "Bridge"))
        elif name == "Bridge":
            expansion._record(local_disc, _edge("Bridge", "Should Still Be Reached"))

    def fake_notable_set(names):
        notable_set_calls.append(set(names))
        return {"Bridge"} if "Bridge" in names else set()

    monkeypatch.setattr(expansion, "_process_person", fake_process_person)
    monkeypatch.setattr(expansion, "is_filtering_active", lambda: False)
    monkeypatch.setattr(expansion.ORCH, "notable_set", fake_notable_set)

    # prefer_reachable=False: isolates this test to piece 4's OWN notable_set
    # call. _ranked_expandable's pre-existing reachable-mode fame sort (the
    # DEFAULT) calls notable_set for an unrelated reason -- ranking, not
    # shallow-marking -- and would otherwise make notable_set_calls non-empty
    # regardless of enhanced_professional_search, defeating this assertion.
    expansion.expand_graph(db, "Seed", max_depth=3,
                           enhanced_professional_search=False, prefer_reachable=False)

    assert "Should Still Be Reached" in processed
    assert notable_set_calls == [], "notable_set must not be consulted for shallow-marking outside Alpha"


def _edge(subject: str, person_b: str):
    from app.extraction.schemas import EdgeSignals, ExtractedEdge
    return ExtractedEdge(
        person_a=subject, person_b=person_b, other_kind="person",
        relationship_type="coworker", confidence_base=0.85, confidence_adjusted=0.85,
        source_url=f"http://x/{person_b}", signals=EdgeSignals(explicit_keyword_match=True),
    )
