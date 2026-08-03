""""Explored" data: expand_graph's visited_by_hop, threaded through
connect._expand_both_concurrently and connect_people's "explored" field --
so a caller can visualize what Artemis actually looked at on BOTH sides of
a /connect search even when the two sides never met (there's no found
route to show in that case at all, but the search still did real work
worth surfacing).
"""
from app.extraction.schemas import EdgeSignals, ExtractedEdge
from app.graph import builder, connect as C, expansion


def _edge(subject: str, person_b: str) -> ExtractedEdge:
    return ExtractedEdge(
        person_a=subject, person_b=person_b, other_kind="person",
        relationship_type="coworker", confidence_base=0.85, confidence_adjusted=0.85,
        source_url=f"http://x/{person_b}", signals=EdgeSignals(explicit_keyword_match=True),
    )


# ---------------------------------------------------------------------------
# expand_graph: visited_by_hop
# ---------------------------------------------------------------------------
def test_expand_graph_returns_visited_by_hop(db, monkeypatch):
    def fake_process_person(worker_db, name, hop, local_disc, progress=None,
                            is_person=True, context="", silo_weights=None,
                            enhanced_professional_search=False,
                            professional_only=False,
                            target_person_name="", target_context=""):
        subject = builder.get_or_create_person(worker_db, name)
        subject.processed = 1
        if name == "Seed":
            expansion._record(local_disc, _edge("Seed", "Bridge1"))
            expansion._record(local_disc, _edge("Seed", "Bridge2"))

    monkeypatch.setattr(expansion, "_process_person", fake_process_person)
    monkeypatch.setattr(expansion, "is_filtering_active", lambda: False)

    stats = expansion.expand_graph(db, "Seed", max_depth=2, prefer_reachable=False)

    assert stats["visited_by_hop"][0] == ["Seed"]
    assert sorted(stats["visited_by_hop"][1]) == ["Bridge1", "Bridge2"]


def test_expand_graph_visited_by_hop_reflects_the_selected_frontier_only(db, monkeypatch):
    """A node discovered but never selected for the next hop (e.g. it fails
    is_expandable's confidence bar) must not appear in visited_by_hop --
    that field is "what Artemis actually went and looked at", not "every
    name it ever heard of"."""
    def fake_process_person(worker_db, name, hop, local_disc, progress=None,
                            is_person=True, context="", silo_weights=None,
                            enhanced_professional_search=False,
                            professional_only=False,
                            target_person_name="", target_context=""):
        subject = builder.get_or_create_person(worker_db, name)
        subject.processed = 1
        if name == "Seed":
            expansion._record(local_disc, ExtractedEdge(
                person_a="Seed", person_b="Weak Mention", other_kind="person",
                relationship_type="unknown", confidence_base=0.1, confidence_adjusted=0.1,
                source_url="http://x/weak", signals=EdgeSignals(),
            ))

    monkeypatch.setattr(expansion, "_process_person", fake_process_person)
    monkeypatch.setattr(expansion, "is_filtering_active", lambda: False)

    stats = expansion.expand_graph(db, "Seed", max_depth=2, prefer_reachable=False)

    assert stats["visited_by_hop"][0] == ["Seed"]
    assert stats["visited_by_hop"].get(1, []) == []


# ---------------------------------------------------------------------------
# expand_graph: boundary (candidates the LAST processed hop turned up but
# never got to walk itself, e.g. the famous-side depth-1 cap)
# ---------------------------------------------------------------------------
def test_expand_graph_boundary_captures_last_hop_unwalked_candidates(db, monkeypatch):
    """max_depth=1 means the hop loop ends right after processing the seed --
    same shape as SHALLOW_FAMOUS_DEPTH. The seed's own search still finds and
    persists real candidates; `boundary` is how a caller learns about them
    even though visited_by_hop never gets a hop-1 entry for this run."""
    def fake_process_person(worker_db, name, hop, local_disc, progress=None,
                            is_person=True, context="", silo_weights=None,
                            enhanced_professional_search=False,
                            professional_only=False,
                            target_person_name="", target_context=""):
        subject = builder.get_or_create_person(worker_db, name)
        subject.processed = 1
        if name == "Seed":
            expansion._record(local_disc, _edge("Seed", "Bridge1"))
            expansion._record(local_disc, _edge("Seed", "Bridge2"))

    monkeypatch.setattr(expansion, "_process_person", fake_process_person)
    monkeypatch.setattr(expansion, "is_filtering_active", lambda: False)

    stats = expansion.expand_graph(db, "Seed", max_depth=1, prefer_reachable=False)

    assert list(stats["visited_by_hop"].keys()) == [0]
    assert sorted(stats["boundary"]) == ["Bridge1", "Bridge2"]


def test_expand_graph_boundary_empty_when_nothing_further_found(db, monkeypatch):
    def fake_process_person(worker_db, name, hop, local_disc, progress=None,
                            is_person=True, context="", silo_weights=None,
                            enhanced_professional_search=False,
                            professional_only=False,
                            target_person_name="", target_context=""):
        subject = builder.get_or_create_person(worker_db, name)
        subject.processed = 1

    monkeypatch.setattr(expansion, "_process_person", fake_process_person)
    monkeypatch.setattr(expansion, "is_filtering_active", lambda: False)

    stats = expansion.expand_graph(db, "Seed", max_depth=1, prefer_reachable=False)

    assert stats["boundary"] == []


# ---------------------------------------------------------------------------
# connect._expand_both_concurrently: returns both sides' stats
# ---------------------------------------------------------------------------
def test_expand_both_concurrently_returns_visited_by_hop_for_both_sides(db, monkeypatch):
    def fake_expand_graph(worker_db, name, max_depth, **kwargs):
        return {"visited_by_hop": {0: [name]}, "people_found": 0,
                "organizations_found": 0, "edges_found": 0, "sources_fetched": 0,
                "nodes_processed_per_depth": [1]}

    monkeypatch.setattr(C, "expand_graph", fake_expand_graph)

    result = C._expand_both_concurrently(
        db=db, name_a="Alpha", name_b="Beta", depth_a=1, depth_b=1,
        protected=set(), progress=None, context_a="", context_b="")

    assert result["a"]["visited_by_hop"] == {0: ["Alpha"]}
    assert result["b"]["visited_by_hop"] == {0: ["Beta"]}


# ---------------------------------------------------------------------------
# connect._build_explored
# ---------------------------------------------------------------------------
def test_build_explored_returns_none_with_no_expand_stats():
    assert C._build_explored(None, "Alpha", "Beta") is None


def test_build_explored_shapes_both_sides():
    stats = {"a": {"visited_by_hop": {0: ["Alpha"], 1: ["Bridge"]}},
            "b": {"visited_by_hop": {0: ["Beta"]}}}

    result = C._build_explored(stats, "Alpha", "Beta")

    assert result == {
        "a": {"seed": "Alpha", "by_hop": {0: ["Alpha"], 1: ["Bridge"]}, "boundary": []},
        "b": {"seed": "Beta", "by_hop": {0: ["Beta"]}, "boundary": []},
    }


# ---------------------------------------------------------------------------
# connect_people: the "explored" field end to end
# ---------------------------------------------------------------------------
def test_connect_people_includes_explored_when_no_route_found(db, monkeypatch):
    builder.get_or_create_person(db, "Alpha")
    builder.get_or_create_person(db, "Beta")
    db.commit()

    monkeypatch.setattr(C, "_route_exists", lambda *a, **k: False)
    monkeypatch.setattr(C, "_direct_pair_search", lambda *a, **k: (False, False))

    def fake_expand_both(db_arg, name_a, name_b, depth_a, depth_b, *rest, **kwargs):
        return {
            "a": {"visited_by_hop": {0: ["Alpha"], 1: ["Alpha Bridge"]}},
            "b": {"visited_by_hop": {0: ["Beta"], 1: ["Beta Bridge"]}},
        }

    monkeypatch.setattr(C, "_expand_both_concurrently", fake_expand_both)

    result = C.connect_people(db, "Alpha", "Beta", depth=2)

    assert result["connected"] is False
    assert result["explored"] == {
        "a": {"seed": "Alpha", "by_hop": {0: ["Alpha"], 1: ["Alpha Bridge"]}, "boundary": []},
        "b": {"seed": "Beta", "by_hop": {0: ["Beta"], 1: ["Beta Bridge"]}, "boundary": []},
    }


def test_connect_people_explored_is_none_when_route_already_known(db, monkeypatch):
    """No fresh expansion ran at all (the cheap _route_exists check already
    found a path) -- nothing new was explored, so there's nothing to show
    beyond the route itself."""
    alpha = builder.get_or_create_person(db, "Alpha")
    beta = builder.get_or_create_person(db, "Beta")
    from app.models import RelationshipEdge
    db.add(RelationshipEdge(person_a_id=alpha.id, person_b_id=beta.id,
                            relationship_type="coworker", confidence_raw=0.8,
                            status="candidate",
                            signals={"sentence_cooccurrence": True}))
    db.commit()

    monkeypatch.setattr(C, "_route_exists", lambda *a, **k: True)

    result = C.connect_people(db, "Alpha", "Beta", depth=2)

    assert result["connected"] is True
    assert result["explored"] is None
