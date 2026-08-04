"""Regression tests for the trace-route ("/connect") bugs fixed together:

  1. A seed could be pruned mid-run: expand_graph's noise-shape prune only
     exempted its OWN seed, but connect_people calls it twice (once per
     endpoint) into one shared graph -- the second call's prune saw the
     first call's seed as an ordinary, unprotected node.
  2. _merge_aliases could promote a scraped-chrome surface ("Bill Gates -
     Wikipedia") to canonical_name, because it shares the real seed's
     norm_name -- which then makes the SAME node fail its own shape check.
  3. _prune_invalid_nodes deleted a person via raw, un-flushed-aware bulk
     SQL, leaving that person's edges (added to the session but not yet
     flushed at commit time) pointing at an id that no longer exists.
  4. _path_worthy hard-excluded every 'weak'-status or 'unknown'-typed
     edge, rather than costing them -- on a sparsely-evidenced real graph
     this excluded most of it and routinely reported two linked people as
     "not connected" with no path to show at all.

Bug 3's fallout (a dangling edge surviving after its endpoint is gone) is
also guarded defensively in _adjacency, independent of what caused it --
so already-corrupted data degrades gracefully instead of leaking a raw
UUID into a rendered route.
"""
from app import config
from app.graph import builder, connect as C
from app.graph.expansion import _prune_invalid_nodes
from app.models import Person, RelationshipEdge
from app.utils.names import person_norm_key


def _person(db, name, **kw):
    p = Person(canonical_name=name, norm_name=person_norm_key(name), **kw)
    db.add(p)
    db.flush()
    return p


def _edge(db, a, b, rel="coworker", status="candidate", conf=0.5, signals=None):
    # Default signals represent realistic weak-but-real evidence (the two
    # names DO cooccur in some sentence, just without an explicit keyword) --
    # not the "two coincidental mentions on the same page, never actually
    # cooccurring" case _untraversable specifically excludes (see connect.py).
    # A test that wants THAT exact shape passes signals={} explicitly.
    e = RelationshipEdge(person_a_id=a.id, person_b_id=b.id,
                         relationship_type=rel, status=status, confidence_raw=conf,
                         signals={"sentence_cooccurrence": True} if signals is None else signals)
    db.add(e)
    return e


# --- bug 1: prune must exempt every protected endpoint, not just one -------
def test_prune_protects_all_given_norms_not_just_one(db):
    seed_a = _person(db, "Alpha Person")
    seed_b = _person(db, "Beta Person")
    junk = _person(db, "Some Page - LinkedIn")  # fails the shape check
    db.commit()

    protected = {person_norm_key("Alpha Person"), person_norm_key("Beta Person")}
    _prune_invalid_nodes(db, protected)

    assert db.get(Person, seed_a.id) is not None
    assert db.get(Person, seed_b.id) is not None
    assert db.get(Person, junk.id) is None


def test_prune_with_single_protected_norm_still_deletes_the_other_seed(db):
    """Sanity check that the OLD single-seed behavior really is the bug:
    protecting only one endpoint's norm lets the other be pruned as junk."""
    seed_a = _person(db, "Alpha Person")
    noisy_seed_b = _person(db, "Beta Person - Wikipedia")  # e.g. after a bad
    db.commit()                                            # canonical-name promotion

    _prune_invalid_nodes(db, {person_norm_key("Alpha Person")})

    assert db.get(Person, seed_a.id) is not None
    assert db.get(Person, noisy_seed_b.id) is None  # the bug, reproduced


# --- bug 2: canonical_name must never be overwritten by a noisy surface ---
def test_merge_aliases_never_promotes_a_noisy_surface_to_canonical(db):
    seed = _person(db, "Bill Gates")
    builder._merge_aliases(seed, "Bill Gates - Wikipedia")
    assert seed.canonical_name == "Bill Gates"
    assert "Bill Gates - Wikipedia" in (seed.aliases or [])


def test_merge_aliases_still_promotes_a_longer_clean_surface(db):
    seed = _person(db, "Bill Gates")
    builder._merge_aliases(seed, "William Henry Gates III")
    assert seed.canonical_name == "William Henry Gates III"


# --- bug 3: pending (unflushed) edges must not be orphaned by the prune ---
def test_prune_flushes_before_deleting_so_pending_edges_are_not_orphaned(db):
    junk = _person(db, "Some Page - LinkedIn")
    good = _person(db, "Good Person")
    db.commit()
    _edge(db, junk, good)  # added to the session, NOT flushed yet

    _prune_invalid_nodes(db, set())
    db.commit()

    surviving_edges = db.query(RelationshipEdge).all()
    people_ids = {p.id for p in db.query(Person).all()}
    for e in surviving_edges:
        assert e.person_a_id in people_ids
        assert e.person_b_id is None or e.person_b_id in people_ids


# --- _adjacency must degrade gracefully on already-dangling edges ---------
def test_adjacency_skips_edges_with_a_missing_endpoint(db):
    good = _person(db, "Good Person")
    db.commit()
    ghost_id = "not-a-real-person-id"
    db.add(RelationshipEdge(person_a_id=ghost_id, person_b_id=good.id,
                            relationship_type="coworker", status="candidate",
                            confidence_raw=0.5))
    db.commit()

    adj, person_by_id, src_by_id, degree = C._adjacency(db)
    assert ghost_id not in adj
    assert adj.get(good.id, []) == []


# --- bug 4: weak/unknown edges must be traversable (costed, not excluded) -
def test_weak_status_edge_is_traversable_when_it_is_the_only_route(db):
    a = _person(db, "A Person")
    b = _person(db, "B Person")
    _edge(db, a, b, rel="coworker", status="weak", conf=0.3)
    db.commit()

    adj, person_by_id, src_by_id, degree = C._adjacency(db)
    routes = C._diverse_paths(adj, a.id, b.id, 3, 1, person_by_id, degree)
    assert routes, "a weak-status edge must still form a path when it's the only one"


def test_unknown_type_edge_is_traversable_when_it_is_the_only_route(db):
    a = _person(db, "A Person")
    b = _person(db, "B Person")
    _edge(db, a, b, rel="unknown", status="candidate", conf=0.4)
    db.commit()

    adj, person_by_id, src_by_id, degree = C._adjacency(db)
    routes = C._diverse_paths(adj, a.id, b.id, 3, 1, person_by_id, degree)
    assert routes, "an unknown-typed edge must still form a path when it's the only one"


# --- bug 5: a phantom edge from two coincidental same-page mentions must
# not be traversable, even as the only route -- live case: "Amit Sharma" (a
# real person, DIFFERENT from the Trinamix Amit Sharma) reported as
# "directly connected" to Mark Zuckerberg off two unrelated sentences on
# one page, confidence 0.10, no cooccurrence, no explicit keyword, unknown
# type. Distinct in kind (not just degree) from bug 4's weak/unknown edges,
# which DO have real -- if weak -- cooccurring evidence.
def test_phantom_no_cooccurrence_edge_is_not_traversable_even_as_the_only_route(db):
    a = _person(db, "A Person")
    b = _person(db, "B Person")
    _edge(db, a, b, rel="unknown", status="weak", conf=0.1, signals={
        "sentence_cooccurrence": False, "explicit_keyword_match": False,
    })
    db.commit()

    adj, person_by_id, src_by_id, degree = C._adjacency(db)
    routes = C._diverse_paths(adj, a.id, b.id, 3, 1, person_by_id, degree)
    assert not routes, ("two names that never actually cooccur must not form a "
                        "path just because nothing else competes with it")


def test_route_exists_is_false_for_a_phantom_no_cooccurrence_edge(db):
    a = _person(db, "A Person")
    b = _person(db, "B Person")
    _edge(db, a, b, rel="unknown", status="weak", conf=0.1, signals={
        "sentence_cooccurrence": False, "explicit_keyword_match": False,
    })
    db.commit()

    assert C._route_exists(db, "A Person", "B Person", 3) is False


def test_edge_with_cooccurrence_but_no_keyword_stays_traversable(db):
    """The narrow cut is EITHER cooccurrence OR an explicit keyword, not
    both -- an edge that cooccurs but has no strength keyword (the common
    'weak, real mention' shape) must still pass."""
    a = _person(db, "A Person")
    b = _person(db, "B Person")
    _edge(db, a, b, rel="unknown", status="weak", conf=0.29, signals={
        "sentence_cooccurrence": True, "explicit_keyword_match": False,
    })
    db.commit()

    assert C._route_exists(db, "A Person", "B Person", 3) is True


def test_a_typed_edge_stays_traversable_even_with_no_cooccurrence_signal(db):
    """The cut only applies when relationship_type is ALSO 'unknown' -- a
    typed edge (however it got typed) is a real assertion, not a phantom
    coincidence, even if the cooccurrence signal wasn't recorded."""
    a = _person(db, "A Person")
    b = _person(db, "B Person")
    _edge(db, a, b, rel="coworker", status="weak", conf=0.2, signals={
        "sentence_cooccurrence": False, "explicit_keyword_match": False,
    })
    db.commit()

    assert C._route_exists(db, "A Person", "B Person", 3) is True


def test_typed_candidate_edge_still_beats_an_unknown_shortcut(db):
    """The relaxation must not erase the preference for well-evidenced edges:
    given a choice, routing still prefers the typed edge over an untyped one
    of otherwise-equal confidence."""
    a = _person(db, "A Person")
    b = _person(db, "B Person")
    typed = _edge(db, a, b, rel="coworker", status="candidate", conf=0.5)
    db.commit()
    assert C._edge_cost(typed) < C._edge_cost(
        RelationshipEdge(person_a_id=a.id, person_b_id=b.id,
                         relationship_type="unknown", status="candidate",
                         confidence_raw=0.5)
    )


def test_rejected_status_edge_is_still_hard_excluded(db):
    a = _person(db, "A Person")
    b = _person(db, "B Person")
    _edge(db, a, b, rel="coworker", status="rejected", conf=0.9)
    db.commit()

    adj, person_by_id, src_by_id, degree = C._adjacency(db)
    assert adj.get(a.id, []) == []


# --- _route_exists: bounded hop-by-hop walk, not a full-graph rebuild ------
def test_route_exists_finds_a_route_within_max_hops(db):
    a = _person(db, "A Person")
    mid = _person(db, "Mid Person")
    b = _person(db, "B Person")
    _edge(db, a, mid)
    _edge(db, mid, b)
    db.commit()

    assert C._route_exists(db, "A Person", "B Person", 2) is True
    # an edge is undirected here, so which endpoint it was stored under
    # must not decide whether the route is found
    assert C._route_exists(db, "B Person", "A Person", 2) is True


def test_route_exists_stops_at_max_hops(db):
    a = _person(db, "A Person")
    m1 = _person(db, "Mid One")
    m2 = _person(db, "Mid Two")
    b = _person(db, "B Person")
    _edge(db, a, m1)
    _edge(db, m1, m2)
    _edge(db, m2, b)
    db.commit()

    assert C._route_exists(db, "A Person", "B Person", 2) is False
    assert C._route_exists(db, "A Person", "B Person", 3) is True


def test_route_exists_hard_excludes_a_rejected_edge(db):
    """Same single rule as _path_worthy: 'rejected' is the only status a route
    may not run through, and the cheap probe applies exactly that rule."""
    a = _person(db, "A Person")
    b = _person(db, "B Person")
    _edge(db, a, b, rel="coworker", status="rejected", conf=0.9)
    db.commit()

    assert C._route_exists(db, "A Person", "B Person", 3) is False


def test_route_exists_still_walks_a_weak_untyped_edge(db):
    """The other half of that rule (bug 4): everything short of 'rejected' is
    traversable -- priced by the scoring pass, never excluded -- so the probe
    must not quietly reintroduce a status/type floor of its own."""
    a = _person(db, "A Person")
    b = _person(db, "B Person")
    _edge(db, a, b, rel="unknown", status="weak", conf=0.3)
    db.commit()

    assert C._route_exists(db, "A Person", "B Person", 3) is True


def test_route_exists_is_true_for_one_and_the_same_person(db):
    _person(db, "Alpha Person")
    db.commit()

    # both surfaces normalise to a single node, so there is nothing to search
    assert C._route_exists(db, "Alpha Person", "alpha  person", 3) is True


def test_route_exists_is_false_when_an_endpoint_is_not_in_the_graph(db):
    _person(db, "A Person")
    db.commit()

    assert C._route_exists(db, "A Person", "Nobody Known", 3) is False


def test_route_exists_ignores_a_route_through_a_pruned_person(db):
    """Bug 3's fallout, now on the probe: a pair of dangling edges must not
    bridge a route _adjacency would refuse to produce. connect_people skips
    the live search entirely when this returns True, so a probe that walks
    further than the scoring pass would report "already connected" and then
    return no path at all."""
    a = _person(db, "A Person")
    b = _person(db, "B Person")
    db.commit()
    ghost_id = "not-a-real-person-id"
    db.add(RelationshipEdge(person_a_id=a.id, person_b_id=ghost_id,
                            relationship_type="coworker", status="candidate",
                            confidence_raw=0.5))
    db.add(RelationshipEdge(person_a_id=ghost_id, person_b_id=b.id,
                            relationship_type="coworker", status="candidate",
                            confidence_raw=0.5))
    db.commit()

    assert C._route_exists(db, "A Person", "B Person", 3) is False


def test_route_exists_never_rebuilds_the_whole_adjacency_map(db, monkeypatch):
    """The point of the probe: it answers from the neighborhood it actually
    walks, hop by hop over an index -- never by loading every person, source
    and edge in the graph the way the once-per-request scoring pass does."""
    a = _person(db, "A Person")
    b = _person(db, "B Person")
    _edge(db, a, b)
    db.commit()

    def _boom(*args, **kwargs):
        raise AssertionError("_route_exists must not rebuild the full adjacency map")

    monkeypatch.setattr(C, "_adjacency", _boom)

    assert C._route_exists(db, "A Person", "B Person", 3) is True


def test_connect_people_skips_search_when_a_route_already_exists(db, monkeypatch):
    """'Go through what's already known first': a route already sitting in
    the graph (e.g. a bridged linkedin_1st edge, or leftover data from an
    earlier /connect or /discover run) must resolve via the zero-cost
    _route_exists check alone -- neither the live direct-pair search nor the
    full neighborhood expansion should run at all."""
    a = _person(db, "Alpha Person")
    b = _person(db, "Beta Person")
    _edge(db, a, b, rel="coworker", status="candidate", conf=0.8)
    db.commit()

    def _boom(*args, **kwargs):
        raise AssertionError("must not search when a route already exists in the graph")

    monkeypatch.setattr(C, "_expand_both_concurrently", _boom)
    monkeypatch.setattr(C, "_direct_pair_search", _boom)

    result = C.connect_people(db, "Alpha Person", "Beta Person", depth=2)

    assert result["connected"] is True


# --- fame-asymmetric expansion depth ----------------------------------------

def test_strip_trailing_context_removes_of_at_and_comma_clauses():
    assert C._strip_trailing_context("Larry Ellison of Oracle") == "Larry Ellison"
    assert C._strip_trailing_context("Larry Ellison at Oracle") == "Larry Ellison"
    assert C._strip_trailing_context("Larry Ellison, Oracle") == "Larry Ellison"
    assert C._strip_trailing_context("Larry Ellison") == "Larry Ellison"


def test_resolve_expansion_depths_shallows_the_notable_side_even_with_context_baked_in(monkeypatch):
    """The exact failure mode that motivated this: the frontend's Route panel
    has no separate context field, so a famous person's name often arrives
    as one combined string ("Larry Ellison of Oracle"), which has no
    Wikipedia page of its own -- only the stripped form does."""
    def fake_notable_set(names):
        return {n for n in names if n in ("Larry Ellison",)}

    monkeypatch.setattr(C.ORCH, "notable_set", fake_notable_set)

    depth_a, depth_b = C._resolve_expansion_depths(
        "Prantik Chakraborty of Trinamix", "Larry Ellison of Oracle", 2)
    assert depth_a == 2
    assert depth_b == C.SHALLOW_FAMOUS_DEPTH


def test_resolve_expansion_depths_shallows_the_notable_side(monkeypatch):
    monkeypatch.setattr(C.ORCH, "notable_set", lambda names: {"Famous Person"})

    depth_a, depth_b = C._resolve_expansion_depths("Famous Person", "Obscure Person", 2)
    assert depth_a == C.SHALLOW_FAMOUS_DEPTH
    assert depth_b == 2

    depth_a, depth_b = C._resolve_expansion_depths("Obscure Person", "Famous Person", 2)
    assert depth_a == 2
    assert depth_b == C.SHALLOW_FAMOUS_DEPTH


def test_resolve_expansion_depths_gives_origin_side_one_extra_hop_at_depth_3(monkeypatch):
    """ORIGIN_EXTRA_HOP_AT_DEPTH: the famous side's SHALLOW_FAMOUS_DEPTH cap
    never scales with the requested depth, so at depth=3 specifically the
    non-famous (origin) side gets depth+1 to partly offset that gap. Scoped
    to exactly 3, not depth>=3 -- deliberately not generalized until there's
    evidence for how 4+ should scale (see the constant's own comment)."""
    monkeypatch.setattr(C.ORCH, "notable_set", lambda names: {"Famous Person"})

    depth_a, depth_b = C._resolve_expansion_depths("Famous Person", "Obscure Person", 3)
    assert depth_a == C.SHALLOW_FAMOUS_DEPTH
    assert depth_b == 4

    depth_a, depth_b = C._resolve_expansion_depths("Obscure Person", "Famous Person", 3)
    assert depth_a == 4
    assert depth_b == C.SHALLOW_FAMOUS_DEPTH


def test_resolve_expansion_depths_symmetric_when_both_notable(monkeypatch):
    monkeypatch.setattr(C.ORCH, "notable_set", lambda names: {"Alpha", "Beta"})
    assert C._resolve_expansion_depths("Alpha", "Beta", 3) == (3, 3)


def test_resolve_expansion_depths_symmetric_when_neither_notable(monkeypatch):
    monkeypatch.setattr(C.ORCH, "notable_set", lambda names: set())
    assert C._resolve_expansion_depths("Alpha", "Beta", 3) == (3, 3)


def test_resolve_expansion_depths_never_deepens_past_the_requested_depth(monkeypatch):
    """A shallow depth-1 request must stay depth 1 for the famous side, not
    get rounded up to SHALLOW_FAMOUS_DEPTH if that's somehow larger. (Depth 3
    is the one deliberate exception -- see
    test_resolve_expansion_depths_gives_origin_side_one_extra_hop_at_depth_3.)
    """
    monkeypatch.setattr(C.ORCH, "notable_set", lambda names: {"Famous Person"})
    depth_a, depth_b = C._resolve_expansion_depths("Famous Person", "Obscure Person", 1)
    assert depth_a == 1
    assert depth_b == 1


def test_resolve_expansion_depths_degrades_to_symmetric_on_lookup_failure(monkeypatch):
    def _boom(names):
        raise RuntimeError("network hiccup")

    monkeypatch.setattr(C.ORCH, "notable_set", _boom)
    assert C._resolve_expansion_depths("Alpha", "Beta", 3) == (3, 3)


def test_connect_people_passes_asymmetric_depths_to_expansion(db, monkeypatch):
    monkeypatch.setattr(C, "_route_exists", lambda *a, **k: False)
    monkeypatch.setattr(C, "_direct_pair_search", lambda *a, **k: (False, False))
    monkeypatch.setattr(C.ORCH, "notable_set", lambda names: {"Famous Person"})

    captured = {}

    def fake_expand_both(db_arg, name_a, name_b, depth_a, depth_b, *rest, **kwargs):
        captured["depths"] = (depth_a, depth_b)

    monkeypatch.setattr(C, "_expand_both_concurrently", fake_expand_both)

    C.connect_people(db, "Famous Person", "Obscure Person", depth=3)

    # depth=3 triggers ORIGIN_EXTRA_HOP_AT_DEPTH: the origin side gets 4, not 3.
    assert captured["depths"] == (C.SHALLOW_FAMOUS_DEPTH, 4)


# ── hop-limited search must memoize (node, hops), not node ────────────────
# _best_path pruned with `best_cost[node]`, the standard Dijkstra rule, which is
# wrong under a hop limit: reaching a node cheaply at the last permitted hop --
# where it can no longer be extended -- permanently blocked reaching that same
# node earlier, from which the target was still in range.
#
# Live consequence: Charlie Warren -> Donald Trump reported "no path within 5
# hops" while a five-hop route sat in the graph, which a plain BFS over the very
# same adjacency found. It also put _route_exists (a BFS) permanently at odds
# with the scoring pass.

def _chain(adj, names, cost_edge):
    """Wire names into a chain in a bare adjacency dict."""
    for x, y in zip(names, names[1:]):
        adj.setdefault(x, []).append((y, cost_edge))
        adj.setdefault(y, []).append((x, cost_edge))


def test_a_reachable_target_is_not_hidden_by_a_cheap_long_detour(db):
    """The regression, reduced.

    'mid' is reachable two ways: cheaply via a long detour that arrives with no
    hops left, and expensively via a short hop from which the target is still
    reachable. Pruning by node alone records the cheap arrival first and then
    refuses the short one, losing the only usable route.
    """
    from app.graph.connect import _best_path
    from app.models import RelationshipEdge

    cheap = RelationshipEdge(relationship_type="coworker", status="strong",
                             confidence_raw=0.95, signals={})
    costly = RelationshipEdge(relationship_type="coworker", status="weak",
                              confidence_raw=0.05, signals={})

    adj = {}
    # long, cheap way in: start -> d1 -> d2 -> d3 -> mid   (4 hops)
    _chain(adj, ["start", "d1", "d2", "d3", "mid"], cheap)
    # short, expensive way in: start -> mid                (1 hop)
    adj.setdefault("start", []).append(("mid", costly))
    adj.setdefault("mid", []).append(("start", costly))
    # and the last leg
    adj.setdefault("mid", []).append(("target", cheap))
    adj.setdefault("target", []).append(("mid", cheap))

    path = _best_path(adj, "start", "target", max_hops=4)

    assert path is not None, "a reachable target must not be hidden"
    assert [n for n, _e in path] == ["start", "mid", "target"]


def test_the_cheap_route_still_wins_when_hops_allow_it(db):
    """The fix must not turn the pathfinder into plain BFS -- with room to
    spare, the cheaper route is still the one chosen."""
    from app.graph.connect import _best_path
    from app.models import RelationshipEdge

    cheap = RelationshipEdge(relationship_type="coworker", status="strong",
                             confidence_raw=0.95, signals={})
    costly = RelationshipEdge(relationship_type="coworker", status="weak",
                              confidence_raw=0.05, signals={})

    adj = {}
    _chain(adj, ["start", "good", "target"], cheap)
    _chain(adj, ["start", "bad", "target"], costly)

    path = _best_path(adj, "start", "target", max_hops=5)
    assert [n for n, _e in path] == ["start", "good", "target"]
