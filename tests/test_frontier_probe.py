"""Ask each frontier node whether IT reaches the target, before walking it.

Expansion walks outward from both endpoints and hopes the frontiers meet. For a
famous endpoint they cannot: SHALLOW_FAMOUS_DEPTH caps that side at one hop
precisely because the neighborhood is too large to enumerate. So the meeting has
to be FOUND, not walked into -- and asking "is this node documented with the
target" costs one search against ~35 to expand the node, on exactly the kind of
person whose ties are written down.

Motivating failure: Charlie Warren -> Donald Trump returned

    Charlie Warren -> Paul Graham -> Drew Houston -> Mark Zuckerberg
                   -> Andreessen Horowitz -> Donald Trump

-- five hops, one of them a video title typed 'family_social', one a venture
firm held as a person -- while never once asking whether Paul Graham, Drew
Houston or Mark Zuckerberg is documented with Trump. The last of those is.
"""
import pytest

from app import config
from app.graph import connect as C


@pytest.fixture(autouse=True)
def _probe_on(monkeypatch):
    monkeypatch.setattr(C.config, "CONNECT_PROBE_FRONTIER", True)
    monkeypatch.setattr(C.config, "CONNECT_PROBE_ONLY_FAMOUS", True)
    monkeypatch.setattr(C.config, "CONNECT_PROBE_MAX_PER_HOP", 5)
    monkeypatch.setattr(C, "is_filtering_active", lambda: False)
    monkeypatch.setattr(C, "_notable_endpoints", lambda a, b: (True, True))
    # triage is exercised in its own tests below; off by default here
    monkeypatch.setattr(C.route_adjudicator, "is_active", lambda: False)


def _make(monkeypatch, far="Donald Trump", **over):
    """_make_prober lives inside _expand_both_concurrently, so exercise it the
    way production does -- through a real call with expansion faked out."""
    seen = {}
    probed = []

    monkeypatch.setattr(C, "_direct_pair_search",
                        lambda _db, a, b, *r, **k: (probed.append((a, b)), (False, False))[1])

    def fake_expand(worker_db, name, side_depth, **kwargs):
        seen[name] = kwargs.get("on_frontier")
        return {}

    monkeypatch.setattr(C, "expand_graph", fake_expand)
    for k, v in over.items():
        monkeypatch.setattr(C.config, k, v)
    return seen, probed


def test_each_side_probes_toward_the_other_endpoint(db, monkeypatch):
    """A side walking out from A asks about B -- the endpoint it is trying to
    reach, not the one it started from."""
    seen, probed = _make(monkeypatch)
    C._expand_both_concurrently(db, "Aa Origin", "Donald Trump", 2, 2,
                                set(), None, "", "")

    seen["Aa Origin"](["Paul Graham", "Drew Houston"], db)
    assert probed == [("Paul Graham", "Donald Trump"),
                      ("Drew Houston", "Donald Trump")]


def test_the_target_is_never_probed_against_itself(db, monkeypatch):
    """The far endpoint can appear in its own frontier; asking whether Trump
    reaches Trump costs a search and answers nothing."""
    seen, probed = _make(monkeypatch)
    C._expand_both_concurrently(db, "Aa Origin", "Donald Trump", 2, 2,
                                set(), None, "", "")

    seen["Aa Origin"](["Donald Trump", "Paul Graham"], db)
    assert probed == [("Paul Graham", "Donald Trump")]


def test_probes_are_capped_per_hop(db, monkeypatch):
    """The cap IS the spend: one search each, so an unbounded frontier would
    reintroduce the cost this exists to avoid."""
    seen, probed = _make(monkeypatch, CONNECT_PROBE_MAX_PER_HOP=2)
    C._expand_both_concurrently(db, "Aa Origin", "Donald Trump", 2, 2,
                                set(), None, "", "")

    seen["Aa Origin"]([f"Person {i}" for i in range(9)], db)
    assert len(probed) == 2


def test_junk_frontier_nodes_are_not_probed(db, monkeypatch):
    """"General Manager" and "Andreessen Horowitz" are in this graph as people.
    Probing them spends a search and risks writing another junk edge, so the
    frontier goes through the entity filter first."""
    monkeypatch.setattr(C, "is_filtering_active", lambda: True)
    monkeypatch.setattr(C, "filter_entities", lambda names, kind: {"Paul Graham"})
    seen, probed = _make(monkeypatch)
    C._expand_both_concurrently(db, "Aa Origin", "Donald Trump", 2, 2,
                                set(), None, "", "")

    seen["Aa Origin"](["General Manager", "Andreessen Horowitz", "Paul Graham"], db)
    assert probed == [("Paul Graham", "Donald Trump")]


def test_no_probing_toward_an_unknown_person(db, monkeypatch):
    """The argument for probing is that a DOCUMENTED person answers in one
    query. For an obscure endpoint the search is better spent on the walk."""
    monkeypatch.setattr(C, "_notable_endpoints", lambda a, b: (False, False))
    seen, probed = _make(monkeypatch)
    C._expand_both_concurrently(db, "Aa Origin", "Bb Nobody", 2, 2,
                                set(), None, "", "")

    assert seen["Aa Origin"] is None, "no prober should be installed at all"
    assert probed == []


def test_probing_can_be_switched_off(db, monkeypatch):
    seen, probed = _make(monkeypatch, CONNECT_PROBE_FRONTIER=False)
    C._expand_both_concurrently(db, "Aa Origin", "Donald Trump", 2, 2,
                                set(), None, "", "")
    assert seen["Aa Origin"] is None


def test_a_failing_probe_does_not_fail_the_walk(db, monkeypatch):
    """Opportunistic: the expansion it interrupts is still the real answer."""
    def _boom(*a, **k):
        raise RuntimeError("provider down")

    seen, _probed = _make(monkeypatch)
    monkeypatch.setattr(C, "_direct_pair_search", _boom)
    C._expand_both_concurrently(db, "Aa Origin", "Donald Trump", 2, 2,
                                set(), None, "", "")

    seen["Aa Origin"](["Paul Graham"], db)   # must not raise


# ── one model call instead of N searches ──────────────────────────────────
# Reaching a well-connected person does not mean they reach the target, and
# searching each of them to find that out is how a hop spends five queries to
# learn nothing. The frontier is triaged first, as the same matching question
# the adjudicator uses: these people on the left, the target alone on the right.

def _triage(monkeypatch, keep):
    monkeypatch.setattr(C.route_adjudicator, "is_active", lambda: True)
    monkeypatch.setattr(C.route_adjudicator, "decide", lambda **k: {
        "action": "probe", "expand": [], "why": "only these are documented",
        "pairs": [{"a": n, "b": k["right"][0]} for n in k["left"] if n in keep]})


def test_only_the_nodes_the_model_keeps_are_searched(db, monkeypatch):
    _triage(monkeypatch, keep={"Mark Zuckerberg"})
    seen, probed = _make(monkeypatch)
    C._expand_both_concurrently(db, "Aa Origin", "Donald Trump", 2, 2,
                                set(), None, "", "")

    seen["Aa Origin"](["Paul Graham", "Mark Zuckerberg", "Drew Houston"], db)

    assert probed == [("Mark Zuckerberg", "Donald Trump")], \
        "two useless searches must not be spent"


def test_the_model_cannot_add_a_name_to_the_frontier(db, monkeypatch):
    """It selects from what the walk found; a name it invents is not searched,
    so the worst case is that a hop costs one call and no queries."""
    monkeypatch.setattr(C.route_adjudicator, "is_active", lambda: True)
    monkeypatch.setattr(C.route_adjudicator, "decide", lambda **k: {
        "action": "probe", "expand": [], "why": "x",
        "pairs": [{"a": "Someone Invented", "b": "Donald Trump"}]})
    seen, probed = _make(monkeypatch)
    C._expand_both_concurrently(db, "Aa Origin", "Donald Trump", 2, 2,
                                set(), None, "", "")

    seen["Aa Origin"](["Paul Graham"], db)
    assert probed == []


def test_an_unavailable_triage_falls_back_to_probing_everything(db, monkeypatch):
    """Fails open, not closed: the probe is the feature, triage is the saving."""
    monkeypatch.setattr(C.route_adjudicator, "is_active", lambda: True)
    monkeypatch.setattr(C.route_adjudicator, "decide", lambda **k: None)
    seen, probed = _make(monkeypatch)
    C._expand_both_concurrently(db, "Aa Origin", "Donald Trump", 2, 2,
                                set(), None, "", "")

    seen["Aa Origin"](["Paul Graham", "Drew Houston"], db)
    assert len(probed) == 2


def test_the_probe_uses_the_workers_session_not_the_callers(db, monkeypatch):
    """Regression: the prober closed over the OUTER session while running
    inside both side workers, which raises "this session is provisioning a new
    connection; concurrent operations are not permitted"."""
    monkeypatch.setattr(C.route_adjudicator, "is_active", lambda: False)
    seen, _probed = _make(monkeypatch)
    used = []
    # after _make, which installs its own recorder
    monkeypatch.setattr(C, "_direct_pair_search",
                        lambda _db, a, b, *r, **k: (used.append(_db), (False, False))[1])
    C._expand_both_concurrently(db, "Aa Origin", "Donald Trump", 2, 2,
                                set(), None, "", "")

    sentinel = object()
    seen["Aa Origin"](["Paul Graham"], sentinel)
    assert used == [sentinel], "the probe must run on the session it was handed"


# ── the hook has to be REACHABLE ───────────────────────────────────────────
# It originally sat after the next hop's frontier was ranked, which every real
# walk exits before: `stop_after_node` breaks the moment should_stop trips
# (true of any walk that starts with a route already believed to exist),
# `hop == max_depth - 1` breaks on the last hop before ranking, and the node cap
# breaks too. The feature existed and never once ran.

def _expand_capturing(db, monkeypatch, **kw):
    """Run the real expand_graph with the network stubbed out, recording what
    on_frontier is handed."""
    from app.graph import expansion as E
    offered = []
    monkeypatch.setattr(E, "_ranked_expandable",
                        lambda *a, **k: ["Mark Zuckerberg", "Paul Graham"])
    monkeypatch.setattr(E, "_process_person", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(E, "_prune_invalid_nodes", lambda *a, **k: None)
    monkeypatch.setattr(E, "_retype_unknown_edges", lambda *a, **k: None)
    E.expand_graph(db, "Aa Origin", kw.pop("depth", 1),
                   on_frontier=lambda names, _db: offered.append(list(names)),
                   **kw)
    return offered


def test_the_hook_fires_on_a_single_hop_walk(db, monkeypatch):
    """depth=1 breaks at `hop == max_depth - 1` before any ranking, so nothing
    discovered on the only hop was ever offered."""
    assert _expand_capturing(db, monkeypatch, depth=1) == \
        [["Mark Zuckerberg", "Paul Graham"]]


def test_the_hook_fires_even_when_the_walk_stops_early(db, monkeypatch):
    """`stop_after_node` is the exit that won every observed run: the stop
    condition trips partway through a hop, after nodes have been processed and
    candidates discovered. Those discoveries are exactly what is worth asking
    about, and they were being thrown away."""
    calls = {"n": 0}

    def stop_after_first_node(_db):
        calls["n"] += 1
        return calls["n"] > 1        # let the hop start, then trip

    offered = _expand_capturing(db, monkeypatch, depth=2,
                                should_stop=stop_after_first_node)
    assert offered == [["Mark Zuckerberg", "Paul Graham"]]


def test_a_name_is_offered_once_across_hops(db, monkeypatch):
    """The ranking is cumulative, so without deduping, hop 2 re-offers
    everything hop 1 already asked about and the caller pays again."""
    offered = _expand_capturing(db, monkeypatch, depth=3)
    assert offered == [["Mark Zuckerberg", "Paul Graham"]], offered
