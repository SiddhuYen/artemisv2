"""Never answer "no connection" without asking whether the walk stopped early.

connect_people used to return not-connected the moment its own machinery ran
out. By then it is holding exactly the context needed to judge otherwise: who
was explored on each side, which routes were proposed, and the verifier's own
words for rejecting them.

Charlie Warren -> Donald Trump is the motivating failure. The walk quit while
Sam Altman sat unexpanded in the graph with 34 edges, and neither "Charlie
Warren and Sam Altman are both Y Combinator" nor "Sam Altman has met Trump
repeatedly" was ever a query.

The containment is the PRICE ASYMMETRY between the two moves:

  probe  - one search; may name anyone, because the search is what decides.
  expand - ~35 searches; may only name nodes the walk already ranked, by index
           into the shortlist it was handed.
"""
import pytest

from app import config
from app.extraction import route_adjudicator as RA
from app.graph import connect as C


@pytest.fixture(autouse=True)
def _active(monkeypatch):
    monkeypatch.setattr(RA.config, "CONNECT_ADJUDICATE_NO_ROUTE", True)
    monkeypatch.setattr(RA, "claude_available", lambda: True)


def _answer(monkeypatch, payload):
    monkeypatch.setattr(RA, "call_json", lambda *a, **k: payload)


# --- the verdict ------------------------------------------------------------
def test_a_probe_may_name_someone_not_in_the_graph(monkeypatch):
    """One search, and the search decides -- so an unknown name is allowed. It
    is the mechanism that would have asked about Sam Altman and Trump."""
    _answer(monkeypatch, {"action": "probe", "expand": [], "why": "YC route untested",
                          "pairs": [{"a": "Sam Altman", "b": "Donald Trump"}]})
    out = RA.decide("Charlie Warren", "Donald Trump", shortlist=["Paul Graham"])
    assert out["action"] == "probe"
    assert out["pairs"] == [{"a": "Sam Altman", "b": "Donald Trump"}]


def test_an_expansion_may_only_name_a_shortlisted_node(monkeypatch):
    """~35 searches each, so the model picks by INDEX into what the walk
    already ranked. A hallucinated name cannot become an expansion."""
    _answer(monkeypatch, {"action": "expand", "pairs": [], "why": "central node",
                          "expand": [2]})
    out = RA.decide("Aa", "Bb", shortlist=["Paul Graham", "Sam Altman", "Jane Doe"])
    assert out["expand"] == ["Sam Altman"]


def test_an_out_of_range_expansion_index_is_dropped_not_clamped(monkeypatch):
    """Clamping would silently spend ~35 searches on whichever node happened to
    sit at the boundary."""
    _answer(monkeypatch, {"action": "expand", "pairs": [], "why": "x",
                          "expand": [99, 0, -1]})
    out = RA.decide("Aa", "Bb", shortlist=["Paul Graham"])
    assert out["expand"] == []
    assert out["action"] == "none", "an expansion with nothing valid left is not an action"


def test_the_two_endpoints_are_not_proposed_as_their_own_probe(monkeypatch):
    """That pair is the question that already failed."""
    _answer(monkeypatch, {"action": "probe", "expand": [], "why": "x",
                          "pairs": [{"a": "Charlie Warren", "b": "Donald Trump"},
                                    {"a": "Sam Altman", "b": "Donald Trump"}]})
    out = RA.decide("Charlie Warren", "Donald Trump")
    assert out["pairs"] == [{"a": "Sam Altman", "b": "Donald Trump"}]


def test_probes_are_capped(monkeypatch):
    monkeypatch.setattr(RA.config, "CONNECT_ADJUDICATE_MAX_PROBES", 2)
    _answer(monkeypatch, {"action": "probe", "expand": [], "why": "x",
                          "pairs": [{"a": f"P{i}", "b": f"Q{i}"} for i in range(9)]})
    assert len(RA.decide("Aa", "Bb")["pairs"]) == 2


def test_expansion_can_be_disabled_entirely(monkeypatch):
    """The cost-capped setting: probing stays available, ~35-search expansions
    do not."""
    monkeypatch.setattr(RA.config, "CONNECT_ADJUDICATE_MAX_EXPAND", 0)
    _answer(monkeypatch, {"action": "expand", "pairs": [], "why": "x", "expand": [1]})
    out = RA.decide("Aa", "Bb", shortlist=["Paul Graham"])
    assert out["expand"] == [] and out["action"] == "none"


@pytest.mark.parametrize("payload", [
    None, {}, {"action": "nonsense", "pairs": [], "expand": [], "why": ""}])
def test_fails_closed_to_the_old_behaviour(monkeypatch, payload):
    _answer(monkeypatch, payload)
    assert RA.decide("Aa", "Bb") is None


def test_inactive_when_switched_off(monkeypatch):
    monkeypatch.setattr(RA.config, "CONNECT_ADJUDICATE_NO_ROUTE", False)
    monkeypatch.setattr(RA, "call_json",
                        lambda *a, **k: pytest.fail("must not call the model"))
    assert RA.decide("Aa", "Bb") is None


# --- how connect_people uses it --------------------------------------------
def _people(db, *names):
    from app.graph import builder
    for n in names:
        builder.get_or_create_person(db, n)
    db.commit()


def _no_route_walk(monkeypatch):
    monkeypatch.setattr(C, "_ensure_origin_enriched", lambda *a, **k: {})
    monkeypatch.setattr(C, "_expand_both_concurrently", lambda *a, **k: {})
    monkeypatch.setattr(C, "_route_exists", lambda *a, **k: False)
    monkeypatch.setattr(C, "_adjacency", lambda db, *a: ({}, {}, {}, {}))
    monkeypatch.setattr(C, "_diverse_paths", lambda *a, **k: [])
    monkeypatch.setattr(C.bridge_hypothesis, "propose", lambda *a, **k: [])


def test_no_connection_is_never_returned_without_asking(db, monkeypatch):
    """The requirement."""
    _no_route_walk(monkeypatch)
    _people(db, "Aa One", "Bb Two")
    asked = []
    monkeypatch.setattr(C.route_adjudicator, "is_active", lambda: True)
    monkeypatch.setattr(C.route_adjudicator, "decide",
                        lambda *a, **k: (asked.append(1),
                                         {"action": "none", "pairs": [], "expand": [],
                                          "why": "genuinely unrelated"})[1])
    monkeypatch.setattr(C, "_direct_pair_search", lambda *a, **k: (False, False))

    out = C.connect_people(db, "Aa One", "Bb Two", depth=2)

    assert asked, "the model must be consulted before answering no"
    assert out["connected"] is False
    assert out["adjudication"]["why"] == "genuinely unrelated"


def test_its_probes_are_actually_run(db, monkeypatch):
    _no_route_walk(monkeypatch)
    _people(db, "Charlie Warren", "Donald Trump")
    probed = []
    monkeypatch.setattr(C, "_direct_pair_search",
                        lambda _db, a, b, *r, **k: (probed.append((a, b)), (False, False))[1])
    monkeypatch.setattr(C.route_adjudicator, "is_active", lambda: True)
    monkeypatch.setattr(C.route_adjudicator, "decide", lambda *a, **k: {
        "action": "probe", "expand": [], "why": "untested YC route",
        "pairs": [{"a": "Sam Altman", "b": "Donald Trump"}]})

    out = C.connect_people(db, "Charlie Warren", "Donald Trump", depth=2)

    assert ("Sam Altman", "Donald Trump") in probed
    # and the answer says what was tried, rather than a bare refusal
    assert "untested YC route" in out["reason"]


def test_a_failing_probe_does_not_fail_the_answer(db, monkeypatch):
    """A last-resort pass must not turn 'no route' into a 500."""
    _no_route_walk(monkeypatch)
    _people(db, "Aa One", "Bb Two")

    def _boom(_db, a, b, *rest, **kw):
        # stage 1 (the endpoints themselves) must still work; it is the
        # adjudicator's follow-up probe that fails here.
        if a == "Sam Altman":
            raise RuntimeError("provider down")
        return (False, False)

    monkeypatch.setattr(C, "_direct_pair_search", _boom)
    monkeypatch.setattr(C.route_adjudicator, "is_active", lambda: True)
    monkeypatch.setattr(C.route_adjudicator, "decide", lambda *a, **k: {
        "action": "probe", "expand": [], "why": "x",
        "pairs": [{"a": "Sam Altman", "b": "Donald Trump"}]})

    assert C.connect_people(db, "Aa One", "Bb Two", depth=2)["connected"] is False
