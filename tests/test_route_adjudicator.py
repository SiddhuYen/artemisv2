"""Never answer "no connection" without asking whether the walk stopped early.

connect_people used to return not-connected the moment its own machinery ran
out. By then it holds both sides of the failed search -- who is around the
origin, who is around the target, and hop_verify's own words for rejecting each
candidate route -- and nothing ever looked at that before answering.

Charlie Warren -> Donald Trump is the motivating failure: the walk quit while
Sam Altman sat unexpanded in the graph with 34 edges, and neither "Charlie
Warren and Sam Altman are both Y Combinator" nor "Sam Altman has met Trump
repeatedly" was ever a query.

THE QUESTION IS A MATCHING PROBLEM, AND THAT IS THE CONTAINMENT.

Both lists come from the caller and BOTH sides of every pairing are chosen by
index into them, so the model cannot name anyone at all. The first version let
it propose free-text names and it immediately produced two failures this shape
makes impossible: it paired "Convex" -- the CONTEXT STRING the operator typed
for Charlie Warren, a company -- against Donald Trump, and it repeatedly paired
the origin with whatever famous names were in front of it (Lip-Bu Tan, Arnold
Schwarzenegger), because the list it was shown was ranked over the whole
database rather than over this walk.
"""
import pytest

from app.extraction import route_adjudicator as RA
from app.graph import connect as C

LEFT = ["Paul Graham", "Sam Altman", "Patrick Collison"]
RIGHT = ["Donald Trump", "Elon Musk", "Mark Zuckerberg"]


@pytest.fixture(autouse=True)
def _active(monkeypatch):
    monkeypatch.setattr(RA.config, "CONNECT_ADJUDICATE_NO_ROUTE", True)
    monkeypatch.setattr(RA, "claude_available", lambda: True)


def _answer(monkeypatch, payload):
    monkeypatch.setattr(RA, "call_json", lambda *a, **k: payload)


def _decide(**kw):
    kw.setdefault("left", LEFT)
    kw.setdefault("right", RIGHT)
    return RA.decide("Charlie Warren", "Donald Trump", **kw)


# --- the matching -----------------------------------------------------------
def test_a_pairing_resolves_by_index_on_both_sides(monkeypatch):
    """The mechanism that would have asked about Sam Altman and Trump."""
    _answer(monkeypatch, {"action": "probe", "expand": [], "why": "YC route untested",
                          "pairs": [{"left": 2, "right": 1}]})
    out = _decide()
    assert out["action"] == "probe"
    assert out["pairs"] == [{"a": "Sam Altman", "b": "Donald Trump"}]


def test_the_model_has_no_field_in_which_to_name_anyone(monkeypatch):
    """Free text is what produced a search for "Convex" -- a company, lifted
    from the operator's own context field -- against Donald Trump. Both sides
    of a pairing are now integers, so that failure has nowhere to arrive."""
    props = RA._SCHEMA["properties"]["pairs"]["items"]["properties"]
    assert {k: v["type"] for k, v in props.items()} == {"left": "integer",
                                                       "right": "integer"}


def test_out_of_range_indices_are_dropped_not_clamped(monkeypatch):
    """Clamping would silently spend a query -- or ~35 of them for an expansion
    -- on whichever entry happened to sit at the boundary."""
    _answer(monkeypatch, {"action": "probe", "expand": [], "why": "x",
                          "pairs": [{"left": 99, "right": 1},
                                    {"left": 0, "right": 1},
                                    {"left": 2, "right": 1}]})
    assert _decide()["pairs"] == [{"a": "Sam Altman", "b": "Donald Trump"}]


def test_pairing_someone_with_themselves_is_dropped(monkeypatch):
    _answer(monkeypatch, {"action": "probe", "expand": [], "why": "x",
                          "pairs": [{"left": 1, "right": 1}]})
    out = RA.decide("Charlie Warren", "Donald Trump",
                    left=["Donald Trump"], right=["Donald Trump"])
    assert out["action"] == "none"


def test_pairings_are_capped(monkeypatch):
    monkeypatch.setattr(RA.config, "CONNECT_ADJUDICATE_MAX_PROBES", 2)
    _answer(monkeypatch, {"action": "probe", "expand": [], "why": "x",
                          "pairs": [{"left": l, "right": r}
                                    for l in (1, 2, 3) for r in (1, 2, 3)]})
    assert len(_decide()["pairs"]) == 2


def test_an_expansion_names_a_left_hand_person_by_index(monkeypatch):
    """~35 searches, so it can only ever be someone the walk already found."""
    _answer(monkeypatch, {"action": "expand", "pairs": [], "why": "central",
                          "expand": [3]})
    assert _decide()["expand"] == ["Patrick Collison"]


def test_expansion_can_be_disabled_entirely(monkeypatch):
    """The cost-capped setting: pairings stay available, ~35-search expansions
    do not."""
    monkeypatch.setattr(RA.config, "CONNECT_ADJUDICATE_MAX_EXPAND", 0)
    _answer(monkeypatch, {"action": "expand", "pairs": [], "why": "x", "expand": [1]})
    out = _decide()
    assert out["expand"] == [] and out["action"] == "none"


@pytest.mark.parametrize("payload", [
    None, {}, {"action": "nonsense", "pairs": [], "expand": [], "why": ""}])
def test_fails_closed_to_the_old_behaviour(monkeypatch, payload):
    _answer(monkeypatch, payload)
    assert _decide() is None


def test_nothing_to_match_means_nothing_to_ask(monkeypatch):
    """No call at all when either side is empty -- there is no question."""
    monkeypatch.setattr(RA, "call_json",
                        lambda *a, **k: pytest.fail("must not call the model"))
    assert RA.decide("Aa", "Bb", left=[], right=RIGHT) is None
    assert RA.decide("Aa", "Bb", left=LEFT, right=[]) is None


def test_inactive_when_switched_off(monkeypatch):
    monkeypatch.setattr(RA.config, "CONNECT_ADJUDICATE_NO_ROUTE", False)
    monkeypatch.setattr(RA, "call_json",
                        lambda *a, **k: pytest.fail("must not call the model"))
    assert _decide() is None


# --- how connect_people builds the two sides -------------------------------
def _people(db, *names):
    from app.graph import builder
    out = [builder.get_or_create_person(db, n) for n in names]
    db.commit()
    return out


def _edge(db, x, y):
    from app.models import RelationshipEdge
    db.add(RelationshipEdge(person_a_id=x.id, person_b_id=y.id,
                            relationship_type="coworker", status="strong",
                            confidence_raw=0.8, method="test",
                            evidence_snippet="ev", signals={}))


def _no_route_walk(monkeypatch):
    monkeypatch.setattr(C, "_ensure_origin_enriched", lambda *a, **k: {})
    monkeypatch.setattr(C, "_expand_both_concurrently", lambda *a, **k: {})
    monkeypatch.setattr(C, "_route_exists", lambda *a, **k: False)
    monkeypatch.setattr(C, "_diverse_paths", lambda *a, **k: [])
    monkeypatch.setattr(C.bridge_hypothesis, "propose", lambda *a, **k: [])
    monkeypatch.setattr(C, "is_filtering_active", lambda: False)
    monkeypatch.setattr(C, "_direct_pair_search", lambda *a, **k: (False, False))


def _capture(monkeypatch, verdict=None):
    seen = {}
    monkeypatch.setattr(C.route_adjudicator, "is_active", lambda: True)
    monkeypatch.setattr(
        C.route_adjudicator, "decide",
        lambda *a_, **k: (seen.update(k),
                          verdict or {"action": "none", "pairs": [],
                                      "expand": [], "why": "genuinely unrelated"})[1])
    return seen


def test_no_connection_is_never_returned_without_asking(db, monkeypatch):
    """The requirement."""
    a, _b, near = _people(db, "Charlie Warren", "Donald Trump", "Paul Graham")
    _edge(db, a, near)
    db.commit()
    _no_route_walk(monkeypatch)
    seen = _capture(monkeypatch)

    out = C.connect_people(db, "Charlie Warren", "Donald Trump", depth=2)

    assert seen, "the model must be consulted before answering no"
    assert out["connected"] is False
    assert out["adjudication"]["why"] == "genuinely unrelated"


def test_the_target_leads_the_right_hand_list(db, monkeypatch):
    """"Does one of these people know the TARGET" closes the gap in one hop,
    and is only askable if the target is on the list."""
    a, _b, near = _people(db, "Charlie Warren", "Donald Trump", "Paul Graham")
    _edge(db, a, near)
    db.commit()
    _no_route_walk(monkeypatch)
    seen = _capture(monkeypatch)

    C.connect_people(db, "Charlie Warren", "Donald Trump", depth=2)

    assert seen["right"][0] == "Donald Trump"
    assert "Paul Graham" in seen["left"]


def test_each_side_is_scoped_to_this_walk_not_the_whole_graph(db, monkeypatch):
    """The regression. A node that is merely popular in the database -- because
    an earlier, unrelated search inflated its degree -- must not appear. That
    is what put Lip-Bu Tan in front of the model on a Charlie Warren query."""
    a, _b, near, stranger = _people(
        db, "Charlie Warren", "Donald Trump", "Paul Graham", "Lip-Bu Tan")
    _edge(db, a, near)
    for i in range(6):
        (filler,) = _people(db, f"Filler {i}")
        _edge(db, stranger, filler)
    db.commit()
    _no_route_walk(monkeypatch)
    seen = _capture(monkeypatch)

    C.connect_people(db, "Charlie Warren", "Donald Trump", depth=2)

    assert "Paul Graham" in seen["left"], "an endpoint's own neighbour belongs"
    assert "Lip-Bu Tan" not in set(seen["left"]) | set(seen["right"]), \
        "a popular node unrelated to this query must never be offered"


def test_its_pairings_are_actually_searched(db, monkeypatch):
    a, _b, near = _people(db, "Charlie Warren", "Donald Trump", "Sam Altman")
    _edge(db, a, near)
    db.commit()
    _no_route_walk(monkeypatch)
    probed = []
    monkeypatch.setattr(C, "_direct_pair_search",
                        lambda _db, x, y, *r, **k: (probed.append((x, y)),
                                                    (False, False))[1])
    _capture(monkeypatch, {"action": "probe", "expand": [], "why": "untested YC route",
                           "pairs": [{"a": "Sam Altman", "b": "Donald Trump"}]})

    out = C.connect_people(db, "Charlie Warren", "Donald Trump", depth=2)

    assert ("Sam Altman", "Donald Trump") in probed
    assert "untested YC route" in out["reason"]


def test_a_failing_pairing_does_not_fail_the_answer(db, monkeypatch):
    """A last-resort pass must not turn 'no route' into a 500."""
    a, _b, near = _people(db, "Charlie Warren", "Donald Trump", "Sam Altman")
    _edge(db, a, near)
    db.commit()
    _no_route_walk(monkeypatch)

    def _boom(_db, x, y, *rest, **kw):
        if x == "Sam Altman":
            raise RuntimeError("provider down")
        return (False, False)

    monkeypatch.setattr(C, "_direct_pair_search", _boom)
    _capture(monkeypatch, {"action": "probe", "expand": [], "why": "x",
                           "pairs": [{"a": "Sam Altman", "b": "Donald Trump"}]})

    assert C.connect_people(db, "Charlie Warren", "Donald Trump",
                            depth=2)["connected"] is False
