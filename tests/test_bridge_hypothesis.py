"""Ask who stands between them -- and let SEARCH decide.

The cheap pair search asks one question, "are A and B named together", and for
a pair three hops apart the honest answer is no. Sanjay Ghemawat and Larry Page
are the case: nine results for the pair query, none stating a direct tie, and
Jeff Dean named on every single one. The intermediary was on the page; nothing
was looking for an intermediary.

This stage asks for one. Its whole safety property is that the model's answer
is never an edge -- it is a search query. "Which famous people know each other"
is exactly where a language model is most fluent and least accountable, so a
confidently wrong name has to cost a wasted search and nothing more.
"""
import pytest

from app import config
from app.extraction import bridge_hypothesis as BH
from app.graph import connect as C


@pytest.fixture(autouse=True)
def _active(monkeypatch):
    monkeypatch.setattr(config, "CONNECT_ASK_CLAUDE_BRIDGE", True)
    monkeypatch.setattr(BH.config, "CONNECT_ASK_CLAUDE_BRIDGE", True)
    monkeypatch.setattr(BH, "claude_available", lambda: True)


def _answer(monkeypatch, candidates):
    monkeypatch.setattr(BH, "call_json",
                        lambda *a, **k: {"candidates": candidates})


# --- what comes back from the model ----------------------------------------
def test_proposes_names_in_order(monkeypatch):
    _answer(monkeypatch, [{"name": "Jeff Dean", "why": "co-wrote MapReduce with A; Google SVP under B"}])
    out = BH.propose("Sanjay Ghemawat", "Larry Page")
    assert [c["name"] for c in out] == ["Jeff Dean"]
    assert "MapReduce" in out[0]["why"]


def test_endpoints_are_never_proposed_as_their_own_bridge(monkeypatch):
    """"A connects to B via A" is not a bridge, and searching it would re-ask
    the question the pair search just answered."""
    _answer(monkeypatch, [{"name": "Larry Page", "why": "is B"},
                          {"name": "Sanjay Ghemawat", "why": "is A"},
                          {"name": "Jeff Dean", "why": "real"}])
    assert [c["name"] for c in BH.propose("Sanjay Ghemawat", "Larry Page")] == ["Jeff Dean"]


def test_single_token_names_are_dropped(monkeypatch):
    """A lone token is nearly always a first name, an org, or a fragment --
    searching it returns noise, not a person."""
    _answer(monkeypatch, [{"name": "Google", "why": "the company"},
                          {"name": "Jeff", "why": "a first name"},
                          {"name": "Jeff Dean", "why": "a person"}])
    assert [c["name"] for c in BH.propose("A Person", "B Person")] == ["Jeff Dean"]


def test_duplicates_collapse(monkeypatch):
    _answer(monkeypatch, [{"name": "Jeff Dean", "why": "one"},
                          {"name": "jeff  dean", "why": "same person"}])
    assert len(BH.propose("A Person", "B Person")) == 1


def test_respects_the_configured_ceiling(monkeypatch):
    """The ceiling is the spend cap: each name costs up to two pair searches."""
    monkeypatch.setattr(BH.config, "CONNECT_BRIDGE_HYPOTHESES", 2)
    _answer(monkeypatch, [{"name": f"Person {i}", "why": "x"} for i in range(9)])
    assert len(BH.propose("A Person", "B Person")) == 2


@pytest.mark.parametrize("payload", [None, {}, {"candidates": None}, {"candidates": []}])
def test_fails_closed(monkeypatch, payload):
    """No key, a refusal, a timeout, a malformed answer -- the caller must get
    an empty list and carry on to the expansion it would have run anyway."""
    monkeypatch.setattr(BH, "call_json", lambda *a, **k: payload)
    assert BH.propose("A Person", "B Person") == []


def test_inactive_when_switched_off(monkeypatch):
    monkeypatch.setattr(BH.config, "CONNECT_ASK_CLAUDE_BRIDGE", False)
    monkeypatch.setattr(BH, "call_json",
                        lambda *a, **k: pytest.fail("must not call the model"))
    assert BH.propose("A Person", "B Person") == []


# --- how connect_people uses it --------------------------------------------
def test_a_proposed_name_is_searched_never_written(db, monkeypatch):
    """THE containment property. The model names Jeff Dean; the only thing that
    happens is two pair searches. Nothing about the model's claim reaches the
    graph -- if the searches find nothing, neither does Artemis."""
    monkeypatch.setattr(C.bridge_hypothesis, "propose",
                        lambda *a, **k: [{"name": "Jeff Dean", "why": "documented with both"}])
    searched = []
    monkeypatch.setattr(C, "_direct_pair_search",
                        lambda _db, a, b, *rest, **kw: (searched.append((a, b)), (False, False))[1])
    monkeypatch.setattr(C, "_ensure_origin_enriched", lambda *a, **k: {})
    monkeypatch.setattr(C, "_expand_both_concurrently", lambda *a, **k: {})
    monkeypatch.setattr(C, "_adjacency", lambda db, *a: ({}, {}, {}, {}))

    C.connect_people(db, "Sanjay Ghemawat", "Larry Page", depth=2)

    # the pair itself, then BOTH halves of the proposed bridge
    assert searched == [("Sanjay Ghemawat", "Larry Page"),
                        ("Sanjay Ghemawat", "Jeff Dean"),
                        ("Jeff Dean", "Larry Page")]
    from app.models import RelationshipEdge
    assert db.query(RelationshipEdge).count() == 0, \
        "a model-proposed name must never become an edge on its own"


def test_both_halves_are_searched_before_giving_up_on_a_candidate(db, monkeypatch):
    """Half a bridge is not one. An intermediary documented with A but not with
    B leaves the pair exactly as far apart as before, so the second half runs
    even when the first found nothing."""
    monkeypatch.setattr(C.bridge_hypothesis, "propose",
                        lambda *a, **k: [{"name": "Mid Person", "why": "x"}])
    searched = []
    monkeypatch.setattr(C, "_direct_pair_search",
                        lambda _db, a, b, *rest, **kw: (searched.append((a, b)), (False, False))[1])
    monkeypatch.setattr(C, "_ensure_origin_enriched", lambda *a, **k: {})
    monkeypatch.setattr(C, "_expand_both_concurrently", lambda *a, **k: {})
    monkeypatch.setattr(C, "_adjacency", lambda db, *a: ({}, {}, {}, {}))

    C.connect_people(db, "Aa One", "Bb Two", depth=2)

    assert ("Aa One", "Mid Person") in searched
    assert ("Mid Person", "Bb Two") in searched


def test_expansion_is_skipped_once_a_bridge_is_borne_out(db, monkeypatch):
    """The point of the stage: a few searches instead of ~35 queries per node
    across two neighborhoods. The pathfinder's own rule decides it worked --
    not the searches' `found`, which says only that something was written."""
    monkeypatch.setattr(C.bridge_hypothesis, "propose",
                        lambda *a, **k: [{"name": "Mid Person", "why": "x"}])
    monkeypatch.setattr(C, "_direct_pair_search", lambda *a, **k: (True, True))
    monkeypatch.setattr(C, "_ensure_origin_enriched", lambda *a, **k: {})
    calls = []
    monkeypatch.setattr(C, "_expand_both_concurrently",
                        lambda *a, **k: (calls.append("expand"), {})[1])
    monkeypatch.setattr(C, "_adjacency", lambda db, *a: ({}, {}, {}, {}))
    # route only appears after the bridge halves are searched
    seen = {"n": 0}

    def route_exists(_db, *a, **k):
        seen["n"] += 1
        return seen["n"] > 2

    monkeypatch.setattr(C, "_route_exists", route_exists)

    C.connect_people(db, "Aa One", "Bb Two", depth=2)

    assert calls == [], "a borne-out bridge must skip the full expansion"


def test_the_stage_sits_between_the_pair_search_and_expansion(db, monkeypatch):
    """Cheapest first: it only makes sense once the pair search has established
    the two are NOT documented together, and it must come before the expansion
    it exists to avoid."""
    order = []
    monkeypatch.setattr(C, "_direct_pair_search",
                        lambda *a, **k: (order.append("direct"), (False, False))[1])
    monkeypatch.setattr(C.bridge_hypothesis, "propose",
                        lambda *a, **k: (order.append("ask"), [])[1])
    monkeypatch.setattr(C, "_ensure_origin_enriched",
                        lambda *a, **k: (order.append("origin"), {})[1])
    monkeypatch.setattr(C, "_expand_both_concurrently",
                        lambda *a, **k: (order.append("expand"), {})[1])
    monkeypatch.setattr(C, "_adjacency", lambda db, *a: ({}, {}, {}, {}))

    C.connect_people(db, "Aa One", "Bb Two", depth=2)

    assert order == ["direct", "ask", "origin", "expand"]
