"""Hop-0 bridge strategy: reasoning about WHICH of the operator's contacts to
walk first, before any expansion has happened.

The deterministic ranker (network/ranking.score_contacts) still produces the
shortlist and still bounds the queue. This layer only reorders the top of it,
so every test here is really asking one question: when the model is absent,
wrong, or malicious, does the walk still get exactly what it would have got
without it?
"""
import pytest

from app import config
from app.extraction import bridge_strategy
from app.graph import connect as C
from app.models import LocalProfile
from app.network.ranking import BridgeTarget
from app.utils.names import person_norm_key


def _contact(db, name, company, school=None):
    p = LocalProfile(canonical_name=name, norm_name=person_norm_key(name),
                     companies=[company] if company else [],
                     schools=[school] if school else [],
                     titles=["Engineer"])
    db.add(p)
    db.flush()
    return p


def _target(name="Larry Ellison", context="Oracle", companies=("Oracle",)):
    return BridgeTarget(name=name, context=context, companies=list(companies))


@pytest.fixture
def contacts(db):
    """Five eligible contacts at distinct employers, so company decay doesn't
    reorder them and each test's expectations stay about strategy alone."""
    for i, company in enumerate(["Oracle", "Trinamix", "Acme", "Globex", "Initech"]):
        _contact(db, f"Contact {i}", company)
    db.commit()


# ---------------------------------------------------------------------------
# choose() -- validation of what comes back from the model
# ---------------------------------------------------------------------------
def _fake_payload(monkeypatch, payload):
    monkeypatch.setattr(config, "BRIDGE_STRATEGY_ENABLED", True)
    monkeypatch.setattr(bridge_strategy, "claude_available", lambda: True)
    monkeypatch.setattr(bridge_strategy, "call_json", lambda *a, **k: payload)


class _Cand:
    def __init__(self, name):
        self.display_name = name
        self.context = "SomeCo"
        self.bridge_reasons = []
        self.local_profile_id = name


def test_out_of_range_picks_are_dropped_not_clamped(monkeypatch):
    """A model that answers 40 for a list of 15 wasn't making a near-miss
    judgment about contact 14 -- clamping would invent a decision nobody made."""
    _fake_payload(monkeypatch, {"angle": "shared_employer", "picks": [40, 1, -3],
                                "why": "because"})
    out = bridge_strategy.choose("A", "", "B", "", [], [_Cand("x"), _Cand("y")])
    assert out["picks"] == [1]


def test_duplicate_picks_collapse(monkeypatch):
    _fake_payload(monkeypatch, {"angle": "generic", "picks": [0, 0, 1], "why": "w"})
    out = bridge_strategy.choose("A", "", "B", "", [], [_Cand("x"), _Cand("y")])
    assert out["picks"] == [0, 1]


def test_picks_are_capped_at_the_configured_count(monkeypatch):
    _fake_payload(monkeypatch, {"angle": "generic", "picks": [0, 1, 2], "why": "w"})
    out = bridge_strategy.choose("A", "", "B", "", [],
                                 [_Cand("x"), _Cand("y"), _Cand("z")], n_picks=2)
    assert out["picks"] == [0, 1]


def test_an_unknown_angle_degrades_to_generic(monkeypatch):
    _fake_payload(monkeypatch, {"angle": "vibes", "picks": [], "why": "w"})
    out = bridge_strategy.choose("A", "", "B", "", [], [_Cand("x")])
    assert out["angle"] == "generic"


def test_inactive_or_failed_calls_return_none(monkeypatch):
    monkeypatch.setattr(config, "BRIDGE_STRATEGY_ENABLED", False)
    assert bridge_strategy.choose("A", "", "B", "", [], [_Cand("x")]) is None

    _fake_payload(monkeypatch, None)          # call_json's failure contract
    assert bridge_strategy.choose("A", "", "B", "", [], [_Cand("x")]) is None

    _fake_payload(monkeypatch, {"angle": "generic", "picks": [0], "why": "w"})
    assert bridge_strategy.choose("A", "", "B", "", [], []) is None  # nothing to choose


# ---------------------------------------------------------------------------
# _bridge_contacts -- reordering only, never truncation
# ---------------------------------------------------------------------------
def test_reasoning_promotes_its_picks_to_the_front(db, contacts, monkeypatch):
    """Picks are INDEXES into the shortlist the ranker built, so resolve them
    through that same shortlist rather than assuming a fixed order -- ties are
    no longer alphabetical (see ranking._tiebreak)."""
    from app.network.ranking import BridgeTarget, score_contacts
    target = _target()
    shortlist = [c.display_name for c in score_contacts(db, target=target)
                 if c.skip_reason is None]

    _fake_payload(monkeypatch, {"angle": "industry_adjacency", "picks": [3, 1],
                                "why": "consulting shop in the target's ecosystem"})
    out = C._bridge_contacts(db, target, limit=5)
    assert [c.display_name for c in out][:2] == [shortlist[3], shortlist[1]]


def test_the_rest_stay_queued_behind_the_picks(db, contacts, monkeypatch):
    """The fallback that makes a wrong pick survivable: expansion runs the
    queue sequentially and stops on a route, so unpromoted contacts cost
    nothing unless the picks fail to close it."""
    from app.network.ranking import score_contacts
    target = _target()
    shortlist = [c.display_name for c in score_contacts(db, target=target)
                 if c.skip_reason is None]

    _fake_payload(monkeypatch, {"angle": "generic", "picks": [4], "why": "w"})
    out = C._bridge_contacts(db, target, limit=5)
    assert len(out) == 5
    assert out[0].display_name == shortlist[4]
    # every other eligible contact is still present, none dropped
    assert {c.display_name for c in out} == {f"Contact {i}" for i in range(5)}


def test_no_claude_reproduces_the_deterministic_order_exactly(db, contacts, monkeypatch):
    monkeypatch.setattr(config, "BRIDGE_STRATEGY_ENABLED", True)
    monkeypatch.setattr(bridge_strategy, "claude_available", lambda: False)
    reasoned = C._bridge_contacts(db, _target(), limit=5)

    monkeypatch.setattr(config, "BRIDGE_STRATEGY_ENABLED", False)
    plain = C._bridge_contacts(db, _target(), limit=5)
    assert [c.display_name for c in reasoned] == [c.display_name for c in plain]


def test_a_raising_strategy_call_never_fails_the_front(db, contacts, monkeypatch):
    monkeypatch.setattr(config, "BRIDGE_STRATEGY_ENABLED", True)
    monkeypatch.setattr(bridge_strategy, "claude_available", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("strategy exploded")
    monkeypatch.setattr(bridge_strategy, "choose", boom)

    out = C._bridge_contacts(db, _target(), limit=5)
    assert len(out) == 5  # degraded to the deterministic ranking, not empty


def test_empty_picks_leave_the_ranking_untouched(db, contacts, monkeypatch):
    """'None of these is a plausible bridge' is a legitimate answer and must
    not be read as 'promote nothing to the front and reshuffle anyway'."""
    monkeypatch.setattr(config, "BRIDGE_STRATEGY_ENABLED", False)
    plain = [c.display_name for c in C._bridge_contacts(db, _target(), limit=5)]

    _fake_payload(monkeypatch, {"angle": "generic", "picks": [], "why": "nothing fits"})
    out = [c.display_name for c in C._bridge_contacts(db, _target(), limit=5)]
    assert out == plain


def test_the_queue_length_is_unchanged_by_reasoning(db, contacts, monkeypatch):
    """Reasoning reorders; CONNECT_BRIDGE_CONTACTS still bounds the spend."""
    _fake_payload(monkeypatch, {"angle": "shared_employer", "picks": [2, 3], "why": "w"})
    assert len(C._bridge_contacts(db, _target(), limit=2)) == 2
    assert len(C._bridge_contacts(db, _target(), limit=5)) == 5


def test_the_origin_is_never_ranked_as_their_own_bridge(db, contacts, monkeypatch):
    """Confirmed live: without owner_name, score_contacts compared every
    contact against "" and the operator came back as their own #1 bridge.
    Front A already expands that node, so the slot bought a duplicate walk
    instead of a route."""
    monkeypatch.setattr(config, "BRIDGE_STRATEGY_ENABLED", False)
    _contact(db, "Abhimanyu Sharma", "Pantheon")
    db.commit()

    out = C._bridge_contacts(db, _target(), limit=5, origin_name="Abhimanyu Sharma")
    assert "Abhimanyu Sharma" not in {c.display_name for c in out}

    # ...and they ARE present when someone else is the origin, since then they
    # really are just a contact like any other.
    out = C._bridge_contacts(db, _target(), limit=6, origin_name="Someone Else")
    assert "Abhimanyu Sharma" in {c.display_name for c in out}


# ---------------------------------------------------------------------------
# _title_score -- seniority is a proxy for web footprint, so a title that
# doesn't name a job level must not buy one. Every case below came from the
# live ranking that put four student-club officers above a company CTO.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("title,expected", [
    # committee roles: bare "chair" used to match all of these at founder tier
    ("Events Chair", 0.0),
    ("Corporate Outreach Chair", 0.0),
    ("Corporate Committee Chair", 0.0),
    # ...while the real thing still counts
    ("Chair", 3.0),
    ("Board Chair", 3.0),
    ("Chairman", 3.0),
    # "partner" attributively is a department, not a rung
    ("Partner Solutions Manager", 1.0),
    ("CSP Partner Marketing Intern", 0.0),
    ("Partner", 3.0),
    ("General Partner", 3.0),
    # an intern is not senior whatever surrounds it -- but "internal" isn't intern
    ("Software Engineer Intern", 0.0),
    ("Internal Auditor", 0.0),
    # demoting prefixes drop ONE tier rather than disqualifying: a deputy
    # director is still senior. Before this, tier 3's bare "president" matched
    # "vice president" first and made the tier-2 entry unreachable.
    ("Vice President", 2.0),
    ("Deputy Director", 1.0),
    ("Assistant Director", 1.0),
    # unambiguous levels are untouched
    ("CEO", 3.0),
    ("Co-Founder", 3.0),
    ("CTO", 2.0),
    ("Head of Engineering", 2.0),
    ("Senior Engineer", 1.0),
])
def test_title_score(title, expected):
    from app.network.ranking import _title_score
    assert _title_score([title]) == expected
