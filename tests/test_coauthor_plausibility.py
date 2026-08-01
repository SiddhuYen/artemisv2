"""Coauthor plausibility gate: a cheaper, PRIOR question to phase 4b's
existing domain_conflict check. Given what's already known about a subject
(from phases 0-4, already in hand -- no new data needed), is an OpenAlex
coauthor lookup even worth attempting? Skips the call entirely for subjects
who clearly aren't the academic-publishing type, closing the homonym-
collision risk a layer earlier instead of only catching it after OpenAlex
has already resolved a name and returned a coauthor list.

Fail-open by design, same philosophy as disambiguate.domain_conflict: only
an EXPLICIT plausible=False skips the OpenAlex call. Inactive, no signal, or
a failed call all default to "proceed as before" -- this is a cost/risk
optimization, not a safety gate, so it must never be able to block a search
that would otherwise have been fine.
"""
from app.extraction import coauthor_plausibility


# ---------------------------------------------------------------------------
# coauthor_plausibility.check -- unit tests, Claude call mocked
# ---------------------------------------------------------------------------
def test_check_returns_none_when_inactive(monkeypatch):
    monkeypatch.setattr(coauthor_plausibility, "is_active", lambda: False)
    result = coauthor_plausibility.check("Jane Phillips", "Senior SWE at Microsoft", "some signal")
    assert result is None


def test_check_returns_none_with_no_context_or_signal(monkeypatch):
    """Nothing to judge from at all -- a brand-new node. Fail open rather
    than make a call that has no real evidence to reason about."""
    monkeypatch.setattr(coauthor_plausibility, "is_active", lambda: True)
    result = coauthor_plausibility.check("Someone New", "", "")
    assert result is None


def test_check_accepts_an_implausible_verdict(monkeypatch):
    monkeypatch.setattr(coauthor_plausibility, "is_active", lambda: True)
    monkeypatch.setattr(coauthor_plausibility, "call_json", lambda *a, **k: {
        "plausible": False,
        "why": "Senior software engineer at a tech company, no signal of academic research.",
    })
    result = coauthor_plausibility.check(
        "Jane Phillips", "Senior SWE at Microsoft", "Jane Phillips, Senior Software Engineer")
    assert result == {
        "plausible": False,
        "why": "Senior software engineer at a tech company, no signal of academic research.",
    }


def test_check_accepts_a_plausible_verdict(monkeypatch):
    monkeypatch.setattr(coauthor_plausibility, "is_active", lambda: True)
    monkeypatch.setattr(coauthor_plausibility, "call_json", lambda *a, **k: {
        "plausible": True, "why": "Listed as a research scientist with published work.",
    })
    result = coauthor_plausibility.check("Jaya Sharma", "", "Jaya Sharma, cancer researcher")
    assert result["plausible"] is True


def test_check_returns_none_when_call_fails(monkeypatch):
    monkeypatch.setattr(coauthor_plausibility, "is_active", lambda: True)
    monkeypatch.setattr(coauthor_plausibility, "call_json", lambda *a, **k: None)
    result = coauthor_plausibility.check("Someone", "context", "signal")
    assert result is None


def test_check_defaults_plausible_true_when_field_missing(monkeypatch):
    """Malformed/incomplete payload should default toward NOT blocking --
    same conservative direction as everywhere else in this guard."""
    monkeypatch.setattr(coauthor_plausibility, "is_active", lambda: True)
    monkeypatch.setattr(coauthor_plausibility, "call_json", lambda *a, **k: {"why": "unclear"})
    result = coauthor_plausibility.check("Someone", "context", "signal")
    assert result["plausible"] is True
