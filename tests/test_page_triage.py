"""Which pages earn a whole-page model call.

Per-source extraction sent every fetched page to the strong model. Nothing
decided whether a page was worth that. Measured on one /connect -- Charlie
Warren -> Donald Trump, depth 2 -- it was 1,043 Sonnet calls, 1.72M input
tokens, $10.41 of a $10.49 route. The 218 searches cost $0.22. Reading the
pages WAS the route's cost, and the answer was "no route".

Every other guard in the system governs which NODES to walk or which NAMES to
keep. This is the only one that governs which PAGES to read.
"""
import pytest

from app.extraction import page_triage as PT


PAGES = [
    {"title": "Weisselberg sentenced", "snippet": "the former CFO was sentenced"},
    {"title": "Trump Org leadership team", "snippet": "Weisselberg, Calamari and McConney"},
    {"title": "Market commentary", "snippet": "shares fell on the news"},
]


@pytest.fixture(autouse=True)
def _active(monkeypatch):
    monkeypatch.setattr(PT.config, "EXTRACT_PAGE_TRIAGE", True)
    monkeypatch.setattr(PT, "claude_available", lambda: True)


def _answer(monkeypatch, payload):
    monkeypatch.setattr(PT, "call_json", lambda *a, **k: payload)


def test_selection_is_by_index_into_the_pages_fetched(monkeypatch):
    """The model chooses among pages that were actually fetched; it has no
    field in which to conjure a URL."""
    _answer(monkeypatch, {"keep": [2], "why": "only one names other people"})
    assert PT.select("Allen Weisselberg", "Charlie Warren", PAGES) == [1]


def test_out_of_range_indices_are_dropped_not_clamped(monkeypatch):
    """A clamped index spends a whole-page call on whichever result happened to
    sit at the boundary."""
    _answer(monkeypatch, {"keep": [99, 0, -1, 2], "why": "x"})
    assert PT.select("Aa", "Bb", PAGES) == [1]


def test_keeps_are_capped(monkeypatch):
    monkeypatch.setattr(PT.config, "EXTRACT_DEEP_MAX_PAGES", 1)
    _answer(monkeypatch, {"keep": [1, 2, 3], "why": "x"})
    assert len(PT.select("Aa", "Bb", PAGES)) == 1


def test_an_empty_verdict_is_a_verdict_not_a_failure(monkeypatch):
    """[] means "none of these name anyone" and must be distinguishable from
    None, which means "no answer" -- the caller does opposite things with them."""
    _answer(monkeypatch, {"keep": [], "why": "none name anyone"})
    assert PT.select("Aa", "Bb", PAGES) == []


@pytest.mark.parametrize("payload", [None, {}])
def test_no_verdict_is_none(monkeypatch, payload):
    _answer(monkeypatch, payload)
    assert PT.select("Aa", "Bb", PAGES) is None


def test_inactive_when_switched_off(monkeypatch):
    monkeypatch.setattr(PT.config, "EXTRACT_PAGE_TRIAGE", False)
    monkeypatch.setattr(PT, "call_json",
                        lambda *a, **k: pytest.fail("must not call the model"))
    assert PT.select("Aa", "Bb", PAGES) is None


# --- the per-call switch on extract() ---------------------------------------
def test_deep_false_skips_the_claude_tier(monkeypatch):
    """The whole saving. Same page, same silo -- only the switch differs."""
    from app import config
    from app.silos import SILOS

    monkeypatch.setattr(config, "CLAUDE_EXTRACT", True)
    import app.extraction as EX
    monkeypatch.setattr(EX, "claude_available", lambda: True)
    called = []
    monkeypatch.setattr(EX, "claude_extract",
                        lambda *a, **k: called.append("claude") or None)
    monkeypatch.setattr(EX, "spacy_available", lambda: True)
    monkeypatch.setattr(EX, "spacy_extract",
                        lambda *a, **k: called.append("spacy") or
                        EX.ExtractionOutput(extractor="spacy"))
    silo = SILOS[0]

    EX.extract("Aa", "text", silo, deep=True)
    EX.extract("Aa", "text", silo, deep=False)

    assert called == ["claude", "spacy", "spacy"], called
