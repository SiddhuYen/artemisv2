"""Per-source Claude extraction is paid for per PAGE, not per (query, silo) tuple.

expansion._process_person's phase 4 iterates (query, result, silo) tuples while
page fetches are deduped through a set, so one URL returned by several of the
~35 silo queries reaches claude_extract several times carrying byte-identical
text. The prompt depends only on (subject, text) -- silo/evidence/source_url
never reach the model -- so those were byte-identical requests billed at full
page size each, and _dedup_and_cap collapsed the resulting edges anyway.

These tests pin the two halves of the fix: only one of those calls is paid for,
and each caller's own post-processing still runs against the shared verdict.
"""
import pytest

from app import config
from app.extraction import claude_extractor
from app.silos.definitions import COLLEAGUE_SILO, SILOS

_NEWS_SILO = SILOS[0]

_VERDICT = {
    "people": ["Molly Chakraborty"],
    "organizations": ["Trinamix"],
    "relationships": [
        {
            "other": "Molly Chakraborty",
            "kind": "person",
            "evidence": (
                "Molly Chakraborty, Cofounder and President of Trinamix, has "
                "worked alongside Prantik Chakraborty for a decade."
            ),
        }
    ],
}


@pytest.fixture(autouse=True)
def verdict_cache(monkeypatch):
    """A dict standing in for the SQLite verdict cache."""
    store = {}
    monkeypatch.setattr("app.providers.cache.get",
                        lambda key, track=True: store.get(key))
    monkeypatch.setattr("app.providers.cache.set",
                        lambda key, kind, value, ttl: store.__setitem__(key, value))
    return store


def _record_calls(monkeypatch, verdict):
    """Replace the API call with a recorder returning `verdict`."""
    prompts = []

    def _fake(prompt, schema=None, model="", max_tokens=4096, system="", effort=""):
        prompts.append(prompt)
        return verdict

    monkeypatch.setattr(claude_extractor, "call_json", _fake)
    return prompts


def test_one_page_is_extracted_once_across_silos_and_queries(monkeypatch):
    """The whole point: five silo queries returning the same URL bought five
    identical full-page requests."""
    prompts = _record_calls(monkeypatch, _VERDICT)
    text = "Prantik Chakraborty leads sales at Trinamix. " * 20

    first = claude_extractor.claude_extract(
        "Prantik Chakraborty", text, _NEWS_SILO, "snippet from query A",
        "https://example.com/team")
    second = claude_extractor.claude_extract(
        "Prantik Chakraborty", text, COLLEAGUE_SILO, "snippet from query B",
        "https://example.com/team")

    assert len(prompts) == 1
    assert [e.person_b for e in first.edges] == ["Molly Chakraborty"]
    assert [e.person_b for e in second.edges] == ["Molly Chakraborty"]


def test_each_caller_still_gets_its_own_silo_applied(monkeypatch):
    """Sharing the REQUEST must not share the post-processing -- the silo shapes
    the edge (type, confidence multiplier) after the verdict comes back."""
    _record_calls(monkeypatch, _VERDICT)
    text = "Prantik Chakraborty leads sales at Trinamix."

    news = claude_extractor.claude_extract("Prantik Chakraborty", text, _NEWS_SILO)
    colleague = claude_extractor.claude_extract(
        "Prantik Chakraborty", text, COLLEAGUE_SILO)

    assert f"silo '{_NEWS_SILO.key}'" in news.edges[0].method
    assert f"silo '{COLLEAGUE_SILO.key}'" in colleague.edges[0].method


def test_source_url_is_not_shared_between_callers(monkeypatch):
    """Two URLs can serve identical text. The verdict is about the text, but the
    edge must still cite the page it was actually found on -- _dedup_and_cap
    keys on source_url, and a wrong one is an uncheckable citation."""
    _record_calls(monkeypatch, _VERDICT)
    text = "Prantik Chakraborty leads sales at Trinamix."

    a = claude_extractor.claude_extract(
        "Prantik Chakraborty", text, _NEWS_SILO, "", "https://example.com/a")
    b = claude_extractor.claude_extract(
        "Prantik Chakraborty", text, _NEWS_SILO, "", "https://example.com/b")

    assert a.edges[0].source_url == "https://example.com/a"
    assert b.edges[0].source_url == "https://example.com/b"


def test_a_failed_extraction_is_not_remembered(monkeypatch, verdict_cache):
    """None means 'fall back to spaCy for this page'. Caching that would pin the
    page to the deterministic extractor for the whole TTL on one timeout."""
    prompts = _record_calls(monkeypatch, None)

    assert claude_extractor.claude_extract("Prantik", "some text", _NEWS_SILO) is None
    assert verdict_cache == {}

    claude_extractor.claude_extract("Prantik", "some text", _NEWS_SILO)
    assert len(prompts) == 2  # retried, not stuck on the failure


def test_changing_the_extraction_model_does_not_reuse_old_verdicts(monkeypatch):
    """CLAUDE_EXTRACT_MODEL is the obvious cost lever once this stage dominates
    the bill; switching it must re-ask, not serve the old model's answers."""
    prompts = _record_calls(monkeypatch, _VERDICT)

    monkeypatch.setattr(config, "CLAUDE_EXTRACT_MODEL", "claude-opus-5")
    claude_extractor.claude_extract("Prantik", "some text", _NEWS_SILO)
    monkeypatch.setattr(config, "CLAUDE_EXTRACT_MODEL", "claude-haiku-4-5")
    claude_extractor.claude_extract("Prantik", "some text", _NEWS_SILO)

    assert len(prompts) == 2


def test_different_subjects_on_one_page_are_separate_verdicts(monkeypatch):
    """The subject is interpolated into the prompt, so a shared roster page asks
    a genuinely different question for each person expanded from it."""
    prompts = _record_calls(monkeypatch, _VERDICT)

    claude_extractor.claude_extract("Person A", "a shared roster page", _NEWS_SILO)
    claude_extractor.claude_extract("Person B", "a shared roster page", _NEWS_SILO)

    assert len(prompts) == 2


def test_key_is_computed_on_the_text_actually_sent(monkeypatch):
    """Two pages differing only PAST MAX_PAGE_CHARS render the same prompt, so
    keying on the raw text would miss the cache on a request already paid for."""
    prompts = _record_calls(monkeypatch, _VERDICT)
    body = "Prantik Chakraborty. " * config.MAX_PAGE_CHARS

    claude_extractor.claude_extract("Prantik", body + "tail one", _NEWS_SILO)
    claude_extractor.claude_extract("Prantik", body + "tail two", _NEWS_SILO)

    assert len(prompts) == 1


def test_empty_text_never_reaches_the_api(monkeypatch):
    prompts = _record_calls(monkeypatch, _VERDICT)
    out = claude_extractor.claude_extract("Prantik", "", _NEWS_SILO)
    assert prompts == []
    assert out is not None and out.edges == []
