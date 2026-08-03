"""A failed Wikipedia/Wikidata lookup must never be cached the same way a
confirmed empty result is.

Concrete motivating case (live, this session): a single failed
wikidata_id("Mark Zuckerberg") call got cached as qid=None for the full
30-day TTL. notable_set() silently treated one of the most famous people
alive as non-notable for the rest of that window, which
_resolve_expansion_depths reads as "no clear asymmetry to exploit" and
expands both sides at full depth instead of capping the famous side to one
hop -- a large, invisible cost and correctness regression from one transient
API hiccup.
"""
from app.providers import wikipedia as wikipedia_module


class _FakeResponse:
    def __init__(self, status_code, payload=None, raise_on_json=False):
        self.status_code = status_code
        self._payload = payload
        self._raise_on_json = raise_on_json

    def json(self):
        if self._raise_on_json:
            raise ValueError("malformed JSON")
        return self._payload


def _provider():
    return wikipedia_module.WikipediaProvider()


# ---------------------------------------------------------------------------
# wikidata_id
# ---------------------------------------------------------------------------
def test_wikidata_id_does_not_cache_on_no_response(monkeypatch):
    monkeypatch.setattr(wikipedia_module, "request_with_retry", lambda *a, **k: None)
    cache_set_calls = []
    monkeypatch.setattr(wikipedia_module.cache, "get", lambda *a, **k: None)
    monkeypatch.setattr(wikipedia_module.cache, "set",
                        lambda *a, **k: cache_set_calls.append((a, k)))
    result = _provider().wikidata_id("Mark Zuckerberg")
    assert result is None
    assert cache_set_calls == []  # a failed lookup must not poison the cache


def test_wikidata_id_does_not_cache_on_non_200(monkeypatch):
    monkeypatch.setattr(wikipedia_module, "request_with_retry",
                        lambda *a, **k: _FakeResponse(503))
    cache_set_calls = []
    monkeypatch.setattr(wikipedia_module.cache, "get", lambda *a, **k: None)
    monkeypatch.setattr(wikipedia_module.cache, "set",
                        lambda *a, **k: cache_set_calls.append((a, k)))
    result = _provider().wikidata_id("Mark Zuckerberg")
    assert result is None
    assert cache_set_calls == []


def test_wikidata_id_does_not_cache_on_malformed_json(monkeypatch):
    monkeypatch.setattr(wikipedia_module, "request_with_retry",
                        lambda *a, **k: _FakeResponse(200, raise_on_json=True))
    cache_set_calls = []
    monkeypatch.setattr(wikipedia_module.cache, "get", lambda *a, **k: None)
    monkeypatch.setattr(wikipedia_module.cache, "set",
                        lambda *a, **k: cache_set_calls.append((a, k)))
    result = _provider().wikidata_id("Mark Zuckerberg")
    assert result is None
    assert cache_set_calls == []


def test_wikidata_id_caches_a_genuine_success(monkeypatch):
    payload = {"query": {"pages": {"123": {"pageprops": {"wikibase_item": "Q23716"}}}}}
    monkeypatch.setattr(wikipedia_module, "request_with_retry",
                        lambda *a, **k: _FakeResponse(200, payload))
    cache_set_calls = []
    monkeypatch.setattr(wikipedia_module.cache, "get", lambda *a, **k: None)
    monkeypatch.setattr(wikipedia_module.cache, "set",
                        lambda *a, **k: cache_set_calls.append((a, k)))
    result = _provider().wikidata_id("Mark Zuckerberg")
    assert result == "Q23716"
    assert len(cache_set_calls) == 1


def test_wikidata_id_still_caches_a_genuine_empty_result(monkeypatch):
    """A real 200 response that legitimately has no linked Wikidata item is a
    confirmed answer, not a failure -- it should still be cached, same as
    before this fix. Only actual failures (no response, non-200, malformed
    JSON) must skip the cache."""
    payload = {"query": {"pages": {"123": {"pageprops": {}}}}}
    monkeypatch.setattr(wikipedia_module, "request_with_retry",
                        lambda *a, **k: _FakeResponse(200, payload))
    cache_set_calls = []
    monkeypatch.setattr(wikipedia_module.cache, "get", lambda *a, **k: None)
    monkeypatch.setattr(wikipedia_module.cache, "set",
                        lambda *a, **k: cache_set_calls.append((a, k)))
    result = _provider().wikidata_id("Some Non-Wikidata Page")
    assert result is None
    assert len(cache_set_calls) == 1  # confirmed empty, still cached


def test_wikidata_id_recovers_on_the_next_call_after_a_failure(monkeypatch):
    """The actual bug: a failed call must not permanently block a later,
    working call from getting the real answer."""
    payload = {"query": {"pages": {"123": {"pageprops": {"wikibase_item": "Q23716"}}}}}
    responses = iter([None, _FakeResponse(200, payload)])
    monkeypatch.setattr(wikipedia_module, "request_with_retry",
                        lambda *a, **k: next(responses))
    monkeypatch.setattr(wikipedia_module.cache, "get", lambda *a, **k: None)
    monkeypatch.setattr(wikipedia_module.cache, "set", lambda *a, **k: None)

    provider = _provider()
    first = provider.wikidata_id("Mark Zuckerberg")
    second = provider.wikidata_id("Mark Zuckerberg")
    assert first is None
    assert second == "Q23716"


# ---------------------------------------------------------------------------
# summary / article_text / links -- same fix, same shape
# ---------------------------------------------------------------------------
def test_summary_does_not_cache_on_failure(monkeypatch):
    monkeypatch.setattr(wikipedia_module, "request_with_retry", lambda *a, **k: None)
    cache_set_calls = []
    monkeypatch.setattr(wikipedia_module.cache, "get", lambda *a, **k: None)
    monkeypatch.setattr(wikipedia_module.cache, "set",
                        lambda *a, **k: cache_set_calls.append((a, k)))
    assert _provider().summary("Mark Zuckerberg") == ""
    assert cache_set_calls == []


def test_article_text_does_not_cache_on_failure(monkeypatch):
    monkeypatch.setattr(wikipedia_module, "request_with_retry",
                        lambda *a, **k: _FakeResponse(500))
    cache_set_calls = []
    monkeypatch.setattr(wikipedia_module.cache, "get", lambda *a, **k: None)
    monkeypatch.setattr(wikipedia_module.cache, "set",
                        lambda *a, **k: cache_set_calls.append((a, k)))
    assert _provider().article_text("Mark Zuckerberg") == ""
    assert cache_set_calls == []


def test_links_does_not_cache_on_failure(monkeypatch):
    monkeypatch.setattr(wikipedia_module, "request_with_retry",
                        lambda *a, **k: _FakeResponse(200, raise_on_json=True))
    cache_set_calls = []
    monkeypatch.setattr(wikipedia_module.cache, "get", lambda *a, **k: None)
    monkeypatch.setattr(wikipedia_module.cache, "set",
                        lambda *a, **k: cache_set_calls.append((a, k)))
    assert _provider().links("Mark Zuckerberg") == []
    assert cache_set_calls == []
