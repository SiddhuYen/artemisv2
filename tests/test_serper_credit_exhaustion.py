"""Serper's out-of-credits response.

Serper signals an empty balance as 400 {"message": "Not enough credits"} rather
than the 402/429 the exhaustion path checks for. Without recognising it the
provider never marks itself unavailable, so the orchestrator re-calls it for
every query, gets the same 400, and falls through to Brave — measured on a live
10-contact run as 282 wasted Serper calls against 282 real Brave ones, doubling
latency and rate-limiter pressure for zero results.
"""
from app.providers.serper import SerperProvider


class _Resp:
    def __init__(self, status_code, text='{"organic": []}'):
        self.status_code, self.text = status_code, text

    def json(self):
        import json
        return json.loads(self.text)


def _provider(monkeypatch, resp):
    p = SerperProvider()
    p._exhausted = False
    monkeypatch.setattr(p, "available", lambda: True)
    monkeypatch.setattr("app.providers.serper._do_request", lambda q: resp)
    monkeypatch.setattr("app.providers.serper._mark_state", lambda s: None)
    return p


def test_out_of_credits_marks_the_provider_exhausted(monkeypatch):
    p = _provider(monkeypatch,
                  _Resp(400, '{"message":"Not enough credits","statusCode":400}'))
    assert p._search_uncached("anything") == []
    assert p._exhausted is True   # so the orchestrator stops calling it


def test_a_malformed_query_400_stays_retryable(monkeypatch):
    """Not every 400 is an empty balance — a bad query must not disable the
    provider for the rest of the run."""
    p = _provider(monkeypatch, _Resp(400, '{"message":"Invalid query"}'))
    assert p._search_uncached("anything") == []
    assert p._exhausted is False


def test_a_successful_response_still_parses(monkeypatch):
    body = ('{"organic":[{"title":"T","link":"https://e.com","snippet":"S"}]}')
    p = _provider(monkeypatch, _Resp(200, body))
    results = p._search_uncached("anything")
    assert [r.url for r in results] == ["https://e.com"]
    assert p._exhausted is False
