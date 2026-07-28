"""The Claude extraction stages must FAIL CLOSED.

Every LLM stage in Artemis (per-source extraction, entity filter, relationship
classifier) is an accuracy upgrade layered on top of a deterministic pipeline
that already works. The contract that makes that safe is one-directional: an
LLM failure may cost the graph some cleanup, but it must never fail a build,
raise into a caller, or -- worst of all -- delete real data because a call
didn't come back.

These tests pin that contract with no network access and no API key, which is
also exactly the configuration a bare dev checkout and a keyless deploy run in.
"""
import json

import pytest

from app import config
from app.extraction import claude_client, entity_filter, relation_classifier


@pytest.fixture(autouse=True)
def _no_credentials(monkeypatch):
    """Every test here runs as if no key were configured anywhere."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    claude_client.reset_availability_cache()
    yield
    claude_client.reset_availability_cache()


def _force_call_failure(monkeypatch, exc):
    """Make every call_json go through the real error path, raising `exc`."""
    class _Boom:
        class messages:
            @staticmethod
            def create(**_kwargs):
                raise exc

    monkeypatch.setattr(claude_client, "_get_client", lambda: _Boom())


# --- the no-op path (no key configured) -----------------------------------
def test_entity_filter_keeps_everything_when_claude_is_unavailable(monkeypatch):
    """Keep-all, never drop-all: a filter that can't run must not prune nodes."""
    monkeypatch.setattr(claude_client, "_get_client", lambda: None)
    names = ["Satya Nadella", "Bill Gates", "Ada Lovelace"]
    assert entity_filter.validate(names, "person") == set(names)
    assert entity_filter.is_filtering_active() is False


def test_deterministic_noise_filter_runs_without_claude(monkeypatch):
    """The name-shape filter is NOT gated on Claude -- it's the keyless floor."""
    monkeypatch.setattr(claude_client, "_get_client", lambda: None)
    kept = entity_filter.validate(["Satya Nadella", "Cookie Policy"], "person")
    assert kept == {"Satya Nadella"}


def test_classifier_returns_unknown_for_every_item_when_unavailable(monkeypatch):
    monkeypatch.setattr(claude_client, "_get_client", lambda: None)
    items = [
        {"a": "A", "b": "B", "evidence": "A and B co-founded X."},
        {"a": "C", "b": "D", "evidence": "C sits on D's board."},
    ]
    out = relation_classifier.classify(items)
    assert out == [{"type": "unknown", "confidence": 0.0}] * 2
    assert relation_classifier.is_active() is False


def test_classify_result_length_always_matches_input_length(monkeypatch):
    """expansion._retype_unknown_edges zips verdicts against its edge list, so a
    short result would silently mislabel edges rather than skip them."""
    monkeypatch.setattr(claude_client, "_get_client", lambda: None)
    items = [{"a": "A", "b": "B", "evidence": "e"} for _ in range(7)]
    assert len(relation_classifier.classify(items)) == 7


# --- the failure path (a key is present but calls fail) -------------------
@pytest.mark.parametrize("exc", [
    TimeoutError("read timed out"),
    ValueError("malformed response"),
    RuntimeError("connection reset"),
])
def test_call_json_swallows_transport_failures(monkeypatch, exc):
    _force_call_failure(monkeypatch, exc)
    assert claude_client.call_json("p", {"type": "object"}, model="m") is None


def test_transport_failure_does_not_prune_nodes(monkeypatch):
    """A timeout mid-build must not be read as 'none of these are real people'."""
    _force_call_failure(monkeypatch, TimeoutError("read timed out"))
    names = ["Satya Nadella", "Bill Gates"]
    assert entity_filter.validate(names, "person") == set(names)


def test_transport_failure_verdicts_are_not_cached(monkeypatch):
    """A non-judgment cached for CACHE_TTL_WIKI would keep junk (or drop a real
    node) for 30 days on the strength of one bad response."""
    calls = []

    def _fake_set(key, kind, value, ttl):
        calls.append(key)

    _force_call_failure(monkeypatch, TimeoutError("boom"))
    monkeypatch.setattr("app.providers.cache.set", _fake_set)
    entity_filter.validate(["Satya Nadella"], "person")
    assert calls == []


# --- the auth latch --------------------------------------------------------
def test_auth_failure_latches_claude_off(monkeypatch):
    """One local, network-free auth failure disables the rest of the build.

    Without the latch every subsequent batch re-attempts and re-fails, adding
    a round trip per batch to a build that was always going to fall back.
    """
    attempts = []

    class _Unauthed:
        class messages:
            @staticmethod
            def create(**_kwargs):
                attempts.append(1)
                raise TypeError(
                    "Could not resolve authentication method. Expected one of "
                    "api_key, auth_token, or credentials to be set."
                )

    monkeypatch.setattr(claude_client, "_get_client", lambda: _Unauthed())
    monkeypatch.setattr(config, "CLAUDE_FILTER_BATCH", 1)

    entity_filter.validate(["Ada Lovelace", "Grace Hopper", "Alan Turing"], "person")
    assert len(attempts) == 1, "auth failure must not be retried per batch"
    assert claude_client.claude_available() is False
    assert entity_filter.is_filtering_active() is False


def test_non_auth_failure_does_not_latch(monkeypatch):
    """A timeout is transient -- it must not disable Claude for the process."""
    _force_call_failure(monkeypatch, TimeoutError("read timed out"))
    claude_client.call_json("p", {"type": "object"}, model="m")
    assert claude_client.claude_available() is True


# --- response handling -----------------------------------------------------
class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Resp:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_Block(text)] if text is not None else []
        self.stop_reason = stop_reason


def _respond_with(monkeypatch, resp):
    class _Client:
        class messages:
            @staticmethod
            def create(**_kwargs):
                return resp

    monkeypatch.setattr(claude_client, "_get_client", lambda: _Client())


def test_call_json_parses_a_good_response(monkeypatch):
    _respond_with(monkeypatch, _Resp(json.dumps({"verdicts": [{"name": "A", "valid": True}]})))
    assert claude_client.call_json("p", {"type": "object"}, model="m") == {
        "verdicts": [{"name": "A", "valid": True}]
    }


@pytest.mark.parametrize("resp", [
    _Resp("", stop_reason="end_turn"),                      # empty content
    _Resp('{"verdicts": [{"name":', stop_reason="max_tokens"),  # truncated JSON
    _Resp("I can't help with that.", stop_reason="refusal"),    # safety refusal
    _Resp("here you go: not json"),                             # unparseable
    _Resp("[1, 2, 3]"),                                         # right JSON, wrong shape
])
def test_call_json_rejects_unusable_responses(monkeypatch, resp):
    _respond_with(monkeypatch, resp)
    assert claude_client.call_json("p", {"type": "object"}, model="m") is None


# --- effort compatibility --------------------------------------------------
# `output_config.effort` is not universally supported: Haiku 4.5 -- the default
# model for the two batched stages -- rejects it with a 400. Sending it anyway
# would 400 every call, return None, and silently no-op the stage while looking
# exactly like a working fallback.
def _capture_requests(monkeypatch, fail_on_effort=False):
    seen = []

    class _Client:
        class messages:
            @staticmethod
            def create(**kwargs):
                seen.append(kwargs)
                if fail_on_effort and "effort" in kwargs["output_config"]:
                    import anthropic
                    raise anthropic.BadRequestError(
                        message="effort: unsupported parameter for this model",
                        response=_FakeResponse(400),
                        body=None,
                    )
                return _Resp(json.dumps({"ok": True}))

    monkeypatch.setattr(claude_client, "_get_client", lambda: _Client())
    return seen


class _FakeResponse:
    """Minimal stand-in for the httpx response an APIStatusError carries."""

    def __init__(self, status_code):
        self.status_code = status_code
        self.headers = {}
        self.request = None


def test_effort_is_omitted_for_models_that_reject_it(monkeypatch):
    seen = _capture_requests(monkeypatch)
    claude_client.call_json("p", {"type": "object"}, model="claude-haiku-4-5")
    assert "effort" not in seen[0]["output_config"]
    assert seen[0]["output_config"]["format"]["type"] == "json_schema"


def test_effort_is_sent_for_models_that_support_it(monkeypatch):
    seen = _capture_requests(monkeypatch)
    monkeypatch.setattr(config, "CLAUDE_EFFORT", "low")
    claude_client.call_json("p", {"type": "object"}, model="claude-opus-5")
    assert seen[0]["output_config"]["effort"] == "low"


def test_unexpected_effort_rejection_retries_once_without_it(monkeypatch):
    """A model not on the seeded deny-list must self-correct, not no-op."""
    monkeypatch.setattr(claude_client, "_EFFORT_UNSUPPORTED", set())
    monkeypatch.setattr(config, "CLAUDE_EFFORT", "low")
    seen = _capture_requests(monkeypatch, fail_on_effort=True)

    out = claude_client.call_json("p", {"type": "object"}, model="claude-future-1")
    assert out == {"ok": True}, "the retry should have succeeded"
    assert len(seen) == 2
    assert "effort" in seen[0]["output_config"]
    assert "effort" not in seen[1]["output_config"]

    # ...and the next call skips the doomed first attempt entirely.
    claude_client.call_json("p", {"type": "object"}, model="claude-future-1")
    assert len(seen) == 3
    assert "effort" not in seen[2]["output_config"]


def test_batched_stages_default_to_the_cheap_model():
    """The filter and classifier are high-volume; extraction reads whole pages."""
    assert config.CLAUDE_FILTER_MODEL == config.CLAUDE_BATCH_MODEL
    assert config.CLAUDE_CLASSIFY_MODEL == config.CLAUDE_BATCH_MODEL
    assert config.CLAUDE_EXTRACT_MODEL == config.CLAUDE_MODEL


def test_entity_filter_matches_verdicts_by_normalized_name(monkeypatch):
    """A model asked to echo a candidate back can still normalize whitespace;
    an exact-string-only lookup would miss the verdict and default to KEEP."""
    _respond_with(monkeypatch, _Resp(json.dumps({
        "verdicts": [{"name": "satya  nadella", "valid": False}]
    })))
    monkeypatch.setattr("app.providers.cache.get", lambda *a, **k: None)
    monkeypatch.setattr("app.providers.cache.set", lambda *a, **k: None)
    assert entity_filter.validate(["Satya Nadella"], "person") == set()


def test_classifier_rejects_a_type_outside_the_allowed_set(monkeypatch):
    _respond_with(monkeypatch, _Resp(json.dumps({
        "labels": [{"index": 1, "type": "podcast_guest", "confidence": 0.9}]
    })))
    monkeypatch.setattr("app.providers.cache.get", lambda *a, **k: None)
    monkeypatch.setattr("app.providers.cache.set", lambda *a, **k: None)
    out = relation_classifier.classify([{"a": "A", "b": "B", "evidence": "e"}])
    # 'podcast_guest' is asserted only by an RSS feed; never guessed onto prose.
    assert out[0]["type"] == "unknown"


def test_classifier_clamps_confidence_into_range(monkeypatch):
    _respond_with(monkeypatch, _Resp(json.dumps({
        "labels": [{"index": 1, "type": "coworker", "confidence": 4.2}]
    })))
    monkeypatch.setattr("app.providers.cache.get", lambda *a, **k: None)
    monkeypatch.setattr("app.providers.cache.set", lambda *a, **k: None)
    out = relation_classifier.classify([{"a": "A", "b": "B", "evidence": "e"}])
    assert out[0] == {"type": "coworker", "confidence": 1.0}
