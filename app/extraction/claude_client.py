"""Shared Claude (Anthropic API) client for the extraction layer.

Every LLM stage in Artemis — per-source extraction, the entity filter, the
relationship classifier — goes through `call_json()` here. It exists so those
three stages share one client, one concurrency budget, and one failure
contract:

    call_json(...) -> parsed dict on success, None on ANY failure.

`None` is the whole error protocol. A missing API key, a network blip, a rate
limit the SDK exhausted its retries on, a safety refusal, a truncated
response — all collapse to None, and every caller treats None the same way it
used to treat an unreachable local daemon: fall back to the deterministic path
and keep going. No LLM failure can take down a graph build.

Responses are constrained with a JSON schema (`output_config.format`), so a
successful call is guaranteed to be valid JSON matching the caller's shape.
That replaces the regex-and-hope parsing the local-model code needed, and it
also sidesteps the failure mode where a model narrates around its own answer.
"""
from __future__ import annotations

import json
import threading
from typing import Optional

from .. import config
from . import usage

# One client for the process. Built lazily (not at import) so that merely
# importing the app never requires a key -- the CLI, the tests, and a
# no-key deployment all import this module and run fine without one.
_client = None
_client_lock = threading.Lock()
_availability: Optional[bool] = None

# Bound Claude calls in flight, process-wide. Expansion runs
# EXPAND_NODE_CONCURRENCY nodes at once and connect_people runs two of those
# side by side, so without a cap a single build can fan out far past the
# account's rate limit -- and unlike a search provider, every one of those
# calls costs money. Callers queue here instead of firing a burst that 429s.
_SLOTS = threading.Semaphore(config.CLAUDE_MAX_CONCURRENCY)


def _get_client():
    """The shared Anthropic client, or None when no credentials resolve.

    The SDK resolves credentials itself (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
    or an `ant auth login` profile), so an unset ARTEMIS-side key does not by
    itself mean there are no credentials -- construct and let the SDK decide.
    """
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        try:
            import anthropic
        except ImportError:
            return None
        try:
            kwargs = {
                "timeout": config.CLAUDE_TIMEOUT,
                "max_retries": config.CLAUDE_MAX_RETRIES,
            }
            if config.ANTHROPIC_API_KEY:
                kwargs["api_key"] = config.ANTHROPIC_API_KEY
            _client = anthropic.Anthropic(**kwargs)
        except Exception:
            # No resolvable credentials (or a malformed one). Availability is
            # cached below, so this is attempted once, not per call.
            return None
        return _client


# Models that reject `output_config.effort` outright (400) rather than
# ignoring it. Seeded with the ones known not to support it — Haiku 4.5 is the
# default CLAUDE_BATCH_MODEL, so this is the common path, not an edge case —
# and extended at runtime by _note_effort_unsupported() so a model released
# after this list was written self-corrects after one failed call instead of
# silently no-opping a whole stage.
_EFFORT_UNSUPPORTED = {"claude-haiku-4-5", "claude-sonnet-4-5"}
_effort_lock = threading.Lock()


def _supports_effort(model: str) -> bool:
    return model not in _EFFORT_UNSUPPORTED


def _note_effort_unsupported(model: str) -> None:
    with _effort_lock:
        _EFFORT_UNSUPPORTED.add(model)


def _is_effort_rejection(exc: Exception) -> bool:
    """True when the API rejected the request specifically over `effort`.

    Narrow on purpose: a 400 is otherwise a bug in our request that should stay
    visible as a failure, not get silently retried in a different shape.
    """
    try:
        import anthropic
    except ImportError:
        return False
    if not isinstance(exc, anthropic.BadRequestError):
        return False
    return "effort" in str(exc).lower()


def _is_auth_failure(exc: Exception) -> bool:
    """True when `exc` means 'these credentials will never work', not 'try again'.

    Two distinct shapes. With no credential resolvable at all the SDK raises a
    plain TypeError while BUILDING the request (no network involved); with a
    credential that the server rejects it raises Authentication/PermissionDenied.
    Neither is retryable and neither will fix itself mid-build.
    """
    if isinstance(exc, TypeError):
        return "authentication" in str(exc).lower()
    try:
        import anthropic
    except ImportError:
        return False
    return isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError))


def _mark_unavailable() -> None:
    """Latch Claude off for the rest of the process after an auth failure.

    Without this, every subsequent batch in a build would re-attempt and
    re-fail — dozens of pointless round trips per graph, each one delaying the
    fallback the caller is going to take anyway.
    """
    global _availability
    _availability = False


def claude_available() -> bool:
    """True when Claude calls are worth attempting.

    Deliberately does NOT make a network call — this is read on the health
    endpoint and before every batch. It answers "is a credential resolvable",
    which has two tiers:

      - An api_key/auth_token the SDK already resolved from the environment is
        a definite yes.
      - Otherwise a credential may still exist in a form only visible at
        request time (an `ant auth login` profile, workload identity). Rather
        than reimplement the SDK's resolution order here and get it subtly
        wrong, assume yes and let the first real call settle it: an auth
        failure there latches this to False via _mark_unavailable(), so the
        cost of guessing wrong is one local, network-free exception.
    """
    global _availability
    if _availability is not None:
        return _availability
    client = _get_client()
    if client is None:
        _availability = False
    elif getattr(client, "api_key", None) or getattr(client, "auth_token", None):
        _availability = True
    else:
        return True  # unproven — left uncached so the first call can settle it
    return _availability


def credential_state() -> str:
    """What we actually KNOW about the credential, for display on /status.

    `claude_available()` is deliberately optimistic — it guesses yes so a
    profile-based credential still gets a chance — which is right for "should I
    attempt this call" and wrong for "tell the operator what's configured".
    This is the honest version:

      'configured'  — a key or token resolved; calls will be attempted
      'unavailable' — no credential, or one the API rejected; stages are off
      'unverified'  — nothing resolved locally, but a profile/workload identity
                      might still work; settled by the first real call
    """
    if _availability is False:
        return "unavailable"
    client = _get_client()
    if client is None:
        return "unavailable"
    if getattr(client, "api_key", None) or getattr(client, "auth_token", None):
        return "configured"
    return "unverified"


def reset_availability_cache() -> None:
    """Re-probe on the next call. For tests and for config changed at runtime."""
    global _availability, _client
    with _client_lock:
        _availability = None
        _client = None


def call_json(
    prompt: str,
    schema: dict,
    model: str,
    max_tokens: int = 4096,
    system: str = "",
    effort: str = "",
) -> Optional[dict]:
    """One schema-constrained Claude call. Returns the parsed object, or None.

    `schema` must be a JSON Schema object (structured outputs require
    `additionalProperties: false` on every object and an explicit `required`
    list). On success the response is guaranteed to parse and to match it.
    """
    client = _get_client()
    if client is None:
        return None

    def _attempt(with_effort: bool):
        output_config = {"format": {"type": "json_schema", "schema": schema}}
        chosen_effort = effort or config.CLAUDE_EFFORT
        if with_effort and chosen_effort:
            output_config["effort"] = chosen_effort
        return client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or config.CLAUDE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_config=output_config,
        )

    sent_effort = _supports_effort(model)
    with _SLOTS:
        try:
            resp = _attempt(sent_effort)
        except Exception as exc:  # noqa: BLE001 -- every failure is one contract
            # An `effort` rejection is our request's fault, not the caller's,
            # and it's the same for every subsequent call to this model. Drop
            # the parameter, remember that, and retry once -- otherwise the
            # very first batch would silently no-op an entire stage.
            if sent_effort and _is_effort_rejection(exc):
                _note_effort_unsupported(model)
                try:
                    resp = _attempt(with_effort=False)
                except Exception as retry_exc:  # noqa: BLE001
                    if _is_auth_failure(retry_exc):
                        _mark_unavailable()
                    return None
            else:
                # Rate limit past the SDK's own retries, connection, 5xx -- all
                # the same to the caller: no verdict this time. An AUTH failure
                # is different in one way only: it will fail identically
                # forever, so latch Claude off rather than re-failing on every
                # remaining batch.
                if _is_auth_failure(exc):
                    _mark_unavailable()
                return None

    # Before any verdict check, deliberately. A refusal and a max_tokens
    # truncation are both billed responses; skipping them here would make the
    # ledger disagree with the invoice in exactly the cases -- prompts that
    # overrun, prompts that trip a classifier -- worth noticing. See usage.py.
    usage.record(model, getattr(resp, "usage", None))

    # A safety refusal is a successful HTTP response with no usable content;
    # a max_tokens stop means the JSON is cut off mid-object. Neither is a
    # judgment, so neither should be cached or acted on.
    if getattr(resp, "stop_reason", None) in ("refusal", "max_tokens"):
        return None

    text = next((b.text for b in resp.content if b.type == "text"), "")
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None
