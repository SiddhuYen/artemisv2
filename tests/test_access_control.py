"""Access control: the shared-token gate and the per-client rate limiter.

Artemis spends real money per build (search quota + Anthropic tokens), so an
open URL is a bill anyone with the link can run up. These tests pin the two
properties that make a public deployment safe: nothing but the allowlist is
reachable without the token, and no single caller can run unbounded builds even
with it.
"""
import pytest
from fastapi.testclient import TestClient

from app import auth, config
import app.main as M


TOKEN = "test-token-abc123"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(config, "ACCESS_TOKEN", TOKEN)
    auth.build_limiter.reset()
    auth.login_limiter.reset()
    with TestClient(M.app) as c:
        yield c
    auth.build_limiter.reset()
    auth.login_limiter.reset()


@pytest.fixture
def open_client(monkeypatch):
    """No token configured — a local checkout, deliberately wide open."""
    monkeypatch.setattr(config, "ACCESS_TOKEN", "")
    auth.build_limiter.reset()
    with TestClient(M.app) as c:
        yield c
    auth.build_limiter.reset()


# --- the gate ---------------------------------------------------------------
def test_api_request_without_a_token_is_rejected(client):
    res = client.get("/status")
    assert res.status_code == 401
    assert res.json()["detail"] == "authentication required"


def test_bearer_token_is_accepted(client):
    res = client.get("/status", headers={"Authorization": f"Bearer {TOKEN}"})
    assert res.status_code == 200


def test_wrong_token_is_rejected(client):
    res = client.get("/status", headers={"Authorization": "Bearer wrong"})
    assert res.status_code == 401


def test_healthz_is_reachable_without_a_token(client):
    """A load balancer must be able to probe a locked app or it will never
    route traffic to it in the first place."""
    assert client.get("/healthz").status_code == 200


def test_browser_navigation_redirects_to_login(client):
    res = client.get("/ui/", headers={"Accept": "text/html"}, follow_redirects=False)
    assert res.status_code == 303
    assert res.headers["location"] == "/login"


def test_login_page_is_reachable_while_locked_out(client):
    res = client.get("/login", headers={"Accept": "text/html"})
    assert res.status_code == 200
    assert "Access token" in res.text


def test_static_ui_is_gated_too(client):
    """The middleware covers the mounted StaticFiles app, not just API routes —
    otherwise the whole frontend is readable by anyone with the URL."""
    res = client.get("/ui/app.js")
    assert res.status_code in (401, 303)


def test_no_token_configured_leaves_everything_open(open_client):
    assert open_client.get("/status").status_code == 200
    assert open_client.get("/status").json()["auth"]["required"] is False


# --- login flow -------------------------------------------------------------
def test_login_sets_a_session_cookie_that_authenticates(client):
    res = client.post("/login", json={"token": TOKEN})
    assert res.status_code == 200
    assert config.SESSION_COOKIE in res.cookies
    # the cookie now carries the session; no Authorization header needed
    assert client.get("/status").status_code == 200


def test_session_cookie_never_contains_the_raw_token(client):
    """A cookie is stored on disk and easy to leak; the token must not be in it."""
    res = client.post("/login", json={"token": TOKEN})
    assert TOKEN not in res.cookies[config.SESSION_COOKIE]
    assert TOKEN not in res.headers.get("set-cookie", "")


def test_session_cookie_is_httponly(client):
    res = client.post("/login", json={"token": TOKEN})
    assert "httponly" in res.headers["set-cookie"].lower()


def test_login_with_a_bad_token_401s(client):
    assert client.post("/login", json={"token": "nope"}).status_code == 401


def test_logout_clears_the_session(client):
    client.post("/login", json={"token": TOKEN})
    assert client.get("/status").status_code == 200
    client.post("/logout")
    client.cookies.clear()
    assert client.get("/status").status_code == 401


def test_login_attempts_are_rate_limited(monkeypatch, client):
    """A shared token on a public URL has no lockout of its own."""
    monkeypatch.setattr(auth.login_limiter, "limit", 3)
    auth.login_limiter.reset()
    codes = [client.post("/login", json={"token": "wrong"}).status_code
             for _ in range(5)]
    assert 429 in codes, f"brute force was never throttled: {codes}"


# --- rate limiting ----------------------------------------------------------
def test_build_endpoints_are_rate_limited(monkeypatch, open_client):
    """The limiter applies even without auth: the token is shared, so holding
    it is not evidence a caller should get unbounded builds."""
    monkeypatch.setattr(auth.build_limiter, "limit", 2)
    auth.build_limiter.reset()
    # never actually build -- reserve() is enough to prove admission ran
    monkeypatch.setattr(M.threading, "Thread", lambda **kw: type(
        "T", (), {"start": lambda self: None})())

    codes = [open_client.post("/discover", json={"person_name": "X"}).status_code
             for _ in range(4)]
    assert codes[:2] == [200, 200]
    assert 429 in codes[2:], f"expected throttling after 2 builds, got {codes}"


def test_rate_limited_response_carries_retry_after(monkeypatch, open_client):
    monkeypatch.setattr(auth.build_limiter, "limit", 1)
    auth.build_limiter.reset()
    monkeypatch.setattr(M.threading, "Thread", lambda **kw: type(
        "T", (), {"start": lambda self: None})())
    open_client.post("/discover", json={"person_name": "X"})
    res = open_client.post("/discover", json={"person_name": "Y"})
    assert res.status_code == 429
    assert int(res.headers["Retry-After"]) >= 0


def test_reads_are_never_rate_limited(monkeypatch, open_client):
    """Throttling a dashboard just makes it look broken."""
    monkeypatch.setattr(auth.build_limiter, "limit", 1)
    auth.build_limiter.reset()
    for _ in range(10):
        assert open_client.get("/status").status_code == 200


# --- limiter unit behavior --------------------------------------------------
def test_limiter_buckets_are_per_client():
    lim = auth.RateLimiter(limit=1, window_s=3600)
    assert lim.check("1.1.1.1")[0] is True
    assert lim.check("1.1.1.1")[0] is False
    assert lim.check("2.2.2.2")[0] is True, "one client must not exhaust another's budget"


def test_limiter_refills_over_time():
    lim = auth.RateLimiter(limit=60, window_s=60)  # 1 token/second
    for _ in range(60):
        lim.check("c")
    assert lim.check("c")[0] is False
    lim._buckets["c"] = (0.0, lim._buckets["c"][1] - 5.0)  # simulate 5s passing
    assert lim.check("c")[0] is True


def test_client_key_prefers_the_rightmost_forwarded_hop(monkeypatch):
    """Leftmost XFF entries are caller-supplied and forgeable; the rightmost is
    the address our own proxy observed."""
    monkeypatch.setattr(config, "TRUST_PROXY_HEADERS", True)
    headers = {"x-forwarded-for": "9.9.9.9, 203.0.113.7"}
    assert auth.client_key("10.0.0.1", headers) == "203.0.113.7"


def test_client_key_ignores_forwarded_headers_when_not_behind_a_proxy(monkeypatch):
    monkeypatch.setattr(config, "TRUST_PROXY_HEADERS", False)
    headers = {"x-forwarded-for": "9.9.9.9"}
    assert auth.client_key("10.0.0.1", headers) == "10.0.0.1"
