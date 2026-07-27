"""Headless-browser rendering for JavaScript-only pages (OPTIONAL).

Many firm team-roster pages are single-page apps: a plain HTTP GET returns an
empty shell, so firms.py scrapes zero members and that firm's roster edges
never enter the graph. Rendering the page with a real browser recovers the DOM
the JS builds, then the SAME extraction (firms.py's guards + name-shape
filtering) runs on it. This changes nothing structurally — a rendered roster
is the identical assertion (the page lists its team), just finally readable.

OPTIONAL by design. Playwright + a Chromium binary are a heavy dependency
(not in requirements.txt), so the whole system runs without them —
`available()` returns False and every caller (see base.fetch_page) falls back
to the plain fetch. Enable with:

    pip install playwright && python -m playwright install chromium
"""
from __future__ import annotations

import threading
from typing import Optional, Tuple

from .. import config

_lock = threading.Lock()
_playwright = None
_browser = None
_available: Optional[bool] = None


def available() -> bool:
    """True iff Playwright is importable AND a Chromium binary is installed.

    Checked once and cached. A missing browser binary is the common case
    after `pip install playwright` without `playwright install`, so a launch
    is actually attempted rather than trusting the import alone.
    """
    global _available
    if _available is not None:
        return _available
    with _lock:
        if _available is not None:
            return _available
        try:
            _ensure_browser_locked()
            _available = True
        except Exception:
            _available = False
    return _available


def _ensure_browser_locked():
    """Launch (or reuse) a headless Chromium. Caller must hold `_lock`."""
    global _playwright, _browser
    if _browser is not None and _browser.is_connected():
        return
    from playwright.sync_api import sync_playwright  # optional dependency

    if _playwright is None:
        _playwright = sync_playwright().start()
    _browser = _playwright.chromium.launch(headless=True)


def _reset_locked():
    global _browser
    try:
        if _browser is not None:
            _browser.close()
    except Exception:
        pass
    _browser = None


def render_html(url: str) -> Tuple[str, int]:
    """Fully-rendered HTML for `url`, or ("", 0) when rendering is unavailable
    or fails. Waits for the network to settle so JS-built rosters are present.

    A blank first attempt retries once with a fresh browser: a crashed or
    disconnected singleton must not permanently disable rendering.
    """
    if not available():
        return "", 0
    for _attempt in range(2):
        with _lock:
            try:
                _ensure_browser_locked()
                page = _browser.new_page(
                    user_agent=config.USER_AGENT,
                    viewport={"width": 1280, "height": 2400})
                try:
                    resp = page.goto(url, wait_until="domcontentloaded",
                                     timeout=int(config.BROWSER_TIMEOUT_S * 1000))
                    status = resp.status if resp is not None else 0
                    # Give client-side rendering a moment to populate the DOM;
                    # networkidle is best-effort and may time out on chatty pages.
                    try:
                        page.wait_for_load_state(
                            "networkidle",
                            timeout=int(config.BROWSER_SETTLE_S * 1000))
                    except Exception:
                        pass
                    html = page.content()
                    return html[: config.MAX_HTML_CHARS], status
                finally:
                    page.close()
            except Exception:
                _reset_locked()  # drop the (possibly dead) browser and retry once
    return "", 0


def shutdown() -> None:
    """Release the browser and Playwright driver (for a clean process exit)."""
    global _playwright
    with _lock:
        _reset_locked()
        try:
            if _playwright is not None:
                _playwright.stop()
        except Exception:
            pass
        _playwright = None
