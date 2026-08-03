"""Measure what subject-window narrowing saves, on real pages.

ARTEMIS_CLAUDE_EXTRACT sends a page per (subject, page) pair to the model, and
extraction/subject_windows.py narrows that to the passages about the subject.
How much that saves depends entirely on page shape, so the knobs
(ARTEMIS_SUBJECT_WINDOW_SENTENCES, _MIN_CHARS, _LOOKBACK) should be tuned
against measurements rather than intuition. This is the harness for that.

    python -m scripts.measure_subject_windows
    ARTEMIS_SUBJECT_WINDOW_SENTENCES=1 python -m scripts.measure_subject_windows

Spends NO Anthropic tokens: it assembles the real prompt locally and measures
its length. Token figures are the usual ~4 chars/token approximation and are
labelled as estimates -- exact counts would need the token-counting endpoint,
and nothing here touches the API. Everything after the fetch is the production
path (html_to_text, then subject_windows.focus, then _PROMPT_TEMPLATE), so what
is measured is what would actually be sent.

Install spaCy + en_core_web_sm before trusting the numbers: without it
sentence segmentation falls back to a regex that splits on abbreviations, which
moves every window boundary.
"""
from __future__ import annotations

import sys

import httpx

from app import config
from app.extraction import subject_windows
from app.extraction.claude_extractor import _PROMPT_TEMPLATE
from app.extraction.spacy_extractor import spacy_available
from app.utils.htmltext import html_to_text

# (subject, url, what page shape this is exercising). Wikipedia is the dense
# end of the range -- a person's own biography is the case with least to strip.
# Roster and directory pages sit at the other end, and matter more: they are
# what the Alpha walk actually scrapes for a non-famous subject.
CASES = [
    ("Robert Redfield", "https://en.wikipedia.org/wiki/Robert_R._Redfield",
     "own bio: dense, pronoun-heavy, surname re-mentions"),
    ("Anthony Fauci", "https://en.wikipedia.org/wiki/Anthony_Fauci",
     "own bio: least room to narrow"),
    ("Robert Redfield", "https://en.wikipedia.org/wiki/Anthony_Fauci",
     "another person's bio, subject absent"),
    ("Robert Redfield",
     "https://en.wikipedia.org/wiki/Centers_for_Disease_Control_and_Prevention",
     "org page: subject is one name among many"),
    ("Satya Nadella", "https://en.wikipedia.org/wiki/Microsoft",
     "large org page, subject mentioned a handful of times"),
    ("Tim Cook", "https://en.wikipedia.org/wiki/Apple_Inc.",
     "large org page, well-known executive"),
    ("Reid Hoffman", "https://en.wikipedia.org/wiki/LinkedIn",
     "company page naming its founder"),
    ("Jeff Clavier", "https://www.uncorkcapital.com/team/",
     "VC roster, subject listed"),
    ("Sandra Whitfield", "https://www.uncorkcapital.com/team/",
     "VC roster, subject not listed"),
    ("Byron Deeter", "https://www.bvp.com/team",
     "large roster, subject listed"),
]


def _sent_chars(subject: str, body: str) -> int:
    """Characters actually sent, instruction wrapper included."""
    return len(_PROMPT_TEMPLATE.format(subject=subject, text=body))


def main() -> int:
    if not spacy_available():
        print("WARNING: spaCy unavailable -- sentence splitting falls back to a "
              "regex that breaks on abbreviations; numbers below will not match "
              "production.\n", file=sys.stderr)

    print(f"MAX_PAGE_CHARS={config.MAX_PAGE_CHARS} "
          f"WINDOW={config.SUBJECT_WINDOW_SENTENCES} "
          f"MIN_CHARS={config.SUBJECT_WINDOW_MIN_CHARS} "
          f"LOOKBACK={config.SUBJECT_WINDOW_PRONOUN_LOOKBACK}\n")

    header = (f"{'subject':<18}{'page':<34}{'sent':>5}{'anch':>6}{'pron':>5}"
              f"{'seg':>5}{'chars in':>10}{'chars out':>11}{'saved':>8}")
    print(header)
    print("-" * len(header))

    total_before = total_after = 0
    notes = []
    headers = {"User-Agent": config.USER_AGENT}
    wiki_headers = {"User-Agent": config.WIKIMEDIA_USER_AGENT}
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for subject, url, note in CASES:
            try:
                resp = client.get(
                    url, headers=wiki_headers if "wikipedia.org" in url else headers)
                resp.raise_for_status()
            except Exception as exc:  # noqa: BLE001 -- a dead URL is not a failure
                print(f"{subject:<18}{url:<34}  FETCH FAILED: "
                      f"{exc.__class__.__name__}")
                continue

            text = html_to_text(resp.text)
            focused = subject_windows.focus(subject, text)
            before = _sent_chars(subject, text)
            after = 0 if focused.empty else _sent_chars(subject, focused.text)
            total_before += before
            total_after += after
            saved = 100.0 * (before - after) / before if before else 0.0

            page = url.rstrip("/").rsplit("/", 1)[-1][:32]
            print(f"{subject:<18}{page:<34}{focused.total_sentences:>5}"
                  f"{len(focused.anchors):>6}{len(focused.pronoun_anchors):>5}"
                  f"{focused.segments:>5}{before:>10,}{after:>11,}{saved:>7.1f}%")
            notes.append((subject, page, note, focused.reason, saved))

    if not total_before:
        print("\nno pages fetched")
        return 1

    print("-" * len(header))
    saved = 100.0 * (total_before - total_after) / total_before
    print(f"{'TOTAL':<57}{total_before:>10,}{total_after:>11,}{saved:>7.1f}%")
    print(f"\nest. input tokens (~4 chars/token): "
          f"{total_before // 4:,} -> {total_after // 4:,}")

    print("\nper-page:")
    for subject, page, note, reason, pct in notes:
        print(f"  {pct:5.1f}%  {subject} @ {page} -- {note} [{reason}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
