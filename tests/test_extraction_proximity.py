"""Proximity gate for spacy_extract / heuristic_extract (config.ENTITY_
PROXIMITY_WINDOW). Concrete live bug this closes: searching "Eric Domski"
(a real Trinamix/Oracle-partner exec) turned up a page-wide "Oracle
Featured Speakers" conference roster. Buried elsewhere on that same long
page, entirely unrelated to Domski, was a sentence about a DIFFERENT
speaker: "Thomas Kurian is President of Oracle Product Development and
reports to ... Larry Ellison." Nothing checked that Eric Domski was
mentioned anywhere near that sentence -- so it got attributed to him
directly, producing a fabricated "Eric Domski, board_member, Larry
Ellison" edge at live, real confidence.

The fix restricts entity discovery (and the evidence sentence chosen for
it) to sentences within ENTITY_PROXIMITY_WINDOW of an actual occurrence of
the SUBJECT's own name -- unless the subject's name never appears in the
text at all, in which case there's no proximity signal to restrict by.

That fallback itself had a second live gap, found re-verifying this exact
fix against the real page: html_to_text() truncates to
config.MAX_PAGE_CHARS BEFORE either extractor ever sees the text, and on
the actual live page, Eric Domski's own bio section sat past that cutoff
while the unrelated Ellison sentence survived it -- so "no subject
mention -> accept everything" fired anyway, on a text that LOOKED
subject-free only because it had been cut off, not because it never
mentioned him. Fixed by only falling back to "accept everything" when the
text wasn't actually truncated; a truncated text with no subject mention
now rejects everything from that page instead.
"""
from app import config
from app.extraction.heuristic import heuristic_extract
from app.extraction.spacy_extractor import spacy_available, spacy_extract
from app.silos import SILO_BY_KEY

_COMPANY_SILO = SILO_BY_KEY["company"]

# Mirrors the real page shape: subject mentioned early, several unrelated
# filler sentences, then a DIFFERENT person's relationship to a third
# party far enough away to exceed the default window (2).
_ROSTER_TEXT = (
    "Featured Speakers at the conference this year include a range of "
    "industry leaders. Eric Domski will present on supply chain "
    "innovation and cloud strategy. "
    "The venue offers modern facilities for all attendees. "
    "Registration opens at eight in the morning each day. "
    "Lunch will be served in the main hall after the keynote. "
    "Thomas Kurian is President of Oracle Product Development and "
    "reports to Oracle Executive Chairman of the Board and Chief "
    "Technology Officer Larry Ellison."
)

# Same names, but Kurian/Ellison's sentence sits right next to Domski's own
# mention -- should be picked up, not filtered. Ellison named AFTER his
# title (not "Chief Technology Officer Larry Ellison" run together), since
# heuristic_extract's simple capitalized-run regex has its own separate,
# pre-existing limitation combining a leading title into one over-long
# candidate phrase and rejecting it -- irrelevant to the proximity fix this
# test targets, so worth not tripping over here.
_NEARBY_TEXT = (
    "Eric Domski will present on supply chain innovation. "
    "Larry Ellison serves as Oracle's Executive Chairman and Chief "
    "Technology Officer, and Thomas Kurian reports to him."
)

# Subject's name never appears at all -- a synthetic/paraphrased case with
# no proximity signal to restrict by.
_NO_SUBJECT_MENTION_TEXT = (
    "Larry Ellison serves as Oracle's Executive Chairman and Chief "
    "Technology Officer, and Thomas Kurian reports to him."
)


def _names(out):
    return {e.person_b for e in out.edges if e.other_kind == "person"}


# ---------------------------------------------------------------------------
# spacy_extract (the active production extractor)
# ---------------------------------------------------------------------------
def test_spacy_extract_rejects_an_entity_far_from_any_subject_mention():
    if not spacy_available():
        import pytest
        pytest.skip("spaCy model not installed in this environment")
    out = spacy_extract("Eric Domski", _ROSTER_TEXT, _COMPANY_SILO)
    assert "Larry Ellison" not in _names(out), (
        "Larry Ellison's mention is 3+ sentences from Eric Domski's own "
        "mention, on an unrelated speaker's sentence -- must not be wired "
        "to Eric Domski as if it were evidence about him")


def test_spacy_extract_keeps_an_entity_within_the_window():
    if not spacy_available():
        import pytest
        pytest.skip("spaCy model not installed in this environment")
    out = spacy_extract("Eric Domski", _NEARBY_TEXT, _COMPANY_SILO)
    assert "Larry Ellison" in _names(out)


def test_spacy_extract_falls_back_to_unrestricted_when_subject_never_named():
    if not spacy_available():
        import pytest
        pytest.skip("spaCy model not installed in this environment")
    out = spacy_extract("Eric Domski", _NO_SUBJECT_MENTION_TEXT, _COMPANY_SILO)
    assert "Larry Ellison" in _names(out), (
        "no occurrence of the subject's name anywhere in the text means "
        "no proximity signal to restrict by -- this must degrade to the "
        "old, unrestricted behavior, not silently drop everything")


# ---------------------------------------------------------------------------
# heuristic_extract (fallback when spaCy is unavailable)
# ---------------------------------------------------------------------------
def test_heuristic_extract_rejects_an_entity_far_from_any_subject_mention():
    out = heuristic_extract("Eric Domski", _ROSTER_TEXT, _COMPANY_SILO)
    assert "Larry Ellison" not in _names(out)


def test_heuristic_extract_keeps_an_entity_within_the_window():
    out = heuristic_extract("Eric Domski", _NEARBY_TEXT, _COMPANY_SILO)
    assert "Larry Ellison" in _names(out)


def test_heuristic_extract_falls_back_to_unrestricted_when_subject_never_named():
    out = heuristic_extract("Eric Domski", _NO_SUBJECT_MENTION_TEXT, _COMPANY_SILO)
    assert "Larry Ellison" in _names(out)


# ---------------------------------------------------------------------------
# Truncation edge case: a page cut to MAX_PAGE_CHARS where the subject's own
# mention didn't survive the cut, but an unrelated entity's did.
# ---------------------------------------------------------------------------
def _truncated_text_missing_subject() -> str:
    """Builds text that is exactly config.MAX_PAGE_CHARS long, containing
    the Kurian/Ellison sentence but NOT "Eric Domski" anywhere -- simulating
    a real fetched page where the subject's own section fell past the
    truncation point."""
    ellison_sentence = (
        "Thomas Kurian is President of Oracle Product Development and "
        "reports to Oracle Executive Chairman of the Board and Chief "
        "Technology Officer Larry Ellison. "
    )
    filler = "The venue offers modern facilities for all attendees. "
    text = ellison_sentence
    while len(text) < config.MAX_PAGE_CHARS:
        text += filler
    return text[: config.MAX_PAGE_CHARS]


def test_spacy_extract_rejects_everything_when_truncated_with_no_subject_mention():
    if not spacy_available():
        import pytest
        pytest.skip("spaCy model not installed in this environment")
    text = _truncated_text_missing_subject()
    assert "eric domski" not in text.lower()  # sanity: subject really isn't in this text
    out = spacy_extract("Eric Domski", text, _COMPANY_SILO)
    assert "Larry Ellison" not in _names(out), (
        "a truncated page with no subject mention must NOT fall back to "
        "accepting everything -- that's the exact live failure this closes")


def test_spacy_extract_still_falls_back_when_short_text_has_no_subject_mention():
    """The original fallback still has to work for the case it was actually
    designed for: a short text (nowhere near the truncation cap) that
    genuinely never repeats the subject's literal name."""
    if not spacy_available():
        import pytest
        pytest.skip("spaCy model not installed in this environment")
    assert len(_NO_SUBJECT_MENTION_TEXT) < config.MAX_PAGE_CHARS
    out = spacy_extract("Eric Domski", _NO_SUBJECT_MENTION_TEXT, _COMPANY_SILO)
    assert "Larry Ellison" in _names(out)


def test_heuristic_extract_rejects_everything_when_truncated_with_no_subject_mention():
    text = _truncated_text_missing_subject()
    out = heuristic_extract("Eric Domski", text, _COMPANY_SILO)
    assert "Larry Ellison" not in _names(out)
