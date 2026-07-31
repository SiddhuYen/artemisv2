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
text at all, in which case there's no proximity signal to restrict by and
behavior is unchanged from before this fix (a synthetic enrichment string
or heavily-pronomial paragraph shouldn't lose everything just because the
literal name string isn't repeated).
"""
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
