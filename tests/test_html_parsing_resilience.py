"""Tests that unparseable markup degrades to an empty document instead of
raising.

soup_of is the chokepoint every HTML consumer goes through, so an exception
there propagates all the way up and aborts a whole person's expansion --
losing the other ~34 pages' evidence because one page was malformed. A real
YC cache build lost Sam Altman's entire hop-0 to a ParserRejectedMarkup.
"""
import unittest.mock as mock

from bs4 import BeautifulSoup
from bs4.exceptions import ParserRejectedMarkup

from app.utils import htmltext

_REAL_BS = BeautifulSoup


def _reject_once():
    """Patch BeautifulSoup so the parse of the caller's markup is rejected
    while soup_of's own empty-document fallback still constructs normally."""
    def fake(markup, *args, **kwargs):
        if markup:
            raise ParserRejectedMarkup("refused")
        return _REAL_BS(markup, *args, **kwargs)
    return mock.patch.object(htmltext, "BeautifulSoup", side_effect=fake)


def test_soup_of_returns_empty_document_on_rejected_markup():
    with _reject_once():
        assert str(htmltext.soup_of("<broken")) == ""


def test_html_to_text_yields_nothing_rather_than_raising():
    with _reject_once():
        assert htmltext.html_to_text("<broken") == ""


def test_text_blocks_and_jsonld_survive_rejected_markup():
    """Every soup_of consumer, not just html_to_text — firms.py reads rosters
    through text_blocks/jsonld_names on pages it did not choose."""
    with _reject_once():
        assert htmltext.text_blocks("<broken") == []
        assert htmltext.jsonld_names("<broken") == []


def test_well_formed_html_is_unaffected():
    assert htmltext.html_to_text("<p>Hello <b>world</b></p>") == "Hello world"
    assert htmltext.html_to_text("<script>junk()</script><p>Kept</p>") == "Kept"


def test_empty_input_is_not_an_error():
    assert htmltext.html_to_text("") == ""
    assert htmltext.text_blocks("") == []
