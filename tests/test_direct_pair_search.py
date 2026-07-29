"""_direct_pair_search's backfill-on-failed-extraction behavior.

Previously this took a fixed slice of the top SCRAPE_TOP_N search results and
processed exactly those, so a dud result (fetch failure, or a page that just
didn't name the second person) meant fewer successes rather than reaching
into the next result to compensate. It now tracks SUCCESSFUL pages and keeps
going -- bounded by however many results the search actually returned -- until
it hits the target count, then stops (so it doesn't over-fetch once satisfied
either).
"""
from dataclasses import dataclass

from app import config
from app.extraction.schemas import EdgeSignals, ExtractedEdge, ExtractionOutput
from app.graph import connect as C


@dataclass
class _FakePage:
    content: str


@dataclass
class _FakeResult:
    title: str
    url: str
    snippet: str
    provider: str = "serper"


def _edge_to(name_a, name_b):
    out = ExtractionOutput(extractor="fake")
    out.edges.append(ExtractedEdge(
        person_a=name_a, person_b=name_b, other_kind="person",
        relationship_type="coworker", method="fake", evidence_snippet="fake evidence",
        source_url="", confidence_base=0.5, confidence_adjusted=0.5,
        signals=EdgeSignals(),
    ))
    return out


def test_continues_past_a_dud_result_to_reach_the_target_success_count(db, monkeypatch):
    """Result #1 yields nothing; result #2 does. With SCRAPE_TOP_N=1, that one
    success must be enough -- result #3 should never even be fetched."""
    monkeypatch.setattr(config, "SCRAPE_TOP_N", 1)

    results = [
        _FakeResult("Dud", "https://dud.example/", "no mention here"),
        _FakeResult("Hit", "https://hit.example/", "names both people"),
        _FakeResult("Unreached", "https://unreached.example/", "should never be fetched"),
    ]
    fetched = []

    def fake_search(query, is_person=True):
        return results

    def fake_fetch(url):
        fetched.append(url)
        return _FakePage(content=f"<p>text for {url}</p>")

    def fake_extract(name_a, text, silo, evidence, source_url):
        if "hit.example" in text:
            return _edge_to(name_a, "Beta Person")
        return ExtractionOutput(extractor="fake")  # no edges -- a dud

    monkeypatch.setattr(C.ORCH, "search", fake_search)
    monkeypatch.setattr(C.ORCH, "fetch", fake_fetch)
    monkeypatch.setattr(C, "extract", fake_extract)

    found = C._direct_pair_search(db, "Alpha Person", "Beta Person")

    assert found is True
    assert "https://dud.example/" in fetched
    assert "https://hit.example/" in fetched
    assert "https://unreached.example/" not in fetched, \
        "must stop once the target success count is reached"


def test_processes_every_result_when_none_succeed(db, monkeypatch):
    """No result ever names person_b -- must try all of them, then report
    not found, without erroring."""
    monkeypatch.setattr(config, "SCRAPE_TOP_N", 2)

    results = [
        _FakeResult("A", "https://a.example/", "..."),
        _FakeResult("B", "https://b.example/", "..."),
        _FakeResult("C", "https://c.example/", "..."),
    ]
    fetched = []

    def fake_fetch(url):
        fetched.append(url)
        return _FakePage(content="<p>never mentions the target</p>")

    monkeypatch.setattr(C.ORCH, "search", lambda query, is_person=True: results)
    monkeypatch.setattr(C.ORCH, "fetch", fake_fetch)
    monkeypatch.setattr(C, "extract", lambda *a, **k: ExtractionOutput(extractor="fake"))

    found = C._direct_pair_search(db, "Alpha Person", "Beta Person")

    assert found is False
    assert len(fetched) == 3, "must try every returned result when none succeed"
