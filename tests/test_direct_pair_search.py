"""_direct_pair_search's relationship-labeling behavior.

The original version ran each page's text through all 9 silos' independent
keyword-table guesses on a single spaCy-picked "nearest sentence" per page,
so a plain co-occurrence sentence with no recognized keyword ("In office ...
President Donald Trump") got mislabeled by whichever silo's intent_default
fired first (e.g. `news` -> 'interview'), regardless of whether that label
was actually right -- and the REAL sentence stating the relationship
("He was appointed to the post by President Donald Trump...") was never even
looked at, because (a) the Wikipedia result came back via Serper carrying
`provider="serper"`, not `"wikipedia"`, so it got raw-HTML-scraped instead of
using the clean article API, and (b) only one sentence per page was ever
extracted at all.

Now (when Claude is configured): every sentence, in every fetched result,
that mentions BOTH people by name (full name or surname) is a candidate --
not just spaCy's single nearest-sentence guess -- and ALL candidates are
classified in one combined batched call to the same Claude relationship
classifier _retype_unknown_edges already uses elsewhere
(extraction.relation_classifier). A Claude "unknown" verdict is trusted as
an honest answer, not treated as a non-judgment to guess around. Wikipedia
results use the full article body (wikipedia.article_text), detected by URL
rather than by an provider label that never actually fires for this flow.

Falls back to the original single-extraction-per-page keyword guess when
Claude isn't configured at all.
"""
from dataclasses import dataclass

from app.extraction.schemas import EdgeSignals, ExtractedEdge, ExtractionOutput
from app.graph import connect as C


@dataclass
class _FakeResult:
    title: str
    url: str
    snippet: str
    provider: str = "serper"


def _edge_to(name_a, name_b, rel_type="unknown", confidence=0.1):
    out = ExtractionOutput(extractor="fake")
    out.edges.append(ExtractedEdge(
        person_a=name_a, person_b=name_b, other_kind="person",
        relationship_type=rel_type, method="fake", evidence_snippet="fake evidence",
        source_url="", confidence_base=confidence, confidence_adjusted=confidence,
        signals=EdgeSignals(),
    ))
    return out


# ── helpers ──────────────────────────────────────────────────────────────

def test_name_mention_pattern_matches_surname_alone():
    pat, _conflict = C._name_mention_pattern("Robert R Redfield")
    assert pat.search("Redfield later said...")
    assert pat.search("Robert R Redfield was born...")
    assert not pat.search("Redfielder said...")  # word boundary, not substring


def test_name_mention_pattern_conflict_catches_a_different_same_surname_person():
    """A bare surname is ambiguous for anyone sharing it with someone else
    notable -- 'Trump' alone matches Ivanka Trump, Fred Trump, Trump Tower,
    not just Donald Trump."""
    _pat, conflict = C._name_mention_pattern("Donald Trump")
    assert conflict.search("Ivanka Trump attended the event.")
    assert conflict.search("Fred Trump built the company.")
    assert not conflict.search("Donald Trump signed the order.")
    assert not conflict.search("Trump signed the order.")  # bare surname, no conflict


def test_name_mention_pattern_no_conflict_pattern_for_a_mononym():
    _pat, conflict = C._name_mention_pattern("Madonna")
    assert conflict is None


def test_split_sentences_handles_basic_prose():
    text = "He was appointed by Trump. Later, he resigned. A third sentence here."
    sentences = C._split_sentences(text)
    assert sentences == [
        "He was appointed by Trump.",
        "Later, he resigned.",
        "A third sentence here.",
    ]


def test_wikipedia_title_from_url():
    assert C._wikipedia_title_from_url(
        "https://en.wikipedia.org/wiki/Robert_R._Redfield") == "Robert R. Redfield"
    assert C._wikipedia_title_from_url("https://baltimoresun.com/some/article") is None


def test_fetch_result_text_uses_article_text_for_wikipedia_urls(monkeypatch):
    """A Wikipedia URL surfaced via a non-wikipedia provider (e.g. serper)
    must still use the clean full-article API, not raw HTML scraping."""
    res = _FakeResult("Robert R. Redfield", "https://en.wikipedia.org/wiki/Robert_R._Redfield",
                      "snippet", provider="serper")
    monkeypatch.setattr(C.ORCH.wikipedia, "article_text", lambda title: f"CLEAN ARTICLE: {title}")

    def _boom(url):
        raise AssertionError("must not raw-fetch a Wikipedia URL when article_text succeeds")

    monkeypatch.setattr(C.ORCH, "fetch", _boom)

    text = C._fetch_result_text(res)
    assert text == "CLEAN ARTICLE: Robert R. Redfield"


# ── the Claude-backed path ──────────────────────────────────────────────

def _stub_search(monkeypatch, results):
    monkeypatch.setattr(C.ORCH, "search", lambda query, is_person=True: results)


def test_claude_path_scans_every_co_mentioning_window_in_one_batch(db, monkeypatch):
    """Windows of consecutive sentences (not just one sentence at a time)
    catch a relationship stated across a sentence boundary via a pronoun --
    "Redfield studied virology. He was appointed by Trump." never says
    "Redfield" and "Trump" in the SAME sentence, but the 2-sentence window
    does, exactly like the real Wikipedia article that motivated this."""
    results = [_FakeResult("Redfield", "https://example.com/article", "...")]
    _stub_search(monkeypatch, results)
    monkeypatch.setattr(
        C, "_fetch_result_text",
        lambda res: ("Redfield studied virology. "
                     "He was appointed to the post by President Trump. "
                     "Trump later contradicted Redfield on masks."))
    monkeypatch.setattr(C.relation_classifier, "is_active", lambda: True)

    classify_calls = []

    def fake_classify(items):
        classify_calls.append(items)
        return [{"type": "unknown", "confidence": 0.0} for _ in items]

    monkeypatch.setattr(C.relation_classifier, "classify", fake_classify)

    C._direct_pair_search(db, "Robert Redfield", "Donald Trump")

    assert len(classify_calls) == 1, "must be exactly one batched call"
    evidences = [item["evidence"] for item in classify_calls[0]]
    assert "Redfield studied virology. He was appointed to the post by President Trump." in evidences, \
        "the pronoun-spanning window must be a candidate"


def test_claude_verdict_is_used_even_when_it_says_unknown(db, monkeypatch):
    """A Claude 'unknown' is an honest answer, not a signal to fall back to
    a keyword guess -- the persisted edge must stay 'unknown', not get some
    invented label."""
    results = [_FakeResult("A", "https://example.com/a", "...")]
    _stub_search(monkeypatch, results)
    monkeypatch.setattr(C, "_fetch_result_text",
                        lambda res: "Person Alpha met Person Beta at a conference.")
    monkeypatch.setattr(C.relation_classifier, "is_active", lambda: True)
    monkeypatch.setattr(C.relation_classifier, "classify",
                        lambda items: [{"type": "unknown", "confidence": 0.0} for _ in items])

    found, confident = C._direct_pair_search(db, "Person Alpha", "Person Beta")

    assert found is True  # something was persisted (honestly labeled)
    assert confident is False

    from sqlalchemy import select
    from app.models import RelationshipEdge
    edges = db.execute(select(RelationshipEdge)).scalars().all()
    assert all(e.relationship_type == "unknown" for e in edges)


def test_claude_confident_verdict_is_persisted(db, monkeypatch):
    results = [_FakeResult("A", "https://example.com/a", "...")]
    _stub_search(monkeypatch, results)
    monkeypatch.setattr(
        C, "_fetch_result_text",
        lambda res: "Person Alpha was appointed to the role by Person Beta.")
    monkeypatch.setattr(C.relation_classifier, "is_active", lambda: True)
    monkeypatch.setattr(C.relation_classifier, "classify",
                        lambda items: [{"type": "appointee", "confidence": 0.9} for _ in items])

    found, confident = C._direct_pair_search(db, "Person Alpha", "Person Beta")

    assert found is True
    assert confident is True

    from sqlalchemy import select
    from app.models import RelationshipEdge
    edges = db.execute(select(RelationshipEdge)).scalars().all()
    assert any(e.relationship_type == "appointee" for e in edges)


def test_claude_low_confidence_verdict_is_stored_as_unknown(db, monkeypatch):
    """Below CLAUDE_CLASSIFY_MIN_CONF, a specific-but-shaky verdict must not
    be trusted as that type -- store it honestly as unknown instead."""
    results = [_FakeResult("A", "https://example.com/a", "...")]
    _stub_search(monkeypatch, results)
    monkeypatch.setattr(
        C, "_fetch_result_text",
        lambda res: "Person Alpha was near Person Beta at some point.")
    monkeypatch.setattr(C.relation_classifier, "is_active", lambda: True)
    monkeypatch.setattr(
        C.relation_classifier, "classify",
        lambda items: [{"type": "coworker", "confidence": 0.2} for _ in items])  # < MIN_CONF

    C._direct_pair_search(db, "Person Alpha", "Person Beta")

    from sqlalchemy import select
    from app.models import RelationshipEdge
    edges = db.execute(select(RelationshipEdge)).scalars().all()
    assert all(e.relationship_type == "unknown" for e in edges)


def test_unknown_verdict_never_counts_as_confident_even_with_nonzero_confidence(db, monkeypatch):
    """Defensive: nothing in the classifier's schema guarantees type='unknown'
    always pairs with confidence 0.0 (every real call this session happened
    to, but that's not a contract). A non-zero confidence on an 'unknown'
    verdict must still never be reported as a confident match."""
    results = [_FakeResult("A", "https://example.com/a", "...")]
    _stub_search(monkeypatch, results)
    monkeypatch.setattr(C, "_fetch_result_text",
                        lambda res: "Person Alpha met Person Beta once.")
    monkeypatch.setattr(C.relation_classifier, "is_active", lambda: True)
    monkeypatch.setattr(
        C.relation_classifier, "classify",
        lambda items: [{"type": "unknown", "confidence": 0.9} for _ in items])

    found, confident = C._direct_pair_search(db, "Person Alpha", "Person Beta")

    assert found is True
    assert confident is False, \
        "an 'unknown'-typed edge must never count as confident, regardless of its confidence value"


def test_claude_path_excludes_a_window_naming_a_different_same_surname_person(db, monkeypatch):
    """'Ivanka Trump attended...' alongside a Redfield mention must NOT be
    treated as evidence about Donald Trump, even though it contains the bare
    surname 'Trump'."""
    results = [_FakeResult("A", "https://example.com/a", "...")]
    _stub_search(monkeypatch, results)
    monkeypatch.setattr(
        C, "_fetch_result_text",
        lambda res: "Redfield attended the gala. Ivanka Trump was also present at the event.")
    monkeypatch.setattr(C.relation_classifier, "is_active", lambda: True)

    def _boom(items):
        raise AssertionError("must not classify a window that conflicts on surname")

    monkeypatch.setattr(C.relation_classifier, "classify", _boom)

    found, confident = C._direct_pair_search(db, "Robert Redfield", "Donald Trump")

    assert found is False
    assert confident is False


def test_claude_path_finds_nothing_when_no_sentence_mentions_both(db, monkeypatch):
    results = [_FakeResult("A", "https://example.com/a", "...")]
    _stub_search(monkeypatch, results)
    monkeypatch.setattr(C, "_fetch_result_text",
                        lambda res: "Person Alpha did something unrelated entirely.")
    monkeypatch.setattr(C.relation_classifier, "is_active", lambda: True)

    def _boom(items):
        raise AssertionError("must not call classify with zero candidates")

    monkeypatch.setattr(C.relation_classifier, "classify", _boom)

    found, confident = C._direct_pair_search(db, "Person Alpha", "Person Beta")

    assert found is False
    assert confident is False


# ── the keyword fallback (Claude not configured) ────────────────────────

def test_dispatches_to_keyword_fallback_when_claude_inactive(db, monkeypatch):
    results = [_FakeResult("A", "https://a.example/", "...")]
    _stub_search(monkeypatch, results)
    monkeypatch.setattr(C, "_fetch_result_text", lambda res: "some text")
    monkeypatch.setattr(C.relation_classifier, "is_active", lambda: False)
    monkeypatch.setattr(
        C, "extract",
        lambda name_a, *a, **k: _edge_to(name_a, "Beta Person", rel_type="coworker", confidence=0.7))

    found, confident = C._direct_pair_search(db, "Alpha Person", "Beta Person")

    assert found is True
    assert confident is True  # the raw guess's own confidence (0.7) stands unmodified


def test_keyword_fallback_extracts_once_per_result_not_once_per_silo(db, monkeypatch):
    results = [_FakeResult("A", "https://a.example/", "...")]
    _stub_search(monkeypatch, results)
    monkeypatch.setattr(C, "_fetch_result_text", lambda res: "some text")
    monkeypatch.setattr(C.relation_classifier, "is_active", lambda: False)

    calls = []

    def fake_extract(name_a, text, silo, evidence, source_url):
        calls.append(silo)
        return _edge_to(name_a, "Beta Person")

    monkeypatch.setattr(C, "extract", fake_extract)

    C._direct_pair_search(db, "Alpha Person", "Beta Person")

    assert len(calls) == 1, "must extract once per result, not once per silo"


def test_keyword_fallback_no_candidates_returns_false_false(db, monkeypatch):
    results = [_FakeResult("A", "https://a.example/", "...")]
    _stub_search(monkeypatch, results)
    monkeypatch.setattr(C, "_fetch_result_text", lambda res: "some text")
    monkeypatch.setattr(C.relation_classifier, "is_active", lambda: False)
    monkeypatch.setattr(C, "extract", lambda *a, **k: ExtractionOutput(extractor="fake"))

    found, confident = C._direct_pair_search(db, "Alpha Person", "Beta Person")

    assert found is False
    assert confident is False
