"""Phase 1 of the homonym-guard fix: verify a person's identity against
whatever signal is available (user-given context, or evidence already found
this same pass) before trusting an OpenAlex coauthor match -- instead of
either blanket-trusting it (the old behavior) or blanket-skipping it whenever
context happens to be given (also the old behavior, and just as blind, in
the opposite direction).

Concrete motivating case (live, this session): searching /connect for
"Prantik Chakraborty" (a Trinamix VP Sales) picked up ~170 "coauthor" edges
belonging to an unrelated ISRO radar researcher who happens to share the
name, because OpenAlex's bare-name search resolved to the wrong author and
nothing checked whether that author's own affiliation had anything to do
with the actual subject.
"""
from app.graph import builder, disambiguate, expansion
from app.models import RelationshipEdge
from app.providers import openalex as openalex_module


# ---------------------------------------------------------------------------
# disambiguate.py: the new "business" domain bucket
# ---------------------------------------------------------------------------
def test_business_domain_detects_a_vp_sales_title():
    assert "business" in disambiguate.domains_of(
        "works at Trinamix Inc as Vice President Sales & Strategy")


def test_business_and_science_are_reported_as_conflicting():
    signal = "Vice President Sales & Strategy at Trinamix Inc, San Ramon, California."
    candidate = "Prantik Chakraborty, affiliated with Indian Space Research Organisation, researcher."
    assert disambiguate.domain_conflict(signal, candidate) is True


def test_business_domain_does_not_collide_with_venture():
    # "managing director" (business) is a materially different role from
    # "general partner" (venture) -- both being present in the same short
    # bio is common (e.g. a PE-adjacent operator) and should NOT be forced
    # into a false conflict by an overly broad bucket.
    text = "managing director and general partner"
    assert {"business", "venture"} <= disambiguate.domains_of(text)


# ---------------------------------------------------------------------------
# providers/openalex.py: identity_text alongside author resolution
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload

    def json(self):
        return self._payload


def _patch_openalex_cache_miss(monkeypatch):
    monkeypatch.setattr(openalex_module.cache, "get", lambda key, track=True: None)
    monkeypatch.setattr(openalex_module.cache, "set", lambda *a, **k: None)


def test_resolve_author_returns_institution_as_identity_text(monkeypatch):
    _patch_openalex_cache_miss(monkeypatch)
    provider = openalex_module.OpenAlexProvider()

    def fake_request(method, url, provider=None, params=None, **kw):
        return _FakeResponse({"results": [{
            "id": "https://openalex.org/A123",
            "display_name": "Prantik Chakraborty",
            "works_count": 12,
            "last_known_institutions": [{"display_name": "Indian Space Research Organisation"}],
        }]})

    monkeypatch.setattr(openalex_module, "request_with_retry", fake_request)
    author_id, identity = provider._resolve_author("Prantik Chakraborty")

    assert author_id == "A123"
    assert "Indian Space Research Organisation" in identity
    # "academic" is load-bearing, not decorative: an institution's bare NAME
    # has no profession keyword in it on its own, so without an explicit
    # "academic author" framing, disambiguate.domains_of() on this string
    # comes back empty and domain_conflict silently never fires -- confirmed
    # against the REAL live OpenAlex API for this exact name before this
    # wording was added (it resolves to the same wrong ISRO-affiliated
    # author in production right now).
    assert "academic" in identity.lower()


def test_identity_text_still_says_academic_author_with_no_institution_on_record(monkeypatch):
    """No institution on record doesn't mean no signal: passing the
    works_count/name-similarity guard IS a real published record on its
    own, so "an academic author" still holds and still gives
    domain_conflict something to match on (see the "academic" comment in
    _resolve_author for why a bare institution name alone isn't enough)."""
    _patch_openalex_cache_miss(monkeypatch)
    provider = openalex_module.OpenAlexProvider()

    def fake_request(method, url, provider=None, params=None, **kw):
        return _FakeResponse({"results": [{
            "id": "https://openalex.org/A999",
            "display_name": "Someone Else",
            "works_count": 12,
            "last_known_institutions": [],
        }]})

    monkeypatch.setattr(openalex_module, "request_with_retry", fake_request)
    identity = provider.identity_text("Someone Else")
    assert "academic" in identity.lower()
    assert "affiliated with" not in identity  # no institution to name


def test_identity_text_empty_when_name_does_not_resolve(monkeypatch):
    _patch_openalex_cache_miss(monkeypatch)
    provider = openalex_module.OpenAlexProvider()
    monkeypatch.setattr(openalex_module, "request_with_retry",
                        lambda *a, **k: _FakeResponse({"results": []}))
    assert provider.identity_text("Nobody") == ""


# ---------------------------------------------------------------------------
# graph/expansion.py: _process_person's phase 4b gate, end to end
# ---------------------------------------------------------------------------
def _silence_everything_but_openalex(monkeypatch, coauthors_text="", identity_text=""):
    """Stub every ORCH call _process_person makes EXCEPT coauthors_enrichment,
    so a test can isolate the new gating behavior without mocking the entire
    search/scrape/extract pipeline."""
    monkeypatch.setattr(expansion.ORCH, "enrich_person", lambda name: None)
    monkeypatch.setattr(expansion.ORCH, "officer_enrichment", lambda name: {"officers_text": ""})
    monkeypatch.setattr(expansion.ORCH, "edgar_enrichment", lambda name: {"edgar_text": ""})
    monkeypatch.setattr(expansion.ORCH, "firm_enrichment", lambda name: {"firms": []})
    monkeypatch.setattr(expansion.config, "PODCASTS_ENABLED", False)
    monkeypatch.setattr(expansion.ORCH, "dedup", lambda pairs: ([], {}))
    monkeypatch.setattr(expansion.ORCH, "coauthors_enrichment", lambda name: {
        "coauthors": [{"name": "Some Coauthor", "count": 1}] if coauthors_text else [],
        "coauthors_text": coauthors_text,
        "identity_text": identity_text,
    })


def test_conflicting_openalex_identity_is_rejected(db, monkeypatch):
    _silence_everything_but_openalex(
        monkeypatch,
        coauthors_text="Prantik Chakraborty coauthor of Some Coauthor.",
        identity_text="Prantik Chakraborty, affiliated with Indian Space Research Organisation, researcher.",
    )

    expansion._process_person(
        db, "Prantik Chakraborty", 0, {},
        context="Vice President Sales & Strategy at Trinamix Inc",
    )

    subject = builder.get_or_create_person(db, "Prantik Chakraborty")
    assert (subject.meta or {}).get("openalex_rejected") is not None
    # the bogus coauthor must never have been persisted as any edge at all
    # (checked via the edge table, not a specific Person/Org lookup, since
    # spaCy's NER on a bare "X coauthor of Y." template sometimes tags a
    # real person's name as an organization -- a separate, pre-existing
    # extraction-accuracy quirk this fix isn't trying to address)
    assert db.query(RelationshipEdge).count() == 0


def test_non_conflicting_openalex_identity_is_accepted(db, monkeypatch):
    _silence_everything_but_openalex(
        monkeypatch,
        coauthors_text="Larry Ellison coauthor of Harshita Tolani.",
        identity_text="",  # no institution on record -- nothing to conflict with
    )

    expansion._process_person(db, "Larry Ellison", 0, {}, context="")

    assert db.query(RelationshipEdge).count() == 1, \
        "with no conflicting signal, the coauthor edge should land"


def test_no_signal_at_all_degrades_to_accepting_openalex(db, monkeypatch):
    """No context AND no other evidence gathered this pass (the hardest,
    genuinely-unsolved case: a brand-new node with nothing else known yet) --
    domain_conflict is silent when either side is unclear, so this can't be
    caught here. Documenting the honest limit, not just the win."""
    _silence_everything_but_openalex(
        monkeypatch,
        coauthors_text="Devang Mankad coauthor of Shrija Bhattacharyya.",
        identity_text="Devang Mankad, affiliated with Some University, researcher.",
    )

    expansion._process_person(db, "Devang Mankad", 0, {}, context="")

    assert db.query(RelationshipEdge).count() == 1
