"""Org-keyed company/sector directories (providers/directory.py, phase 4f).

The safety story these tests pin down, in one line: a directory page is
evidence about WHO IT LISTS, and only the subject's own presence on it makes
it evidence about the SUBJECT. The implementation this replaced got that
wrong in both directions at once (see providers/directory.py's docstring),
so the subject-listed / subject-absent split is tested explicitly rather
than left implied.
"""
import pytest

from app import config
from app.extraction import node_profiler
from app.graph import builder, expansion
from app.models import Organization, Person, RelationshipEdge
from app.providers import directory as D
from app.providers import rosters
from app.providers.base import Page, SearchResult
from app.utils.names import org_norm_key, person_norm_key


# ---------------------------------------------------------------------------
# rosters.py — the shared guards, and the leadership subset
# ---------------------------------------------------------------------------
def test_is_roster_url_is_unchanged_without_extra_hints():
    """The extraction into rosters.py must not have widened firms.py's
    behavior: sector vocabulary is opt-in via extra_hints."""
    assert rosters.is_roster_url("https://acme.com/team")
    assert not rosters.is_roster_url("https://acme.com/attorneys")
    assert rosters.is_roster_url("https://acme.com/attorneys",
                                 extra_hints=rosters.DIRECTORY_HINTS)


def test_is_roster_url_still_rejects_homepages_aggregators_and_negatives():
    assert not rosters.is_roster_url("https://acme.com/")
    assert not rosters.is_roster_url("https://linkedin.com/company/acme/people")
    assert not rosters.is_roster_url("https://acme.com/team/careers")


def test_guard2_fails_closed_when_the_org_name_has_no_distinctive_tokens():
    """Regression from a live run. An org extracted as "GA" has no tokens
    longer than 2 chars, so org_tokens() is empty -- and Guard 2 used to
    `return True` on that ("nothing to check against"), silently disabling
    the identity check for exactly the names most likely to collide. It
    accepted https://doas.ga.gov/leadership-council, the Georgia state
    government's leadership council, and 11 of its staff were written into
    the graph. With nothing distinctive to match on, only an EXACT identity
    match is good enough."""
    assert not rosters.page_belongs_to_org(
        "https://doas.ga.gov/leadership-council",
        "<html><head><title>DOAS</title></head></html>", "GA")
    # an exact domain-stem match is still accepted
    assert rosters.page_belongs_to_org(
        "https://ga.gov/team", "<html><head><title>GA</title></head></html>", "GA")
    # ...and a name made entirely of generic words gets the same treatment
    assert not rosters.page_belongs_to_org(
        "https://someunrelatedfirm.com/team",
        "<html><head><title>Some Unrelated Firm</title></head></html>", "The Fund")


def test_domain_stem_uses_the_registrable_domain_not_the_first_label():
    """It used to return the FIRST host label, correct only for a bare
    two-label host and silently wrong for every subdomain. Guard 2 compares
    this against the org's name tokens, so any org serving its team page from
    a subdomain -- which large ones usually do -- could never verify. Live,
    all three of Georgia Tech's legitimate leadership pages were rejected."""
    assert rosters.domain_stem("https://www.hustlefund.vc/team") == "hustlefund"
    assert rosters.domain_stem("https://btv.vc/team") == "btv"
    assert rosters.domain_stem("https://research.gatech.edu/leadership") == "gatech"
    assert rosters.domain_stem("https://www.gtri.gatech.edu/about/leadership") == "gatech"
    assert rosters.domain_stem("https://team.firm.co.uk/people") == "firm"


def test_guard2_verifies_a_team_page_on_a_subdomain():
    plain = "<html><head><title>x</title></head></html>"
    assert rosters.page_belongs_to_org(
        "https://careers.thoughtbot.com/team", plain, "Thoughtbot")
    assert rosters.page_belongs_to_org(
        "https://people.uncorkcapital.com/team", plain, "Uncork Capital")


def test_org_name_from_page_can_refuse_the_domain_stem_fallback():
    """The fallback returns the domain stem when the title yields nothing --
    fine for display, circular for identity. Comparing that "declared name"
    to the org name just re-runs the domain check, which is how org "GA"
    verified against doas.ga.gov even after the domain branch was length-
    guarded against that exact match."""
    plain = "<html><head><title>x</title></head></html>"
    assert rosters.org_name_from_page(plain, "https://doas.ga.gov/x") == "Ga"
    assert rosters.org_name_from_page(
        plain, "https://doas.ga.gov/x", allow_stem_fallback=False) == ""


def test_org_chart_labels_are_not_scraped_as_people():
    """A directory interleaves section headings with actual people, and two
    capitalised words is all looks_like_person_name needs -- so "Executive
    Team" and "President's Cabinet" were being persisted as Person rows with
    employment edges. Live, the St. Thomas directory yielded exactly one
    "member": "President's Cabinet"."""
    junk = ["Executive Team", "Deputy Commissioner", "Human Resources Administration",
            "Information Technology", "President’s Cabinet", "Board of Directors",
            "Senior Management", "Office of the President"]
    assert rosters.clean_roster_names(junk) == []


def test_role_noun_surnames_are_still_kept():
    """The filter requires EVERY token to be org-chart vocabulary, precisely
    so real people whose surnames double as role nouns survive. An any-token
    rule would delete "Dean Martin" as a job title."""
    real = ["Gwen Middleton", "Dean Martin", "Bishop Allen", "Grace Yu",
            "Xiaoyu Zhang", "Oluwaseun Adeyemi"]
    assert rosters.clean_roster_names(real) == real


def test_is_leadership_url_is_strictly_narrower_than_is_roster_url():
    """A large org must get its exec page, never its full staff list."""
    assert rosters.is_leadership_url("https://acme.com/leadership")
    assert rosters.is_leadership_url("https://acme.com/about/executive-team")
    # a roster, but not a leadership roster
    assert rosters.is_roster_url("https://acme.com/staff")
    assert not rosters.is_leadership_url("https://acme.com/staff")
    assert not rosters.is_leadership_url("https://acme.com/")


# ---------------------------------------------------------------------------
# sector pack selection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("industry,expected_query_fragment", [
    ("corporate law firm", "attorneys"),
    ("regional hospital network", "physicians"),
    ("Oracle ERP consulting", "consultants"),
    ("state university", "faculty"),
    ("", "team page"),
    ("unknown", "team page"),
    ("something with no pack at all", "team page"),
])
def test_pack_for_routes_industry_to_its_sector_queries(industry, expected_query_fragment):
    pack = D.pack_for(industry)
    assert any(expected_query_fragment in q for q in pack["queries"])


def test_pack_for_falls_back_to_default_rather_than_nothing():
    """An unmatched industry still gets generic queries -- degrading to the
    default pack, not to no search at all."""
    assert D.pack_for("zzzz") == config.DIRECTORY_PACKS["default"]


# ---------------------------------------------------------------------------
# DirectoryProvider — size gating and the guards
# ---------------------------------------------------------------------------
def _provider(monkeypatch, results, pages):
    """A DirectoryProvider whose search returns `results` and whose fetches
    resolve from the `pages` {url: html} map."""
    calls = []

    def fake_search(query):
        calls.append(query)
        return results

    def fake_fetch(url):
        return Page(url=url, status_code=200, content=pages.get(url, ""))

    monkeypatch.setattr(D, "fetch_readable", fake_fetch)
    provider = D.DirectoryProvider(search=fake_search)
    return provider, calls


_ACME_TEAM_HTML = """
<html><head><title>Our Team | Acme</title></head><body>
<div><h3>Dana Whitfield</h3><p>Chief Executive Officer</p></div>
<div><h3>Prantik Chakraborty</h3><p>Vice President Sales</p></div>
<div><h3>Molly Iyer</h3><p>Cofounder and President</p></div>
</body></html>
"""


def test_small_org_uses_the_sector_pack_and_scrapes_the_full_directory(monkeypatch):
    url = "https://acme.com/team"
    provider, calls = _provider(
        monkeypatch,
        results=[SearchResult("Our Team | Acme", url, "snippet", "serper")],
        pages={url: _ACME_TEAM_HTML},
    )
    monkeypatch.setattr(D.cache, "get", lambda key: None)
    monkeypatch.setattr(D.cache, "set", lambda *a, **k: None)

    found = provider.directory("Acme", industry="Oracle ERP consulting", size_tier="small")

    assert found["url"] == url
    assert found["leadership_only"] is False
    assert "Prantik Chakraborty" in found["members"]
    assert "Molly Iyer" in found["members"]


def test_large_org_is_restricted_to_its_leadership_page(monkeypatch):
    """A full staff directory for a large org is a haystack of weak bridges,
    so only a leadership-shaped URL is accepted."""
    staff = "https://acme.com/staff"
    provider, calls = _provider(
        monkeypatch,
        results=[SearchResult("Staff | Acme", staff, "snippet", "serper")],
        pages={staff: _ACME_TEAM_HTML},
    )
    monkeypatch.setattr(D.cache, "get", lambda key: None)
    monkeypatch.setattr(D.cache, "set", lambda *a, **k: None)

    found = provider.directory("Acme", industry="enterprise software", size_tier="large")

    assert found["url"] == ""
    assert found["members"] == []
    assert found["leadership_only"] is True
    assert all("leadership" in q or "executive" in q for q in calls)


def test_unknown_size_tier_is_treated_as_conservatively_as_large(monkeypatch):
    """An ungrounded profile must not unlock the full-directory path."""
    provider, calls = _provider(monkeypatch, results=[], pages={})
    monkeypatch.setattr(D.cache, "get", lambda key: None)
    monkeypatch.setattr(D.cache, "set", lambda *a, **k: None)

    found = provider.directory("Acme", industry="", size_tier="unknown")

    assert found["leadership_only"] is True
    assert all("leadership" in q or "executive" in q for q in calls)


def test_a_page_that_does_not_belong_to_the_org_is_rejected(monkeypatch):
    """Guard 2, inherited from rosters.py: keyword presence never attaches a
    roster -- the page must be the org's by domain or declared name."""
    url = "https://calmstorm.vc/team"
    provider, _ = _provider(
        monkeypatch,
        results=[SearchResult("Team | CalmStorm", url, "snippet", "serper")],
        pages={url: "<html><head><title>Team | CalmStorm</title></head>"
                    "<body><h3>Dana Whitfield</h3></body></html>"},
    )
    monkeypatch.setattr(D.cache, "get", lambda key: None)
    monkeypatch.setattr(D.cache, "set", lambda *a, **k: None)

    found = provider.directory("Storm Ventures", industry="", size_tier="small")
    assert found["url"] == ""
    assert found["members"] == []


def test_overflow_is_reported_when_the_page_lists_more_than_the_cap(monkeypatch):
    names = "".join(f"<h3>Person Number{i}</h3>" for i in range(80))
    url = "https://acme.com/team"
    provider, _ = _provider(
        monkeypatch,
        results=[SearchResult("Our Team | Acme", url, "s", "serper")],
        pages={url: f"<html><head><title>Our Team | Acme</title></head>"
                    f"<body>{names}</body></html>"},
    )
    monkeypatch.setattr(D.cache, "get", lambda key: None)
    monkeypatch.setattr(D.cache, "set", lambda *a, **k: None)

    found = provider.directory("Acme", industry="", size_tier="small")
    assert found["overflow"] is True
    assert len(found["members"]) == config.DIRECTORY_MAX_MEMBERS


# ---------------------------------------------------------------------------
# expansion phase 4f — the evidence rule
# ---------------------------------------------------------------------------
def _silence_everything_but_directory(monkeypatch, found):
    monkeypatch.setattr(expansion.ORCH, "enrich_person", lambda name: None)
    monkeypatch.setattr(expansion.ORCH, "officer_enrichment", lambda name: {"officers_text": ""})
    monkeypatch.setattr(expansion.ORCH, "edgar_enrichment", lambda name: {"edgar_text": ""})
    monkeypatch.setattr(expansion.ORCH, "firm_enrichment", lambda name: {"firms": []})
    monkeypatch.setattr(expansion.ORCH, "coauthors_enrichment", lambda name: {
        "coauthors": [], "coauthors_text": "", "identity_text": "",
    })
    monkeypatch.setattr(expansion.coauthor_plausibility, "is_active", lambda: False)
    monkeypatch.setattr(expansion.config, "PODCASTS_ENABLED", False)
    monkeypatch.setattr(expansion.ORCH, "dedup", lambda pairs: ([], {}))
    monkeypatch.setattr(expansion, "_repeat_candidates", lambda edges: [])
    monkeypatch.setattr(node_profiler, "is_active", lambda: False)
    monkeypatch.setattr(expansion.search_strategy, "is_active", lambda: False)
    monkeypatch.setattr(expansion.ORCH, "search", lambda query, is_person=True: [])
    monkeypatch.setattr(expansion.ORCH, "directory_enrichment",
                        lambda org, industry="", size_tier="": found)


def _org_affiliation_edge(org_name):
    from app.extraction.schemas import EdgeSignals, ExtractedEdge
    return ExtractedEdge(
        person_a="Prantik Chakraborty", person_b="", other_kind="organization",
        organization=org_name, relationship_type="employee",
        confidence_base=0.7, confidence_adjusted=0.7,
        evidence_snippet="Prantik Chakraborty, VP Sales at Acme.",
        source_url="http://x/affiliation", signals=EdgeSignals(explicit_keyword_match=True),
    )


def _edges_for(db, name):
    person = db.query(Person).filter(
        Person.norm_name == person_norm_key(name)).one_or_none()
    if person is None:
        return []
    return db.query(RelationshipEdge).filter(
        RelationshipEdge.person_a_id == person.id).all()


def test_subject_listed_on_the_directory_yields_colleague_edges(db, monkeypatch):
    _silence_everything_but_directory(monkeypatch, {
        "org": "Acme", "url": "https://acme.com/team", "overflow": False,
        "members": ["Prantik Chakraborty", "Molly Iyer", "Dana Whitfield"],
    })
    monkeypatch.setattr(expansion, "_best_org_affiliation_edge",
                        lambda edges: _org_affiliation_edge("Acme"))

    expansion._process_person(db, "Prantik Chakraborty", 0, {},
                              enhanced_professional_search=True)

    subject_edges = _edges_for(db, "Prantik Chakraborty")
    peers = {e.person_b.canonical_name for e in subject_edges if e.person_b_id}
    assert "Molly Iyer" in peers
    assert "Dana Whitfield" in peers


def test_directory_colleagues_bypass_ner_and_keep_non_anglo_names(db, monkeypatch):
    """Regression: phase 4f must NOT round-trip scraped roster names through
    the prose extractor.

    It used to build "X coworker of Y at Org." sentences and re-extract them,
    which put spaCy NER in the path of names already accepted by
    rosters.clean_roster_names. en_core_web_sm tags "Dana Whitfield" PERSON
    but "Molly Iyer" and "Prantik Chakraborty" ORG, so the round trip
    silently dropped the non-Anglo names off a page that structurally
    asserted every one of them -- a systematic bias against precisely the
    non-famous people this feature exists to find. These exact names are the
    ones that exposed it; they are the test on purpose.
    """
    _silence_everything_but_directory(monkeypatch, {
        "org": "Acme", "url": "https://acme.com/team", "overflow": False,
        "members": ["Prantik Chakraborty", "Molly Iyer", "Dana Whitfield",
                    "Xiaoyu Zhang", "Oluwaseun Adeyemi"],
    })
    monkeypatch.setattr(expansion, "_best_org_affiliation_edge",
                        lambda edges: _org_affiliation_edge("Acme"))

    expansion._process_person(db, "Prantik Chakraborty", 0, {},
                              enhanced_professional_search=True)

    peers = {e.person_b.canonical_name
             for e in _edges_for(db, "Prantik Chakraborty") if e.person_b_id}
    assert peers == {"Molly Iyer", "Dana Whitfield", "Xiaoyu Zhang",
                     "Oluwaseun Adeyemi"}, (
        "every roster member except the subject must become a colleague edge, "
        "regardless of what NER would have made of their name"
    )


def test_directory_colleague_edges_are_candidate_tier_not_strong(db, monkeypatch):
    """Co-listing on one roster is a shared affiliation, not a close working
    relationship -- it must not manufacture strong-tier edges."""
    _silence_everything_but_directory(monkeypatch, {
        "org": "Acme", "url": "https://acme.com/team", "overflow": False,
        "members": ["Prantik Chakraborty", "Molly Iyer"],
    })
    monkeypatch.setattr(expansion, "_best_org_affiliation_edge",
                        lambda edges: _org_affiliation_edge("Acme"))

    expansion._process_person(db, "Prantik Chakraborty", 0, {},
                              enhanced_professional_search=True)

    edges = [e for e in _edges_for(db, "Prantik Chakraborty") if e.person_b_id]
    assert edges
    for e in edges:
        assert e.confidence_raw <= config.STRONG_MIN
        assert e.status != "strong"


def test_subject_absent_from_the_directory_yields_no_colleague_claim(db, monkeypatch):
    """The core rule. A directory that does not list the subject says those
    people work there -- it says nothing about whether they know the subject,
    and the implementation this replaced asserted exactly that."""
    _silence_everything_but_directory(monkeypatch, {
        "org": "Acme", "url": "https://acme.com/leadership", "overflow": False,
        "members": ["Molly Iyer", "Dana Whitfield"],
    })
    monkeypatch.setattr(expansion, "_best_org_affiliation_edge",
                        lambda edges: _org_affiliation_edge("Acme"))

    expansion._process_person(db, "Prantik Chakraborty", 0, {},
                              enhanced_professional_search=True)

    subject_edges = _edges_for(db, "Prantik Chakraborty")
    peers = {e.person_b.canonical_name for e in subject_edges if e.person_b_id}
    assert "Molly Iyer" not in peers
    assert "Dana Whitfield" not in peers

    # ...but the employment facts ARE recorded, against the members themselves
    molly_edges = _edges_for(db, "Molly Iyer")
    assert len(molly_edges) == 1
    assert molly_edges[0].relationship_type == "employee"
    assert molly_edges[0].organization_id is not None
    assert molly_edges[0].person_b_id is None


def test_an_overflowing_directory_never_materializes_a_clique(db, monkeypatch):
    """Even with the subject listed, too many members means membership-only:
    a 200-person directory is not 200 mutual colleagues."""
    members = ["Prantik Chakraborty"] + [f"Person Number{i}" for i in range(60)]
    _silence_everything_but_directory(monkeypatch, {
        "org": "Acme", "url": "https://acme.com/team", "overflow": True,
        "members": members,
    })
    monkeypatch.setattr(expansion, "_best_org_affiliation_edge",
                        lambda edges: _org_affiliation_edge("Acme"))

    expansion._process_person(db, "Prantik Chakraborty", 0, {},
                              enhanced_professional_search=True)

    subject_edges = _edges_for(db, "Prantik Chakraborty")
    assert not [e for e in subject_edges if e.person_b_id]


def test_phase_4f_is_off_for_the_famous_shallow_side(db, monkeypatch):
    """Same gating as 4c/4d/4e: the famous side of an asymmetric walk gets a
    1-hop immediate-circle expansion, not directory enumeration."""
    calls = []
    _silence_everything_but_directory(monkeypatch, {
        "org": "Oracle", "url": "https://oracle.com/leadership", "overflow": False,
        "members": ["Someone Else"],
    })
    monkeypatch.setattr(expansion.ORCH, "directory_enrichment",
                        lambda org, industry="", size_tier="": calls.append(org) or {
                            "org": org, "url": "", "members": [], "overflow": False})
    monkeypatch.setattr(expansion, "_best_org_affiliation_edge",
                        lambda edges: _org_affiliation_edge("Oracle"))

    expansion._process_person(db, "Larry Ellison", 0, {},
                              enhanced_professional_search=False)

    assert calls == []


def test_phase_4f_passes_the_grounded_profile_through_to_the_provider(db, monkeypatch):
    """Size/industry gating only works if 4d's profile actually reaches the
    provider -- and a stale-version profile must not."""
    captured = {}

    def fake_enrichment(org, industry="", size_tier=""):
        captured.update(org=org, industry=industry, size_tier=size_tier)
        return {"org": org, "url": "", "members": [], "overflow": False}

    _silence_everything_but_directory(monkeypatch, {})
    monkeypatch.setattr(expansion.ORCH, "directory_enrichment", fake_enrichment)
    monkeypatch.setattr(expansion, "_best_org_affiliation_edge",
                        lambda edges: _org_affiliation_edge("Acme"))

    org = builder.get_or_create_org(db, "Acme")
    org.meta = {"profile": {"v": config.NODE_PROFILE_VERSION, "size_tier": "small",
                            "industry": "Oracle ERP consulting",
                            "summary": "", "grounded": True}}
    db.commit()

    expansion._process_person(db, "Prantik Chakraborty", 0, {},
                              enhanced_professional_search=True)

    assert captured["size_tier"] == "small"
    assert captured["industry"] == "Oracle ERP consulting"


def test_phase_4f_ignores_a_stale_profile_and_gates_conservatively(db, monkeypatch):
    captured = {}

    def fake_enrichment(org, industry="", size_tier=""):
        captured.update(org=org, industry=industry, size_tier=size_tier)
        return {"org": org, "url": "", "members": [], "overflow": False}

    _silence_everything_but_directory(monkeypatch, {})
    monkeypatch.setattr(expansion.ORCH, "directory_enrichment", fake_enrichment)
    monkeypatch.setattr(expansion, "_best_org_affiliation_edge",
                        lambda edges: _org_affiliation_edge("Acme"))

    org = builder.get_or_create_org(db, "Acme")
    org.meta = {"profile": {"v": config.NODE_PROFILE_VERSION - 1, "size_tier": "small",
                            "industry": "Oracle ERP consulting",
                            "summary": "", "grounded": True}}
    db.commit()

    expansion._process_person(db, "Prantik Chakraborty", 0, {},
                              enhanced_professional_search=True)

    # stale -> treated as no profile -> leadership-only path in the provider
    assert captured["size_tier"] == ""
    assert captured["industry"] == ""
