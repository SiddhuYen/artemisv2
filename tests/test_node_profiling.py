"""Alpha piece 1: node profiling (extraction/node_profiler.py + expansion.py
phase 4d) -- "understand current node": how big is the subject's own org,
what industry is it in.

This is the highest hallucination-risk piece of Alpha: unlike
relation_classifier (one evidence sentence, one label from a fixed enum),
profiling asks for an open-ended judgment (size tier, industry) that an LLM
can fill from training-data priors when the fetched snippets say nothing
concrete. The tests below check the two-layer guard against that: the model
must self-report `grounded`, and an ungrounded response is downgraded to
"unknown" regardless of what fields it still filled in -- a plausible-
sounding guess that admits it isn't supported must not survive looking like
a fact.
"""
from app import config
from app.extraction import node_profiler
from app.extraction.schemas import EdgeSignals, ExtractedEdge
from app.graph import builder, expansion
from app.models import Organization
from app.providers.base import SearchResult


def _org_edge(org_name: str, evidence: str = "") -> ExtractedEdge:
    return ExtractedEdge(
        person_a="Subject", organization=org_name, other_kind="organization",
        relationship_type="employee", confidence_base=0.7, confidence_adjusted=0.7,
        evidence_snippet=evidence, signals=EdgeSignals(),
    )


# ---------------------------------------------------------------------------
# node_profiler.profile_org -- unit tests, Claude call mocked
# ---------------------------------------------------------------------------
def test_profile_org_returns_none_with_no_snippets(monkeypatch):
    monkeypatch.setattr(node_profiler, "is_active", lambda: True)
    assert node_profiler.profile_org("Trinamix", []) is None


def test_profile_org_returns_none_when_inactive(monkeypatch):
    monkeypatch.setattr(node_profiler, "is_active", lambda: False)
    assert node_profiler.profile_org("Trinamix", ["some snippet"]) is None


def test_profile_org_accepts_a_grounded_verdict(monkeypatch):
    monkeypatch.setattr(node_profiler, "is_active", lambda: True)
    monkeypatch.setattr(node_profiler, "call_json", lambda *a, **k: {
        "size_tier": "mid",
        "industry": "Oracle ERP consulting",
        "summary": "A mid-sized Oracle implementation consultancy.",
        "grounded": True,
    })
    profile = node_profiler.profile_org("Trinamix", ["Trinamix, a mid-size Oracle consultancy..."])
    assert profile == {
        "v": config.NODE_PROFILE_VERSION,
        "size_tier": "mid",
        "industry": "Oracle ERP consulting",
        "summary": "A mid-sized Oracle implementation consultancy.",
        "grounded": True,
    }


def test_profile_org_downgrades_ungrounded_verdict_to_unknown(monkeypatch):
    """The core guard: even if the model fills size_tier/industry in with
    something plausible-sounding, grounded=False must wipe both back to
    'unknown' -- a guess the model itself flagged as unsupported must not
    reach the caller looking like a fact."""
    monkeypatch.setattr(node_profiler, "is_active", lambda: True)
    monkeypatch.setattr(node_profiler, "call_json", lambda *a, **k: {
        "size_tier": "large",
        "industry": "enterprise software",
        "summary": "Likely a large enterprise software company.",
        "grounded": False,
    })
    profile = node_profiler.profile_org("Trinamix", ["some thin, unrelated snippet"])
    assert profile["size_tier"] == "unknown"
    assert profile["industry"] == "unknown"
    assert profile["grounded"] is False


def test_profile_org_returns_none_when_claude_call_fails(monkeypatch):
    monkeypatch.setattr(node_profiler, "is_active", lambda: True)
    monkeypatch.setattr(node_profiler, "call_json", lambda *a, **k: None)
    assert node_profiler.profile_org("Trinamix", ["a snippet"]) is None


def test_profile_org_normalizes_an_out_of_vocabulary_size_tier(monkeypatch):
    monkeypatch.setattr(node_profiler, "is_active", lambda: True)
    monkeypatch.setattr(node_profiler, "call_json", lambda *a, **k: {
        "size_tier": "gigantic",  # not one of the allowed enum values
        "industry": "consulting",
        "summary": "x",
        "grounded": True,
    })
    profile = node_profiler.profile_org("Trinamix", ["a snippet"])
    assert profile["size_tier"] == "unknown"


def test_profile_org_includes_identity_check_block_when_known_context_given(monkeypatch):
    captured = {}

    def fake_call_json(prompt, schema, model, max_tokens=4096, **kw):
        captured["prompt"] = prompt
        return {"size_tier": "unknown", "industry": "unknown", "summary": "", "grounded": False}

    monkeypatch.setattr(node_profiler, "is_active", lambda: True)
    monkeypatch.setattr(node_profiler, "call_json", fake_call_json)

    node_profiler.profile_org("Trinamix", ["a snippet"],
                              known_context="VP Sales at Trinamix Inc.")
    assert "identity check" in captured["prompt"].lower()
    assert "VP Sales at Trinamix Inc." in captured["prompt"]


def test_profile_org_omits_identity_check_block_with_no_known_context(monkeypatch):
    captured = {}

    def fake_call_json(prompt, schema, model, max_tokens=4096, **kw):
        captured["prompt"] = prompt
        return {"size_tier": "unknown", "industry": "unknown", "summary": "", "grounded": False}

    monkeypatch.setattr(node_profiler, "is_active", lambda: True)
    monkeypatch.setattr(node_profiler, "call_json", fake_call_json)

    node_profiler.profile_org("Trinamix", ["a snippet"])
    # the rules always reference "the identity check, if one was given" --
    # what must be absent with no known_context is the identity block
    # itself (the concrete anchor text and live trinamiX GmbH example).
    assert "Known context about the REAL organization" not in captured["prompt"]
    assert "trinamiX GmbH" not in captured["prompt"]


def test_profile_org_truncates_to_max_snippets_and_chars(monkeypatch):
    captured = {}

    def fake_call_json(prompt, schema, model, max_tokens=4096, **kw):
        captured["prompt"] = prompt
        return {"size_tier": "unknown", "industry": "unknown", "summary": "", "grounded": False}

    monkeypatch.setattr(node_profiler, "is_active", lambda: True)
    monkeypatch.setattr(node_profiler, "call_json", fake_call_json)

    long_snippet = "x" * (config.NODE_PROFILE_SNIPPET_CHARS + 200)
    snippets = [long_snippet] * (config.NODE_PROFILE_MAX_SNIPPETS + 5)
    node_profiler.profile_org("Trinamix", snippets)

    prompt = captured["prompt"]
    # only MAX_SNIPPETS numbered entries should appear
    assert f"[{config.NODE_PROFILE_MAX_SNIPPETS}]" in prompt
    assert f"[{config.NODE_PROFILE_MAX_SNIPPETS + 1}]" not in prompt
    # each entry truncated to NODE_PROFILE_SNIPPET_CHARS, not the full 200-longer string
    assert "x" * (config.NODE_PROFILE_SNIPPET_CHARS + 1) not in prompt


# ---------------------------------------------------------------------------
# expansion._process_person's phase 4d, end to end
# ---------------------------------------------------------------------------
def _silence_everything_but_profiling(monkeypatch, search_results=None, fetched_text=""):
    """Same shape as test_targeted_recheck.py's helper: stub every ORCH call
    except .search/.fetch, and disable phase 4c's own re-query loop
    (_repeat_candidates -> []) so phase 4d's queries are the only ones that
    hit the stubbed search/fetch."""
    monkeypatch.setattr(expansion.ORCH, "enrich_person", lambda name: None)
    monkeypatch.setattr(expansion.ORCH, "officer_enrichment", lambda name: {"officers_text": ""})
    monkeypatch.setattr(expansion.ORCH, "edgar_enrichment", lambda name: {"edgar_text": ""})
    monkeypatch.setattr(expansion.ORCH, "firm_enrichment", lambda name: {"firms": []})
    monkeypatch.setattr(expansion.ORCH, "coauthors_enrichment", lambda name: {
        "coauthors": [], "coauthors_text": "", "identity_text": "",
    })
    # Unrelated to what this file tests -- disable so it doesn't make a real
    # network call in a test.
    monkeypatch.setattr(expansion.coauthor_plausibility, "is_active", lambda: False)
    monkeypatch.setattr(expansion.config, "PODCASTS_ENABLED", False)
    monkeypatch.setattr(expansion.ORCH, "dedup", lambda pairs: ([], {}))
    monkeypatch.setattr(expansion, "_repeat_candidates", lambda edges: [])

    class _Page:
        def __init__(self, content):
            self.content = content

    monkeypatch.setattr(expansion.ORCH, "search",
                        lambda query, is_person=True: search_results or [])
    monkeypatch.setattr(expansion.ORCH, "fetch", lambda url: _Page(fetched_text))


def _get_org(db, name):
    from app.utils.names import org_norm_key
    from sqlalchemy import select
    return db.execute(
        select(Organization).where(Organization.norm_name == org_norm_key(name))
    ).scalar_one_or_none()


def test_phase_4d_profiles_and_caches_the_subjects_org(db, monkeypatch):
    _silence_everything_but_profiling(
        monkeypatch,
        search_results=[SearchResult(
            "Trinamix | LinkedIn", "https://linkedin.com/company/trinamix", "snippet", "serper")],
        fetched_text="Trinamix has 201-500 employees on LinkedIn, an Oracle ERP consultancy.",
    )
    monkeypatch.setattr(expansion, "_best_org_affiliation_edge",
                        lambda edges: _org_edge("Trinamix", "works at Trinamix"))
    monkeypatch.setattr(node_profiler, "is_active", lambda: True)
    monkeypatch.setattr(node_profiler, "profile_org", lambda org, snippets, known_context="": {
        "size_tier": "mid", "industry": "Oracle ERP consulting",
        "summary": "A mid-sized Oracle consultancy.", "grounded": True,
    })

    expansion._process_person(db, "Prantik Chakraborty", 0, {},
                              enhanced_professional_search=True)

    org = _get_org(db, "Trinamix")
    assert org is not None
    assert org.meta["profile"]["size_tier"] == "mid"
    assert org.meta["profile"]["industry"] == "Oracle ERP consulting"


def test_phase_4d_passes_the_subjects_own_affiliation_evidence_as_known_context(db, monkeypatch):
    """The identity-check fix: node_profiler needs something concrete to
    check fetched snippets against, or a search for "Trinamix" pulling in
    the unrelated "trinamiX GmbH" (the real live case this closes) has no
    way to be caught. The subject's own affiliation edge evidence is that
    anchor, and it must actually reach profile_org, not just exist."""
    captured = {}
    _silence_everything_but_profiling(
        monkeypatch,
        search_results=[SearchResult("Trinamix", "https://linkedin.com/company/trinamix",
                                     "snippet", "serper")],
        fetched_text="some fetched text",
    )
    monkeypatch.setattr(expansion, "_best_org_affiliation_edge", lambda edges: _org_edge(
        "Trinamix", "Prantik Chakraborty, Vice President Sales & Strategy at Trinamix Inc."))
    monkeypatch.setattr(node_profiler, "is_active", lambda: True)

    def fake_profile_org(org, snippets, known_context=""):
        captured["known_context"] = known_context
        return {"size_tier": "unknown", "industry": "unknown", "summary": "", "grounded": False}

    monkeypatch.setattr(node_profiler, "profile_org", fake_profile_org)

    expansion._process_person(db, "Prantik Chakraborty", 0, {},
                              enhanced_professional_search=True)

    assert captured["known_context"] == \
        "Prantik Chakraborty, Vice President Sales & Strategy at Trinamix Inc."


def test_phase_4d_is_off_for_the_famous_shallow_side(db, monkeypatch):
    """enhanced_professional_search=False (professional_only side) must not
    profile anything -- mirrors phase 4c's own gating, same reasoning: the
    famous side's notability already came from the Wikidata check upstream."""
    calls = []
    _silence_everything_but_profiling(monkeypatch)
    monkeypatch.setattr(expansion, "_best_org_affiliation_edge",
                        lambda edges: _org_edge("Oracle"))
    monkeypatch.setattr(node_profiler, "profile_org",
                        lambda org, snippets, known_context="": calls.append(org) or None)

    expansion._process_person(db, "Larry Ellison", 0, {},
                              enhanced_professional_search=False)

    assert calls == []
    assert _get_org(db, "Oracle") is None


def test_phase_4d_skips_when_no_known_org(db, monkeypatch):
    calls = []
    _silence_everything_but_profiling(monkeypatch)
    monkeypatch.setattr(expansion, "_best_org_affiliation_edge", lambda edges: None)
    monkeypatch.setattr(node_profiler, "profile_org",
                        lambda org, snippets, known_context="": calls.append(org) or None)

    expansion._process_person(db, "Nobody Notable", 0, {},
                              enhanced_professional_search=True)

    assert calls == []


def test_phase_4d_does_not_reprofile_an_already_profiled_org(db, monkeypatch):
    """Cost control: an org profiled by one colleague's expansion must not be
    re-profiled (re-searched, re-Claude-called) by the next colleague who
    happens to work there."""
    org = builder.get_or_create_org(db, "Trinamix")
    org.meta = {"profile": {"v": config.NODE_PROFILE_VERSION,
                            "size_tier": "mid", "industry": "Oracle ERP consulting",
                            "summary": "already known", "grounded": True}}
    db.commit()

    search_calls = []
    _silence_everything_but_profiling(monkeypatch)
    monkeypatch.setattr(expansion.ORCH, "search",
                        lambda query, is_person=True: search_calls.append(query) or [])
    monkeypatch.setattr(expansion, "_best_org_affiliation_edge",
                        lambda edges: _org_edge("Trinamix", "works at Trinamix"))
    monkeypatch.setattr(node_profiler, "is_active", lambda: True)

    expansion._process_person(db, "Molly Chakraborty", 0, {},
                              enhanced_professional_search=True)

    assert search_calls == [], "already-cached org profile must not trigger new searches"
    assert _get_org(db, "Trinamix").meta["profile"]["summary"] == "already known"


def test_phase_4d_skips_search_entirely_when_claude_is_not_active(db, monkeypatch):
    """No point spending Serper queries on a profile nothing will read --
    node_profiler.is_active() False must short-circuit before any search."""
    search_calls = []
    _silence_everything_but_profiling(monkeypatch)
    monkeypatch.setattr(expansion.ORCH, "search",
                        lambda query, is_person=True: search_calls.append(query) or [])
    monkeypatch.setattr(expansion, "_best_org_affiliation_edge",
                        lambda edges: _org_edge("Trinamix", "works at Trinamix"))
    monkeypatch.setattr(node_profiler, "is_active", lambda: False)

    expansion._process_person(db, "Prantik Chakraborty", 0, {},
                              enhanced_professional_search=True)

    assert search_calls == []
    # The org ROW is still resolved even with profiling off: phase 4e needs
    # the same row, and nesting the lookup inside node_profiler.is_active()
    # is what made ARTEMIS_NODE_PROFILE=0 silently disable the strategy stage
    # too. No profile is written -- that's the part that must not happen.
    org = _get_org(db, "Trinamix")
    assert org is not None
    assert "profile" not in (org.meta or {})


def test_a_stale_profile_is_reprofiled_rather_than_reused(db, monkeypatch):
    """Profiles are cached on the org row with no TTL, so a verdict produced
    under an older prompt/guard set would otherwise stand in forever for one
    the current code would never have produced -- concretely, a profile
    written before the org-identity guard existed. An unstamped or
    old-version profile must be treated as absent."""
    org = builder.get_or_create_org(db, "Trinamix")
    org.meta = {"profile": {"v": config.NODE_PROFILE_VERSION - 1,
                            "size_tier": "large", "industry": "biometric sensors",
                            "summary": "pre-identity-guard verdict", "grounded": True}}
    db.commit()

    _silence_everything_but_profiling(
        monkeypatch,
        search_results=[SearchResult("Trinamix", "https://linkedin.com/company/trinamix",
                                     "snippet", "serper")],
        fetched_text="Trinamix is an Oracle ERP consultancy with 201-500 employees.",
    )
    monkeypatch.setattr(expansion, "_best_org_affiliation_edge",
                        lambda edges: _org_edge("Trinamix", "works at Trinamix"))
    monkeypatch.setattr(node_profiler, "is_active", lambda: True)
    monkeypatch.setattr(node_profiler, "profile_org", lambda org, snippets, known_context="": {
        "v": config.NODE_PROFILE_VERSION, "size_tier": "mid",
        "industry": "Oracle ERP consulting", "summary": "fresh", "grounded": True,
    })

    expansion._process_person(db, "Prantik Chakraborty", 0, {},
                              enhanced_professional_search=True)

    profile = _get_org(db, "Trinamix").meta["profile"]
    assert profile["summary"] == "fresh"
    assert profile["v"] == config.NODE_PROFILE_VERSION


def test_profile_org_stamps_the_current_version(monkeypatch):
    monkeypatch.setattr(node_profiler, "is_active", lambda: True)
    monkeypatch.setattr(node_profiler, "call_json", lambda *a, **k: {
        "size_tier": "mid", "industry": "Oracle ERP consulting",
        "summary": "ok", "grounded": True,
    })
    profile = node_profiler.profile_org("Trinamix", ["a snippet"])
    assert profile["v"] == config.NODE_PROFILE_VERSION
    assert node_profiler.is_current(profile)


def test_is_current_rejects_unstamped_and_superseded_profiles():
    assert not node_profiler.is_current(None)
    assert not node_profiler.is_current({})
    assert not node_profiler.is_current({"size_tier": "mid", "grounded": True})
    assert not node_profiler.is_current({"v": config.NODE_PROFILE_VERSION - 1})
    assert node_profiler.is_current({"v": config.NODE_PROFILE_VERSION})
