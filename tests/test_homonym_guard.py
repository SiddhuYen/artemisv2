"""The homonym-disambiguation guard: builder.get_or_create_person() must not
silently fuse two different real people who happen to share a name.

Two scenarios this guards against:

  1. (QID-adoption path) node "Jordan Lee" already has evidence in the graph
     anchoring them as a venture capitalist. Later, a name-matched Wikidata
     lookup resolves a QID for a completely different "Jordan Lee" -- a
     test-prep educator. Without a check, get_or_create_person's QID-
     adoption path ("case 2": a name match with no QID yet) would stamp the
     VC node with the educator's identity, permanently fusing two strangers
     into one node.

  2. (plain counterpart-merge path, no QID at all) node "Donald Trump" (the
     sitting US president) already has evidence in the graph. Later, an
     OpenAlex coauthor search for an unrelated subject turns up an academic
     coauthor who ALSO happens to be named "Donald Trump" -- and since
     counterpart resolution during edge persistence never had a QID to
     check identity against in the first place, it silently merged onto the
     president's node. This is the live case that shipped and got caught:
     Amit Sharma -> Abhimanyu Sharma -> Jaya Sharma -> (real cancer-research
     coauthors) -> "Donald Trump" (wrong merge) -> Larry Ellison, a path
     stitched together out of two real, unrelated relationships bridged by
     one shared name.

app.graph.disambiguate.domain_conflict is the deterministic, keyword-lexicon
backstop that catches a CLEAR cross-domain mismatch; app.graph.builder wires
it into BOTH of get_or_create_person's merge paths via _homonym_conflict.
Both are conservative by construction: silent (no conflict, adopt/merge as
before) whenever either side has no signal, so a false negative just
preserves today's fuse-by-name behavior and a false positive costs one
extra, disambiguated node.
"""
from app import config
from app.graph import builder, disambiguate
from app.models import RelationshipEdge


def _add_evidence(db, person, snippet: str) -> None:
    """Attach a bare relationship edge to `person` carrying `snippet` as
    evidence -- simulating that this node was already discovered (as someone
    else's counterpart) before any identity was ever proposed for it."""
    db.add(RelationshipEdge(
        person_a_id=person.id,
        relationship_type="unknown",
        evidence_snippet=snippet,
    ))
    db.commit()


# --- disambiguate.domain_conflict: the deterministic keyword check ---------
def test_domain_conflict_fires_on_a_clear_cross_domain_mismatch():
    signal = "Jordan Lee is a venture capitalist and general partner at a seed fund."
    candidate = "Jordan Lee is a test-prep educator running a tutoring academy."
    assert disambiguate.domain_conflict(signal, candidate) is True


def test_domain_conflict_is_silent_when_domains_overlap():
    signal = "Jordan Lee is a venture capitalist at Acme Capital."
    candidate = "Jordan Lee, general partner and investor, founded a fund."
    assert disambiguate.domain_conflict(signal, candidate) is False


def test_domain_conflict_is_silent_when_either_side_is_unanchored():
    assert disambiguate.domain_conflict("", "test-prep educator") is False
    assert disambiguate.domain_conflict("venture capitalist", "") is False
    assert disambiguate.domain_conflict("just some words", "more plain words") is False


def test_politics_domain_detects_the_us_president_phrase():
    text = ("U.S. President Donald Trump and CDC Director Robert R. Redfield "
           "participate in the daily briefing on the coronavirus.")
    assert "politics" in disambiguate.domains_of(text)


def test_bare_president_does_not_match_politics():
    """Deliberately excluded -- a bare "president" collides constantly with
    the business bucket's own corporate-officer language (company
    president, club president, university president)."""
    assert "politics" not in disambiguate.domains_of(
        "Jane Doe is president of a mid-sized manufacturing company.")


def test_us_president_and_academic_coauthor_are_reported_as_conflicting():
    signal = "U.S. President Donald Trump addressed the nation on Tuesday."
    candidate = ("Jaya Sharma coauthor of Donald Trump. "
                "(an academic author, from a research coauthorship)")
    assert disambiguate.domain_conflict(signal, candidate) is True


# --- builder.get_or_create_person: the QID-adoption path --------------------
def test_conflicting_domains_split_into_separate_nodes(db):
    original = builder.get_or_create_person(db, "Jordan Lee")
    _add_evidence(db, original,
                  "Jordan Lee is a venture capitalist and investor at a seed-stage fund.")

    resolved = builder.get_or_create_person(
        db, "Jordan Lee", qid="Q999",
        identity_text="Jordan Lee is a test-prep educator who runs an admissions academy.")

    assert resolved is not None
    assert resolved.id != original.id                 # a genuinely separate node
    assert resolved.wikidata_qid == "Q999"
    assert resolved.norm_name.endswith("#Q999")
    assert original.wikidata_qid is None               # the VC node's identity untouched
    assert original.meta.get("homonym_rejected") is not None


def test_overlapping_domains_adopt_the_qid_onto_the_existing_node(db):
    original = builder.get_or_create_person(db, "Jordan Lee")
    _add_evidence(db, original, "Jordan Lee is a venture capitalist at Acme Capital.")

    resolved = builder.get_or_create_person(
        db, "Jordan Lee", qid="Q999",
        identity_text="Jordan Lee, general partner and investor, founded a fund.")

    assert resolved is not None
    assert resolved.id == original.id                  # same node, no split
    assert resolved.wikidata_qid == "Q999"
    assert "homonym_rejected" not in (resolved.meta or {})


def test_a_brand_new_node_with_no_prior_evidence_adopts_freely(db):
    """Nothing to dispute the match with yet -- fails open, same as before
    this guard existed."""
    builder.get_or_create_person(db, "Alex Rivera")  # creates the node, no edges

    resolved = builder.get_or_create_person(
        db, "Alex Rivera", qid="Q42",
        identity_text="Alex Rivera is a professional footballer.")

    assert resolved is not None
    assert resolved.wikidata_qid == "Q42"


def test_guard_can_be_disabled_via_config(db, monkeypatch):
    monkeypatch.setattr(config, "IDENTITY_VERIFY_ENABLED", False)
    original = builder.get_or_create_person(db, "Jordan Lee")
    _add_evidence(db, original,
                  "Jordan Lee is a venture capitalist and investor at a seed-stage fund.")

    resolved = builder.get_or_create_person(
        db, "Jordan Lee", qid="Q999",
        identity_text="Jordan Lee is a test-prep educator who runs an admissions academy.")

    assert resolved.id == original.id
    assert resolved.wikidata_qid == "Q999"


def test_no_identity_text_leaves_adoption_unconditional(db):
    """Callers with no candidate description (e.g. plain counterpart
    resolution during edge extraction, which never passes a qid at all) get
    the pre-guard behavior -- there's nothing to check the name against."""
    original = builder.get_or_create_person(db, "Jordan Lee")
    _add_evidence(db, original,
                  "Jordan Lee is a venture capitalist and investor at a seed-stage fund.")

    resolved = builder.get_or_create_person(db, "Jordan Lee", qid="Q999")

    assert resolved.id == original.id
    assert resolved.wikidata_qid == "Q999"


def test_conflicting_qid_still_splits_regardless_of_identity_text(db):
    """Case 3 (an existing DIFFERENT qid) is unrelated to this guard and must
    keep working exactly as before."""
    first = builder.get_or_create_person(db, "Jordan Lee", qid="Q1")
    second = builder.get_or_create_person(
        db, "Jordan Lee", qid="Q2", identity_text="anything at all")

    assert second.id != first.id
    assert second.wikidata_qid == "Q2"


# ---------------------------------------------------------------------------
# builder.get_or_create_person: the plain (no-QID) counterpart-merge path --
# the fix for the live Donald Trump mixup.
# ---------------------------------------------------------------------------
def test_no_qid_merge_splits_on_a_real_conflict(db):
    president = builder.get_or_create_person(db, "Donald Trump")
    _add_evidence(db, president,
                  "U.S. President Donald Trump and CDC Director Robert R. Redfield "
                  "participate in the daily briefing on the coronavirus.")

    resolved = builder.get_or_create_person(
        db, "Donald Trump",
        identity_text="Jaya Sharma coauthor of Donald Trump. "
                      "(an academic author, from a research coauthorship)")

    assert resolved is not None
    assert resolved.id != president.id            # a genuinely separate node
    assert resolved.norm_name != president.norm_name
    assert resolved.canonical_name == "Donald Trump"  # still displays under the same name
    assert president.meta.get("homonym_rejected") is not None


def test_no_qid_merge_converges_repeat_conflicting_mentions_onto_one_node(db):
    """A SECOND, differently-worded academic-coauthor mention of "Donald
    Trump" must land on the SAME disambiguated node as the first -- not
    fragment into a third node every time the wording differs slightly."""
    president = builder.get_or_create_person(db, "Donald Trump")
    _add_evidence(db, president, "U.S. President Donald Trump addressed the nation.")

    first = builder.get_or_create_person(
        db, "Donald Trump",
        identity_text="Some Author coauthor of Donald Trump. "
                      "(an academic author, from a research coauthorship)")
    second = builder.get_or_create_person(
        db, "Donald Trump",
        identity_text="Another Researcher coauthor of Donald Trump. "
                      "(an academic author, from a research coauthorship)")

    assert first.id == second.id
    assert first.id != president.id


def test_no_qid_merge_adopts_when_no_conflict(db):
    original = builder.get_or_create_person(db, "Jaya Sharma")
    _add_evidence(db, original, "Jaya Sharma coauthor of Lynn Hlatky. "
                                "(an academic author, from a research coauthorship)")

    resolved = builder.get_or_create_person(
        db, "Jaya Sharma",
        identity_text="Jaya Sharma coauthor of Michael Coyle. "
                      "(an academic author, from a research coauthorship)")

    assert resolved.id == original.id
    assert "homonym_rejected" not in (resolved.meta or {})


def test_no_qid_merge_is_silent_with_no_identity_text(db):
    """The pre-fix behavior for every caller that doesn't pass identity_text
    at all must be completely unchanged."""
    president = builder.get_or_create_person(db, "Donald Trump")
    _add_evidence(db, president, "U.S. President Donald Trump addressed the nation.")

    resolved = builder.get_or_create_person(db, "Donald Trump")

    assert resolved.id == president.id


def test_no_qid_merge_is_silent_when_domains_overlap(db):
    original = builder.get_or_create_person(db, "Jordan Lee")
    _add_evidence(db, original, "Jordan Lee is a venture capitalist at Acme Capital.")

    resolved = builder.get_or_create_person(
        db, "Jordan Lee",
        identity_text="Jordan Lee, general partner and investor, founded a fund.")

    assert resolved.id == original.id


def test_no_qid_merge_respects_the_config_toggle(db, monkeypatch):
    monkeypatch.setattr(config, "IDENTITY_VERIFY_ENABLED", False)
    president = builder.get_or_create_person(db, "Donald Trump")
    _add_evidence(db, president, "U.S. President Donald Trump addressed the nation.")

    resolved = builder.get_or_create_person(
        db, "Donald Trump",
        identity_text="Jaya Sharma coauthor of Donald Trump. "
                      "(an academic author, from a research coauthorship)")

    assert resolved.id == president.id
