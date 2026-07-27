"""The homonym-disambiguation guard: builder.get_or_create_person() must not
silently fuse two different real people who happen to share a name.

Scenario this guards against: node "Jordan Lee" already has evidence in the
graph anchoring them as a venture capitalist. Later, a name-matched Wikidata
lookup resolves a QID for a completely different "Jordan Lee" -- a test-prep
educator. Without a check, get_or_create_person's QID-adoption path ("case 2":
a name match with no QID yet) would stamp the VC node with the educator's
identity, or vice versa, permanently fusing two strangers into one node.

app.graph.disambiguate.domain_conflict is the deterministic, keyword-lexicon
backstop that catches a CLEAR cross-domain mismatch; app.graph.builder wires
it into get_or_create_person's adoption path via _homonym_conflict. Both are
conservative by construction: silent (no conflict, adopt as before) whenever
either side has no signal, so a false negative just preserves today's
fuse-by-name behavior and a false positive costs one extra, QID-suffixed node.
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
