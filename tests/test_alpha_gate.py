"""Which side of a /connect gets Alpha, and why.

Alpha -- the targeted recheck (phase 4c), the reasoning-selected search angles
(4e), and the narrowing to ALPHA_TOP_CANDIDATES instead of the generic
EXPAND_TOP_STRONG beam (step 7) -- belongs to a side walking TOWARD a famous
target. That is the situation the targeted phases exist for: there is a
specific, well-documented person to aim queries at.

It used to be inferred from `depth_a > depth_b`. Depth asymmetry is set by
_resolve_expansion_depths only when EXACTLY ONE endpoint is notable, so on a
famous<->famous pair both differences were zero and Alpha silently switched
itself off on BOTH sides -- for the pairs most likely to need it. Sanjay
Ghemawat <-> Larry Page resolved to depths (2, 2), so the top-5 narrowing and
every targeted phase were unreachable and the walk fell back to the generic
15-node beam with no targeted recheck at all.

The question is now asked directly: is the OTHER endpoint notable.
"""
from app.graph import connect as C


def _enhanced_for(db, monkeypatch, name_a, name_b, notable):
    """(enhanced_a, enhanced_b) as _expand_both_concurrently computes them.

    Intercepts at expand_graph rather than asserting on a private flag, so this
    pins what each side is actually TOLD, which is what the targeted phases
    read.
    """
    monkeypatch.setattr(C.ORCH, "notable_set",
                        lambda names: {n for n in names if n in notable})
    seen = {}

    def fake_expand(worker_db, name, side_depth, **kwargs):
        seen[name] = kwargs["enhanced_professional_search"]
        return {}

    monkeypatch.setattr(C, "expand_graph", fake_expand)
    depth_a, depth_b = C._resolve_expansion_depths(name_a, name_b, 2)
    C._expand_both_concurrently(db, name_a, name_b, depth_a, depth_b,
                                set(), None, "", "")
    return seen.get(name_a), seen.get(name_b)


def test_alpha_fires_on_both_sides_when_both_endpoints_are_famous(db, monkeypatch):
    """The regression: two famous people are each walking toward a famous
    target, so both sides qualify -- but symmetric depth used to read as
    'no asymmetry to exploit' and turned Alpha off entirely."""
    assert _enhanced_for(db, monkeypatch, "Sanjay Ghemawat", "Larry Page",
                         notable={"Sanjay Ghemawat", "Larry Page"}) == (True, True)


def test_famous_pair_still_expands_symmetrically(db, monkeypatch):
    """Alpha is now decoupled from depth, so enabling it on both sides must NOT
    drag the depth asymmetry along with it -- neither famous endpoint should be
    capped to a shallow immediate-circle walk on the other's account."""
    monkeypatch.setattr(C.ORCH, "notable_set", lambda names: set(names))
    assert C._resolve_expansion_depths("Sanjay Ghemawat", "Larry Page", 2) == (2, 2)


def test_alpha_fires_only_toward_the_famous_side_when_one_is_famous(db, monkeypatch):
    """Unchanged: the non-famous origin walks toward a famous target and gets
    Alpha; the famous side, capped to its immediate circle, does not."""
    assert _enhanced_for(db, monkeypatch, "Ordinary Person", "Larry Page",
                         notable={"Larry Page"}) == (True, False)


def test_alpha_stays_off_when_neither_endpoint_is_famous(db, monkeypatch):
    """No famous target to walk toward means the targeted phases have nothing
    to aim at -- unchanged."""
    assert _enhanced_for(db, monkeypatch, "Ordinary One", "Ordinary Two",
                         notable=set()) == (False, False)


def test_alpha_stays_off_when_the_notability_lookup_fails(monkeypatch):
    """Best-effort signal: a transient provider error degrades to the plain
    walk, not to a guess in either direction."""
    def _boom(names):
        raise RuntimeError("wikipedia down")

    monkeypatch.setattr(C.ORCH, "notable_set", _boom)
    assert C._notable_endpoints("Sanjay Ghemawat", "Larry Page") == (False, False)


def test_notability_matches_a_name_carrying_trailing_context(monkeypatch):
    """'Larry Ellison of Oracle' never has its own Wikipedia page, so the
    context-stripped form has to count -- otherwise a UI that appends context
    silently disables Alpha."""
    monkeypatch.setattr(C.ORCH, "notable_set",
                        lambda names: {n for n in names if n == "Larry Ellison"})
    assert C._notable_endpoints("Larry Ellison of Oracle", "Nobody Here") == (True, False)
