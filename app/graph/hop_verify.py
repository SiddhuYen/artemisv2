"""Path-assembly-time hop verification -- the "deferred Claude verification
stage" referenced in claude_extractor.py / relation_classifier.py /
entity_filter.py's docstrings, and in connect_people's "Requires Claude
verification before activation" warning.

relation_classifier.py TYPES an edge from its evidence sentence; entity_filter
validates entity-hood; NEITHER ever asks whether the edge is actually true.
This is that missing check: does this edge's own evidence genuinely support
the claimed relationship, at all.

Deliberately edge-intrinsic, not path-aware: the verdict is written back onto
the edge (verified_status/verified_at/verified_reason) and reused across
EVERY path that edge appears in, so it cannot depend on any one path's
surrounding context -- an edge that's "genuine" has to mean genuine full
stop, not "plausible in this particular chain". That's a narrower question
than judging a whole path's narrative, but it's the one a per-edge cache can
actually answer consistently.

Runs only against hops in a candidate route that's already been found
(connect._verified_routes) -- never against every edge considered during
search -- so cost scales with what's shown to users, not with the graph
explored to find it.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from .. import config
from ..extraction.claude_client import call_json, claude_available
from ..models import RelationshipEdge

_PROMPT = """Does this evidence genuinely support {a} and {b} having a real \
"{rel}" relationship (or knowing / having known each other)?

Evidence: "{evidence}"

Rules:
- Judge ONLY what the evidence actually states -- not outside knowledge about
  either person.
- A source that's about one of them but doesn't actually connect the two
  names is NOT genuine evidence (e.g. an article about A that merely quotes
  or mentions B in an unrelated context).
- Two names appearing in the same piece with no stated relationship between
  them is NOT genuine evidence.
- reason: one short sentence explaining the judgment either way.
"""
_SCHEMA = {
    "type": "object",
    "properties": {
        "genuine": {
            "type": "boolean",
            "description": "true only if the evidence actually supports the claim",
        },
        "reason": {"type": "string"},
    },
    "required": ["genuine", "reason"],
    "additionalProperties": False,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Marks a rejection as authored by a person, not by the model. Stored as a
# prefix on verified_reason rather than a new column so it needs no migration
# on a live shared graph, and so the reason a human gave is still readable
# right where every other verdict's reason already is.
OPERATOR_PREFIX = "operator: "


def is_operator_rejected(edge: RelationshipEdge) -> bool:
    return (edge.verified_status == "rejected"
            and (edge.verified_reason or "").startswith(OPERATOR_PREFIX))


def _is_stale(edge: RelationshipEdge) -> bool:
    if not edge.verified_status or not edge.verified_at:
        return True
    # A person looked at this edge and said it was wrong. TTL expiry exists so
    # a MODEL's verdict can be revisited as prompts and evidence improve; there
    # is no equivalent reason to re-litigate a human one, and doing so would
    # let the reconsideration path silently restore a hand-rejected edge to
    # 'candidate' (see verify's stale-rejection branch) -- the exact failure
    # the operator used this to prevent.
    if is_operator_rejected(edge):
        return False
    try:
        checked = datetime.fromisoformat(edge.verified_at)
    except (TypeError, ValueError):
        return True
    ttl = (config.HOP_VERIFY_TTL_REJECTED if edge.verified_status == "rejected"
           else config.HOP_VERIFY_TTL_GENUINE)
    return (datetime.now(timezone.utc) - checked).total_seconds() > ttl


def reject_edges(db: Session, edge_ids, reason: str = "") -> dict:
    """Mark edges false because a person said so. Returns per-id outcomes.

    Writes the SAME two fields the model's own negative verdict writes
    (status='rejected' + verified_status='rejected'), because those are
    already load-bearing everywhere a path is assembled: connect._untraversable
    excludes them from traversal, connect._route_exists stops reporting the
    pair as already-connected, and network.paths drops them too. Nothing new
    has to learn about operator feedback for it to take effect.

    That last one is the point of the whole endpoint. A bogus edge doesn't just
    rank badly -- _route_exists short-circuits the entire paid walk on a hit,
    so until the edge is excluded it actively PREVENTS the search that would
    find the real answer. Rejecting it is what re-opens that pair to discovery.

    Idempotent: re-rejecting an already-rejected edge is a no-op that still
    reports ok, so a double-click can't produce an error the operator has to
    interpret.
    """
    results = {}
    changed = 0
    for edge_id in edge_ids:
        edge = db.get(RelationshipEdge, edge_id)
        if edge is None:
            results[edge_id] = "not_found"
            continue
        if is_operator_rejected(edge):
            results[edge_id] = "already_rejected"
            continue
        edge.status = "rejected"
        edge.verified_status = "rejected"
        edge.verified_at = _now_iso()
        edge.verified_reason = (OPERATOR_PREFIX + (reason.strip() or "marked wrong by operator"))[:500]
        db.add(edge)
        results[edge_id] = "rejected"
        changed += 1
    if changed:
        db.commit()
    return {"rejected": changed, "results": results}


def verify(db: Session, edge: RelationshipEdge, name_a: str, name_b: str) -> bool:
    """True if `edge` should stay traversable.

    Any Claude failure (no credential, rate limit, refusal -- see
    claude_client.call_json's contract) resolves to True and leaves the edge
    exactly as it was: no LLM failure may make a real connection disappear
    from a path, and an unresolved call isn't cached as either verdict so
    it's retried, not stuck "checked" when it wasn't.

    An edge with no evidence_snippet at all (e.g. a structural assertion like
    linkedin_1st always has one in practice, but nothing guarantees every
    edge does) also resolves to True without calling Claude -- same
    fail-open philosophy as coauthor_plausibility.check: there's nothing to
    judge from, so asking the model to rule on an empty string isn't a real
    verification, it's just noise with a verdict attached."""
    if not config.CLAUDE_VERIFY_HOPS or not claude_available():
        return True
    if not edge.evidence_snippet:
        return True
    if not _is_stale(edge):
        return edge.verified_status != "rejected"

    payload = call_json(
        _PROMPT.format(a=name_a, b=name_b, rel=edge.relationship_type,
                       evidence=(edge.evidence_snippet or "")[:500]),
        schema=_SCHEMA,
        model=config.HOP_VERIFY_MODEL,
        max_tokens=256,
    )
    if payload is None:
        return True

    genuine = bool(payload.get("genuine", True))
    edge.verified_status = "genuine" if genuine else "rejected"
    edge.verified_at = _now_iso()
    edge.verified_reason = str(payload.get("reason", ""))[:500]
    if not genuine:
        # The existing, load-bearing exclusion field -- see
        # connect._untraversable / network.paths._PublicEdges.__init__, both
        # of which already treat status=='rejected' as "not a real
        # connection" and exclude it from every path, not just this one.
        edge.status = "rejected"
    elif edge.status == "rejected":
        # A stale rejection got reconsidered and reversed. The edge's
        # original confidence tier was overwritten when it was rejected and
        # can't be recovered, so it's restored to a safe, traversable middle
        # tier rather than left permanently excluded despite the new verdict.
        edge.status = "candidate"
    db.add(edge)
    db.commit()
    return genuine
