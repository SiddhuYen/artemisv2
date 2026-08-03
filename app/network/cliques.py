"""Wave 0 of initial enrichment: structural edges derivable from the contact
export ALONE — zero search queries, zero provider calls, zero Claude tokens.

A LinkedIn/vCard export already asserts far more than "you know these people".
Every row carries a self-reported employer and school, and two of your contacts
listing the same small employer is a structural claim that they are colleagues
— the same kind of claim scripts/build_yc_cache.py materializes from a firm's
team page. This module harvests exactly that, so the graph is meaningfully
thicker before a single query is spent.

What it creates, all from `LocalProfile` rows:
  - an Organization per distinct employer / school
  - `employee` / `student` membership edges (contact -> org)
  - `coworker` cliques among contacts at the SAME employer, capped

What it deliberately does NOT create:
  - school cliques. A school is a directory, not a team, and the export carries
    no graduation year to bound the era — two contacts who attended the same
    university 20 years apart are not connected by that fact. Schools get
    membership edges only.
  - cliques for oversized or generic employers (see _GENERIC_EMPLOYERS and
    config.CONTACT_CLIQUE_MAX).

Idempotent. Every edge is keyed to a stable synthetic source URL, so
save_source's dedup-by-URL and add_edge_from_extraction's dedup-by-(a, b, type,
source) mean re-running converges instead of piling up duplicates — the same
property ingest.backfill_graph_edges relies on.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import config
from ..extraction.schemas import EdgeSignals, ExtractedEdge
from ..models import LocalProfile, Person
from ..providers.base import SearchResult
from ..utils.names import org_norm_key

# Confidence for wave-0 edges. Both land above config.STRONG_MIN (0.6), i.e.
# 'strong' status, because the assertion comes from the operator's own export
# rather than from inferred prose — but both sit BELOW the 0.95 an explicit
# linkedin_1st connection gets, and coworker sits below membership because
# "we both listed this employer" is one inference step removed from "I listed
# this employer".
_MEMBERSHIP_CONFIDENCE = 0.85
_COWORKER_CONFIDENCE = 0.75

# Employer values that are a job status, not an organization. Left unfiltered
# these are catastrophic: "Self-Employed" is one of the most common values in a
# real LinkedIn export, and it would fuse every independent contractor the
# operator knows into one dense false clique of mutual "coworkers".
# Compared post-org_norm_key, so punctuation/case/legal suffixes are already
# folded ("Self-Employed" -> "self employed").
_GENERIC_EMPLOYERS = frozenset({
    "self employed", "selfemployed", "self", "self employment",
    "freelance", "freelancer", "freelancing", "independent",
    "independent consultant", "independent contractor", "contractor",
    "consultant", "consulting", "sole proprietor", "owner",
    "retired", "unemployed", "student", "none", "n a", "na", "unknown",
    "various", "multiple", "private", "home", "personal",
    # "Stealth Startup" is LinkedIn's placeholder for declining to say where
    # you work, and it is COMMON — 12 contacts in a real 1,025-contact export,
    # which was enough to rank it third for a wave-2 org sweep. Treating it as
    # an employer fuses a dozen unrelated people into one fictional company.
    "stealth", "stealth startup", "stealth mode", "stealth mode startup",
    "stealth co", "confidential", "undisclosed", "tbd", "looking",
    "open to work", "seeking opportunities", "not currently working",
})


def _source_url(kind: str, norm_key: str) -> str:
    """Stable provenance key per (kind, org).

    `contact-export://` mirrors ingest.py's `linkedin-import://` scheme: not a
    fetchable URL, just a stable identifier saying "this came from the
    operator's own contact export, not the web". Stable is the whole point —
    it is what makes a re-run converge instead of duplicating.
    """
    return f"contact-export://{kind}/{norm_key}"


def _synthetic_source(db: Session, kind: str, norm_key: str) -> object:
    from ..graph import builder  # local import: avoids a network<->graph cycle

    url = _source_url(kind, norm_key)
    res = SearchResult(norm_key, url, "contact_export", "contact_export")
    return builder.save_source(db, res, f"wave0:{kind}")


def _membership_edge(db: Session, person: Person, org, rel_type: str,
                     source, source_url: str) -> None:
    from ..graph import builder

    edge = ExtractedEdge(
        person_a=person.canonical_name, organization=org.name,
        other_kind="organization", org_type=org.type,
        relationship_type=rel_type, method="contact export",
        source_url=source_url,
        evidence_snippet=(f"{person.canonical_name} lists {org.name} in the "
                          f"operator's contact export."),
        confidence_base=_MEMBERSHIP_CONFIDENCE,
        confidence_adjusted=_MEMBERSHIP_CONFIDENCE,
        signals=EdgeSignals(trusted=True, explicit_keyword_match=True),
    )
    builder.add_edge_from_extraction(db, person, edge, 0, source, org)


def _coworker_edge(db: Session, subject: Person, other: Person, org_name: str,
                   source, source_url: str) -> None:
    from ..graph import builder

    edge = ExtractedEdge(
        person_a=subject.canonical_name, person_b=other.canonical_name,
        other_kind="person", relationship_type="coworker",
        method="contact export (shared employer)",
        source_url=source_url,
        evidence_snippet=(f"{subject.canonical_name} and {other.canonical_name} "
                          f"both list {org_name} as their employer in the "
                          f"operator's contact export."),
        confidence_base=_COWORKER_CONFIDENCE,
        confidence_adjusted=_COWORKER_CONFIDENCE,
        signals=EdgeSignals(trusted=True, explicit_keyword_match=True),
    )
    builder.add_edge_from_extraction(db, subject, edge, 0, source, other)


class _Participant(NamedTuple):
    """A contact, or the operator themselves, as far as cliques care.

    The operator belongs in their own employer's clique: without them they sit
    outside every org cluster, connected only by linkedin_1st edges, which
    understates the network they actually have at their own company.
    """
    key: str            # LocalProfile id, or "owner"
    name: str
    companies: List[str]
    schools: List[str]


def _participants(profiles: List[LocalProfile], owner=None) -> List[_Participant]:
    out = [_Participant(p.id, p.canonical_name,
                        list(p.companies or []), list(p.schools or []))
           for p in profiles]
    if owner is not None and (owner.name or "").strip():
        out.append(_Participant(
            "owner", owner.name.strip(),
            [owner.company] if owner.company else [],
            [owner.school] if owner.school else []))
    return out


def _group_by_org(participants: List[_Participant], field: str,
                  ) -> Dict[str, Tuple[str, List[_Participant]]]:
    """norm_key -> (display name, members). The first surface form wins as the
    display name, matching get_or_create_org's first-writer-wins behavior."""
    groups: Dict[str, Tuple[str, List[_Participant]]] = {}
    for participant in participants:
        seen_here = set()
        for raw in (getattr(participant, field, None) or []):
            norm = org_norm_key(raw or "")
            # Dedup on the NORM key, not the raw string: a merged profile
            # carries several surface forms of one employer ("Acme", "Acme
            # Inc."), and listing a member twice would inflate the group past
            # the clique cap and duplicate their membership edge.
            if not norm or norm in _GENERIC_EMPLOYERS or norm in seen_here:
                continue
            seen_here.add(norm)
            _display, members = groups.setdefault(norm, (raw.strip(), []))
            members.append(participant)
    return groups


def materialize_contact_cliques(db: Session, progress=None, owner=None) -> dict:
    """Build wave-0 structural edges from every imported LocalProfile.

    `owner` (an OwnerProfile, optional) puts the operator into their OWN
    employer's and school's clusters. Without it the operator sits outside
    every org cluster, joined to the graph only by linkedin_1st edges, which
    understates the network they have at their own company.

    Returns {organizations, membership_edges, coworker_edges, cliques,
    skipped_oversize, skipped_generic}. Safe to call repeatedly and safe to
    call before any enrichment has run — it never touches the network.
    """
    from ..graph import builder

    profiles = list(db.execute(select(LocalProfile)).scalars())
    participants = _participants(profiles, owner)
    if not participants:
        return {"organizations": 0, "membership_edges": 0, "coworker_edges": 0,
                "cliques": 0, "skipped_oversize": 0, "skipped_generic": 0}

    # Resolve every participant to a graph Person ONCE. Contacts are the
    # operator's own ground truth — the nodes every route ultimately has to run
    # through — so unlike discovered nodes they are not gated on
    # builder.at_node_cap, exactly as ingest.ingest_rows already treats them.
    person_by_key: Dict[str, Person] = {}
    for participant in participants:
        person = builder.get_or_create_person(db, participant.name)
        if person is not None:
            person_by_key[participant.key] = person

    counts = {"organizations": 0, "membership_edges": 0, "coworker_edges": 0,
              "cliques": 0, "skipped_oversize": 0, "skipped_generic": 0}

    # generic employers are counted for reporting, not silently dropped
    for participant in participants:
        for raw in participant.companies:
            if org_norm_key(raw or "") in _GENERIC_EMPLOYERS:
                counts["skipped_generic"] += 1

    for field, org_type, rel_type, label in (
        ("companies", "company", "employee", "employer"),
        ("schools", "school", "student", "school"),
    ):
        for norm, (display, members) in _group_by_org(participants, field).items():
            org = builder.get_or_create_org(db, display, org_type=org_type)
            if org is None:
                continue
            counts["organizations"] += 1
            source = _synthetic_source(db, label, norm)
            source_url = _source_url(label, norm)

            people = [person_by_key[p.key] for p in members
                      if p.key in person_by_key]
            for person in people:
                _membership_edge(db, person, org, rel_type, source, source_url)
                counts["membership_edges"] += 1

            # Only employers produce cliques (see module docstring), and only
            # small ones. Above the cap the shared employer is a directory
            # artifact, so membership is kept and the clique is dropped.
            if label != "employer":
                continue
            if len(people) < 2:
                continue
            if len(people) > config.CONTACT_CLIQUE_MAX:
                counts["skipped_oversize"] += 1
                if progress:
                    progress(f"  ⊘ {display}: {len(people)} contacts exceeds the "
                             f"clique cap ({config.CONTACT_CLIQUE_MAX}) — "
                             f"membership only, no coworker edges")
                continue

            # One row per PAIR, not per ordered pair. Both readers of a
            # coworker edge are undirected — connect._adjacency always was, and
            # expansion._reuse_existing_neighbors matches person_a_id OR
            # person_b_id — so the mirrored row this used to write asserted
            # nothing the first one didn't. Writing it anyway doubled every
            # clique and, now that coworker is symmetric for dedup purposes
            # (models.SYMMETRIC_RELATIONSHIP_TYPES), would collapse into the
            # same row regardless.
            for i, subject in enumerate(people):
                for other in people[i + 1:]:
                    _coworker_edge(db, subject, other, display, source, source_url)
                    counts["coworker_edges"] += 1
            counts["cliques"] += 1
            if progress:
                pairs = len(people) * (len(people) - 1) // 2
                progress(f"  ✓ {display}: {len(people)} contacts → "
                         f"{pairs} coworker edges")

    db.commit()
    if progress:
        progress(f"wave 0: {counts['organizations']} orgs, "
                 f"{counts['membership_edges']} membership + "
                 f"{counts['coworker_edges']} coworker edges, "
                 f"{counts['cliques']} cliques")
    return counts
