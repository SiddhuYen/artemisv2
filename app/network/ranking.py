"""Deciding WHICH contacts are worth spending searches on, and in what order.

At ~35 queries per person (9 silos x MAX_QUERIES_PER_SILO, deduped) a
1,000-contact export is tens of thousands of provider calls. Enriching everyone
is not an option, so the run is a ranked prefix of the contact list and this
module produces that ranking.

The objective is NOT "who is most important to the operator" — it is "where
does the next search query buy the most new reachable people". That leads to
two rules that look surprising until you hold the objective in mind:

  1. Seniority is a proxy for WEB FOOTPRINT, not for value. The silos in
     silos/definitions.py query for board seats, funding, appointments and
     press. A founder or partner returns something for those 35 queries; a
     junior IC at a private company returns nothing, at identical cost.

  2. Contacts at an employer the operator ALREADY has contacts at are worth
     less, not more, and are damped (see _COMPANY_DECAY). The tenth person you
     know at one company opens very little territory the first nine didn't,
     because their colleagues largely overlap. Ranking by raw score alone would
     spend the whole budget inside one company; the decay makes the selection
     greedily cover distinct organizations instead.

Everything here is pure and local — no provider calls, no Claude — so a run can
be planned and inspected before a cent is spent. Notability (the one signal
that needs the network) is applied separately by the caller via
`apply_notability`, which takes an already-fetched name set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .. import config
from ..models import LocalProfile, Person, RelationshipEdge
from ..utils.names import normalize, org_norm_key, person_norm_key
from .cliques import _GENERIC_EMPLOYERS
from .silo_weights import initial_weights

# --- scoring weights -------------------------------------------------------
# Every eligible contact starts here so the per-employer decay always has
# something to act on, and so ordering among otherwise-identical contacts is
# decided by the decay rather than by ties.
_BASE = 1.0

# Highest matching tier wins; these do NOT stack. Titles are compared after
# normalize(), which folds case and turns "Co-Founder" into "co founder".
_SENIORITY_TIERS = (
    (3.0, ("founder", "co founder", "cofounder", "ceo", "chief executive",
           "president", "managing partner", "general partner", "managing director",
           "chairman", "chairwoman", "chair", "owner", "partner")),
    (2.0, ("vp", "vice president", "svp", "evp", "head of", "director", "chief",
           "cto", "cfo", "coo", "cmo", "cro", "principal", "board member")),
    (1.0, ("senior", "lead", "manager", "staff")),
)

# The web has already said something about this person that the graph captured
# — the strongest available evidence that another 35 queries will also land.
_HAS_PUBLIC_EDGES = 1.5
# Shared affiliation with the operator: a real colleague or classmate, not a
# conference badge scan, so the tie is worth building out from.
_SHARED_EMPLOYER = 1.5
_SHARED_SCHOOL = 0.75
# Identity anchors. These buy disambiguation rather than yield: a contact with
# a profile URL is far less likely to be confused with a namesake.
_HAS_LINKEDIN_URL = 0.5
_HAS_EMAIL = 0.25

# Applied once per already-selected contact at the same employer (rule 2
# above): 1st contact at an org scores x1, 2nd x0.6, 3rd x0.36…
_COMPANY_DECAY = 0.6

# Multiplier for a contact the notability check says the web knows about.
# Applied by apply_notability, never here — it needs a provider call.
_NOTABLE_BOOST = 1.4

_EDGE_TYPES_THAT_PROVE_NOTHING = {"linkedin_1st", "coworker", "employee", "student"}


@dataclass
class ScoredContact:
    """One contact's place in the plan. `skip_reason` set => never enriched."""
    local_profile_id: str
    display_name: str
    norm_name: str
    context: str                      # employer/school passed as seed_context
    score: float
    already_enriched: bool = False    # Person.processed — nothing left to do
    skip_reason: Optional[str] = None
    # Which silos are worth running for this contact, and how much of each.
    # Computed here rather than in a second pass because this is the one place
    # the contact's LocalProfile is already in hand. See silo_weights.py.
    silo_weights: Dict[str, float] = field(default_factory=dict)


def _title_score(titles: List[str]) -> float:
    blob = " ".join(normalize(t) for t in (titles or []) if t)
    if not blob:
        return 0.0
    for weight, keywords in _SENIORITY_TIERS:
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", blob):
                return weight
    return 0.0


def _real_orgs(values: List[str]) -> List[str]:
    """Employers that name an actual organization.

    Reuses cliques' generic-employer list: "Self-Employed" is a job status, so
    it neither identifies a company nor disambiguates the person.
    """
    out = []
    for raw in (values or []):
        norm = org_norm_key(raw or "")
        if norm and norm not in _GENERIC_EMPLOYERS:
            out.append(raw.strip())
    return out


def _people_with_public_evidence(db: Session, contact_norms: Set[str]) -> Set[str]:
    """Which of `contact_norms` the public graph holds web-sourced evidence
    about.

    Edges the operator's own export produced (linkedin_1st, and wave 0's
    coworker/employee/student) are excluded — they exist for every contact by
    construction, so counting them would flatten this signal to a constant.

    Scoped to `contact_norms` rather than scanning every Person/
    RelationshipEdge in the graph: the discovery graph is shared across every
    operator and grows with every run anyone does, but this only ever needs
    an answer for the contacts actually being ranked right now. Planning is
    supposed to cost nothing — scanning the whole shared graph here would
    make every operator's "free" plan slower in proportion to everyone
    else's enrichment history, not their own contact list.
    """
    if not contact_norms:
        return set()
    people = {
        p.id: p.norm_name
        for p in db.execute(
            select(Person).where(Person.norm_name.in_(contact_norms))
        ).scalars()
    }
    if not people:
        return set()
    person_ids = list(people)
    found: Set[str] = set()
    for e in db.execute(
        select(RelationshipEdge).where(
            or_(RelationshipEdge.person_a_id.in_(person_ids),
                RelationshipEdge.person_b_id.in_(person_ids))
        )
    ).scalars():
        if e.relationship_type in _EDGE_TYPES_THAT_PROVE_NOTHING:
            continue
        for pid in (e.person_a_id, e.person_b_id):
            if pid and pid in people:
                found.add(people[pid])
    return found


def score_contacts(db: Session, owner_name: str = "", owner_company: str = "",
                   owner_school: str = "") -> List[ScoredContact]:
    """Rank every imported contact. Returns ALL of them, best first, with
    ineligible ones carrying a `skip_reason` so the plan stays auditable
    (a contact that will never be enriched should say so, not vanish)."""
    profiles = list(db.execute(select(LocalProfile)).scalars())
    if not profiles:
        return []

    owner_norm = person_norm_key(owner_name) if owner_name else ""
    owner_co = org_norm_key(owner_company) if owner_company else ""
    owner_sch = org_norm_key(owner_school) if owner_school else ""

    contact_norms = {
        (profile.norm_name or person_norm_key(profile.canonical_name))
        for profile in profiles
    }
    contact_norms.discard("")
    contact_norms.discard(owner_norm)

    with_evidence = _people_with_public_evidence(db, contact_norms)
    processed = {p.norm_name for p in db.execute(
        select(Person).where(Person.processed == 1,
                             Person.norm_name.in_(contact_norms))).scalars()}

    scored: List[ScoredContact] = []
    for profile in profiles:
        norm = profile.norm_name or person_norm_key(profile.canonical_name)
        if not norm or norm == owner_norm:
            continue  # the operator is not their own contact

        companies = _real_orgs(profile.companies or [])
        schools = _real_orgs(profile.schools or [])
        # Context must be an ORGANIZATION. A title alone ("VP of Engineering")
        # narrows nothing — there are thousands — so it cannot protect against
        # attaching a namesake's network, which is the failure this guards.
        context = companies[0] if companies else (schools[0] if schools else "")
        if not context:
            reason = "generic_only" if (profile.companies or []) else "no_context"
            scored.append(ScoredContact(
                local_profile_id=profile.id, display_name=profile.canonical_name,
                norm_name=norm, context="", score=0.0, skip_reason=reason))
            continue

        score = _BASE + _title_score(profile.titles or [])
        if norm in with_evidence:
            score += _HAS_PUBLIC_EDGES
        if owner_co and any(org_norm_key(c) == owner_co for c in companies):
            score += _SHARED_EMPLOYER
        if owner_sch and any(org_norm_key(s) == owner_sch for s in schools):
            score += _SHARED_SCHOOL
        if profile.linkedin_url:
            score += _HAS_LINKEDIN_URL
        if profile.email:
            score += _HAS_EMAIL

        scored.append(ScoredContact(
            local_profile_id=profile.id, display_name=profile.canonical_name,
            norm_name=norm, context=context, score=score,
            already_enriched=norm in processed,
            silo_weights=(initial_weights(
                titles=profile.titles, companies=profile.companies,
                schools=profile.schools, email=profile.email or "")
                if config.ENRICH_SILO_WEIGHTS_ENABLED else {})))

    return _apply_company_decay(scored)


def _apply_company_decay(scored: List[ScoredContact]) -> List[ScoredContact]:
    """Damp each additional contact at an employer already represented above it.

    Greedy submodular coverage: walk the list best-first and charge each
    contact for how much of its employer is already covered. Without this a
    single large employer's contacts occupy the whole budget.
    """
    eligible = [c for c in scored if c.skip_reason is None]
    skipped = [c for c in scored if c.skip_reason is not None]
    # deterministic: name breaks score ties so two runs plan identically
    eligible.sort(key=lambda c: (-c.score, c.norm_name))

    seen_per_org: Dict[str, int] = {}
    for contact in eligible:
        key = org_norm_key(contact.context)
        n = seen_per_org.get(key, 0)
        contact.score = round(contact.score * (_COMPANY_DECAY ** n), 4)
        seen_per_org[key] = n + 1

    eligible.sort(key=lambda c: (-c.score, c.norm_name))
    skipped.sort(key=lambda c: c.norm_name)
    return eligible + skipped


@dataclass
class OrgSweep:
    """An employer worth expanding once, instead of per-contact."""
    name: str
    norm_name: str
    contacts: int


def org_sweep_candidates(db: Session,
                         min_contacts: Optional[int] = None,
                         limit: Optional[int] = None) -> List[OrgSweep]:
    """Employers shared by enough contacts to be worth one sweep.

    Ranked by how many of the operator's contacts sit there — that count IS the
    coverage the sweep buys. Below `min_contacts` the org is not worth a
    dedicated expansion: sweeping those one or two contacts directly costs the
    same and returns their actual network rather than the org's public face.

    Generic employers are excluded on the same grounds as everywhere else —
    "Self-Employed" is not an organization to expand.
    """
    if min_contacts is None:
        min_contacts = config.ENRICH_ORG_MIN_CONTACTS
    if limit is None:
        limit = config.ENRICH_ORG_MAX_SWEEPS

    counts: Dict[str, int] = {}
    display: Dict[str, str] = {}
    for profile in db.execute(select(LocalProfile)).scalars():
        # One vote per contact per org. Deduped on the NORM key, not the raw
        # string: a merged profile routinely carries several surface forms of
        # one employer ("Acme", "Acme Inc."), and counting those separately
        # would let a single contact push an org over the threshold alone.
        seen_here = set()
        for raw in _real_orgs(profile.companies or []):
            key = org_norm_key(raw)
            if not key or key in seen_here:
                continue
            seen_here.add(key)
            counts[key] = counts.get(key, 0) + 1
            display.setdefault(key, raw)

    sweeps = [OrgSweep(display[k], k, n) for k, n in counts.items()
              if n >= min_contacts]
    sweeps.sort(key=lambda s: (-s.contacts, s.norm_name))
    return sweeps[:limit] if limit > 0 else sweeps


def apply_notability(scored: List[ScoredContact], notable: Set[str]) -> List[ScoredContact]:
    """Boost contacts the web demonstrably knows about, then re-rank.

    Split out from score_contacts because it is the one signal that costs a
    provider call (ORCH.notable_set, one batched lookup for many names). The
    caller fetches the set; this stays pure and testable.

    Note this pushes the ranking in the OPPOSITE direction from expansion's
    default EXPAND_PREFER_REACHABLE, which walks toward the least-famous nodes.
    Both are right: expansion is looking for a warm path down the fame gradient
    to a stranger, whereas here we are choosing whose 35 queries will actually
    return something.
    """
    by_name = {c.display_name for c in scored} & set(notable)
    if not by_name:
        return scored
    for contact in scored:
        if contact.skip_reason is None and contact.display_name in by_name:
            contact.score = round(contact.score * _NOTABLE_BOOST, 4)
    eligible = [c for c in scored if c.skip_reason is None]
    skipped = [c for c in scored if c.skip_reason is not None]
    eligible.sort(key=lambda c: (-c.score, c.norm_name))
    return eligible + skipped
