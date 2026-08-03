"""Deciding WHICH contacts are worth spending searches on, and in what order.

At ~35 queries per person (9 silos x MAX_QUERIES_PER_SILO, deduped) a
1,000-contact export is tens of thousands of provider calls. Enriching everyone
is not an option, so the run is a ranked prefix of the contact list and this
module produces that ranking.

There are two objectives, and which one applies depends on whether a TARGET is
supplied (see BridgeTarget).

UNTARGETED (the cold-start batch): "where does the next search query buy the
most new reachable people". Nothing is known about who the operator will
eventually want to reach, so the ranking maximises breadth of coverage.

TARGETED (a specific /connect): "which of the operator's contacts most
plausibly bridges to THIS person". Breadth stops being the goal the moment
there is a destination — the best bridge to a stranger is usually a contact who
shares their employer, school or field, not the operator's most notable
contact. Passing a target adds those signals and exempts the target's own
employer from the coverage decay below.

The untargeted objective leads to two rules that look surprising until you hold
it in mind:

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
class BridgeTarget:
    """Who a ranking is being computed FOR — the person a walk wants to reach.

    Ranking without one answers "whose 35 queries return the most", which is
    the right question for a cold-start batch and the wrong one for a specific
    /connect: the contact most likely to bridge to a particular stranger is
    rarely the operator's most-notable contact, it is the one who shares that
    stranger's employer, school or field. Supplying this re-asks the question
    as "who most plausibly bridges to THIS person".

    Every field is optional and cheap — whatever /connect already knows about
    the far endpoint. With all of them empty the target contributes nothing and
    scoring degrades exactly to the untargeted ranking.
    """
    name: str = ""
    context: str = ""                       # the disambiguation hint, if any
    companies: List[str] = field(default_factory=list)
    schools: List[str] = field(default_factory=list)
    silo_weights: Dict[str, float] = field(default_factory=dict)

    def orgs(self) -> Set[str]:
        keys = {org_norm_key(c) for c in self.companies if c}
        if self.context:
            keys.add(org_norm_key(self.context))
        return {k for k in keys if k and k not in _GENERIC_EMPLOYERS}

    def school_keys(self) -> Set[str]:
        return {k for k in (org_norm_key(s) for s in self.schools if s)
                if k and k not in _GENERIC_EMPLOYERS}


# --- target-conditioned weights ---------------------------------------------
# A contact at the TARGET's own employer is the single strongest bridge signal
# available without spending a query: colleagues are the relationship the graph
# most reliably recovers, so this outweighs every operator-side signal.
_TARGET_SHARED_EMPLOYER = 4.0
# Same school as the target — real, but weaker: cohorts span decades and
# cliques.py already refuses to build coworker edges from schools for that
# reason.
_TARGET_SHARED_SCHOOL = 1.5
# The contact's silos and the target's silos overlap (both academics, both in
# government). Not a shared institution, just a shared WORLD — the weakest of
# the three, and scaled by how much they actually overlap.
_TARGET_SILO_AFFINITY = 1.25
# Affinity below this still nudges the score but isn't worth CLAIMING as a
# reason — "both have a nonzero news weight" explains nothing to an operator.
_SILO_AFFINITY_MIN_REASON = 0.5


def _silo_affinity(contact: Dict[str, float], target: Dict[str, float]) -> float:
    """Cosine-ish overlap of two silo-weight vectors, 0..1.

    Deliberately ignores the `company` silo: it is ~1.0 for every contact by
    construction (see silo_weights.initial_weights), so counting it would give
    every pair a large constant affinity and flatten the signal to noise.
    """
    keys = (set(contact) | set(target)) - {"company"}
    if not keys:
        return 0.0
    shared = sum(min(contact.get(k, 0.0), target.get(k, 0.0)) for k in keys)
    total = sum(max(contact.get(k, 0.0), target.get(k, 0.0)) for k in keys)
    return (shared / total) if total > 0 else 0.0


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
    # Why a target-conditioned ranking lifted this contact ("shared_employer",
    # "shared_school", "silo_affinity"). Empty for an untargeted ranking. Kept
    # so a /connect can explain WHICH of the operator's contacts it chose to
    # spend queries on and why, instead of surfacing an opaque reordering.
    bridge_reasons: List[str] = field(default_factory=list)


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
                   owner_school: str = "",
                   target: Optional[BridgeTarget] = None) -> List[ScoredContact]:
    """Rank every imported contact. Returns ALL of them, best first, with
    ineligible ones carrying a `skip_reason` so the plan stays auditable
    (a contact that will never be enriched should say so, not vanish).

    `target`, when given, re-asks the question as "who bridges to THIS person"
    rather than "whose queries return the most" — see BridgeTarget. Omitting it
    reproduces the untargeted ranking exactly, which is what the cold-start
    batch run still wants.
    """
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

        weights = (initial_weights(
            titles=profile.titles, companies=profile.companies,
            schools=profile.schools, email=profile.email or "")
            if config.ENRICH_SILO_WEIGHTS_ENABLED else {})

        # Target conditioning, applied on top of the operator-side score rather
        # than replacing it: a contact who shares the target's employer but has
        # no web footprint at all still cannot be expanded into anything, so
        # the footprint signals above have to keep mattering.
        reasons: List[str] = []
        if target is not None:
            target_orgs = target.orgs()
            if target_orgs and any(org_norm_key(c) in target_orgs for c in companies):
                score += _TARGET_SHARED_EMPLOYER
                reasons.append("shared_employer")
            target_schools = target.school_keys()
            if target_schools and any(org_norm_key(s) in target_schools for s in schools):
                score += _TARGET_SHARED_SCHOOL
                reasons.append("shared_school")
            if target.silo_weights and weights:
                affinity = _silo_affinity(weights, target.silo_weights)
                if affinity > 0:
                    score += _TARGET_SILO_AFFINITY * affinity
                    if affinity >= _SILO_AFFINITY_MIN_REASON:
                        reasons.append("silo_affinity")

        scored.append(ScoredContact(
            local_profile_id=profile.id, display_name=profile.canonical_name,
            norm_name=norm, context=context, score=score,
            already_enriched=norm in processed, bridge_reasons=reasons,
            silo_weights=weights))

    return _apply_company_decay(scored, target=target)


def _apply_company_decay(scored: List[ScoredContact],
                         target: Optional[BridgeTarget] = None) -> List[ScoredContact]:
    """Damp each additional contact at an employer already represented above it.

    Greedy submodular coverage: walk the list best-first and charge each
    contact for how much of its employer is already covered. Without this a
    single large employer's contacts occupy the whole budget.

    The TARGET's own employer is exempt. The decay exists to stop one company
    monopolising a budget meant to cover many organizations — but when the walk
    is aimed at a specific person, their employer is not one org among many, it
    is the destination. Damping it would penalise each additional colleague of
    the target precisely when a second and third route into that building is
    the most valuable thing the ranking can buy. Every other employer still
    decays normally, so coverage is still spread everywhere it should be.
    """
    eligible = [c for c in scored if c.skip_reason is None]
    skipped = [c for c in scored if c.skip_reason is not None]
    # deterministic: name breaks score ties so two runs plan identically
    eligible.sort(key=lambda c: (-c.score, c.norm_name))

    exempt = target.orgs() if target is not None else set()
    seen_per_org: Dict[str, int] = {}
    for contact in eligible:
        key = org_norm_key(contact.context)
        if key in exempt:
            continue
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
