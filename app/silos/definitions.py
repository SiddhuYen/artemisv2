"""The modular silo search engine.

Each Silo is a declarative bundle of:
  - which providers to query,
  - query templates ({person} / {company} substituted at runtime),
  - signal keywords mapping evidence text -> relationship_type,
  - a default relationship_type when no keyword matches,
  - a `strength` weight used when ranking nodes for expansion.

Adding a new silo = appending one Silo() entry. Nothing else changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Silo:
    key: str
    title: str
    providers: List[str]
    queries: List[str]
    # evidence keyword (lowercase) -> relationship_type
    signals: Dict[str, str] = field(default_factory=dict)
    # relationship types this silo is most likely to surface (ordered).
    priority_relationship_types: List[str] = field(default_factory=list)
    # per-silo confidence bias applied multiplicatively (1.0 = neutral).
    confidence_multiplier: float = 1.0
    default_relationship: str = "unknown"
    strength: float = 0.5  # relative value of edges from this silo (0..1)
    # If True, the default_relationship applies even when no signal keyword is
    # present (the query intent itself is the evidence — e.g. news/interview).
    # If False, an absent keyword yields 'unknown' rather than fabricating a
    # specific structural relationship (no inference without evidence).
    intent_default: bool = False

    def render_queries(self, person: str, company: str = "") -> List[str]:
        out = []
        for tmpl in self.queries:
            try:
                out.append(tmpl.format(person=person, company=company).strip())
            except (KeyError, IndexError):
                out.append(tmpl.replace("{person}", person))
        return out

    def spec(self, person_name: str, company: str = "") -> Dict[str, object]:
        """Structured silo contract for a given person (the required output).

        {
          "queries": [...],
          "priority_relationship_types": [...],
          "confidence_multiplier": float
        }
        """
        return {
            "queries": self.render_queries(person_name, company),
            "priority_relationship_types": list(self.priority_relationship_types),
            "confidence_multiplier": self.confidence_multiplier,
        }


SILOS: List[Silo] = [
    Silo(
        key="news",
        title="News / Media",
        providers=["duckduckgo"],
        queries=[
            '"{person}" interview',
            '"{person}" profile',
            '"{person}" appointed',
            '"{person}" joins board',
            '"{person}" named',
            '"{person}" speaks',
            '"{person}" podcast',
        ],
        signals={
            "interview": "interview",
            "podcast": "interview",
            "appointed": "appointee",
            "joins board": "board_member",
            "board": "board_member",
            "named": "employee",
        },
        priority_relationship_types=["interview", "employee", "board_member", "appointee"],
        confidence_multiplier=1.0,
        default_relationship="interview",
        strength=0.5,
        intent_default=True,
    ),
    Silo(
        key="company",
        title="Company / Executive",
        providers=["duckduckgo", "wikipedia"],
        queries=[
            '"{person}" CEO',
            '"{person}" board',
            '"{person}" founded',
            '"{person}" executive',
            '"{person}" investor',
            '"{person}" cofounder',
        ],
        signals={
            "co-founder": "cofounder",
            "cofounder": "cofounder",
            "founded": "cofounder",
            "ceo": "employee",
            "chief": "employee",
            "executive": "employee",
            "investor": "investor",
            "invested": "investor",
            "board": "board_member",
        },
        priority_relationship_types=["cofounder", "employee", "investor", "board_member"],
        confidence_multiplier=1.2,
        default_relationship="employee",
        strength=0.8,
    ),
    Silo(
        key="board_nonprofit",
        title="Board / Nonprofit / Foundations",
        providers=["duckduckgo", "wikipedia"],
        queries=[
            '"{person}" board member',
            '"{person}" trustee',
            '"{person}" foundation',
            '"{person}" advisory board',
            '"{person}" nonprofit',
        ],
        signals={
            "board member": "board_member",
            "board of directors": "board_member",
            "trustee": "board_member",
            "advisory board": "advisor",
            "advisor": "advisor",
            "adviser": "advisor",
            "foundation": "board_member",
        },
        priority_relationship_types=["board_member", "advisor"],
        confidence_multiplier=1.4,  # governance ties are high-value & specific
        default_relationship="board_member",
        strength=0.9,  # high value for warm intros
    ),
    Silo(
        key="education",
        title="University / Education",
        providers=["duckduckgo", "wikipedia"],
        queries=[
            '"{person}" professor',
            '"{person}" alumni',
            '"{person}" faculty',
            '"{person}" lecturer',
            '"{person}" PhD',
            '"{person}" research',
        ],
        signals={
            "professor": "faculty",
            "faculty": "faculty",
            "lecturer": "faculty",
            "alumni": "student",
            "alumnus": "student",
            "graduated": "student",
            "phd": "student",
            "advisor": "advisor",
            "co-author": "coauthor",
        },
        priority_relationship_types=["faculty", "student", "advisor", "coauthor"],
        confidence_multiplier=1.0,
        default_relationship="faculty",
        strength=0.6,
    ),
    Silo(
        key="events",
        title="Events / Conferences",
        providers=["duckduckgo"],
        queries=[
            '"{person}" speaker conference',
            '"{person}" panel',
            '"{person}" keynote',
            '"{person}" summit',
        ],
        signals={
            "keynote": "speaker",
            "speaker": "speaker",
            "panel": "speaker",
            "panelist": "speaker",
            "summit": "speaker",
        },
        priority_relationship_types=["speaker"],
        confidence_multiplier=0.7,  # co-presence is weak unless repeated
        default_relationship="speaker",
        strength=0.3,  # weaker unless repeated
    ),
    Silo(
        key="publications",
        title="Publications / Media Appearances",
        providers=["duckduckgo", "wikipedia"],
        queries=[
            '"{person}" coauthor',
            '"{person}" published',
            '"{person}" paper',
            '"{person}" article',
        ],
        signals={
            "co-author": "coauthor",
            "coauthor": "coauthor",
            "co-authored": "coauthor",
            "authored": "author",
            "wrote": "author",
            "published": "author",
        },
        priority_relationship_types=["coauthor", "author"],
        confidence_multiplier=1.0,
        default_relationship="author",
        strength=0.5,
    ),
    # --- Friends & Family (personal ties) ---------------------------------
    # Widen the target's DIRECT (depth-1) circle with personal connections.
    # Split into two silos so each gets its own MAX_QUERIES_PER_SILO budget
    # (more personal queries run per build). Both classify to `family_social`
    # (we don't distinguish "friend" from "relative" as a separate type), but
    # now carry explicit signal keywords: an evidenced spouse/sibling/friend is
    # an explicit-keyword match (higher confidence + eligible for expansion),
    # while bare co-occurrence still falls back to low-confidence family_social.
    Silo(
        key="family",
        title="Family / Relatives",
        providers=["duckduckgo", "wikipedia"],
        queries=[
            '"{person}" wife OR husband OR spouse',
            '"{person}" son OR daughter OR children',
            '"{person}" brother OR sister OR sibling',
            '"{person}" father OR mother OR parents',
            '"{person}" family OR relatives',
            '"{person}" married OR wedding',
        ],
        signals={
            "wife": "family_social", "husband": "family_social", "spouse": "family_social",
            "married": "family_social", "wedding": "family_social",
            "son": "family_social", "daughter": "family_social",
            "child": "family_social", "children": "family_social",
            "brother": "family_social", "sister": "family_social",
            "sibling": "family_social",
            "father": "family_social", "mother": "family_social",
            "parent": "family_social", "parents": "family_social",
            "family": "family_social", "relative": "family_social",
            "cousin": "family_social",
        },
        priority_relationship_types=["family_social"],
        confidence_multiplier=0.8,  # personal ties are noisy; keep modest
        default_relationship="family_social",
        strength=0.5,
        intent_default=True,
    ),
    Silo(
        key="friends",
        title="Friends / Personal",
        providers=["duckduckgo"],
        queries=[
            '"{person}" "close friend" OR "best friend"',
            '"{person}" "childhood friend" OR "longtime friend"',
            '"{person}" friend OR friendship',
            '"{person}" confidant OR mentor',
        ],
        signals={
            "close friend": "family_social", "best friend": "family_social",
            "childhood friend": "family_social", "longtime friend": "family_social",
            "friend": "family_social", "friendship": "family_social",
            "confidant": "family_social",
        },
        priority_relationship_types=["family_social"],
        confidence_multiplier=0.7,
        default_relationship="family_social",
        strength=0.4,
        intent_default=True,
    ),
    Silo(
        key="government",
        title="Government / Public Office",
        providers=["duckduckgo", "wikipedia"],
        queries=[
            '"{person}" appointed',
            '"{person}" administration',
            '"{person}" department',
            '"{person}" committee',
        ],
        signals={
            "appointed": "appointee",
            "nominated": "appointee",
            "administration": "advisor",
            "committee": "advisor",
            "secretary": "appointee",
        },
        priority_relationship_types=["appointee", "advisor"],
        confidence_multiplier=1.1,
        default_relationship="appointee",
        strength=0.6,
    ),
]


SILO_BY_KEY = {s.key: s for s in SILOS}


# Extraction-only silo (NOT searched) for structured Wikipedia/Wikidata facts.
# Broad signal coverage so a single pass classifies any relationship type, with
# a high confidence multiplier because the source data is structured/reliable.
STRUCTURED_SILO = Silo(
    key="structured",
    title="Structured (Wikipedia / Wikidata)",
    providers=[],
    queries=[],
    signals={
        "co-founder": "cofounder", "cofounder": "cofounder", "founder of": "cofounder",
        "founded": "cofounder",
        "employer": "employee", "employee": "employee", "chief executive": "employee",
        "ceo": "employee", "executive": "employee", "works at": "employee",
        "board": "board_member", "chairperson": "board_member", "trustee": "board_member",
        "director": "board_member",
        "advisor": "advisor", "adviser": "advisor", "student of": "advisor",
        "educated at": "student", "alumni": "student", "graduated": "student",
        "student": "student", "studied": "student",
        "professor": "faculty", "faculty": "faculty",
        "coworker of": "coworker", "coworker": "coworker", "colleague": "coworker",
        "classmate": "student",
        "co-author": "coauthor", "coauthor": "coauthor",
        "spouse": "family_social", "married": "family_social", "child": "family_social",
        "father": "family_social", "mother": "family_social", "sibling": "family_social",
        "son": "family_social", "daughter": "family_social", "family": "family_social",
        "officeholder": "appointee", "position held": "appointee", "appointed": "appointee",
    },
    priority_relationship_types=["cofounder", "board_member", "employee", "family_social"],
    confidence_multiplier=2.0,  # direct structured facts are high-trust
    default_relationship="unknown",
    strength=1.0,
    intent_default=False,
)


# For "colleague"/co-member edges from REVERSE lookups (shared employer/org via
# Wikidata, SEC co-insiders, OpenAlex coauthors, ProPublica boards). These are
# inferences from shared affiliation, NOT direct facts — so a much lower
# multiplier keeps them at 'candidate' tier (never 'strong').
COLLEAGUE_SILO = Silo(
    key="colleague",
    title="Shared-affiliation colleagues",
    providers=[],
    queries=[],
    signals=dict(STRUCTURED_SILO.signals),
    priority_relationship_types=["coworker", "board_member", "coauthor"],
    confidence_multiplier=1.3,  # shared-affiliation inference -> candidate, not strong
    default_relationship="unknown",
    strength=0.7,
    intent_default=False,
)
