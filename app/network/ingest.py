"""Local network CSV ingestion + normalization.

Handles inconsistent LinkedIn-style exports (variant/missing headers). Each row
becomes a normalized LocalProfile. Unless the CSV carries explicit relationship
columns, every uploaded profile is treated as directly connected to "You"
(a local_edge with from_profile_id = NULL).

When `owner_name` is supplied, each imported contact ALSO becomes a real
`linkedin_1st` RelationshipEdge in the shared public graph (owner <-> contact),
so /connect and /discover can route through it immediately — a CSV row is a
structural assertion that the connection exists, exactly like DrewGloverDemo's
linkedin_csv ingester. Without `owner_name`, ingestion stays scoped to the
walled-off LocalProfile/LocalEdge tables (today's behavior), since there is no
graph node to anchor the edges to.
"""
from __future__ import annotations

import csv
import io
from typing import Dict, Iterable, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..extraction.schemas import EdgeSignals, ExtractedEdge
from ..models import LocalEdge, LocalProfile, Person
from ..providers.base import SearchResult
from ..utils.names import name_variants, person_norm_key

# Confidence for a linkedin_1st edge — a CSV row IS the assertion (not
# inferred from prose), so it sits above every web-sourced relationship type.
_LINKEDIN_CONFIDENCE = 0.95


def _linkedin_edge(db: Session, owner: Person, contact: Person, url: str) -> None:
    from ..graph import builder  # local import: avoids a network<->graph cycle

    if contact.id == owner.id:
        return
    # Stable per (owner, contact) synthetic URL when the row carries no
    # profile URL, so re-uploading the same export stays idempotent instead
    # of piling up duplicate Source rows / edges each time.
    source_url = url or f"linkedin-import://{owner.norm_name}/{contact.norm_name}"
    res = SearchResult(contact.canonical_name, source_url, "linkedin_csv",
                       "linkedin_csv")
    source = builder.save_source(db, res, "linkedin_csv_import")
    edge = ExtractedEdge(
        person_a=owner.canonical_name, person_b=contact.canonical_name,
        other_kind="person", relationship_type="linkedin_1st",
        method="LinkedIn connections export", source_url=source_url,
        evidence_snippet=(f"{contact.canonical_name} is a verified LinkedIn "
                          f"connection of {owner.canonical_name}."),
        confidence_base=_LINKEDIN_CONFIDENCE, confidence_adjusted=_LINKEDIN_CONFIDENCE,
        signals=EdgeSignals(trusted=True, explicit_keyword_match=True),
    )
    builder.add_edge_from_extraction(db, owner, edge, 0, source, contact)

# header alias -> canonical field (compared case-insensitively, spaces/_ ignored)
_HEADER_ALIASES = {
    "firstname": "first_name", "first": "first_name",
    "lastname": "last_name", "last": "last_name", "surname": "last_name",
    "name": "name", "fullname": "name",
    "company": "company", "organization": "company", "organisation": "company",
    "currentcompany": "company", "employer": "company",
    "position": "title", "title": "title", "role": "title", "headline": "title",
    "email": "email", "emailaddress": "email", "emailaddresses": "email",
    "location": "location", "region": "location",
    "school": "school", "education": "school",
    "connectedon": "connected_on",
    "url": "url", "profileurl": "url", "linkedinurl": "url", "publicurl": "url",
    "notes": "notes",
    # optional explicit relationship columns
    "connectedto": "connected_to", "relationship": "relationship",
}


def _canon_header(h: str) -> str:
    key = "".join(ch for ch in (h or "").lower() if ch.isalnum())
    return _HEADER_ALIASES.get(key, key)


def _row_get(row: Dict[str, str], field: str) -> str:
    val = row.get(field)
    return val.strip() if isinstance(val, str) else ""


def _as_list(value: str) -> List[str]:
    """Split a possibly multi-valued cell on ';' or '|' (not ',' — names/orgs
    legitimately contain commas)."""
    if not value:
        return []
    parts = [p.strip() for p in value.replace("|", ";").split(";")]
    return [p for p in parts if p]


def _profile_from_row(raw_row: Dict[str, str]) -> Optional[dict]:
    row = {_canon_header(k): (v or "") for k, v in raw_row.items()}

    name = _row_get(row, "name")
    if not name:
        first, last = _row_get(row, "first_name"), _row_get(row, "last_name")
        name = " ".join(p for p in (first, last) if p).strip()
    if not name:
        return None  # cannot anchor a profile without a name

    return {
        "canonical_name": name,
        "norm_name": person_norm_key(name),
        "aliases": sorted(v for v in name_variants(name) if v != name),
        "email": (_row_get(row, "email") or None),
        "linkedin_url": (_row_get(row, "url") or None),
        "companies": _as_list(_row_get(row, "company")),
        "titles": _as_list(_row_get(row, "title")),
        "schools": _as_list(_row_get(row, "school")),
        "locations": _as_list(_row_get(row, "location")),
        "notes": (_row_get(row, "notes") or None),
        "raw_row": {k: v for k, v in raw_row.items() if v},
        "connected_on": (_row_get(row, "connected_on") or None),
        "_connected_to": _row_get(row, "connected_to"),
    }


def _dedup_key(p: dict) -> str:
    return (p["email"] or "").lower() or p["norm_name"]


_HEADER_HINTS = ("first name", "last name", "name,", "email", "company", "url")


def _strip_preamble(text: str) -> str:
    """LinkedIn exports prepend a 'Notes:' blurb before the real header row.
    Drop everything before the first line that looks like the column header."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        low = line.lower()
        if ("first name" in low and "last name" in low) or \
           (low.startswith("name,") or ",name," in low) or \
           (sum(h in low for h in _HEADER_HINTS) >= 2 and "," in line):
            return "\n".join(lines[i:])
    return text


def backfill_graph_edges(db: Session, owner_name: str,
                         claim_unowned: bool = False) -> int:
    """Retroactively bridge already-imported LocalProfiles into the shared
    public graph as linkedin_1st edges, for profiles imported before
    `owner_name` was passed on upload (or before that parameter existed at
    all -- every profile imported through the UI prior to this fix has no
    graph edge, since the frontend never sent it).

    Idempotent: _linkedin_edge's stable synthetic source_url plus
    add_edge_from_extraction's upsert-by-(a, b, type, source) dedup mean
    re-running this for the same owner is always safe -- it converges rather
    than piling up duplicate edges.

    Scoped to THIS owner's rows, and strictly: a row with no owner_norm is
    excluded rather than shared. Several people's exports live in one
    local_profiles table, and before this the loop selected all of them --
    confirmed live, where running it wrote 1,188 first-degree edges from one
    operator to another operator's contacts. Those edges are traversable, so
    connect._route_exists will short-circuit a real search and answer through
    people the operator has never met.

    Excluding unowned rows is the same safe direction connect._origin_is_operator
    already takes ("returns False when neither is available"): asserting a
    first-degree tie is a claim about the world, and a row that does not say
    whose contact it is cannot support that claim.

    `claim_unowned` restores the original migration affordance -- rows imported
    before owner_name existed -- but as an explicit opt-in, and it CLAIMS them
    (stamps owner_norm) rather than merely reading them, so the assertion is
    recorded and the next operator to run this cannot make it again. It is
    off by default because the default is what runs unattended on every
    /connect, where a silent land-grab of someone else's export is exactly the
    failure this scoping exists to prevent."""
    from ..graph import builder

    owner_name = (owner_name or "").strip()
    if not owner_name:
        return 0
    owner = builder.get_or_create_person(db, owner_name)
    if owner is None:
        return 0
    owner_key = person_norm_key(owner_name)

    # Whole loop wrapped as one retryable unit, not just the final commit:
    # this function's own docstring already establishes it's safe to redo
    # (idempotent, stable synthetic source_url), which is exactly what a
    # retry needs -- see builder.commit_with_retry.
    def _apply() -> int:
        count = 0
        where = LocalProfile.owner_norm == owner_key
        if claim_unowned:
            where = where | (LocalProfile.owner_norm.is_(None))
        for profile in db.execute(select(LocalProfile).where(where)).scalars():
            contact = builder.get_or_create_person(db, profile.canonical_name)
            if contact is None:
                continue
            _linkedin_edge(db, owner, contact, profile.linkedin_url or "")
            if profile.owner_norm is None:
                profile.owner_norm = owner_key   # claimed, so only once
                db.add(profile)
            count += 1
        return count

    return builder.commit_with_retry(db, _apply) or 0


def ingest_csv(db: Session, content: str, owner_name: str = "") -> dict:
    """Parse + persist a CSV. Returns {created, updated, edges, skipped,
    graph_edges}. `graph_edges` (only non-zero when `owner_name` is given)
    counts real linkedin_1st RelationshipEdge rows added to the shared graph —
    see module docstring."""
    # Skip a possible BOM and any LinkedIn 'Notes:' preamble before the header.
    text = _strip_preamble(content.lstrip("﻿"))
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return {"created": 0, "updated": 0, "edges": 0, "skipped": 0,
                "graph_edges": 0, "error": "empty or headerless CSV"}
    return ingest_rows(db, reader, owner_name=owner_name)


def ingest_rows(db: Session, rows: Iterable[Dict[str, str]],
                owner_name: str = "") -> dict:
    """Persist already-parsed rows — the shared tail of every import path, so a
    vCard import (see /network/contacts/import) de-dupes, links to "You" and
    seeds the public graph on exactly the same terms as a CSV upload. Keys are
    matched through the same `_HEADER_ALIASES`, so a row is just
    {"Name": ..., "Company": ..., "Position": ...}."""
    from ..graph import builder  # local import: avoids a network<->graph cycle

    rows = list(rows)  # buffered so a retry can re-iterate from the start,
                       # not just re-read an already-exhausted CSV reader

    # Whole ingest wrapped as one retryable unit, not just the final commit:
    # LocalProfile/LocalEdge rows are added directly in this loop (unlike the
    # graph module's own writes, which already retry internally via
    # save_source/add_edge_from_extraction), and owner resolution itself can
    # mutate an existing Person's aliases -- all of that is exactly what a
    # forced rollback() would undo/expunge (confirmed empirically, see
    # tests/test_commit_retry.py), so a retry redoes the whole pass rather
    # than just re-committing an empty transaction.
    def _apply() -> dict:
        owner = builder.get_or_create_person(db, owner_name) if owner_name.strip() else None
        owner_key = person_norm_key(owner_name) if owner_name.strip() else ""
        created = updated = edges = skipped = graph_edges = 0
        by_key: Dict[str, LocalProfile] = {}

        for raw in rows:
            parsed = _profile_from_row(raw)
            if parsed is None:
                skipped += 1
                continue
            connected_to = parsed.pop("_connected_to", "")
            key = _dedup_key(parsed)

            existing = by_key.get(key)
            if existing is None:
                existing = db.execute(
                    select(LocalProfile).where(LocalProfile.norm_name == parsed["norm_name"])
                ).scalar_one_or_none()
                if existing and parsed["email"] and existing.email \
                        and existing.email.lower() != parsed["email"].lower():
                    existing = None  # same name, different person

            if existing:
                _merge_profile(existing, parsed)
                updated += 1
            else:
                existing = LocalProfile(**parsed)
                db.add(existing)
                db.flush()
                created += 1
                # default: directly connected to "You"
                db.add(LocalEdge(from_profile_id=None, to_profile_id=existing.id))
                edges += 1
            # Stamp ownership on created AND updated rows: an import is a
            # claim about whose contacts these are, and a row someone else
            # uploaded first is still this operator's contact too. Last writer
            # wins, which is the honest reading of "I also know this person"
            # (the alternative -- first-upload-owns -- would let one operator
            # permanently exclude everyone else's identical contact).
            if owner_key:
                existing.owner_norm = owner_key
            by_key[key] = existing

            if owner is not None:
                contact = builder.get_or_create_person(db, existing.canonical_name)
                if contact is not None:
                    _linkedin_edge(db, owner, contact, existing.linkedin_url or "")
                    graph_edges += 1

        return {"created": created, "updated": updated, "edges": edges,
                "skipped": skipped, "graph_edges": graph_edges}

    return builder.commit_with_retry(db, _apply) or {
        "created": 0, "updated": 0, "edges": 0, "skipped": 0, "graph_edges": 0}


def _merge_profile(existing: LocalProfile, parsed: dict) -> None:
    for field in ("companies", "titles", "schools", "locations", "aliases"):
        merged = sorted(set(existing.__dict__.get(field) or []) | set(parsed[field]))
        setattr(existing, field, merged)
    if not existing.email and parsed["email"]:
        existing.email = parsed["email"]
    if not existing.linkedin_url and parsed["linkedin_url"]:
        existing.linkedin_url = parsed["linkedin_url"]
    if not existing.notes and parsed["notes"]:
        existing.notes = parsed["notes"]
    if not existing.connected_on and parsed["connected_on"]:
        existing.connected_on = parsed["connected_on"]
