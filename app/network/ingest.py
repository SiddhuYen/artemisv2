"""Local network CSV ingestion + normalization.

Handles inconsistent LinkedIn-style exports (variant/missing headers). Each row
becomes a normalized LocalProfile. Unless the CSV carries explicit relationship
columns, every uploaded profile is treated as directly connected to "You"
(a local_edge with from_profile_id = NULL).
"""
from __future__ import annotations

import csv
import io
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import LocalEdge, LocalProfile
from ..utils.names import name_variants, person_norm_key

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


def ingest_csv(db: Session, content: str) -> dict:
    """Parse + persist a CSV. Returns {created, updated, edges, skipped}."""
    # Skip a possible BOM and any LinkedIn 'Notes:' preamble before the header.
    text = _strip_preamble(content.lstrip("﻿"))
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return {"created": 0, "updated": 0, "edges": 0, "skipped": 0,
                "error": "empty or headerless CSV"}

    created = updated = edges = skipped = 0
    by_key: Dict[str, LocalProfile] = {}

    for raw in reader:
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
        by_key[key] = existing

    db.commit()
    return {"created": created, "updated": updated, "edges": edges, "skipped": skipped}


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
