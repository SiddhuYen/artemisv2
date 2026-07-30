"""The operator's own identity, persisted server-side.

Until now "who am I" lived only in the browser's localStorage and was passed
per-request as `owner_name`. Two concrete costs of that: contacts imported
before the frontend started sending it have no graph edges at all (hence
ingest.backfill_graph_edges), and the operator's own employer/school were never
sent by anything, so ranking's shared-affiliation boost could not fire.

Scoped by `owner_id` — the X-Graph-Id header, the same key Boards use — rather
than assumed singleton, so two operators on one deployment do not overwrite
each other. The discovery graph itself stays shared; this is only identity.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import OwnerProfile

_FIELDS = ("name", "company", "title", "school", "linkedin_url", "email")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_owner(db: Session, owner_id: str) -> Optional[OwnerProfile]:
    return db.execute(
        select(OwnerProfile).where(OwnerProfile.owner_id == owner_id)
    ).scalar_one_or_none()


def upsert_owner(db: Session, owner_id: str, **fields) -> OwnerProfile:
    """Create or update the profile. Only supplied fields are touched, so a
    partial save (just the company, say) never blanks the rest."""
    profile = get_owner(db, owner_id)
    values = {k: (fields[k] or "").strip() or None
              for k in _FIELDS if k in fields}
    if profile is None:
        profile = OwnerProfile(owner_id=owner_id, name=values.get("name") or "")
        db.add(profile)
    for key, value in values.items():
        if key == "name" and not value:
            continue        # a profile without a name cannot anchor anything
        setattr(profile, key, value)
    profile.updated_at = _now()
    db.commit()
    return profile


def owner_dict(profile: Optional[OwnerProfile]) -> dict:
    if profile is None:
        return {"configured": False, "name": "", "company": None, "title": None,
                "school": None, "linkedin_url": None, "email": None}
    return {
        "configured": bool((profile.name or "").strip()),
        "name": profile.name, "company": profile.company,
        "title": profile.title, "school": profile.school,
        "linkedin_url": profile.linkedin_url, "email": profile.email,
        "updated_at": profile.updated_at,
    }
