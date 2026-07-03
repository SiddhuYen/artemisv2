"""Fuzzy name similarity for matching local profiles to public people.

Pure-stdlib (difflib) to avoid new dependencies. A "high" similarity requires
both a strong overall ratio AND surname agreement, so "John Smith" vs
"John Brown" does not score high.
"""
from __future__ import annotations

from difflib import SequenceMatcher

from ..utils.names import person_norm_key

HIGH_SIMILARITY = 0.85  # threshold for tiers 2/3 ("fuzzy name similarity high")


def name_similarity(a: str, b: str) -> float:
    """0..1 similarity between two person names (normalised, initials stripped)."""
    na, nb = person_norm_key(a), person_norm_key(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0

    ratio = SequenceMatcher(None, na, nb).ratio()

    ta, tb = na.split(), nb.split()
    # Surname agreement is a strong gate; without it, cap the score so unrelated
    # people who merely share a first name can't reach the 'high' threshold.
    if ta and tb and ta[-1] != tb[-1]:
        ratio = min(ratio, 0.6)

    # Token-set overlap bonus (handles reordering / extra middle tokens).
    sa, sb = set(ta), set(tb)
    if sa and sb:
        overlap = len(sa & sb) / max(len(sa), len(sb))
        ratio = max(ratio, 0.5 * ratio + 0.5 * overlap)

    return round(min(ratio, 1.0), 3)


def is_high(a: str, b: str) -> bool:
    return name_similarity(a, b) >= HIGH_SIMILARITY
