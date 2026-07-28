"""Fuzzy name similarity for matching local profiles to public people.

Pure-stdlib (difflib) to avoid new dependencies. A "high" similarity requires
both a strong overall ratio AND surname agreement, so "John Smith" vs
"John Brown" does not score high.

The "high" bar itself is length-adjusted (similarity_threshold): SequenceMatcher's
ratio is noisier for short names than long ones -- with few characters, a
shared common token (a common surname like "Chen", "Smith", "Lee") can
dominate the ratio even when the OTHER token is a completely different
person's name. Measured empirically: "Mo Chen" vs "Jo Chen" (different
people) scores 0.857 -- ABOVE a flat 0.85 bar -- while "Maria ... Garcia" vs
"Maria ... Rodriguez" (also different people, but longer) scores a safely
low 0.70. Short names get a stricter (higher) bar to compensate; long names
keep the original flat bar unchanged since it isn't showing that problem.
"""
from __future__ import annotations

from difflib import SequenceMatcher

from ..utils.names import person_norm_key

HIGH_SIMILARITY = 0.85  # baseline "high similarity" bar once names are long enough

# similarity_threshold() interpolation: SHORT_NAME_THRESHOLD applies at/below
# MIN_LEN chars (the shorter of the two normalized names), relaxing linearly
# down to HIGH_SIMILARITY by MAX_LEN chars. An exact match always scores 1.0
# in name_similarity() regardless, so this only tightens the FUZZY (non-exact)
# case for short names -- exactly where the false-positive risk above lives.
_LENGTH_ADJUST_MIN_LEN = 6
_LENGTH_ADJUST_MAX_LEN = 16
SHORT_NAME_THRESHOLD = 0.94


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


def similarity_threshold(a: str, b: str) -> float:
    """The 'high similarity' bar for this specific pair, length-adjusted --
    see module docstring. Callers that gate on HIGH_SIMILARITY directly
    (rather than through is_high) should use this instead so the bar and the
    score it's compared against stay consistent."""
    na, nb = person_norm_key(a), person_norm_key(b)
    n = min(len(na), len(nb))
    if n >= _LENGTH_ADJUST_MAX_LEN:
        return HIGH_SIMILARITY
    if n <= _LENGTH_ADJUST_MIN_LEN:
        return SHORT_NAME_THRESHOLD
    frac = (n - _LENGTH_ADJUST_MIN_LEN) / (_LENGTH_ADJUST_MAX_LEN - _LENGTH_ADJUST_MIN_LEN)
    return SHORT_NAME_THRESHOLD - (SHORT_NAME_THRESHOLD - HIGH_SIMILARITY) * frac


def is_high(a: str, b: str) -> bool:
    return name_similarity(a, b) >= similarity_threshold(a, b)
