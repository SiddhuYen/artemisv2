"""Deterministic confidence model + relationship classification signals.

confidence_adjusted = confidence_base
                      × silo.confidence_multiplier
                      × keyword_strength_factor
then clamped to [0, 1] and capped by the evidence rules:
  - no explicit keyword match  -> cannot exceed NO_EXPLICIT_KEYWORD_CEILING
  - sentence co-occurrence only -> cannot exceed COOCCURRENCE_ONLY_CEILING

Tiers (config): < WEAK_MAX = weak, [WEAK_MAX, STRONG_MIN] = candidate,
> STRONG_MIN = strong.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Tuple

from .. import config
from ..utils.names import normalize


@lru_cache(maxsize=None)
def _kw_regex(keyword: str) -> "re.Pattern[str]":
    """Word-boundary regex for a keyword, so e.g. "son" doesn't match inside
    "Johnson" and "board" doesn't match inside "keyboard". Compiled once per
    distinct keyword and cached, since the keyword set is small and fixed."""
    return re.compile(r"\b" + re.escape(keyword) + r"\b")


def keyword_strength_factor(text: str) -> Tuple[float, List[str]]:
    """Return (multiplicative factor, distinct strength keywords found)."""
    lowered = (text or "").lower()
    found = []
    for kw in config.STRENGTH_KEYWORDS:
        if kw not in found and _kw_regex(kw).search(lowered):
            found.append(kw)
    # normalise co-/hyphen variants so "cofounder"/"co-founder" don't double count
    distinct = {k.replace("-", "") for k in found}
    factor = min(
        1.0 + config.STRENGTH_KEYWORD_STEP * len(distinct),
        config.STRENGTH_FACTOR_CEILING,
    )
    return factor, found


def classify_with_signal(evidence_text: str, silo) -> Tuple[str, bool, str]:
    """Classify relationship_type from the evidence and report whether an
    EXPLICIT silo keyword drove it.

    Returns (relationship_type, explicit_keyword_match, matched_keyword).
    - explicit silo signal keyword present -> (mapped_type, True, kw).
    - else intent_default silo -> (default, False, "").
    - else -> ("unknown", False, "").

    Note: the family/friends silos now carry explicit family_social signal
    keywords, so an evidenced personal tie (named spouse/sibling/friend) is a
    true explicit match; bare co-occurrence falls back to their intent default.
    """
    lowered = (evidence_text or "").lower()
    for keyword, rel_type in silo.signals.items():
        if _kw_regex(keyword).search(lowered):
            return rel_type, True, keyword
    if silo.intent_default:
        return silo.default_relationship, False, ""
    return "unknown", False, ""


def sentence_cooccurrence(subject: str, other: str, evidence_text: str) -> bool:
    """True if subject and other both appear in the evidence sentence."""
    low = (evidence_text or "").lower()
    return normalize(subject) != "" and subject.lower() in low and other.lower() in low


def compute_confidence(
    base: float,
    silo_multiplier: float,
    strength_factor: float,
    explicit_keyword_match: bool,
    cooccurrence: bool,
) -> float:
    """Apply the model and the evidence-rule ceilings, then clamp to [0,1].

    Only an explicit relationship keyword co-occurring WITH the pair can
    reach 'strong' -- every other case is ceilinged by how much was actually
    corroborated, weakest to strongest evidence (first matching case wins):
      - explicit keyword, but subject/target never share a sentence -- the
        keyword alone doesn't establish THIS pair (COOCCURRENCE_ONLY_CEILING).
      - no cooccurrence at all and no explicit keyword -- the weakest case
        (NO_COOCCURRENCE_CEILING, kept under WEAK_MAX so it always reads
        'weak').
      - cooccurrence, no strength keyword either (COOCCURRENCE_ONLY_CEILING).
      - cooccurrence AND a strength keyword (e.g. "led", "board"), but that
        keyword never named the actual relationship type
        (GENERIC_STRENGTH_CEILING) -- clearly short of 'strong' rather than
        0.01 away from it on nothing more than a generic verb.
    """
    adjusted = base * silo_multiplier * strength_factor

    if explicit_keyword_match and cooccurrence:
        return round(max(0.0, min(adjusted, 1.0)), 3)

    adjusted = min(adjusted, config.NO_EXPLICIT_KEYWORD_CEILING)  # absolute backstop

    if not cooccurrence:
        ceiling = (config.COOCCURRENCE_ONLY_CEILING if explicit_keyword_match
                   else config.NO_COOCCURRENCE_CEILING)
    elif not _any_strength(strength_factor):
        ceiling = config.COOCCURRENCE_ONLY_CEILING
    else:
        ceiling = config.GENERIC_STRENGTH_CEILING
    adjusted = min(adjusted, ceiling)

    return round(max(0.0, min(adjusted, 1.0)), 3)


def _any_strength(strength_factor: float) -> bool:
    return strength_factor > 1.0 + 1e-9


def tier(confidence: float) -> str:
    """weak / candidate / strong per configured thresholds."""
    if confidence < config.WEAK_MAX:
        return "weak"
    if confidence <= config.STRONG_MIN:
        return "candidate"
    return "strong"
