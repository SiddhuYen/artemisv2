"""Wikidata provider (SECONDARY, structured relationships).

Given a person's Wikidata QID, returns explicit relationships (spouse, employer,
educated-at, board/chair, founder, student-of, …) with resolved entity names.
These are high-precision, source-grounded facts — used before HTML scraping.

Output is plain (relationship_type, name) tuples plus a synthetic evidence text
so the EXISTING extraction/graph pipeline consumes them unchanged.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .. import config
from . import cache
from .base import request_with_retry
from .ratelimit import IntervalLimiter

_ENTITYDATA = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
_WBGET = "https://www.wikidata.org/w/api.php"
_SPARQL = "https://query.wikidata.org/sparql"
_LIMITER = IntervalLimiter(config.WIKI_MIN_INTERVAL)

# properties whose co-holders are "colleagues" of the subject (reverse lookups).
# Deliberately EXCLUDES educated-at (P69) and political party (P102): sharing a
# university or party is not a real relationship (mass false "classmate"/coworker
# edges), per the no-inference-from-shared-institution rule.
_COLLEAGUE_PROPS = {
    "P108": "coworker",      # employer
    "P463": "coworker",      # member of (org / board)
}
_MAX_COLLEAGUE_ORGS = 5
_MAX_COLLEAGUES_PER_ORG = 15
_MAX_COLLEAGUES_TOTAL = 30

# Wikidata property -> (our relationship_type, human phrase used in evidence)
_PROPERTY_MAP: Dict[str, Tuple[str, str]] = {
    "P26": ("family_social", "spouse"),
    "P40": ("family_social", "child"),
    "P22": ("family_social", "father"),
    "P25": ("family_social", "mother"),
    "P3373": ("family_social", "sibling"),
    "P108": ("employee", "employer"),
    "P69": ("student", "educated at"),
    "P1066": ("advisor", "student of"),
    "P802": ("faculty", "student"),
    "P488": ("board_member", "chairperson of"),
    "P169": ("employee", "chief executive of"),
    "P1308": ("appointee", "officeholder"),
    "P39": ("appointee", "position held at"),
    "P112": ("cofounder", "founder of"),
}
_MAX_TARGETS = 40


class WikidataProvider:
    name = "wikidata"

    def is_human(self, qid: str) -> bool:
        """True if the QID is instance-of human (P31=Q5). Guards against treating
        a non-person page (an election, a company, a song) as a person."""
        if not qid:
            return False
        return _is_human(self._entity_claims(qid))

    def relationships(self, qid: str) -> List[dict]:
        """List of {relationship_type, name, prop, phrase} for a person QID."""
        if not qid:
            return []
        key = cache.make_key(self.name, "rel", qid)
        cached = cache.get(key)
        if cached is not None:
            return cached.get("rels", [])

        claims = self._entity_claims(qid)
        # collect (prop, target_qid) for mapped properties
        pairs: List[Tuple[str, str]] = []
        for prop, (_rtype, _phrase) in _PROPERTY_MAP.items():
            for stmt in claims.get(prop, []) or []:
                tgt = _claim_target_qid(stmt)
                if tgt:
                    pairs.append((prop, tgt))
                if len(pairs) >= _MAX_TARGETS:
                    break

        labels = self._labels([t for _p, t in pairs])
        rels: List[dict] = []
        for prop, tgt in pairs:
            rtype, phrase = _PROPERTY_MAP[prop]
            name = labels.get(tgt)
            if name:
                rels.append({"relationship_type": rtype, "name": name,
                             "prop": prop, "phrase": phrase})
        cache.set(key, "rel", {"rels": rels}, config.CACHE_TTL_WIKI)
        return rels

    def evidence_text(self, subject: str, rels: List[dict]) -> str:
        """Synthetic, keyword-rich text so the existing extractor classifies
        each relationship with an explicit-keyword match."""
        sentences = []
        for r in rels:
            sentences.append(f"{subject} {r['phrase']} {r['name']}.")
        return " ".join(sentences)

    def humans_from_titles(self, titles: List[str]) -> List[str]:
        """Of these Wikipedia page titles, return the ones that are PEOPLE
        (Wikidata instance-of human, P31=Q5), as resolved labels."""
        out: List[str] = []
        uniq = [t for t in dict.fromkeys(titles) if t]
        for i in range(0, len(uniq), 50):
            batch = uniq[i:i + 50]
            _LIMITER.acquire()
            resp = request_with_retry(
                "GET", _WBGET, provider=self.name,
                params={"action": "wbgetentities", "sites": "enwiki",
                        "titles": "|".join(batch), "props": "claims|labels",
                        "languages": "en", "format": "json"},
            )
            if resp is None or resp.status_code != 200:
                continue
            try:
                for _qid, ent in resp.json().get("entities", {}).items():
                    if _is_human(ent.get("claims", {})):
                        label = ent.get("labels", {}).get("en", {}).get("value")
                        if label:
                            out.append(label)
            except Exception:
                continue
        return out

    def colleagues(self, qid: str) -> List[dict]:
        """People who share an org (employer/member-of/party) or school with the
        subject — reverse SPARQL lookups. Returns [{name, relationship_type, org}]."""
        if not qid:
            return []
        key = cache.make_key(self.name, "colleagues", qid)
        cached = cache.get(key)
        if cached is not None:
            return cached.get("colleagues", [])

        claims = self._entity_claims(qid)
        org_targets = []
        for prop, rel in _COLLEAGUE_PROPS.items():
            for stmt in claims.get(prop, []) or []:
                tgt = _claim_target_qid(stmt)
                if tgt:
                    org_targets.append((prop, rel, tgt))
        org_targets = org_targets[:_MAX_COLLEAGUE_ORGS]

        results: List[dict] = []
        seen = set()
        for prop, rel, org_qid in org_targets:
            org_label = self._labels([org_qid]).get(org_qid, "")
            query = (
                f"SELECT ?pLabel WHERE {{ ?p wdt:{prop} wd:{org_qid} . "
                f"?p wdt:P31 wd:Q5 . SERVICE wikibase:label "
                f"{{ bd:serviceParam wikibase:language 'en'. }} }} "
                f"LIMIT {_MAX_COLLEAGUES_PER_ORG}"
            )
            for name in self._sparql_names(query):
                k = name.lower()
                if name and k not in seen:
                    seen.add(k)
                    results.append({"name": name, "relationship_type": rel,
                                    "org": org_label, "phrase": _REL_PHRASE.get(rel, rel)})
                if len(results) >= _MAX_COLLEAGUES_TOTAL:
                    break
            if len(results) >= _MAX_COLLEAGUES_TOTAL:
                break
        cache.set(key, "colleagues", {"colleagues": results}, config.CACHE_TTL_WIKI)
        return results

    def colleagues_text(self, subject: str, colleagues: List[dict]) -> str:
        return " ".join(f"{subject} {c['phrase']} {c['name']}." for c in colleagues)

    def _sparql_names(self, query: str) -> List[str]:
        _LIMITER.acquire()
        resp = request_with_retry(
            "GET", _SPARQL, provider=self.name,
            params={"query": query, "format": "json"},
            headers={"Accept": "application/sparql-results+json"},
        )
        if resp is None or resp.status_code != 200:
            return []
        try:
            rows = resp.json().get("results", {}).get("bindings", [])
            return [r["pLabel"]["value"] for r in rows
                    if "pLabel" in r and not r["pLabel"]["value"].startswith("Q")]
        except Exception:
            return []

    # --- internals --------------------------------------------------------
    def _entity_claims(self, qid: str) -> Dict[str, list]:
        _LIMITER.acquire()
        resp = request_with_retry("GET", _ENTITYDATA.format(qid=qid), provider=self.name)
        if resp is None or resp.status_code != 200:
            return {}
        try:
            entity = resp.json().get("entities", {}).get(qid, {})
            return entity.get("claims", {}) or {}
        except Exception:
            return {}

    def _labels(self, qids: List[str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        uniq = list(dict.fromkeys(qids))
        for i in range(0, len(uniq), 50):
            batch = uniq[i:i + 50]
            _LIMITER.acquire()
            resp = request_with_retry(
                "GET", _WBGET, provider=self.name,
                params={"action": "wbgetentities", "ids": "|".join(batch),
                        "props": "labels", "languages": "en", "format": "json"},
            )
            if resp is None or resp.status_code != 200:
                continue
            try:
                for q, ent in resp.json().get("entities", {}).items():
                    label = ent.get("labels", {}).get("en", {}).get("value")
                    if label:
                        out[q] = label
            except Exception:
                continue
        return out


_REL_PHRASE = {"coworker": "coworker of", "student": "classmate of"}


def _is_human(claims: dict) -> bool:
    for stmt in claims.get("P31", []) or []:
        if _claim_target_qid(stmt) == "Q5":
            return True
    return False


def _claim_target_qid(stmt: dict):
    try:
        dv = stmt["mainsnak"]["datavalue"]["value"]
        if isinstance(dv, dict) and dv.get("entity-type") == "item":
            return dv.get("id")
    except Exception:
        return None
    return None
