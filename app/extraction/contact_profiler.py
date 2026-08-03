"""Is this contact worth spending 35 queries on? Answered from the CSV row.

network/ranking's rule 1 states the problem plainly: "Seniority is a proxy for
WEB FOOTPRINT, not for value." The proxy is a title regex, and on a real
1,187-contact export it was the ONLY signal with any spread -- `_HAS_PUBLIC_EDGES`
fired for 11 contacts and notability for none -- so it decided the ranking
alone, and its false positives ("Events Chair", "CSP Partner Marketing Intern")
went straight to the top. This measures the thing the regex was standing in for.

Deliberately target-INDEPENDENT, and that is the whole point of putting it here
rather than at rank time. Whether a contact has a discoverable professional
footprint is a fact about them, true for every target they might ever bridge
toward, so it is computed once at import and reused by every future /connect.
Only the target-dependent half of the judgment -- which of these people bridges
to THIS person -- belongs in the per-connect path (see bridge_strategy).

Costs no searches. That matters because the alternative, running the real
enrichment batch, is ~35 queries x every contact, which is the exact spend the
ranking exists to avoid -- circular: you need the ranking to choose who to
enrich, and enrichment to make the ranking good. This breaks the circle with
one batched pass over text the operator already gave us.

The judgment is about the ROW, not the person. "Does this employer/title
combination suggest someone public sources name individually" is answerable
from a CSV line. "Is this person important" is not, and asking it would invite
exactly the training-data prior the rest of this package spends its guards
defending against ("everyone knows a Goldman VP can reach anyone"). The prompt
asks the first question and the enum only has room for that answer.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from .. import config
from ..providers import cache
from .claude_client import call_json, claude_available

# What a search for this person's NAME would plausibly return. Ordered, and the
# order is the point -- it is what gives the ranking real spread where a title
# regex gave 1,100 contacts the same handful of values.
FOOTPRINTS = ("individual", "org_only", "none")

# The silo vocabulary (app/silos), minus family/friends which are not domains.
# Reusing the silo keys rather than inventing an industry list is deliberate:
# a contact's domain and a target's silo weights have to be comparable, and a
# parallel taxonomy would quietly make _silo_affinity meaningless.
DOMAINS = ("company", "news", "board_nonprofit", "education", "events",
           "publications", "government", "other")

_PROMPT = """For each person below, judge from THEIR ROW ALONE what a web \
search for their name would return. You are judging the row, not the person -- \
do not use outside knowledge about anyone or any company.

THE DECIDING QUESTION IS THE ROLE, NOT THE EMPLOYER. A famous employer does \
not make an individual searchable: an engineer at Google and an engineer at a \
ten-person startup are BOTH org_only, because public sources write about the \
company, not about them. Only promote to individual when the ROLE ITSELF is \
one that gets a person named.

footprint:
- individual: the role is one public sources name a specific person in -- \
founder, co-founder, owner, chief executive or other C-level officer, \
partner, professor, elected or appointed official, published author. \
Searching their name plausibly returns pages about THEM.
- org_only: a real, searchable organization, but an ordinary role inside it. \
This is the DEFAULT for employed people. Engineer, developer, analyst, \
actuary, scientist, consultant, associate, designer, recruiter, manager, \
director of a function, intern, "incoming" anything -- all org_only, however \
prestigious the employer.
- none: the row names no organization that generates public professional \
coverage at all -- a student club, campus chapter or society, a school \
society office, a part-time retail or service job, "self-employed", \
"freelance", or a blank employer.

Most rows in a normal contact export are org_only. If you are unsure between \
two tiers, choose the lower one -- a wrongly-promoted row costs a real search \
budget, a wrongly-demoted one costs nothing that cannot be recovered later.

domain: which world this row belongs to -- one of: {domains}.
Use "other" when the row does not clearly sit in any of them.

why: at most one short clause, quoting what in the row decided it.

Items:
{items}

Return one entry per item, using the item's own number."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "footprint": {"type": "string", "enum": list(FOOTPRINTS)},
                    "domain": {"type": "string", "enum": list(DOMAINS)},
                    "why": {"type": "string"},
                },
                "required": ["index", "footprint", "domain", "why"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def is_active() -> bool:
    return bool(config.CONTACT_PROFILE_ENABLED) and claude_available()


# Bump whenever the prompt changes what a given row SHOULD score. Without it a
# reworded prompt is invisible: verdicts are cached for CACHE_TTL_WIKI keyed on
# the row's own text, so every already-seen contact would return the old answer
# and a re-profile would spend nothing and change nothing. v2 rewrote the
# footprint definitions after v1 read employer prestige as individual coverage
# and promoted 18% of a mostly-junior export to the top tier.
PROMPT_VERSION = 2


def _key(item: dict) -> str:
    raw = (f"v{PROMPT_VERSION}|{item.get('name','')}"
           f"|{item.get('employer','')}|{item.get('title','')}")
    return "contactprofile:" + hashlib.sha1(raw.encode()).hexdigest()


def _render(items: List[dict]) -> str:
    lines = []
    for n, it in enumerate(items, 1):
        lines.append(
            f"{n}. {it.get('name','')} -- employer: {it.get('employer') or 'none'}"
            f" -- title: {it.get('title') or 'none'}"
            + (f" -- school: {it['school']}" if it.get("school") else ""))
    return "\n".join(lines)


def _ask(items: List[dict]) -> Dict[int, dict]:
    payload = call_json(
        _PROMPT.format(items=_render(items), domains=", ".join(DOMAINS)),
        schema=_SCHEMA, model=config.CONTACT_PROFILE_MODEL,
        max_tokens=200 * len(items) + 256)
    out: Dict[int, dict] = {}
    if not payload:
        return out
    for row in (payload.get("results") or []):
        try:
            out[int(row["index"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return out


def profile(items: List[dict]) -> List[dict]:
    """items: [{name, employer, title, school}] -> [{footprint, domain, why}]
    aligned by index. Returns None entries where no verdict was reached.

    A missing verdict stays None rather than defaulting to a footprint, and is
    never cached: persisting a non-judgment would permanently mark a real
    contact unsearchable on one bad response, and the caller can tell the
    difference between "judged low" and "not judged" only if this does.
    """
    results: List[dict] = [None] * len(items)
    if not items or not is_active():
        return results

    pending = []
    for idx, it in enumerate(items):
        cached = cache.get(_key(it), track=False)
        if cached is not None:
            results[idx] = cached
        else:
            pending.append((idx, it))

    size = config.CONTACT_PROFILE_BATCH
    chunks = [pending[s:s + size] for s in range(0, len(pending), size)]
    if not chunks:
        return results

    def _apply(chunk, verdicts: Dict[int, dict]) -> None:
        for n, (orig_idx, it) in enumerate(chunk, 1):
            v = verdicts.get(n)
            if v is None:
                continue
            fp = v.get("footprint")
            dom = v.get("domain")
            if fp not in FOOTPRINTS:
                continue          # an off-enum answer is not a judgment
            out = {"footprint": fp,
                   "domain": dom if dom in DOMAINS else "other",
                   "why": str(v.get("why", "") or "")[:200]}
            results[orig_idx] = out
            cache.set(_key(it), "contactprofile", out, config.CACHE_TTL_WIKI)

    # Same shape as relation_classifier/entity_filter: the first chunk runs
    # alone so a broken credential costs one wasted call rather than one per
    # chunk, and the rest -- independent, no shared state -- run concurrently.
    first, rest = chunks[0], chunks[1:]
    _apply(first, _ask([it for _i, it in first]))
    if not rest or not claude_available():
        return results

    with ThreadPoolExecutor(max_workers=min(4, len(rest))) as ex:
        futures = {ex.submit(_ask, [it for _i, it in chunk]): chunk for chunk in rest}
        for future in as_completed(futures):
            if not claude_available():
                continue
            _apply(futures[future], future.result())
    return results
