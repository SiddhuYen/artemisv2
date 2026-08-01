# `/connect` design: famous ↔ famous

Status: **proposal, not implemented**. Companion to Alpha (nonfamous ↔
famous, merged in PR #25) — this covers the case Alpha explicitly excludes.

## Where this sits in the four-quadrant plan

`/connect` pairs fall into four buckets by notability (Wikidata `wikidata_qid`
presence, via `ORCH.notable_set`):

| A \ B | famous | nonfamous |
|---|---|---|
| **famous** | this doc | Alpha (done) |
| **nonfamous** | Alpha (done) | not yet designed |

Alpha only fires when *exactly one* side is notable —
`_resolve_expansion_depths` in [app/graph/connect.py](../app/graph/connect.py)
checks `a_notable != b_notable` and falls back to symmetric `(depth, depth)`
expansion whenever both are famous (or neither is). This doc proposes the
reasoning for the both-famous cell. Nonfamous ↔ nonfamous is out of scope
here too.

## The problem

Today, both-famous pairs get plain symmetric expansion at the caller's
`depth` on **both** sides — no capping at all. Alpha exists specifically
because one side ballooning (a famous person's huge, expensive,
slow-to-prune network) was already a problem when only one side had that
shape. Both-famous doesn't just fail to get Alpha's mitigation, it's the
worst case for the exact cost Alpha was built to avoid: two large, unrelated
expansions racing to write into the same shared graph, most of which is
irrelevant to the actual bridge.

## Why this isn't "Alpha, applied twice"

Alpha's four pieces solve problems specific to the nonfamous side having no
structured data to reason over. None of those problems are the same shape
here:

- **Depth asymmetry** doesn't apply — there's no natural "origin" side to
  keep full and "famous" side to cap at `SHALLOW_FAMOUS_DEPTH` (1 hop). Both
  sides are the balloon risk simultaneously.
- **`node_profiler.py`** (piece 1) exists because an ordinary person's org
  has no structured facts — Claude has to synthesize size/industry from
  scraped prose, which is why it's grounding-gated against hallucination.
  Famous people already carry structured Wikidata facts (occupation,
  employer, board seats, education, awards). There's little to synthesize;
  this is closer to a lookup/overlap check than an inference call.
- **`search_strategy.py`**'s fixed-enum angle-picking (piece 2) is still a
  reasonable pattern, but the angles themselves need to change (see below).
- **`SENIORITY_BONUS`** (piece 3) doesn't discriminate here — both sides are
  senior by construction.
- **Shallow-bridge capping** (piece 4, stop expanding a candidate who turns
  out to be independently famous) may fire on nearly every candidate in a
  famous-famous walk, since famous-adjacent people are the norm, not the
  exception, in this graph region. The trigger condition needs to be
  sharper than "is this candidate notable at all."

## Proposed design

### 1. Lean harder on `_direct_pair_search` before expanding either side

`_direct_pair_search` (already in `connect.py`) checks whether the two names
are ever mentioned together before paying for a full neighborhood walk. For
an ordinary person paired with a celebrity, that's a cheap long-shot. For two
famous people, genuine interactions between them are *more* likely to be
independently reported (joint interviews, shared board announcements, event
photos) than for an ordinary-person pair — this should be the primary
channel, not a first-pass filter. Concretely: raise `SCRAPE_TOP_N` (currently
10) for this call specifically when both endpoints are notable, and consider
a second direct-pair query pass with a different query template
(e.g. `"{a}" "{b}" board OR event OR interview`) before falling through to
full expansion.

### 2. Replace node profiling with a structured overlap check

Instead of a Claude call synthesizing prose into a size/industry judgment
(node_profiler's shape), pull each side's existing structured signals —
Wikidata occupation/employer/board-membership fields, already available via
the same lookup `notable_set` uses — and compute categorical overlap (shared
industry, overlapping institutions, overlapping era). Lower hallucination
risk than piece 1 by construction: it's a lookup and set-intersection, not
open-ended synthesis over ambiguous snippets. Feeds into step 3 the same way
`node_profiler`'s output feeds `search_strategy`.

### 3. Fixed-enum search strategy, new angle set

Keep piece 2's containment principle — the model picks an angle from a fixed
enum, never writes query text — but the angles change to fit two
already-documented people:

- `shared_board_or_event`
- `same_industry_peers`
- `alma_mater_or_award`
- `generic` (today's untargeted silo search)

Each angle maps to a `config.STRATEGY_ANGLE_QUERIES`-style deterministic
template, same as Alpha.

### 4. Ranking bias: overlap with the target's world, not seniority

`SENIORITY_BONUS` rewards cofounder/board-member edges — not useful when
every candidate is plausibly senior. The equivalent signal here is whether a
candidate's own profile overlaps with the *other endpoint's* specific world
(same industry/institution as the target), which step 2's overlap data
already computes. Candidates with target-relevant overlap should outrank
generically well-evidenced but unrelated ones.

### 5. Shallow-bridge capping: sharpen the trigger

Piece 4 stops expanding a candidate who turns out to be independently
notable. Applied unmodified here, it would fire on most candidates and
collapse the walk to depth 1 everywhere. Proposed trigger instead: only cap
a candidate whose own notability/hub-degree *exceeds* both endpoints' (i.e.
a third celebrity bigger than either target), using the existing
`MEGA_HUB_DEGREE` / `FAME_PENALTY` machinery in `_node_penalty` as the
comparison basis rather than a flat notable/not-notable check.

## Open questions

- Should both sides share one depth cap, or does one endpoint (e.g. whichever
  has denser Wikidata board/employer data) deserve to go deeper, mirroring
  Alpha's asymmetry even though neither side is "nonfamous"?
- Is `MEGA_HUB_DEGREE` (40) already a reasonable threshold for step 5's
  "bigger than either endpoint" comparison, or does that need its own
  constant?
- Does the direct-pair-search boost in step 1 need its own config flag
  (`FAMOUS_FAMOUS_*`, mirroring `STRATEGY_ENABLED`/`NODE_PROFILE_ENABLED`),
  or can it key off the same `notable` check inline?

## Non-goals

- Nonfamous ↔ nonfamous is a separate, undesigned quadrant — not addressed
  here.
- No code changes in this doc's PR — design only, for review before
  implementation starts.
