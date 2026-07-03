# Artemis V2 — Public Relationship Graph Builder (MVP)

Discovers real-world connections between **people** and **organizations** from
public data sources, and builds a growing, **evidence-grounded** knowledge graph.

> **MVP scope (this build):** `search → extraction → graph building → expansion`.
> **Intentionally NOT included yet:** matching against a user's external network,
> and the Claude verification stage. Every relationship is recorded with its
> source URL + evidence snippet; nothing is auto-marked `accepted`.

## Design philosophy

> "Discover everything public, build a messy but evidence-grounded relationship graph."
> NOT "Decide whether two people are connected."

No relationship is inferred without textual evidence. When a silo surfaces a page
but no signal keyword for that silo is present, the edge is recorded as
`unknown` rather than fabricating a specific relationship type.

## Stack

Python 3.9+ · FastAPI · SQLite + SQLAlchemy 2.0 · httpx · BeautifulSoup4 · pydantic v2
· optional Ollama (`http://localhost:11434`).

## Run (terminal — primary)

Give a name, get the relationship graph printed in your terminal. No frontend.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m app.cli "Satya Nadella"            # target's direct relationships (fast)
python -m app.cli "Satya Nadella" --depth 2  # also expand top connections (slower)
python -m app.cli "Satya Nadella" --all      # include low-signal 'unknown' edges
python -m app.cli                            # prompts for a name interactively
```

Flags: `--depth N` (1 = target only [default], 2 = +one hop), `--all` (show every
extracted entity incl. noisy `unknown` edges), `--keep` (accumulate into the
existing graph instead of starting fresh each run).

Each connection is printed with its relationship type, confidence, status,
source domain(s), and the evidence sentence.

### Extraction quality

If a local **Ollama** daemon is reachable (`http://localhost:11434`) it is used
for extraction and the graph is much cleaner. Otherwise the system falls back to
a conservative **low-confidence heuristic extractor** (capitalised-name +
org-suffix detection) — useful but noisy, which is why the CLI hides `unknown`
edges by default.

## Run (optional HTTP API)

```bash
uvicorn app.main:app --reload   # open http://localhost:8000/docs
```

`GET /health` reports which extractor mode is active.

## API

| Method | Path              | Purpose |
|--------|-------------------|---------|
| POST   | `/targets/search` | Run the full pipeline for a target and expand the graph |
| GET    | `/graph`          | Full node/edge graph |
| GET    | `/people`         | All discovered people |
| GET    | `/edges`          | All relationship edges |
| GET    | `/health`         | Liveness + active extractor |

### `POST /targets/search`

```json
// request
{ "target_name": "Tim Cook", "max_depth": 2 }
```

```json
// response
{
  "nodes": [ { "id": "...", "label": "Tim Cook", "kind": "person", "type": null } ],
  "edges": [ {
    "id": "...", "from": "<person_a_id>", "to": "<person_b_or_org_id>",
    "type": "board_member", "confidence": 0.25, "source_url": "https://...",
    "status": "raw", "method": "...", "evidence": "...", "depth": 0
  } ],
  "stats": { "people_found": 4, "organizations_found": 2,
             "edges_found": 22, "sources_fetched": 3 }
}
```

The graph output matches the required wire format: `nodes[{id,label}]` and
`edges[{from,to,type,confidence,source_url}]` (plus extra evidence fields).

## How it works

```
target ─▶ SILOS (8) ─▶ SearchProviders ─▶ scrape/summary ─▶ Extraction ─▶ Graph build ─▶ Expansion (BFS)
```

1. **Silos** ([app/silos/definitions.py](app/silos/definitions.py)) — 8 declarative
   search domains (News, Company, Board/Nonprofit, Education, Events,
   Publications, Social/Family, Government). Each defines providers, query
   templates, signal keywords → relationship types, and a `strength` weight.
   Add a silo by appending one `Silo(...)` entry.
2. **Search providers** ([app/providers/](app/providers/)) — DuckDuckGo HTML
   endpoint (with redirect-link decoding + lite fallback) and the Wikipedia
   MediaWiki API. Shared `User-Agent`, in-memory caching, best-effort (never
   raises on network failure).
3. **Extraction** ([app/extraction/](app/extraction/)) — Ollama strict-JSON
   extractor with a heuristic fallback. Relationship type is chosen only from
   signal keywords actually present in the text.
4. **Graph builder** ([app/graph/builder.py](app/graph/builder.py)) — dedup-aware
   upserts (people/orgs by normalized name, sources by URL, edges by endpoint +
   type + source). Confidence → status policy; `family_social` capped at
   `candidate`; nothing auto-`accepted`.
5. **Expansion** ([app/graph/expansion.py](app/graph/expansion.py)) — BFS that
   runs all silos on the target (hop 0), ranks discovered people by
   `#sources · 2 + max_confidence + strength · 1.5`, then expands only the
   **top 20 per depth** up to `max_depth`.

## Data model

`people`, `organizations`, `sources`, `relationship_edges` — see
[app/models.py](app/models.py). Relationship types, org types, edge statuses and
providers are controlled vocabularies defined there.

## Search provider architecture

Pluggable providers behind one interface ([providers/](app/providers/)):
`search(query) -> [SearchResult]`, `fetch(url) -> Page`.

**Routing** (configurable via `ARTEMIS_ROUTE_PERSON` / `ARTEMIS_ROUTE_DEFAULT`):
for a notable person → **Wikipedia/Wikidata → Brave → DuckDuckGo**; otherwise
**Brave → DuckDuckGo**. First provider with useful results wins.

- **Brave** (PRIMARY web search) — REST API, key from `BRAVE_API_KEY`; respects
  a per-second limit + best-effort monthly quota; on quota/credit exhaustion it
  steps aside and the next provider takes over. *No key → automatically skipped.*
- **Wikipedia + Wikidata** (SECONDARY, structured) — page summary, important
  links, and **structured Wikidata relationships** (spouse, employer, educated-at,
  board, founder…) consulted *before* any HTML scraping. These are high-trust,
  source-grounded facts.
- **DuckDuckGo** (FALLBACK only) — used solely when Brave fails/exhausts/empties.
  Protected by a token bucket + jitter and a **circuit breaker** that trips on
  repeated 429s (stops DDG during a cooldown, auto-retries after).

**Persistent cache** ([providers/cache.py](app/providers/cache.py)) — every
search/page/wiki/wikidata response is cached in SQLite (`artemis_cache.db`,
30-day TTL) and **survives across CLI runs**. Identical requests are never
repeated within the TTL. *(A warm re-run of a depth-1 build: 0 network calls,
<1s.)*

**Query deduplication** — equivalent queries across silos collapse to one
request (canonical: lowercased, punctuation-free, token-sorted), with a mapping
preserved back to every originating silo.

**Reliability** — exponential backoff on 429/5xx only, max 3 retries, honoring
`Retry-After`; provider-specific rate limiting.

**Stats** — every run prints Brave/Wikipedia/Wikidata/DuckDuckGo counts, cache
hits/misses, deduped queries, circuit-breaker trips, average latency, and total
search time.

```bash
export BRAVE_API_KEY=...     # optional; without it, Wikipedia/Wikidata + DDG are used
python -m app.cli "Bill Gates" --depth 2
```

## Production hardening (precision engine)

The pipeline is tuned for **low hallucination + source-grounded edges**:

- **Silo contracts** — each silo exposes `spec(person)` →
  `{queries, priority_relationship_types, confidence_multiplier}`
  ([silos/definitions.py](app/silos/definitions.py)).
- **Confidence model** ([extraction/confidence.py](app/extraction/confidence.py)):
  `adjusted = base × silo_multiplier × keyword_strength_factor`, clamped to
  [0,1], with evidence ceilings — **an explicit keyword is required to exceed
  0.6**, and sentence co-occurrence alone caps under 0.4. Tiers: weak (<0.3),
  candidate (0.3–0.6), strong (>0.6).
- **Structured extraction** ([extraction/schemas.py](app/extraction/schemas.py)):
  every edge carries `evidence_snippet`, `source_url`, `confidence_base`,
  `confidence_adjusted`, and `signals{explicit_keyword_match,
  sentence_cooccurrence, strength_keywords_found}`; rejected candidates are
  recorded, not silently dropped.
- **Normalization & dedup** — people dedup on a middle-initial-stripped key with
  auto-stored aliases; orgs dedup on a suffix-stripped key (Inc/LLC/Corp/
  Foundation/University…); edges dedup on (person_a, counterpart, type,
  source_url).
- **Expansion safety** ([graph/expansion.py](app/graph/expansion.py)) — ranks by
  strong/explicit edges + avg confidence + source diversity; expands only the
  top strong/explicit nodes per hop; caps at 200 nodes/run and 50 edges/node
  (samples top 20 over that).
- **Reliability** — all search/fetch calls cached, retried with exponential
  backoff on 429/5xx, and DuckDuckGo is rate-limited process-wide.
- **Summary** — `GET /summary` and the CLI print top people/orgs, strongest
  edges, confidence distribution, and per-depth expansion counts.

## Local network matching (candidate intros — UNVERIFIED)

Match your own network (a LinkedIn-style CSV) against the discovered public
graph to find candidate intro paths: **You → contact → public person → … →
target**. This stage finds *intersections only* — it never asserts an intro is
real and never calls Claude. Every path is `status: "unverified"`.

```bash
# 1) build a target's public graph first
python -m app.cli "Tim Cook"
# 2) upload your network (persists across future target rebuilds)
python -m app.cli upload-network ~/Downloads/Connections.csv
# 3) match + generate candidate paths
python -m app.cli match "Tim Cook"
# 4) print the full candidate-path JSON
python -m app.cli paths "Tim Cook"
```

CSV handling tolerates inconsistent LinkedIn headers (`First Name`/`Last Name`
or `Name`, `Company`, `Position`, `Email`, `School`, `Location`, `URL`,
`Notes`, …). Without explicit relationship columns, every contact is treated as
directly connected to **You**.

**Match tiers** ([network/matching.py](app/network/matching.py)):

| Tier | Rule | Confidence |
|------|------|-----------|
| 1 `exact_name`   | normalized full name == public canonical name | 0.95 |
| 2 `name_company` | high fuzzy name + company/org overlap | 0.80–0.90 |
| 3 `name_school`  | high fuzzy name + school/location overlap (weak) | 0.60–0.75 |
| – `fuzzy_name`   | high fuzzy name only, no corroboration (weak) | 0.50 |
| 4 `org_overlap`  | local org appears in graph — near-miss, **no person path** | 0.40–0.60 |

Rejected by construction: same-city-only, same-industry-only, title-only,
generic school-only without name similarity.

**Paths** ([network/paths.py](app/network/paths.py)) — best-path search over
public person-person edges (≤4 hops), preferring high-confidence/strong/
non-rejected edges; `score = local_match_confidence × avg_edge_confidence ×
relationship_strength`. Org-overlap matches do not generate person paths.

**Endpoints:** `POST /network/upload`, `GET /network/profiles`,
`POST /match/{target_person_id}`, `GET /matches`, `GET /candidate-paths`,
`GET /candidate-paths/{id}`.

Tables: `local_profiles`, `local_edges`, `graph_matches`, `candidate_paths`
([models.py](app/models.py)). A new target search clears only the public graph
and matches — your uploaded network is preserved.

## Deferred (next stage — not built)

- Claude-based verification of candidate paths (promoting `unverified` →
  `verified`/`accepted`). All paths remain `unverified` until then.
