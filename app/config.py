"""Central configuration for Artemis V2.

All knobs live here so silos / expansion / providers stay declarative.

Secrets/keys: put them in a `.env` file in the project root (e.g.
`ANTHROPIC_API_KEY=sk-ant-...`, `BRAVE_API_KEY=bsa-xxxx`). It is loaded
automatically and never committed.
"""
import os


def _load_dotenv() -> None:
    """Minimal .env loader (no dependency). Existing env vars take precedence."""
    for path in (".env", os.path.join(os.path.dirname(__file__), "..", ".env")):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key, val = key.strip(), val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
        except OSError:
            continue


_load_dotenv()


def _env_bool(name: str, default: str) -> bool:
    """Parse a boolean env var, recognizing common truthy/falsy spellings
    (0/1, true/false, yes/no, on/off) case-insensitively. A value that isn't
    a recognized falsy spelling is truthy, so a typo like "flase" still
    behaves as an explicit override rather than silently doing nothing --
    but "no"/"off" (easy things to type meaning "disable this") are honored
    instead of being silently treated as enabled."""
    val = os.environ.get(name, default).strip().lower()
    return val not in ("0", "false", "no", "off", "n", "")


# --- storage ---------------------------------------------------------------
DB_URL = os.environ.get("ARTEMIS_DB_URL", "sqlite:///./artemis.db")
# Per-session graph isolation (HTTP API): each browser session gets its own
# SQLite file here, so concurrent users never clobber each other's public graph.
GRAPH_DIR = os.environ.get("ARTEMIS_GRAPH_DIR", "./graphs")
# Boards/pages live in their own SQLite file — they never reference the
# discovery-graph tables, and isolating them means a board's frequent
# autosave writes never hit "database is locked" behind a long /connect or
# /targets/search transaction on the (much busier) main DB_URL database.
BOARDS_DB_URL = os.environ.get("ARTEMIS_BOARDS_DB_URL", "sqlite:///./artemis_boards.db")

# --- HTTP ------------------------------------------------------------------
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HTTP_TIMEOUT = float(os.environ.get("ARTEMIS_HTTP_TIMEOUT", "8.0"))
# retry: only on 429/5xx, exponential backoff, honor Retry-After
HTTP_RETRIES = int(os.environ.get("ARTEMIS_HTTP_RETRIES", "3"))
HTTP_BACKOFF_BASE = float(os.environ.get("ARTEMIS_HTTP_BACKOFF", "0.4"))  # seconds
# Random extra delay ADDED on top of every backoff/Retry-After sleep (both in
# request_with_retry). Without it, N threads that all get rate-limited in the
# same instant compute the IDENTICAL deterministic delay and retry in the same
# instant again — a self-synchronizing collision that node/side-level
# concurrency (expansion.py, connect.py) made real: measured live, it turned a
# 429 into a repeating storm rather than a one-time backoff.
HTTP_BACKOFF_JITTER = float(os.environ.get("ARTEMIS_HTTP_BACKOFF_JITTER", "0.75"))
HTTP_RETRY_STATUS = (429, 500, 502, 503, 504)
# Extra seconds ADDED on top of HTTP_TIMEOUT for the hard, unavoidable timeout
# wrapped around every request (see providers.base._call_with_hard_timeout).
# httpx's own `timeout=HTTP_TIMEOUT` is supposed to bound this already, but
# measured live in this environment it did not: a DuckDuckGo call sat in raw
# socket.connect() for 40+ seconds (an unreachable host with packets silently
# dropped rather than rejected, which some networks/sandboxes do) — well past
# its configured 8s timeout, hanging the entire node indefinitely with no
# recourse. The hard timeout below runs the call on a background thread and
# gives up waiting at HTTP_TIMEOUT + this buffer, REGARDLESS of what httpx or
# the OS does; the buffer just avoids racing httpx's own (usually correct)
# timeout and killing a request that would have succeeded a moment later.
HTTP_HARD_TIMEOUT_BUFFER = float(os.environ.get("ARTEMIS_HTTP_HARD_TIMEOUT_BUFFER", "3.0"))
# Hard ceiling on outbound HTTP requests in flight AT ONCE, process-wide,
# regardless of caller — a request_with_retry() call holds one slot for its
# entire attempt (including any backoff sleeps between retries), so a slot
# stuck backing off is a slot NOT handed to a fresh request piling onto the
# same struggling host. This is the actual fix for concurrent expansion
# overwhelming a provider: EXPAND_NODE_CONCURRENCY x SEARCH_WORKERS x (2
# connect_people sides) can want up to ~64 requests at once, far past what
# Serper (5 QPS) or EDGAR/OpenCorporates/ProPublica (well under that) can
# absorb — those callers now queue on this semaphore instead of firing a burst
# that 429s. Deliberately NOT per-provider: two providers each individually
# "under quota" can still jointly overwhelm the local machine's outbound
# connection budget, and provider-specific QPS is already enforced by each
# provider's own limiter (ratelimit.py) on top of this.
MAX_CONCURRENT_REQUESTS = int(os.environ.get("ARTEMIS_MAX_CONCURRENT_REQUESTS", "10"))

# --- providers -------------------------------------------------------------
# Brave Search REST API (PRIMARY web search). Key from env; absent => skipped.
# Serper.dev (Google SERP) — PRIMARY web search (cheap: ~$0.10-1/1k, 2,500/mo
# free). Key from SERPER_API_KEY; absent => skipped (falls back to Brave).
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "").strip()
SERPER_ENDPOINT = "https://google.serper.dev/search"
SERPER_QPS = float(os.environ.get("ARTEMIS_SERPER_QPS", "5.0"))    # Serper allows higher QPS
SERPER_MONTHLY_QUOTA = int(os.environ.get("ARTEMIS_SERPER_QUOTA", "2500"))  # free-tier cap

# Brave Search REST API (BACKUP web search). Key from env; absent => skipped.
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "").strip()
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_QPS = float(os.environ.get("ARTEMIS_BRAVE_QPS", "1.0"))      # free tier ~1 q/s
BRAVE_MONTHLY_QUOTA = int(os.environ.get("ARTEMIS_BRAVE_QUOTA", "1000"))

# Provider routing (configurable). Tried in order; stop at first useful result.
# Serper is primary, Brave is the backup, DuckDuckGo the free last resort.
def _route_list(name: str, default: str) -> list:
    """Comma-split env var into provider keys, trimming whitespace and
    dropping empty entries -- so "serper, brave" (a stray space after the
    comma) doesn't silently produce a " brave" entry that fails to match
    the provider registry and drops out of routing with no error."""
    return [p.strip() for p in os.environ.get(name, default).split(",") if p.strip()]



# NOTE: search() (orchestrator.py) stops at the first provider that returns
# ANY results, so order here is a real priority, not just a fallback list.
# Wikipedia MUST come after serper/brave: MediaWiki's srsearch is a loose
# full-text search that returns *something* for almost any query containing
# a real person's name (even a silo-specific query like "X board member"),
# so putting it first was silently starving Serper/Brave -- the actual
# general web search this route is supposed to be primary -- of every
# person query. Wikipedia still participates as a free structured fallback
# before the aggressively rate-limited DuckDuckGo, just not ahead of primary.
ROUTE_PERSON = _route_list(
    "ARTEMIS_ROUTE_PERSON", "serper,brave,wikipedia,wikidata,duckduckgo")
ROUTE_DEFAULT = _route_list("ARTEMIS_ROUTE_DEFAULT", "serper,brave,duckduckgo")

# Per-provider rate limits (seconds between calls / token bucket)
WIKI_MIN_INTERVAL = float(os.environ.get("ARTEMIS_WIKI_MIN_INTERVAL", "0.1"))
DDG_MIN_INTERVAL = float(os.environ.get("ARTEMIS_DDG_MIN_INTERVAL", "0.6"))
DDG_JITTER = float(os.environ.get("ARTEMIS_DDG_JITTER", "0.4"))  # +0..jitter random
DDG_BUCKET_CAPACITY = int(os.environ.get("ARTEMIS_DDG_BUCKET", "4"))

# DuckDuckGo circuit breaker: trip after N consecutive 429s, cool down, retry.
DDG_BREAKER_THRESHOLD = int(os.environ.get("ARTEMIS_DDG_BREAKER_THRESHOLD", "3"))
DDG_BREAKER_COOLDOWN = float(os.environ.get("ARTEMIS_DDG_BREAKER_COOLDOWN", "300"))  # s

# --- snapshots -------------------------------------------------------------
# Every build writes a JSON snapshot of the target's graph here for safekeeping.
CACHED_GRAPHS_DIR = os.environ.get("ARTEMIS_CACHED_GRAPHS_DIR", "./cached_graphs")

# --- persistent cache ------------------------------------------------------
CACHE_DB = os.environ.get("ARTEMIS_CACHE_DB", "./artemis_cache.db")
CACHE_TTL_SEARCH = int(os.environ.get("ARTEMIS_CACHE_TTL_SEARCH", str(30 * 86400)))
CACHE_TTL_PAGE = int(os.environ.get("ARTEMIS_CACHE_TTL_PAGE", str(30 * 86400)))
CACHE_TTL_WIKI = int(os.environ.get("ARTEMIS_CACHE_TTL_WIKI", str(30 * 86400)))

# --- search behaviour ------------------------------------------------------
RESULTS_PER_QUERY = int(os.environ.get("ARTEMIS_RESULTS_PER_QUERY", "5"))
SCRAPE_TOP_N = int(os.environ.get("ARTEMIS_SCRAPE_TOP_N", "3"))
MAX_PAGE_CHARS = int(os.environ.get("ARTEMIS_MAX_PAGE_CHARS", "20000"))
# How many sentences away from an actual mention of the SUBJECT's own name an
# extracted entity's sentence may be and still count as evidence about the
# subject. Confirmed live as a real gap, not a hypothetical: on a large
# multi-person page (e.g. a conference "Featured Speakers" roster), spaCy/
# heuristic extraction found every person entity on the WHOLE page and
# attributed each one's own nearby sentence as "evidence" with no check that
# the subject was mentioned anywhere near it -- producing edges like "Eric
# Domski, board_member, Larry Ellison" from a sentence that was actually
# about a different speaker (Thomas Kurian) entirely, found via Eric
# Domski's OWN unrelated mention elsewhere on the same long page. A subject
# whose name never appears in the text at all (a synthetic enrichment
# string, or a pronoun-heavy paragraph) has no proximity signal to check
# against and falls back to the old, unrestricted behavior -- this narrows
# what already-working cases accept, it doesn't add a new failure mode.
ENTITY_PROXIMITY_WINDOW = int(os.environ.get("ARTEMIS_ENTITY_PROXIMITY_WINDOW", "2"))
# cap queries run per silo (keeps a single target snappy)
MAX_QUERIES_PER_SILO = int(os.environ.get("ARTEMIS_MAX_QUERIES_PER_SILO", "4"))
# concurrency for the network phase (searches + page fetches)
SEARCH_WORKERS = int(os.environ.get("ARTEMIS_SEARCH_WORKERS", "8"))

# --- expansion -------------------------------------------------------------
DEFAULT_MAX_DEPTH = 2
EXPAND_TOP_N_PER_DEPTH = int(os.environ.get("ARTEMIS_EXPAND_TOP_N", "20"))
# expansion safety / anti-explosion
EXPAND_TOP_STRONG = int(os.environ.get("ARTEMIS_EXPAND_TOP_STRONG", "15"))
# How many frontier nodes a single hop processes CONCURRENTLY (each on its own
# DB session — see expand_graph). Deliberately separate from SEARCH_WORKERS,
# which bounds concurrency INSIDE one node's own searches/page-fetches — the
# two multiply (worst case EXPAND_NODE_CONCURRENCY x SEARCH_WORKERS open
# connections at once, x2 again since connect_people also runs both endpoints
# concurrently), easily wanting 60+ requests in flight. That used to hit
# providers as a real burst; now it queues on MAX_CONCURRENT_REQUESTS
# (base.py) instead, so this can run at its intended value again — the actual
# fix for the self-inflicted request storm is the global cap + jittered
# backoff, not a low number here.
EXPAND_NODE_CONCURRENCY = int(os.environ.get("ARTEMIS_EXPAND_NODE_CONCURRENCY", "4"))
# Reachability mode: when expanding, prefer the LEAST-famous real connections
# (no Wikipedia page, fewer sources) instead of the strongest/most-famous ones.
# This walks the graph DOWN the fame gradient toward a normal person's network,
# which is how a warm-intro path to your own contacts is actually found.
EXPAND_PREFER_REACHABLE = _env_bool("ARTEMIS_EXPAND_PREFER_REACHABLE", "1")
# Down-weight family_social edges so expansion explores PROFESSIONAL connections
# (colleagues, boards, co-founders, investors, political) instead of walking the
# subject's family tree. A path between two people runs through work, not relatives.
DOWNWEIGHT_FAMILY = _env_bool("ARTEMIS_DOWNWEIGHT_FAMILY", "1")
FAMILY_PENALTY = float(os.environ.get("ARTEMIS_FAMILY_PENALTY", "1.5"))
PROFESSIONAL_BONUS = float(os.environ.get("ARTEMIS_PROFESSIONAL_BONUS", "1.0"))
# Shared global map accumulates people across ALL runs, so the cap is high.
# (A single isolated build rarely approaches this; it bounds unbounded growth.)
MAX_TOTAL_NODES = int(os.environ.get("ARTEMIS_MAX_TOTAL_NODES", "50000"))
# connect() builds TWO graphs into one DB. Each side gets its OWN people budget
# so the first (richer) person can't starve the second — the second person's cap
# is raised to (2 x per-side) after the first is built. Total ~= 2 x per-side.
CONNECT_NODE_CAP_PER_SIDE = int(os.environ.get("ARTEMIS_CONNECT_NODE_CAP_PER_SIDE", "1000"))
# how many DIVERSE routes connect() returns (each avoids prior routes' bridges)
CONNECT_MAX_PATHS = int(os.environ.get("ARTEMIS_CONNECT_MAX_PATHS", "3"))
# per-node edge caps (raised: Tier-1/2 structured sources produce 100s of clean
# contacts per person; the old caps were sampling almost all of them away)
MAX_EDGES_PER_NODE = int(os.environ.get("ARTEMIS_MAX_EDGES_PER_NODE", "200"))
EDGE_SAMPLE_LIMIT = int(os.environ.get("ARTEMIS_EDGE_SAMPLE_LIMIT", "150"))

# --- pathfinding cost shaping -----------------------------------------------
# Flat cost added per hop in a path, on top of that hop's edge cost. Without
# this, a 2-hop relay of two strong ties can score identically to (or beat) one
# slightly-weaker direct connection, even though a direct ask is always better
# than asking two people to relay it. Mirrors DrewGloverDemo's HOP_SURCHARGE.
HOP_SURCHARGE = float(os.environ.get("ARTEMIS_HOP_SURCHARGE", "1.5"))

# Penalty added when a path TRANSITS (not starts/ends at) a person who has a
# Wikidata QID — i.e. is independently notable. A real, evidenced edge through
# a celebrity is still a true connection (Rule 0 isn't about this), but it's a
# poor bridge: they're unlikely to relay an intro for a stranger. Endpoints are
# exempt — the fame gate only matters for who a path routes THROUGH.
FAME_PENALTY = float(os.environ.get("ARTEMIS_FAME_PENALTY", "4.0"))

# Node degree (edge count) above which a person is treated as a mega-hub for
# routing purposes — discourages funnelling every path through the same
# well-connected node when a lower-degree alternative exists. Below this,
# a normal well-connected person pays nothing.
MEGA_HUB_DEGREE = int(os.environ.get("ARTEMIS_MEGA_HUB_DEGREE", "40"))
# Per-hop-through-a-mega-hub penalty, scaled by log(degree) so the penalty
# grows gently past the threshold rather than jumping at a hard cliff.
DEGREE_PENALTY_COEF = float(os.environ.get("ARTEMIS_DEGREE_PENALTY_COEF", "0.5"))

# --- Claude (Anthropic API) -------------------------------------------------
# The LLM stages (per-source extraction, entity filter, relationship
# classifier) all run on Claude. Absent a key every one of them is a
# transparent no-op and the pipeline falls back to spaCy/heuristic extraction
# and deterministic filtering — the app runs, the graph is just noisier.
#
# The key is read from the platform's own secret store (`fly secrets set`,
# Render/Railway env vars, `docker run -e`) or a local .env; it is never
# returned to a client. Leave ANTHROPIC_API_KEY unset to let the SDK resolve
# credentials itself (ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile).
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
CLAUDE_MODEL = os.environ.get("ARTEMIS_CLAUDE_MODEL", "claude-opus-5")
# Model for the two HIGH-VOLUME batched stages (entity filter, relationship
# classifier). Both are narrow, schema-constrained judgment calls on short
# strings — "is this a real person's name", "what relationship does this
# sentence state" — where the cheapest current model is sufficient and runs
# ~5x cheaper than CLAUDE_MODEL. Per-source extraction, which reads whole
# pages, stays on CLAUDE_MODEL. Raise this to CLAUDE_MODEL if the filter starts
# dropping real people or the classifier mislabels edges.
CLAUDE_BATCH_MODEL = os.environ.get("ARTEMIS_CLAUDE_BATCH_MODEL", "claude-haiku-4-5")
# Per-call ceiling and retry budget. The SDK retries 429/5xx itself with
# exponential backoff; past that a call returns None and the caller falls back.
CLAUDE_TIMEOUT = float(os.environ.get("ARTEMIS_CLAUDE_TIMEOUT", "60.0"))
CLAUDE_MAX_RETRIES = int(os.environ.get("ARTEMIS_CLAUDE_MAX_RETRIES", "3"))
# Claude calls in flight at once, process-wide (see extraction/claude_client).
# Separate from MAX_CONCURRENT_REQUESTS: that one protects search providers
# with 1–5 QPS limits, this one protects an account rate limit — and the
# budget.
CLAUDE_MAX_CONCURRENCY = int(os.environ.get("ARTEMIS_CLAUDE_MAX_CONCURRENCY", "8"))
# Reasoning depth. These are narrow, high-volume classification calls with a
# schema-constrained answer, not open-ended reasoning, so the default is low —
# it is the main latency and cost lever if a build feels slow.
#
# NOT universally supported: `effort` is rejected outright by some models
# (Haiku 4.5 among them, which is CLAUDE_BATCH_MODEL's default). The client
# omits it for those rather than sending a request that would 400 — see
# extraction/claude_client._supports_effort. Set to "" to never send it.
CLAUDE_EFFORT = os.environ.get("ARTEMIS_CLAUDE_EFFORT", "low")
CLAUDE_SYSTEM_PROMPT = (
    "You extract and validate facts from messy public web text for a "
    "relationship-graph builder. Every answer must be grounded in the text "
    "you are given. Never guess, never infer beyond what is stated, and omit "
    "anything you are unsure about — a missing entry is always better than a "
    "fabricated one."
)

# --- confidence model ------------------------------------------------------
# Confidence ceilings — heuristic extraction is never trusted highly.
HEURISTIC_BASE_CONFIDENCE = 0.25
CLAUDE_BASE_CONFIDENCE = 0.55
# Use Claude for per-source EXTRACTION: one LLM call per scraped page. This is
# the cleanest extraction path and the single biggest accuracy lever, but it is
# also by far the most expensive stage — a depth-2 build scrapes dozens of
# pages per node across ~16 nodes. Default OFF so the cheap, batched filter and
# classifier below can run on their own; turn it on deliberately.
CLAUDE_EXTRACT = _env_bool("ARTEMIS_CLAUDE_EXTRACT", "0")
CLAUDE_EXTRACT_MODEL = os.environ.get("ARTEMIS_CLAUDE_EXTRACT_MODEL", CLAUDE_MODEL)
# spaCy NER extraction (Tier 4): grammar-aware, kills prose-fragment noise.
# Preferred over the heuristic when installed; far cheaper than Claude extraction.
SPACY_EXTRACT = _env_bool("ARTEMIS_SPACY_EXTRACT", "1")
SPACY_BASE_CONFIDENCE = float(os.environ.get("ARTEMIS_SPACY_BASE_CONFIDENCE", "0.35"))
# Claude entity filter: drop extracted nodes that aren't real named people/orgs.
# Auto-enabled when a key is present; no-op otherwise. Batched and cached 30
# days, so each distinct name is paid for once.
CLAUDE_FILTER = _env_bool("ARTEMIS_CLAUDE_FILTER", "1")
CLAUDE_FILTER_MODEL = os.environ.get("ARTEMIS_CLAUDE_FILTER_MODEL", CLAUDE_BATCH_MODEL)
CLAUDE_FILTER_BATCH = int(os.environ.get("ARTEMIS_CLAUDE_FILTER_BATCH", "50"))
# Claude relationship classifier: re-type 'unknown' edges from their evidence.
CLAUDE_CLASSIFY_RELATIONS = _env_bool("ARTEMIS_CLAUDE_CLASSIFY", "1")
CLAUDE_CLASSIFY_MODEL = os.environ.get("ARTEMIS_CLAUDE_CLASSIFY_MODEL", CLAUDE_BATCH_MODEL)
CLAUDE_CLASSIFY_BATCH = int(os.environ.get("ARTEMIS_CLAUDE_CLASSIFY_BATCH", "25"))
CLAUDE_CLASSIFY_MIN_CONF = float(os.environ.get("ARTEMIS_CLAUDE_CLASSIFY_MIN_CONF", "0.5"))
RELATION_CONF_CEILING = float(os.environ.get("ARTEMIS_RELATION_CONF_CEILING", "0.85"))

# --- OpenAlex (academic coauthors) -----------------------------------------
# namesake guards: require a real publication record + a close name match
OPENALEX_MIN_WORKS = int(os.environ.get("ARTEMIS_OPENALEX_MIN_WORKS", "3"))
OPENALEX_NAME_SIM = float(os.environ.get("ARTEMIS_OPENALEX_NAME_SIM", "0.8"))

# --- homonym disambiguation -------------------------------------------------
# builder.get_or_create_person() can ADOPT a name-matched Wikidata QID onto an
# existing node (its "case 2": a same-named node with no QID yet). Before that
# happens, compare the node's own already-accumulated edge evidence against
# the candidate identity's text (see graph.disambiguate.domain_conflict) and
# refuse to fuse them if the two plainly anchor in different professional
# worlds. Default on: the check is deterministic and fails open whenever
# either side lacks signal (a brand-new node, or a candidate with no
# description text), so it costs nothing on the common notable-person path
# and only ever prevents a false "same person" merge — never blocks a real
# one. Disable only to reproduce pre-guard behavior for comparison/debugging.
IDENTITY_VERIFY_ENABLED = _env_bool("ARTEMIS_IDENTITY_VERIFY_ENABLED", "1")
# How many of a node's own existing edge-evidence snippets are folded into the
# signal compared against a proposed identity. Small and cheap on purpose:
# this is a coarse keyword-domain check, not a summarizer — a handful of
# snippets already carries the professional-domain keywords if any are
# present, and capping it keeps one prolific node's lookup bounded.
IDENTITY_SIGNAL_MAX_SNIPPETS = int(
    os.environ.get("ARTEMIS_IDENTITY_SIGNAL_MAX_SNIPPETS", "5"))

# --- targeted professional-network re-search (bounded beam) ----------------
# On the non-famous side of an asymmetric /connect walk (see
# graph.connect._resolve_expansion_depths), generic silo templates find a
# subject's company but then keep guessing broadly instead of following up on
# names that already surfaced repeatedly -- e.g. a real cofounder mentioned
# only in passing LinkedIn posts stays capped at "weak coworker" confidence
# forever, because nothing ever asks a sharper, targeted question about her
# specifically. expansion._process_person's targeted-recheck phase re-queries
# each such repeat name directly (subject + candidate, both names), instead
# of relying on the generic net to state the relationship by accident.
#
# Bounded on purpose: this is a beam search, not full recursion. At
# ENHANCED_SEARCH_MAX_CANDIDATES per hop, cost stays predictable (5 -> 25 ->
# 125 across 3 hops) instead of scaling with however many candidates a node
# happens to discover.
ENHANCED_SEARCH_MAX_CANDIDATES = int(
    os.environ.get("ARTEMIS_ENHANCED_SEARCH_MAX_CANDIDATES", "5"))
# A name has to repeat at least this many times across independently-found
# sources before it's worth a dedicated re-query -- a single passing mention
# isn't the "keeps coming up" signal this is meant to act on.
ENHANCED_SEARCH_MIN_MENTIONS = int(
    os.environ.get("ARTEMIS_ENHANCED_SEARCH_MIN_MENTIONS", "2"))

# --- Alpha step 4/5: node profiling (org size/industry) --------------------
# See extraction/node_profiler.py for the hallucination guard this feeds --
# the model must ground every field in the fetched snippets or the whole
# verdict collapses to "unknown" rather than a plausible-sounding guess.
NODE_PROFILE_ENABLED = _env_bool("ARTEMIS_NODE_PROFILE", "1")
NODE_PROFILE_MODEL = os.environ.get("ARTEMIS_NODE_PROFILE_MODEL", CLAUDE_BATCH_MODEL)
# Structured-source-first queries (LinkedIn's employee-count badge, Crunchbase's
# headcount field) are strongly preferred over generic "about us" marketing
# copy, which almost never states real numbers and just invites the model to
# infer instead of read.
NODE_PROFILE_QUERIES = [
    '"{org}" employees size linkedin',
    '"{org}" crunchbase OR overview company',
]
NODE_PROFILE_MAX_SNIPPETS = int(os.environ.get("ARTEMIS_NODE_PROFILE_MAX_SNIPPETS", "3"))
NODE_PROFILE_SNIPPET_CHARS = int(os.environ.get("ARTEMIS_NODE_PROFILE_SNIPPET_CHARS", "600"))

# --- Alpha step 6: search strategy (which angle of the professional network
# to search) -----------------------------------------------------------------
# See extraction/search_strategy.py: the model picks a FIXED angle, never
# writes query text itself -- STRATEGY_ANGLE_QUERIES below is the actual,
# fully deterministic and inspectable query surface. "generic" intentionally
# maps to no extra queries: the existing broad silo search already runs
# regardless, this only ever ADDS a couple of targeted queries on top.
STRATEGY_ENABLED = _env_bool("ARTEMIS_STRATEGY", "1")
STRATEGY_MODEL = os.environ.get("ARTEMIS_STRATEGY_MODEL", CLAUDE_BATCH_MODEL)
STRATEGY_ANGLE_QUERIES = {
    "current_employer_leadership": [
        '"{org}" leadership team OR executives',
    ],
    "past_employers": [
        '"{subject}" "previously at" OR "formerly at" OR "prior to {org}"',
    ],
    "industry_peers": [
        '"{industry}" industry leaders network',
    ],
    "board_or_advisory": [
        '"{subject}" board member OR advisor OR advisory',
    ],
    "generic": [],
}

# --- OpenCorporates (company officer networks) -----------------------------
# Free-tier token from https://opencorporates.com/api_accounts/new ; absent => skipped.
OPENCORPORATES_API_TOKEN = os.environ.get("OPENCORPORATES_API_TOKEN", "").strip()
OPENCORP_MIN_INTERVAL = float(os.environ.get("ARTEMIS_OPENCORP_MIN_INTERVAL", "0.5"))

# --- SEC EDGAR (public-company insider networks) ---------------------------
# Free, no key — but SEC requires a declared User-Agent with contact info.
EDGAR_ENABLED = _env_bool("ARTEMIS_EDGAR_ENABLED", "1")
EDGAR_USER_AGENT = os.environ.get(
    "ARTEMIS_EDGAR_USER_AGENT", "Artemis Graph Builder research@artemis.local")
EDGAR_MIN_INTERVAL = float(os.environ.get("ARTEMIS_EDGAR_MIN_INTERVAL", "0.2"))

# --- ProPublica Nonprofit Explorer (990 boards) ----------------------------
PROPUBLICA_ENABLED = _env_bool("ARTEMIS_PROPUBLICA_ENABLED", "1")
PROPUBLICA_MIN_INTERVAL = float(os.environ.get("ARTEMIS_PROPUBLICA_MIN_INTERVAL", "0.3"))
PROPUBLICA_MAX_ORGS = int(os.environ.get("ARTEMIS_PROPUBLICA_MAX_ORGS", "3"))

# --- Firms (VC/company team-roster scraping) --------------------------------
# A roster page listing a person is a structural assertion ("this page lists
# its own team"), a much stronger signal than a search-engine snippet.
FIRMS_ENABLED = _env_bool("ARTEMIS_FIRMS_ENABLED", "1")
MAX_ROSTER_MEMBERS = int(os.environ.get("ARTEMIS_MAX_ROSTER_MEMBERS", "40"))
MAX_FIRMS_PER_PERSON = int(os.environ.get("ARTEMIS_MAX_FIRMS_PER_PERSON", "3"))
# raw HTML cached/parsed per page fetch (before html_to_text's further cut to
# MAX_PAGE_CHARS) — a roster grid can sit far down the markup of a long page.
MAX_HTML_CHARS = int(os.environ.get("ARTEMIS_MAX_HTML_CHARS", str(MAX_PAGE_CHARS * 4)))

# --- Podcasts (RSS host<->guest structural edges) ---------------------------
# An episode asserts one thing: this host interviewed this guest. Tier-1 data —
# never a guest<->guest edge (see providers/podcasts.py).
PODCASTS_ENABLED = _env_bool("ARTEMIS_PODCASTS_ENABLED", "1")
PODCAST_MAX_EPISODES = int(os.environ.get("ARTEMIS_PODCAST_MAX_EPISODES", "300"))
PODCAST_EPISODE_SEARCH_LIMIT = int(
    os.environ.get("ARTEMIS_PODCAST_EPISODE_SEARCH_LIMIT", "50"))
MAX_PODCAST_APPEARANCES = int(os.environ.get("ARTEMIS_MAX_PODCAST_APPEARANCES", "6"))
MAX_HOST_FEED_GUESTS = int(os.environ.get("ARTEMIS_MAX_HOST_FEED_GUESTS", "60"))

# --- Funding rounds (org<->org co-investment facts, NOT person edges) -------
# A round announcement names the investors in it — a structural assertion
# about the FIRMS, never chained into a person-to-person edge (no "employee
# of Org A -> Org A co-invested with Org B -> employee of Org B" inference).
# See graph.builder.record_coinvestment, which stores this only on
# Organization.meta["co_investments"] and is never read by pathfinding.
FUNDING_ENABLED = _env_bool("ARTEMIS_FUNDING_ENABLED", "1")
MAX_ROUNDS_PER_FIRM = int(os.environ.get("ARTEMIS_MAX_ROUNDS_PER_FIRM", "5"))
MAX_COINVESTOR_FIRMS = int(os.environ.get("ARTEMIS_MAX_COINVESTOR_FIRMS", "6"))

# --- headless-browser rendering (Playwright, OPTIONAL) -----------------------
# JS-only team pages (React/Vue SPAs) return an empty shell to a plain GET.
# Playwright is a heavy optional dependency: when it (or its Chromium binary)
# isn't installed, browser.available() returns False and every caller falls
# back to the plain fetch — nothing else in the system is affected.
BROWSER_TIMEOUT_S = float(os.environ.get("ARTEMIS_BROWSER_TIMEOUT_S", "12.0"))
BROWSER_SETTLE_S = float(os.environ.get("ARTEMIS_BROWSER_SETTLE_S", "3.0"))

# --- network/upload (My Connections CSV) -------------------------------------
MAX_UPLOAD_BYTES = int(os.environ.get("ARTEMIS_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
# /network/contacts/import takes JSON rows instead of a file, so it needs its own
# ceiling — a phone address book is a few thousand cards at the very top end.
MAX_IMPORT_CONTACTS = int(os.environ.get("ARTEMIS_MAX_IMPORT_CONTACTS", "10000"))

# --- wave 0: structural edges derived from the contact export alone ----------
# Contacts sharing an employer become coworkers, but only when the group is
# small enough to plausibly BE a team. Deliberately tighter than the YC roster
# cap (MAX_CLIQUE_MEMBERS=60 in scripts/build_yc_cache.py): a curated team page
# asserts "these people work together", whereas a CSV's company column is a
# self-reported free-text field, so 200 contacts listing "Google" are a
# directory artifact, not colleagues. Above the cap we keep org membership and
# skip the clique — see network/cliques.py.
CONTACT_CLIQUE_MAX = int(os.environ.get("ARTEMIS_CONTACT_CLIQUE_MAX", "25"))

# --- enrichment runs (waves 1+) ---------------------------------------------
# Build slots an enrichment run may never occupy, so an hours-long run cannot
# starve interactive /connect and /targets/search. With the default capacity of
# 2 and 1 reserved, a background task runs only while nothing else is, and
# leaves a free slot the moment it does — see buildqueue.BuildQueue.
ENRICH_RESERVED_BUILD_SLOTS = int(
    os.environ.get("ARTEMIS_ENRICH_RESERVED_BUILD_SLOTS", "1"))
# Wave 1: how many top-ranked contacts a start with no explicit limit covers.
# Sized to finish in a few minutes at ~2 min/contact so onboarding shows real
# progress; the long tail is a later, budgeted run over the same plan.
ENRICH_WAVE1_SIZE = int(os.environ.get("ARTEMIS_ENRICH_WAVE1_SIZE", "30"))
# How long a background task waits for a build slot before re-checking whether
# its run was cancelled. Short: this is a poll interval, not a timeout.
ENRICH_SLOT_POLL_S = float(os.environ.get("ARTEMIS_ENRICH_SLOT_POLL_S", "2.0"))

# The probe: one query to decide whether a contact's full ~35-query sweep is
# worth paying for at all (see network/probe.py). Most contacts in a real
# export have no web footprint, so this is the largest available saving on a
# long-tail run. Disable to force the full sweep on every contact.
ENRICH_PROBE_ENABLED = _env_bool("ARTEMIS_ENRICH_PROBE_ENABLED", "1")
# Search results that actually mention the contact before the sweep is funded.
ENRICH_PROBE_MIN_HITS = int(os.environ.get("ARTEMIS_ENRICH_PROBE_MIN_HITS", "1"))
# How long a "no footprint" verdict stands before a later run re-probes. People
# do acquire a web presence — a new job, a funding round — just not weekly.
ENRICH_PROBE_TTL_DAYS = int(os.environ.get("ARTEMIS_ENRICH_PROBE_TTL_DAYS", "30"))

# --- wave 2: org sweeps ------------------------------------------------------
# Expanding an employer several contacts share, once, reaches that org's public
# neighbourhood for ONE contact's worth of queries. Only worth it above a few
# contacts: below that, sweeping the contacts themselves is both cheaper and
# more specific to the operator's actual network.
ENRICH_ORG_SWEEPS_ENABLED = _env_bool("ARTEMIS_ENRICH_ORG_SWEEPS_ENABLED", "1")
ENRICH_ORG_MIN_CONTACTS = int(os.environ.get("ARTEMIS_ENRICH_ORG_MIN_CONTACTS", "3"))
# Hard ceiling on org sweeps per run: they run FIRST (highest coverage per
# query), so an unbounded number would spend the whole wave-1 budget before
# reaching a single real contact.
ENRICH_ORG_MAX_SWEEPS = int(os.environ.get("ARTEMIS_ENRICH_ORG_MAX_SWEEPS", "10"))

# --- per-connection silo weights ---------------------------------------------
# Expansion runs the same 9 silos over everyone (36 queries) regardless of who
# the subject is. Weights computed from the contact's own export row decide
# which silos run and how many of their queries — see network/silo_weights.py.
ENRICH_SILO_WEIGHTS_ENABLED = _env_bool("ARTEMIS_ENRICH_SILO_WEIGHTS_ENABLED", "1")
# A silo weighted below this is skipped entirely for that contact. Raising it
# spends less and risks missing a real relationship; 0 runs every silo for
# everyone, i.e. the unweighted behavior.
ENRICH_SILO_MIN_WEIGHT = float(
    os.environ.get("ARTEMIS_ENRICH_SILO_MIN_WEIGHT", "0.25"))

# tier thresholds: < WEAK_MAX = weak, [WEAK_MAX, STRONG_MIN] = candidate,
# > STRONG_MIN = strong (eligible for expansion priority)
WEAK_MAX = 0.3
STRONG_MIN = 0.6
# ceilings enforced by the evidence rules, without an explicit silo keyword --
# ordered weakest to strongest evidence so the cap reflects how much was
# actually corroborated, not just "some cap under STRONG_MIN":
NO_EXPLICIT_KEYWORD_CEILING = 0.59  # absolute backstop: no explicit keyword => never 'strong'
NO_COOCCURRENCE_CEILING = 0.29     # subject/target never share a sentence -- stays 'weak'
COOCCURRENCE_ONLY_CEILING = 0.39   # they co-occur, but no strength keyword either
GENERIC_STRENGTH_CEILING = 0.5     # co-occur + a strength keyword (e.g. "led"), but that
                                    # keyword never named the actual relationship type
# strength keywords boost confidence when present in the evidence text
STRENGTH_KEYWORDS = [
    "cofounder", "co-founder", "board", "appointed", "joined",
    "advisor", "adviser", "partner", "led", "worked with",
]
STRENGTH_KEYWORD_STEP = 0.15  # per distinct strength keyword found
STRENGTH_FACTOR_CEILING = 1.6

# --- access control ---------------------------------------------------------
# Shared-token gate for the whole app (UI + API). UNSET = wide open, which is
# right for a local checkout and wrong for anything with a public URL: every
# build spends real money (search quota + Anthropic tokens), so an open
# deployment is a bill anyone with the link can run up.
#
# Set it the same way as any other secret — `fly secrets set
# ARTEMIS_ACCESS_TOKEN=...`, a Render/Railway env var, or a line in .env.
# Generate one with:  python -c "import secrets; print(secrets.token_urlsafe(32))"
ACCESS_TOKEN = os.environ.get("ARTEMIS_ACCESS_TOKEN", "").strip()
# Browser session cookie set by POST /login. Holds a value DERIVED from the
# token, never the token itself (see auth.session_value).
SESSION_COOKIE = os.environ.get("ARTEMIS_SESSION_COOKIE", "artemis_session")
SESSION_MAX_AGE_S = int(os.environ.get("ARTEMIS_SESSION_MAX_AGE_S", str(30 * 86400)))
# Trust X-Forwarded-For for the client IP. True on any managed host (Fly,
# Render, Railway all terminate TLS in front of the app), so rate limits key on
# the real caller rather than lumping every request under one proxy IP. Set to
# 0 if the app is ever exposed directly — an untrusted client can forge the
# header and evade per-IP limits.
TRUST_PROXY_HEADERS = _env_bool("ARTEMIS_TRUST_PROXY_HEADERS", "1")

# --- rate limiting ----------------------------------------------------------
# Token bucket per client for the endpoints that cost money (/discover,
# /connect, /targets/search). Burst = bucket capacity; the refill rate is
# BUILD_RATE_LIMIT per BUILD_RATE_WINDOW_S. Cheap reads are never limited.
BUILD_RATE_LIMIT = int(os.environ.get("ARTEMIS_BUILD_RATE_LIMIT", "12"))
BUILD_RATE_WINDOW_S = float(os.environ.get("ARTEMIS_BUILD_RATE_WINDOW_S", "3600"))
# Failed /login attempts per client per window, so a shared token can't be
# brute-forced through the public URL.
LOGIN_RATE_LIMIT = int(os.environ.get("ARTEMIS_LOGIN_RATE_LIMIT", "10"))
LOGIN_RATE_WINDOW_S = float(os.environ.get("ARTEMIS_LOGIN_RATE_WINDOW_S", "900"))

# --- build admission --------------------------------------------------------
# Graph builds that may run CONCURRENTLY. Was effectively 1: connect_people
# mutated a config global, so every build had to be serialized behind one
# mutex and a second user simply waited. That global is now a per-call
# argument (expand_graph's `prefer_reachable`), so builds can overlap.
#
# Kept small on purpose — this multiplies against EXPAND_NODE_CONCURRENCY and
# SEARCH_WORKERS. Outbound HTTP is still globally bounded by
# MAX_CONCURRENT_REQUESTS, so raising this adds waiting builds, not provider
# load; the real limits are host RAM and Anthropic/search spend.
MAX_CONCURRENT_BUILDS = int(os.environ.get("ARTEMIS_MAX_CONCURRENT_BUILDS", "2"))
# Builds allowed to QUEUE behind those. Past this a request is rejected
# immediately with 429 instead of joining a line it would time out in — a
# queue nobody will wait through is worse than a clear refusal.
MAX_QUEUED_BUILDS = int(os.environ.get("ARTEMIS_MAX_QUEUED_BUILDS", "8"))
