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
def _normalize_db_url(url: str) -> str:
    """Accept the `postgres://` scheme that hosted providers hand out.

    Supabase, Render and Heroku all still print connection strings starting
    `postgres://`, but SQLAlchemy 1.4+ removed that alias and raises
    NoSuchModuleError ("Can't load plugin: sqlalchemy.dialects:postgres") on
    it. Rewriting here means a copy-pasted dashboard string just works instead
    of failing at import time with an error that reads like a missing driver.
    """
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://"):]
    return url


def _resolve_db_url() -> str:
    """ARTEMIS_DB_URL wins; DATABASE_URL is the fallback; SQLite is the default.

    DATABASE_URL is the convention every managed host and Postgres add-on
    populates automatically (Render, Supabase, Heroku, Fly). Reading it means a
    deploy needs no Artemis-specific database wiring at all -- attach the
    database and the app finds it. ARTEMIS_DB_URL still takes precedence so a
    local override can point somewhere else without unsetting the platform's
    own variable.

    The SQLite default is for local dev and the test suite ONLY. It is not
    safe for a deployment: concurrent expansion has multiple writers, and
    SQLite serializes them behind a single write lock (see DEPLOY.md).
    """
    return _normalize_db_url(
        os.environ.get("ARTEMIS_DB_URL")
        or os.environ.get("DATABASE_URL")
        or "sqlite:///./artemis.db"
    )


DB_URL = _resolve_db_url()
IS_POSTGRES = DB_URL.startswith("postgresql")
# Per-session graph isolation (HTTP API): each browser session gets its own
# SQLite file here, so concurrent users never clobber each other's public graph.
GRAPH_DIR = os.environ.get("ARTEMIS_GRAPH_DIR", "./graphs")
# Boards/pages default to their OWN SQLite file: they never reference the
# discovery-graph tables, and isolating them means a board's frequent autosave
# writes never hit "database is locked" behind a long /connect or
# /targets/search transaction on the (much busier) main DB_URL database.
#
# That whole rationale is SQLite-specific, so on Postgres boards default to
# the SAME database instead -- there is no single write lock to contend for,
# and a deployment should not need a second managed database (or a writable
# local disk, which a container may not even have) just to hold two tables.
BOARDS_DB_URL = _normalize_db_url(
    os.environ.get("ARTEMIS_BOARDS_DB_URL")
    or (DB_URL if IS_POSTGRES else "sqlite:///./artemis_boards.db")
)
# Postgres connection pool (ignored on SQLite, which has no network pool).
# Sized against what expansion actually opens: one Session per concurrently
# processed node (EXPAND_NODE_CONCURRENCY, below), doubled because
# connect_people expands BOTH endpoints at once, plus the API's own
# per-request sessions. Default 10+10 covers that with headroom while staying
# far under the connection cap of a small managed tier -- which is shared with
# psql, migrations, and any other client you have open.
DB_POOL_SIZE = int(os.environ.get("ARTEMIS_DB_POOL_SIZE", "10"))
DB_MAX_OVERFLOW = int(os.environ.get("ARTEMIS_DB_MAX_OVERFLOW", "10"))

# --- HTTP ------------------------------------------------------------------
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Wikimedia (Wikipedia + Wikidata) enforce a robot policy that REJECTS generic
# browser-mimicking user agents with HTTP 403: https://w.wiki/4wJS . A crawler
# must identify itself and provide contact info, the same requirement EDGAR_
# USER_AGENT below already honors for the SEC.
#
# This was found the hard way. Sending USER_AGENT above got a blanket 403 from
# every Wikimedia endpoint, and because every wiki call is best-effort
# (notable_set swallows exceptions, enrich_person returns None on failure) a
# TOTAL outage was indistinguishable from "this person simply isn't famous".
# Silently, that meant: notable_set() always empty, so /connect's asymmetric
# Alpha walk NEVER fired for anyone; enrich_person() always None, so Tier-1
# Wikipedia/Wikidata structured enrichment never ran; no QID ever anchored a
# person, so homonym disambiguation lost its authoritative signal and
# FAME_PENALTY never applied to any transit node.
#
# Scoped to Wikimedia rather than replacing USER_AGENT globally: the browser
# UA is doing real work for ordinary page scraping, where plenty of sites
# refuse non-browser agents -- exactly the opposite constraint.
WIKIMEDIA_USER_AGENT = os.environ.get(
    "ARTEMIS_WIKIMEDIA_USER_AGENT",
    "ArtemisGraphBuilder/2.0 (https://github.com/SiddhuYen/artemisv2; "
    "research@artemis.local)")
WIKIMEDIA_HEADERS = {"User-Agent": WIKIMEDIA_USER_AGENT}
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
# Headcount moves slowly (quarterly at most) and each lookup costs a real
# search + model call, so this outlives the other caches.
CACHE_TTL_HEADCOUNT = int(os.environ.get("ARTEMIS_CACHE_TTL_HEADCOUNT", str(90 * 86400)))

# --- search behaviour ------------------------------------------------------
# RESULTS_PER_QUERY caps how many results a provider even RETURNS per query
# (the "num"/"count" param sent to Serper/Brave/etc) -- SCRAPE_TOP_N below
# can never scrape past this, since ranks beyond it don't exist in the
# results list at all. Raised alongside SCRAPE_TOP_N (both 3->10, live-
# tested: two comparable never-searched people at 3 vs 6 took ~the same
# wall-clock time, 19s vs 17s -- page fetches within a query already run
# concurrently, and Serper bills per QUERY regardless of "num", so this is
# effectively free coverage, not a cost increase).
RESULTS_PER_QUERY = int(os.environ.get("ARTEMIS_RESULTS_PER_QUERY", "10"))
SCRAPE_TOP_N = int(os.environ.get("ARTEMIS_SCRAPE_TOP_N", "10"))
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
# How many times to redo a whole node whose transaction died on a lock or
# deadlock (expansion._process_one). Distinct from the statement-level retries
# in graph.builder, which cannot fix a Postgres deadlock: there both workers
# hold locks the other needs, so only discarding the entire transaction breaks
# the cycle.
#
# 5, not a token 2, because the retries have to OUTLAST the transaction that
# won the deadlock. The loser can only get through once the winner commits and
# releases its locks, and a node's transaction runs for seconds (extraction
# plus a round trip per write). Measured against a real Postgres with two
# workers inserting the same 40 people in opposite orders: at 2 attempts every
# round still lost a worker, because all its retries fired before the winner
# finished; at 5 -- whose backoff spreads ~0.3s to ~4.8s -- nothing was lost
# across four rounds. Retries are cheap relative to losing the node: the
# searches behind one are served from the provider cache, so the cost is
# re-extraction, not re-buying data.
NODE_DB_RETRY_ATTEMPTS = int(os.environ.get("ARTEMIS_NODE_DB_RETRY_ATTEMPTS", "5"))
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
# Alpha step 7: bonus per cofounder/board_member-typed (or business-domain-
# language) edge a candidate has -- ranks "most high up and well connected"
# candidates above equally-evidenced but junior ones. Additive on top of
# score()'s existing strong/explicit/source terms, not a replacement for them.
SENIORITY_BONUS = float(os.environ.get("ARTEMIS_SENIORITY_BONUS", "1.0"))
# Alpha step 7's "pick 5 of the strongest" -- narrower than the general
# EXPAND_TOP_STRONG beam (15). Applied only on the non-famous/origin side of
# an asymmetric /connect walk (enhanced_professional_search), where a
# reasoning-selected angle already narrowed the field; expanding 15 people's
# worth of generic silo search doesn't need the same narrowing.
ALPHA_TOP_CANDIDATES = int(os.environ.get("ARTEMIS_ALPHA_TOP_CANDIDATES", "5"))
# Shared global map accumulates people across ALL runs, so the cap is high.
# (A single isolated build rarely approaches this; it bounds unbounded growth.)
MAX_TOTAL_NODES = int(os.environ.get("ARTEMIS_MAX_TOTAL_NODES", "50000"))
# connect() builds TWO graphs into one DB. Each side gets its OWN people budget
# so the first (richer) person can't starve the second — the second person's cap
# is raised to (2 x per-side) after the first is built. Total ~= 2 x per-side.
CONNECT_NODE_CAP_PER_SIDE = int(os.environ.get("ARTEMIS_CONNECT_NODE_CAP_PER_SIDE", "1000"))
# how many DIVERSE routes connect() returns (each avoids prior routes' bridges)
CONNECT_MAX_PATHS = int(os.environ.get("ARTEMIS_CONNECT_MAX_PATHS", "3"))
# How many of the operator's own contacts /connect enriches as bridge
# candidates, ranked toward THIS target (see graph/connect._bridge_contacts).
# This is what makes enrichment per-connect rather than one global batch: the
# contacts worth spending queries on depend on who is being reached, which the
# cold-start run could not know. They expand sequentially in rank order and
# stop the moment a route is found, so this is a ceiling, not a fixed cost.
# 0 disables the front entirely and restores the pure two-endpoint walk.
CONNECT_BRIDGE_CONTACTS = int(
    os.environ.get("ARTEMIS_CONNECT_BRIDGE_CONTACTS", "5"))
# Step 1 of every /connect: derive the ORIGIN's own first degree into the shared
# graph (linkedin_1st bridge + wave-0 org membership and coworker cliques)
# before any paid search. Both derivations are free and idempotent, so this runs
# unconditionally rather than depending on whether an import or a batch run
# happened to do it -- see graph/connect._ensure_origin_enriched. 0 skips it.
CONNECT_ENRICH_ORIGIN = _env_bool("ARTEMIS_CONNECT_ENRICH_ORIGIN", "1")
# Master switch for step 1's first-degree bridge (every imported contact ->
# the origin, as linkedin_1st). Split from the flag above because the two
# halves make different claims: wave 0 derives ties BETWEEN contacts from their
# own employer columns and asserts nothing about the origin, whereas this
# asserts the origin personally knows all of them.
#
# NOTE this is only a ceiling, not the actual gate. The bridge additionally
# requires the origin to BE the contact owner -- see
# graph/connect._origin_is_operator -- because those contacts are nobody
# else's connections. Turning this off disables the bridge outright; leaving it
# on still bridges nothing when /connect is traced from someone who isn't you.
CONNECT_ORIGIN_BACKFILL = _env_bool("ARTEMIS_CONNECT_ORIGIN_BACKFILL", "1")
# Harvest the OTHER people named on the pages the direct-pair search already
# fetched, instead of reading each page for a single A-B fact and discarding
# everyone else on it.
#
# The pair query ('"Sanjay Ghemawat" "Larry Page"') returns pages about the
# world both endpoints sit in, so the people on them are disproportionately the
# ones who bridge -- and the page is already downloaded and already saved as a
# Source. The keyword path was extracting them and dropping them one line
# later; the Claude path only ever looked at windows naming both endpoints, so
# it never saw them at all. Either way the observed failure was a pair three
# hops apart returning "no connection" while the intermediary sat in HTML we
# had paid for.
#
# NOT free, and that is why it has a switch: per-source extraction runs once
# per endpoint per page, so this is ~2x pages Claude calls added to what the
# docstring calls the cheap first pass. Costed on /status.
CONNECT_HARVEST_PAIR_PAGES = _env_bool("ARTEMIS_CONNECT_HARVEST_PAIR_PAGES", "1")
# Ceiling on pages harvested per direct-pair search, applied to the highest-
# ranked results. Bounds the added spend when a query returns a long tail whose
# relevance is already falling off.
CONNECT_HARVEST_MAX_PAGES = int(
    os.environ.get("ARTEMIS_CONNECT_HARVEST_MAX_PAGES", "6"))
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

# --- subject-window narrowing (extraction/subject_windows.py) ---------------
# CLAUDE_EXTRACT sends a WHOLE page (up to MAX_PAGE_CHARS, ~5k tokens) to answer
# a question about one person, and on a real search result most of that page is
# about someone else. Narrowing to the sentences that name the subject -- or use
# a pronoun resolving to them -- plus SUBJECT_WINDOW_SENTENCES of context either
# side is the difference between paying per page and paying per passage.
#
# Set to 0 to send whole pages again (the pre-narrowing behaviour), which is the
# thing to try first if extraction recall drops after enabling this.
SUBJECT_WINDOW_ENABLED = _env_bool("ARTEMIS_SUBJECT_WINDOW", "1")
# Sentences of context kept either side of an anchor sentence. Defaults to
# ENTITY_PROXIMITY_WINDOW deliberately: that constant already encodes how far
# from a subject mention this codebase is willing to call something evidence
# about the subject (see its comment), and narrowing tighter than the gate that
# judges the result afterwards would just discard text the extractors would
# have been allowed to use.
SUBJECT_WINDOW_SENTENCES = int(
    os.environ.get("ARTEMIS_SUBJECT_WINDOW_SENTENCES", str(ENTITY_PROXIMITY_WINDOW)))
# Below this, don't narrow at all. Short inputs to extract() are not scraped
# pages -- they are the synthetic enrichment strings built in expansion phase 0
# (Wikidata evidence text, roster colleague summaries, OpenAlex coauthor lists).
# Those are already dense and already about the subject, and several never spell
# the subject's name in a sentence at all, so narrowing them risks dropping the
# whole input to save tokens that were never the problem.
SUBJECT_WINDOW_MIN_CHARS = int(
    os.environ.get("ARTEMIS_SUBJECT_WINDOW_MIN_CHARS", "1500"))
# How many sentences back a pronoun's antecedent may be. Past a few sentences
# the "last name mentioned" heuristic stops being evidence of anything -- an
# unbounded walk would happily bind a "he" on line 400 to a name on line 1.
SUBJECT_WINDOW_PRONOUN_LOOKBACK = int(
    os.environ.get("ARTEMIS_SUBJECT_WINDOW_LOOKBACK", "4"))
# Hard ceiling, in sentences, on a single merged span. An author-archive or
# listing page repeats the subject's name as a byline every few sentences, so
# each mention's own window overlaps the next and the merge loop chains them
# into one span covering nearly the whole page -- including whatever unrelated
# teaser text sits between two bylines, with no elision marker to warn the
# model it's reading glued-together boilerplate. Once a chain would grow past
# this many sentences, split it instead: the next anchor starts a new span,
# and the elision marker between them tells the model (correctly) that
# something was cut, rather than presenting repeated-byline noise as one
# continuous passage about the subject.
SUBJECT_WINDOW_MAX_MERGED_SENTENCES = int(
    os.environ.get("ARTEMIS_SUBJECT_WINDOW_MAX_MERGED", "20"))
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

# Wikidata "colleagues" reverse lookup (shared employer/board -> "coworker"):
# an org above this current-headcount size is too large for bare co-membership
# to mean two specific people actually know each other (e.g. "both worked at
# Harvard" spans centuries and thousands of people). Below it, or when the
# headcount can't be determined, the edge stands as before -- see
# providers.wikidata._org_headcount for why "unknown" defaults to small rather
# than large (small/obscure orgs are the ones LEAST likely to have a
# documented headcount, so defaulting the other way would penalize exactly the
# legitimate small-company connections this exists to protect).
WIKIDATA_COLLEAGUE_ORG_MAX_SIZE = int(
    os.environ.get("ARTEMIS_WIKIDATA_COLLEAGUE_ORG_MAX_SIZE", "300"))
HEADCOUNT_MODEL = os.environ.get("ARTEMIS_HEADCOUNT_MODEL", CLAUDE_BATCH_MODEL)
RELATION_CONF_CEILING = float(os.environ.get("ARTEMIS_RELATION_CONF_CEILING", "0.85"))

# The "deferred Claude verification stage" referenced in claude_extractor.py /
# relation_classifier.py / entity_filter.py's docstrings, and in connect_people's
# "Requires Claude verification before activation" warning: judges ONE edge's
# own evidence, independent of any specific path it's walked in (a verdict is
# cached on the edge and reused across every path that includes it -- see
# graph.hop_verify's docstring for why path-level context can't factor in).
CLAUDE_VERIFY_HOPS = _env_bool("ARTEMIS_CLAUDE_VERIFY_HOPS", "1")
HOP_VERIFY_MODEL = os.environ.get("ARTEMIS_HOP_VERIFY_MODEL", CLAUDE_BATCH_MODEL)
# Asymmetric on purpose: a wrongly-REJECTED edge actively hides a real
# connection from every path shown to a user until it's re-checked, so it
# gets a much shorter leash than a wrongly-approved one, which just leaves
# the graph at its pre-verification baseline in the meantime.
HOP_VERIFY_TTL_GENUINE = int(os.environ.get("ARTEMIS_HOP_VERIFY_TTL_GENUINE", str(30 * 86400)))
HOP_VERIFY_TTL_REJECTED = int(os.environ.get("ARTEMIS_HOP_VERIFY_TTL_REJECTED", str(10 * 86400)))

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

# --- coauthor plausibility gate (before OpenAlex, not just after) ----------
# See extraction/coauthor_plausibility.py: a cheaper, prior question than
# phase 4b's existing domain_conflict check -- given what's already known
# about the subject, is an OpenAlex coauthor lookup even worth attempting?
# Skips the call entirely for subjects who clearly aren't the type to have
# academic publications, closing the homonym-collision risk a layer earlier
# instead of only catching it after OpenAlex has already resolved a name.
COAUTHOR_PLAUSIBILITY_ENABLED = _env_bool("ARTEMIS_COAUTHOR_PLAUSIBILITY", "1")
COAUTHOR_PLAUSIBILITY_MODEL = os.environ.get(
    "ARTEMIS_COAUTHOR_PLAUSIBILITY_MODEL", CLAUDE_BATCH_MODEL)

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
# Cached profiles live on Organization.meta with NO TTL (an org's size/industry
# doesn't churn), which means a profile written under an OLDER prompt or guard
# set is indistinguishable from a current one and is reused forever. That is
# not hypothetical: the org-identity block (node_profiler's third guard) was
# added only after a live run conflated "Trinamix Inc" with "trinamiX GmbH",
# so every profile written before it is a pre-guard verdict. BUMP THIS on any
# change to the prompt, the schema, or the post-call guards -- a stale profile
# then simply re-profiles on next encounter instead of silently standing in
# for one the current code would never have produced.
NODE_PROFILE_VERSION = 2
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
# Per-contact "would a search for this name return anything" judgment, made
# once from the uploaded row and stored on the profile (see
# extraction/contact_profiler.py). Target-independent, so it is computed at
# import/backfill time and reused by every future /connect rather than being
# re-derived per connect like the hop-0 strategy below.
CONTACT_PROFILE_ENABLED = _env_bool("ARTEMIS_CONTACT_PROFILE", "1")
CONTACT_PROFILE_MODEL = os.environ.get("ARTEMIS_CONTACT_PROFILE_MODEL", CLAUDE_BATCH_MODEL)
CONTACT_PROFILE_BATCH = int(os.environ.get("ARTEMIS_CONTACT_PROFILE_BATCH", "25"))
# Score contribution per footprint tier. "individual" is what 35 queries are
# actually buying; "none" is negative because spending them on a student-club
# row is worse than neutral -- it displaces a contact who would have returned
# something, and reliably drags in a namesake's network.
CONTACT_FOOTPRINT_SCORES = {"individual": 3.0, "org_only": 0.5, "none": -1.5}

# Hop-0 counterpart to STRATEGY below (see extraction/bridge_strategy.py):
# reasons about WHICH of the operator's contacts to walk first, before any
# expansion has happened, from the origin's and target's contexts.
#
# The deterministic ranker still runs first and still decides the shortlist --
# a 1,188-contact export doesn't fit in a prompt, and its formula encodes real
# signal cheaply and exactly. The model only reorders the top of that ranking.
BRIDGE_STRATEGY_ENABLED = _env_bool("ARTEMIS_BRIDGE_STRATEGY", "1")
BRIDGE_STRATEGY_MODEL = os.environ.get("ARTEMIS_BRIDGE_STRATEGY_MODEL", CLAUDE_BATCH_MODEL)
# How many top-ranked contacts are shown to the model to choose among.
BRIDGE_SHORTLIST = int(os.environ.get("ARTEMIS_BRIDGE_SHORTLIST", "15"))
# How many it may promote to the front of the queue. The REST of the ranked
# contacts stay queued behind them as fallback rather than being dropped:
# expansion is sequential and stops the instant a route closes, so a correct
# pick costs exactly these two and a wrong one degrades to today's behavior
# instead of losing the walk. Cutting the queue to the picks would save nothing
# on a good run and cost recall on a bad one.
BRIDGE_PRIORITY_PICKS = int(os.environ.get("ARTEMIS_BRIDGE_PRIORITY_PICKS", "2"))
STRATEGY_ENABLED = _env_bool("ARTEMIS_STRATEGY", "1")
STRATEGY_MODEL = os.environ.get("ARTEMIS_STRATEGY_MODEL", CLAUDE_BATCH_MODEL)
STRATEGY_ANGLE_QUERIES = {
    # Deliberately EMPTY, not deleted: the angle is still a meaningful thing
    # for the model to choose, it just no longer maps to prose queries. It
    # used to fire '"{org}" leadership team OR executives' into the ordinary
    # extraction path, where the proximity gate decided the outcome -- a long
    # leadership page that never names a VP-level subject had every entity
    # dropped, and a short one had the whole exec roster wired to the subject
    # as unevidenced colleagues. A roster is a structural assertion, so that
    # work now happens in expansion's phase 4f via providers/directory.py,
    # which runs for every Alpha node regardless of the chosen angle.
    "current_employer_leadership": [],
    "past_employers": [
        '"{subject}" "previously at" OR "formerly at" OR "prior to {org}"',
    ],
    # Also empty, for a simpler reason: '"{industry}" industry leaders
    # network' dropped both the subject and the target and returned
    # listicles. The directory pack (DIRECTORY_PACKS) reaches the same
    # people through their actual employers instead of through SEO bait.
    "industry_peers": [],
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
# Raw HTML cached/parsed per page fetch (before html_to_text's further cut to
# MAX_PAGE_CHARS) — a roster grid can sit far down the markup of a long page.
#
# Measured, not guessed. At the old MAX_PAGE_CHARS * 4 (80k) the cap landed
# mid-boilerplate on every large roster tried: uncorkcapital.com/team gave 0
# names truncated vs 14 whole, stthomas.edu 0 vs 16, bvp.com/team 65 vs 252.
# Those pages are NOT single-page apps -- the names were in the plain response
# all along, just past the cut. 250k is where all of them reach full recall.
#
# Affordable only because fetch_page now strips inline script/style bodies
# BEFORE truncating (htmltext.strip_inline_noise), which roughly halves what
# is stored; a bare cap raise would have multiplied page-cache growth instead.
MAX_HTML_CHARS = int(os.environ.get("ARTEMIS_MAX_HTML_CHARS", "250000"))

# --- Directories (org-keyed staff/professional directories) -----------------
# The org-keyed sibling of FIRMS above: firms.py answers "who does this PERSON
# work with", this answers "who works at this ORG". Built for Alpha's
# non-famous side, where the subject's own employer is the densest real source
# of professional connections there is -- and where the previous
# implementation of that idea (the current_employer_leadership strategy angle)
# fed a company leadership page through the PROSE extractor, which either
# dropped everything (long page, subject never named -> proximity gate) or
# wired the whole exec roster to the subject unevidenced (short page). A
# roster is a structural assertion and belongs on the structural path.
DIRECTORY_ENABLED = _env_bool("ARTEMIS_DIRECTORY_ENABLED", "1")
DIRECTORY_MAX_MEMBERS = int(os.environ.get("ARTEMIS_DIRECTORY_MAX_MEMBERS", "60"))
# How many located candidate URLs to verify before giving up on an org.
DIRECTORY_MAX_CANDIDATES = int(os.environ.get("ARTEMIS_DIRECTORY_MAX_CANDIDATES", "6"))

# Sector query packs, selected by keyword match against node_profiler's
# GROUNDED `industry` string (see extraction/node_profiler.py). Same
# containment principle as STRATEGY_ANGLE_QUERIES: the profile picks WHICH
# pre-written pack applies, nothing ever generates query text from a model.
# "default" is used when the industry is unknown or matches nothing, so an
# ungrounded profile degrades to the generic pack rather than to no search.
DIRECTORY_PACKS = {
    "legal": {
        "match": ("law", "legal", "attorney", "litigation", "counsel"),
        "queries": ['"{org}" attorneys directory', '"{org}" our lawyers'],
    },
    "medical": {
        "match": ("health", "medical", "hospital", "clinic", "care", "physician"),
        "queries": ['"{org}" physicians directory', '"{org}" our providers'],
    },
    "academic": {
        "match": ("university", "college", "education", "research institute"),
        "queries": ['"{org}" faculty directory', '"{org}" department people'],
    },
    "consulting": {
        "match": ("consult", "advisory", "professional services", "erp", "systems integrat"),
        "queries": ['"{org}" our team consultants', '"{org}" leadership team'],
    },
    "finance": {
        "match": ("bank", "financial", "invest", "capital", "wealth", "insurance"),
        "queries": ['"{org}" our team', '"{org}" leadership team'],
    },
    "nonprofit": {
        "match": ("nonprofit", "non-profit", "foundation", "charity", "ngo"),
        "queries": ['"{org}" staff directory', '"{org}" our team'],
    },
    "default": {
        "match": (),
        "queries": ['"{org}" team page', '"{org}" our team', '"{org}" staff directory'],
    },
}

# A large org's full directory is a haystack, and everyone in it is a weak
# bridge -- so for `large` (and for `unknown`, which is the conservative
# default when profiling didn't ground) only the leadership page is worth
# fetching. Small/mid orgs get the full sector pack: at that size the
# directory genuinely IS the subject's professional network.
DIRECTORY_LEADERSHIP_QUERIES = ['"{org}" leadership team', '"{org}" executive team']
DIRECTORY_FULL_SIZE_TIERS = ("small", "mid")

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
# A failed task (a transient provider timeout, a rate-limit blip -- expected,
# not exceptional, at ~35 outbound calls per contact) is eligible to be picked
# up again by _next_task as long as its attempts stay under this. Bounded so a
# contact that fails for a PERMANENT reason (a name that reliably crashes the
# extractor) doesn't retry forever — it fails out for good past this ceiling.
ENRICH_MAX_ATTEMPTS = int(os.environ.get("ARTEMIS_ENRICH_MAX_ATTEMPTS", "3"))

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

# --- coverage-based reuse -----------------------------------------------------
# `Person.processed` says a node was expanded but not what it was ASKED. With
# reuse gated on the flag alone, the first walk to touch a node froze it: every
# later walk replayed its stored neighbors, so a /connect toward a target the
# original walk knew nothing about could never ask its own questions. With this
# on, a node records its per-silo query budget (Person.metadata["silo_coverage"])
# and a later walk re-searches ONLY the silos that budget never covered — which
# is what lets enrichment be initialized per connect instead of once, globally.
#
# Set to 0 to restore the old freeze-on-first-expansion behavior. Nodes carrying
# no coverage record (expanded before this shipped) are read as fully covered
# either way, so turning it on does not invalidate a warm graph.
EXPAND_COVERAGE_REUSE = _env_bool("ARTEMIS_EXPAND_COVERAGE_REUSE", "1")

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
