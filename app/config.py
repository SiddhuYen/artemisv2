"""Central configuration for Artemis V2.

All knobs live here so silos / expansion / providers stay declarative.

Secrets/keys: put them in a `.env` file in the project root (e.g.
`BRAVE_API_KEY=bsa-xxxx`). It is loaded automatically and never committed.
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


# --- storage ---------------------------------------------------------------
DB_URL = os.environ.get("ARTEMIS_DB_URL", "sqlite:///./artemis.db")
# Per-session graph isolation (HTTP API): each browser session gets its own
# SQLite file here, so concurrent users never clobber each other's public graph.
GRAPH_DIR = os.environ.get("ARTEMIS_GRAPH_DIR", "./graphs")

# --- HTTP ------------------------------------------------------------------
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HTTP_TIMEOUT = float(os.environ.get("ARTEMIS_HTTP_TIMEOUT", "8.0"))
# retry: only on 429/5xx, exponential backoff, honor Retry-After
HTTP_RETRIES = int(os.environ.get("ARTEMIS_HTTP_RETRIES", "3"))
HTTP_BACKOFF_BASE = float(os.environ.get("ARTEMIS_HTTP_BACKOFF", "0.4"))  # seconds
HTTP_RETRY_STATUS = (429, 500, 502, 503, 504)

# --- providers -------------------------------------------------------------
# Brave Search REST API (PRIMARY web search). Key from env; absent => skipped.
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "").strip()
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_QPS = float(os.environ.get("ARTEMIS_BRAVE_QPS", "1.0"))      # free tier ~1 q/s
BRAVE_MONTHLY_QUOTA = int(os.environ.get("ARTEMIS_BRAVE_QUOTA", "2000"))

# Provider routing (configurable). Tried in order; stop at first useful result.
ROUTE_PERSON = os.environ.get(
    "ARTEMIS_ROUTE_PERSON", "wikipedia,wikidata,brave,duckduckgo").split(",")
ROUTE_DEFAULT = os.environ.get(
    "ARTEMIS_ROUTE_DEFAULT", "brave,duckduckgo").split(",")

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
# cap queries run per silo (keeps a single target snappy)
MAX_QUERIES_PER_SILO = int(os.environ.get("ARTEMIS_MAX_QUERIES_PER_SILO", "4"))
# concurrency for the network phase (searches + page fetches)
SEARCH_WORKERS = int(os.environ.get("ARTEMIS_SEARCH_WORKERS", "8"))

# --- expansion -------------------------------------------------------------
DEFAULT_MAX_DEPTH = 2
EXPAND_TOP_N_PER_DEPTH = int(os.environ.get("ARTEMIS_EXPAND_TOP_N", "20"))
# expansion safety / anti-explosion
EXPAND_TOP_STRONG = int(os.environ.get("ARTEMIS_EXPAND_TOP_STRONG", "15"))
# Reachability mode: when expanding, prefer the LEAST-famous real connections
# (no Wikipedia page, fewer sources) instead of the strongest/most-famous ones.
# This walks the graph DOWN the fame gradient toward a normal person's network,
# which is how a warm-intro path to your own contacts is actually found.
EXPAND_PREFER_REACHABLE = os.environ.get(
    "ARTEMIS_EXPAND_PREFER_REACHABLE", "1") not in ("0", "false", "")
# Down-weight family_social edges so expansion explores PROFESSIONAL connections
# (colleagues, boards, co-founders, investors, political) instead of walking the
# subject's family tree. A path between two people runs through work, not relatives.
DOWNWEIGHT_FAMILY = os.environ.get("ARTEMIS_DOWNWEIGHT_FAMILY", "1") not in ("0", "false", "")
FAMILY_PENALTY = float(os.environ.get("ARTEMIS_FAMILY_PENALTY", "1.5"))
PROFESSIONAL_BONUS = float(os.environ.get("ARTEMIS_PROFESSIONAL_BONUS", "1.0"))
MAX_TOTAL_NODES = int(os.environ.get("ARTEMIS_MAX_TOTAL_NODES", "800"))
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

# --- confidence model ------------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
# Confidence ceilings — heuristic extraction is never trusted highly.
HEURISTIC_BASE_CONFIDENCE = 0.25
OLLAMA_BASE_CONFIDENCE = 0.55
# Use Ollama for per-source EXTRACTION (cleanest but slow — one LLM call per
# page). Default OFF: heuristic extraction stays primary even when Ollama is up,
# so the (cheap, batched) entity filter below can run without hours of latency.
OLLAMA_EXTRACT = os.environ.get("ARTEMIS_OLLAMA_EXTRACT", "0") not in ("0", "false", "")
# spaCy NER extraction (Tier 4): grammar-aware, kills prose-fragment noise.
# Preferred over the heuristic when installed; far cheaper than Ollama extraction.
SPACY_EXTRACT = os.environ.get("ARTEMIS_SPACY_EXTRACT", "1") not in ("0", "false", "")
SPACY_BASE_CONFIDENCE = float(os.environ.get("ARTEMIS_SPACY_BASE_CONFIDENCE", "0.35"))
# Ollama entity filter: drop extracted nodes that aren't real named people/orgs.
# Auto-enabled when Ollama is reachable; no-op otherwise. Cached 30 days.
OLLAMA_FILTER = os.environ.get("ARTEMIS_OLLAMA_FILTER", "1") not in ("0", "false", "")
OLLAMA_FILTER_MODEL = os.environ.get("ARTEMIS_OLLAMA_FILTER_MODEL", OLLAMA_MODEL)
OLLAMA_FILTER_BATCH = int(os.environ.get("ARTEMIS_OLLAMA_FILTER_BATCH", "50"))
# Ollama relationship classifier: re-type 'unknown' edges from their evidence.
OLLAMA_CLASSIFY_RELATIONS = os.environ.get(
    "ARTEMIS_OLLAMA_CLASSIFY", "1") not in ("0", "false", "")
OLLAMA_CLASSIFY_BATCH = int(os.environ.get("ARTEMIS_OLLAMA_CLASSIFY_BATCH", "25"))
OLLAMA_CLASSIFY_MIN_CONF = float(os.environ.get("ARTEMIS_OLLAMA_CLASSIFY_MIN_CONF", "0.5"))
RELATION_CONF_CEILING = float(os.environ.get("ARTEMIS_RELATION_CONF_CEILING", "0.85"))

# --- OpenAlex (academic coauthors) -----------------------------------------
# namesake guards: require a real publication record + a close name match
OPENALEX_MIN_WORKS = int(os.environ.get("ARTEMIS_OPENALEX_MIN_WORKS", "3"))
OPENALEX_NAME_SIM = float(os.environ.get("ARTEMIS_OPENALEX_NAME_SIM", "0.8"))

# --- OpenCorporates (company officer networks) -----------------------------
# Free-tier token from https://opencorporates.com/api_accounts/new ; absent => skipped.
OPENCORPORATES_API_TOKEN = os.environ.get("OPENCORPORATES_API_TOKEN", "").strip()
OPENCORP_MIN_INTERVAL = float(os.environ.get("ARTEMIS_OPENCORP_MIN_INTERVAL", "0.5"))

# --- SEC EDGAR (public-company insider networks) ---------------------------
# Free, no key — but SEC requires a declared User-Agent with contact info.
EDGAR_ENABLED = os.environ.get("ARTEMIS_EDGAR_ENABLED", "1") not in ("0", "false", "")
EDGAR_USER_AGENT = os.environ.get(
    "ARTEMIS_EDGAR_USER_AGENT", "Artemis Graph Builder research@artemis.local")
EDGAR_MIN_INTERVAL = float(os.environ.get("ARTEMIS_EDGAR_MIN_INTERVAL", "0.2"))

# --- ProPublica Nonprofit Explorer (990 boards) ----------------------------
PROPUBLICA_ENABLED = os.environ.get("ARTEMIS_PROPUBLICA_ENABLED", "1") not in ("0", "false", "")
PROPUBLICA_MIN_INTERVAL = float(os.environ.get("ARTEMIS_PROPUBLICA_MIN_INTERVAL", "0.3"))
PROPUBLICA_MAX_ORGS = int(os.environ.get("ARTEMIS_PROPUBLICA_MAX_ORGS", "3"))
# tier thresholds: < WEAK_MAX = weak, [WEAK_MAX, STRONG_MIN] = candidate,
# > STRONG_MIN = strong (eligible for expansion priority)
WEAK_MAX = 0.3
STRONG_MIN = 0.6
# ceilings enforced by the evidence rules
COOCCURRENCE_ONLY_CEILING = 0.39   # sentence co-occurrence alone is weak
NO_EXPLICIT_KEYWORD_CEILING = 0.59  # no explicit keyword => cannot be 'strong'
# strength keywords boost confidence when present in the evidence text
STRENGTH_KEYWORDS = [
    "cofounder", "co-founder", "board", "appointed", "joined",
    "advisor", "adviser", "partner", "led", "worked with",
]
STRENGTH_KEYWORD_STEP = 0.15  # per distinct strength keyword found
STRENGTH_FACTOR_CEILING = 1.6
