"""Name normalisation + lightweight entity-shape heuristics.

Used both for dedup (normalize) and for filtering junk out of heuristic
extraction (looks_like_person_name / org suffix detection).
"""
import re

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")

# Common honorifics / role words that pollute capitalised-token extraction.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "in", "on", "at", "to",
    "mr", "mrs", "ms", "dr", "prof", "sir", "ceo", "cfo", "cto", "president",
    "chairman", "director", "founder", "officer", "company", "inc", "llc",
    "university", "foundation", "news", "report", "said", "according",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    # frequent non-person proper nouns (places / products / media) that the
    # capitalised-token heuristic otherwise mistakes for people
    "united", "states", "kingdom", "new", "york", "san", "los", "angeles",
    "francisco", "city", "north", "south", "east", "west", "street", "avenue",
    "windows", "phone", "office", "server", "cloud", "online", "today",
    "world", "times", "post", "journal", "magazine", "press", "media",
    # generic institutional / descriptor words that form noisy pseudo-names
    "national", "international", "state", "higher", "education", "council",
    "committee", "conference", "symposium", "award", "civilian", "college",
    "academy", "business", "global", "federal", "central", "royal", "public",
    "big", "tech", "higher", "vision", "audio", "music", "shopping", "store",
}

ORG_SUFFIXES = {
    "inc": "company", "inc.": "company", "llc": "company", "ltd": "company",
    "corp": "company", "corporation": "company", "co": "company",
    "company": "company", "group": "company", "holdings": "company",
    "partners": "company", "ventures": "company", "capital": "company",
    "labs": "company", "technologies": "company", "systems": "company",
    "university": "school", "college": "school", "institute": "school",
    "school": "school", "academy": "school",
    "foundation": "nonprofit", "trust": "nonprofit", "fund": "nonprofit",
    "association": "nonprofit", "society": "nonprofit", "nonprofit": "nonprofit",
    "department": "government", "agency": "government", "commission": "government",
    "committee": "government", "bureau": "government", "ministry": "government",
    "conference": "event", "summit": "event", "forum": "event", "expo": "event",
}

# Trailing suffix tokens removed when building an organization dedup key.
# Conservative: legal/structural suffixes only (per spec: Inc, LLC, Ltd,
# Foundation, University, Corp ...). Interior words are never removed.
_ORG_DEDUP_SUFFIXES = {
    "inc", "llc", "ltd", "limited", "corp", "corporation", "co", "company",
    "group", "holdings", "plc", "gmbh", "sa", "ag", "foundation", "university",
}

# Diminutive/nickname -> formal first name, so "Tim Cook" and "Timothy Cook"
# collapse to ONE person node. Applied to the FIRST token when building a
# person key. Conservative & one-directional (nickname -> formal); genuinely
# gender-ambiguous stems (Chris, Pat, Sam, Jamie, Alex) are intentionally
# omitted to avoid wrong merges. Extend as needed.
_DIMINUTIVES = {
    "tim": "timothy", "timmy": "timothy",
    "bill": "william", "billy": "william", "will": "william", "willy": "william",
    "bob": "robert", "bobby": "robert", "rob": "robert", "robbie": "robert",
    "dick": "richard", "rick": "richard", "ricky": "richard", "rich": "richard",
    "tom": "thomas", "tommy": "thomas",
    "mike": "michael", "mikey": "michael",
    "jim": "james", "jimmy": "james",
    "joe": "joseph", "joey": "joseph",
    "dave": "david", "davey": "david",
    "dan": "daniel", "danny": "daniel",
    "matt": "matthew",
    "nick": "nicholas",
    "tony": "anthony",
    "ben": "benjamin", "benji": "benjamin",
    "ed": "edward", "eddie": "edward",
    "ted": "theodore", "teddy": "theodore",
    "andy": "andrew",
    "greg": "gregory",
    "jeff": "jeffrey",
    "ken": "kenneth", "kenny": "kenneth",
    "larry": "lawrence",
    "pete": "peter",
    "phil": "philip", "philip": "phillip",
    "ron": "ronald", "ronnie": "ronald",
    "fred": "frederick", "freddie": "frederick",
    "charlie": "charles", "chuck": "charles",
    "nate": "nathaniel",
    "vince": "vincent",
    "walt": "walter",
    "hank": "henry",
    "liz": "elizabeth", "beth": "elizabeth", "betty": "elizabeth",
    "kate": "katherine", "katie": "katherine", "kathy": "katherine",
    "meg": "margaret", "peggy": "margaret", "maggie": "margaret",
    "sue": "susan", "susie": "susan",
    "jen": "jennifer", "jenny": "jennifer",
    "becky": "rebecca",
    "debbie": "deborah", "deb": "deborah",
    "cindy": "cynthia",
    "vicky": "victoria",
    "abby": "abigail",
}

# Scraped-web boilerplate that the capitalised-token / NER extractors otherwise
# mistake for people or orgs: cookie banners, legal/UI chrome, LinkedIn nav.
# `is_noise_name` runs BEFORE the (optional) LLM entity filter, so this junk is
# dropped even when Ollama is off. Tokens here must NOT collide with real name
# words (kept out: fund/capital/group/trust which are legit org words).
_NOISE_TOKENS = {
    "cookie", "cookies", "policy", "policies", "privacy", "agreement",
    "consent", "gdpr", "copyright", "disclaimer", "trademark",
    "profile", "profiles", "login", "signin", "signup", "logout",
    "newsletter", "subscribe", "unsubscribe", "settings", "preferences",
    "notifications", "sitemap", "homepage", "password", "username",
    "advertisement", "sponsored", "checkout", "wishlist", "captcha",
}

_NOISE_PHRASES = {
    "cookie policy", "cookie settings", "cookie preferences", "manage cookies",
    "accept cookies", "accept all", "reject all", "privacy policy",
    "privacy notice", "privacy statement", "your privacy", "data protection",
    "user agreement", "terms of service", "terms of use", "terms and conditions",
    "all rights reserved", "learn more", "read more", "show more", "see more",
    "sign in", "sign up", "log in", "create account", "join now", "get started",
    "contact us", "about us", "follow us", "skip to content", "personal information",
}


def is_noise_name(name: str) -> bool:
    """True if `name` is scraped boilerplate/navigation chrome rather than a real
    named entity (e.g. "Cookie Policy", "User Agreement", "Fred Volinsky Profile").
    Deterministic — works with or without the Ollama entity filter."""
    norm = normalize(name)
    if not norm:
        return True
    if norm in _NOISE_PHRASES:
        return True
    return any(tok in _NOISE_TOKENS for tok in norm.split())


def normalize(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — the base dedup key."""
    if not name:
        return ""
    s = name.strip().lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def strip_middle_initials(name: str) -> str:
    """Drop single-letter middle initials so name variants collapse together.

    "John F. Kennedy" / "John F Kennedy" -> "John Kennedy". First and last
    tokens are always kept; only interior single-letter tokens are removed.
    """
    parts = name.split()
    if len(parts) <= 2:
        return name.strip()
    kept = [parts[0]]
    for mid in parts[1:-1]:
        token = mid.rstrip(".")
        if len(token) <= 1:  # initial like "F" or "F."
            continue
        kept.append(mid)
    kept.append(parts[-1])
    return " ".join(kept)


def person_norm_key(name: str) -> str:
    """Canonical dedup key for a person: normalised, middle-initials stripped,
    with the first name canonicalised through the diminutive map so nickname
    variants collapse ("Tim Cook" / "Timothy Cook" -> "timothy cook")."""
    base = normalize(strip_middle_initials(name))
    if not base:
        return ""
    parts = base.split()
    parts[0] = _DIMINUTIVES.get(parts[0], parts[0])
    return " ".join(parts)


def strip_org_suffixes(name: str) -> str:
    """Remove trailing org/legal suffix tokens for org dedup.

    "Acme Inc." / "Acme Corporation" -> "Acme". Only trailing suffix tokens are
    removed (repeatedly), never interior words, to limit accidental merges.
    """
    parts = normalize(name).split()
    while len(parts) > 1 and parts[-1] in _ORG_DEDUP_SUFFIXES:
        parts.pop()
    return " ".join(parts)


def org_norm_key(name: str) -> str:
    """Canonical dedup key for an organization (suffix-stripped)."""
    stripped = strip_org_suffixes(name)
    return stripped or normalize(name)


def name_variants(name: str):
    """Surface forms worth storing as aliases (deduped, excluding the input)."""
    variants = set()
    raw = name.strip()
    if raw:
        variants.add(raw)
    smi = strip_middle_initials(raw)
    if smi:
        variants.add(smi)
    return variants


def looks_like_person_name(token: str) -> bool:
    """Heuristic: 2–4 capitalised words, no org suffix, not stopwords/boilerplate."""
    token = token.strip()
    if is_noise_name(token):
        return False
    parts = token.split()
    if not (2 <= len(parts) <= 4):
        return False
    for p in parts:
        if not p[:1].isupper():
            return False
        np = normalize(p)
        if len(np) < 2:          # drop initials / single letters ("John W")
            return False
        if np in _STOPWORDS:
            return False
        if np in ORG_SUFFIXES:
            return False
    return True


def detect_org_type(name: str) -> str:
    """Return an ORG_TYPES value based on the last token's suffix, else unknown."""
    parts = normalize(name).split()
    for p in reversed(parts):
        if p in ORG_SUFFIXES:
            return ORG_SUFFIXES[p]
    return "unknown"


def looks_like_org_name(name: str) -> bool:
    """True if any token matches a known org suffix."""
    parts = normalize(name).split()
    return any(p in ORG_SUFFIXES for p in parts)
