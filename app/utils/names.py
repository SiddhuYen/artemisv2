"""Name normalisation + lightweight entity-shape heuristics.

Used both for dedup (normalize) and for filtering junk out of heuristic
extraction (looks_like_person_name / org suffix detection).
"""
import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")

# Generational suffixes, inconsistently included/omitted across scraped
# sources for the same person ("John Smith" vs "John Smith Jr.") -- stripped
# in strip_middle_initials() so both forms collapse to one dedup key.
_GENERATIONAL_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

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
    # more place-name tokens (region/geography nicknames get referred to as
    # "institutions" in prose — e.g. "the storied Silicon Valley institution"
    # — and pass the 2-word-capitalized shape check just as easily as a name)
    "silicon", "valley", "bay", "area", "coast", "county", "island", "district",
    "harbor", "harbour",
    # generic institutional / descriptor words that form noisy pseudo-names
    "national", "international", "state", "higher", "education", "council",
    "committee", "conference", "symposium", "award", "civilian", "college",
    "academy", "business", "global", "federal", "central", "royal", "public",
    "big", "tech", "higher", "vision", "audio", "music", "shopping", "store",
    # role/title fragments that scraped rosters glue onto names ("Partner Jason
    # Calacanis", "Abhay Mavalankar SVP") — never part of a real personal name.
    # "board"/"member" specifically: structured "roles" enrichment text reads
    # like "Investor at Career Karma, Board Member at Helm" -- the role sits
    # right next to an org name with no marker distinguishing it from a person,
    # so "Board Member" alone clears the 2-word-capitalised shape check clean
    # (seen live: a fake "Board Member" node attached to Garry Tan, sourced
    # from exactly this kind of officer/board-role text).
    "partner", "gp", "vp", "svp", "evp", "coo", "managing", "principal", "head",
    "board", "member",
    # legal/court-filing role words ("Defendant Elon Musk", "Plaintiff Jane Doe")
    "defendant", "plaintiff", "petitioner", "respondent", "appellant",
    "appellee", "witness", "juror",
    # linking/auxiliary verbs — their presence means the "name" is really a
    # sentence fragment ("Diana Hu Is YC"), never part of an actual person's name
    "is", "was", "are", "were", "be", "been", "has", "have",
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
    "phil": "philip",
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
# dropped even when Claude is off. Tokens here must NOT collide with real name
# words (kept out: fund/capital/group/trust which are legit org words).
_NOISE_TOKENS = {
    "cookie", "cookies", "policy", "policies", "privacy", "agreement",
    "consent", "gdpr", "copyright", "disclaimer", "trademark",
    "profile", "profiles", "login", "signin", "signup", "logout",
    "newsletter", "subscribe", "unsubscribe", "settings", "preferences",
    "notifications", "sitemap", "homepage", "password", "username",
    "advertisement", "sponsored", "checkout", "wishlist", "captcha",
    # LinkedIn reaction-count UI text ("1 Reaction", "2 Reactions") butts
    # directly against the NEXT real commenter's name in scraped comment
    # threads, with no punctuation boundary the extractor's capitalised-run
    # regex can see -- it grabs "Reactions <real name>" as one candidate.
    # A token-level match here (not a whole-phrase entry in _NOISE_PHRASES)
    # is required since the attached name is different every time (seen
    # live: 58 distinct fake nodes, one per real commenter this glued onto).
    # The real name is lost along with it -- an acceptable trade: these are
    # incidental commenters on public posts, not the subject being researched.
    "reaction", "reactions",
}

_NOISE_PHRASES = {
    "cookie policy", "cookie settings", "cookie preferences", "manage cookies",
    "accept cookies", "accept all", "reject all", "privacy policy",
    "privacy notice", "privacy statement", "your privacy", "data protection",
    "user agreement", "terms of service", "terms of use", "terms and conditions",
    "all rights reserved", "learn more", "read more", "show more", "see more",
    "sign in", "sign up", "log in", "create account", "join now", "get started",
    "contact us", "about us", "follow us", "skip to content", "personal information",
    # LinkedIn (and similar) comment-thread UI chrome, scraped alongside real
    # commenter names in post/comment text. "Like Reply" in particular reads
    # as a plausible 2-word capitalised name to looks_like_person_name(), so
    # without an exact-phrase block it slips straight through as a fake
    # person node (seen live: 9 separate LinkedIn posts, always this exact
    # button-label text, never an actual person).
    "like reply", "like comment", "love reply", "celebrate reply",
    "support reply", "insightful reply", "curious reply",
    "report this comment", "report this post", "report this",
    "to view or add a comment sign in", "more relevant posts",
}


def is_noise_name(name: str) -> bool:
    """True if `name` is scraped boilerplate/navigation chrome or a page-title
    artifact rather than a real named entity (e.g. "Cookie Policy", "User
    Agreement", "Drew Glover - LinkedIn", "Drew Glover - CEO.com").
    Deterministic — works with or without the Claude entity filter."""
    raw = (name or "").strip()
    if not raw:
        return True
    low = raw.lower()
    # embedded URL / domain / social handle => scraped chrome, not a name
    if ("http" in low or "www." in low or "@" in raw
            or re.search(r"\.(com|org|net|io|ai|co|gov|edu)\b", low)):
        return True
    # "Name - Site" / "Title | Source" / bulleted list artifacts: real personal
    # names never contain a spaced separator (hyphenated surnames have no spaces).
    if any(sep in raw for sep in (" - ", " | ", " – ", " — ", " · ", " • ", "•", "::")):
        return True
    # Multi-word ALL CAPS: a section header or status message run together
    # with real text on the same line, not a name -- e.g. "RESEARCH STARTER
    # Larry Ellison" on a biography page yielded "RESEARCH STARTER" as a
    # second person, with a fabricated relationship to whoever the bio was
    # actually about. Same rule providers/rosters.py's is_org_chart_label
    # already applies to roster pages; general prose extraction had no
    # equivalent guard.
    words = raw.split()
    if len(words) > 1 and raw == raw.upper() and any(c.isalpha() for c in raw):
        return True
    norm = normalize(name)
    if not norm:
        return True
    if norm in _NOISE_PHRASES:
        return True
    return any(tok in _NOISE_TOKENS for tok in norm.split())


def normalize(name: str) -> str:
    """Lowercase, fold diacritics, strip punctuation, collapse whitespace —
    the base dedup key. NFKD-decomposes accented characters into base
    letter + combining mark (e.g. "é" -> "e" + a combining acute accent)
    and drops the combining marks, so "José" and "Jose" -- scraped sources
    are inconsistent about diacritics -- collapse to the same key. A no-op
    for scripts without combining marks (CJK, Cyrillic, ...), so non-Latin
    names are unaffected."""
    if not name:
        return ""
    s = name.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def strip_middle_initials(name: str) -> str:
    """Drop single-letter middle initials and a trailing generational suffix
    so name variants collapse together.

    "John F. Kennedy" / "John F Kennedy" -> "John Kennedy". "John Smith" /
    "John Smith Jr." -> "John Smith" (scraped sources inconsistently
    include/omit Jr./Sr./II/III/IV — the same person shouldn't fork into two
    nodes over it). First and last (post-suffix) tokens are always kept;
    only interior single-letter tokens and a trailing suffix are removed.
    """
    parts = name.split()
    if len(parts) > 1 and parts[-1].rstrip(".").lower() in _GENERATIONAL_SUFFIXES:
        parts = parts[:-1]
    if len(parts) <= 2:
        return " ".join(parts) if parts else name.strip()
    kept = [parts[0]]
    for mid in parts[1:-1]:
        token = mid.rstrip(".")
        if len(token) <= 1:  # initial like "F" or "F."
            continue
        kept.append(mid)
    kept.append(parts[-1])
    return " ".join(kept)


def _resolve_diminutive(token: str) -> str:
    """Follow the diminutive map to its final form. Some entries chain
    (e.g. "phil" -> "philip" -> "phillip"); resolving once would leave "Phil"
    and "Philip" at different keys ("philip" vs "phillip"). Cycle-guarded,
    though the map is hand-authored and acyclic."""
    seen = set()
    while token in _DIMINUTIVES and token not in seen:
        seen.add(token)
        token = _DIMINUTIVES[token]
    return token


def person_norm_key(name: str) -> str:
    """Canonical dedup key for a person: normalised, middle-initials stripped,
    with the first name canonicalised through the diminutive map so nickname
    variants collapse ("Tim Cook" / "Timothy Cook" -> "timothy cook")."""
    base = normalize(strip_middle_initials(name))
    if not base:
        return ""
    parts = base.split()
    parts[0] = _resolve_diminutive(parts[0])
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


# Titles/honorifics that commonly stand in for a first name right before a
# surname ("President Trump", "Dr. Redfield") -- these must NOT trip the
# same-surname conflict check in mention_patterns, since they still refer to
# the one person being searched for, not a different same-surname relative.
# Deliberately common/generic, not exhaustive.
TITLE_WORDS = {
    "President", "Vice", "Senator", "Governor", "Mayor", "Secretary",
    "Director", "Chairman", "Chairwoman", "Chair", "Judge", "Justice",
    "General", "Admiral", "Colonel", "Captain", "Sergeant", "Officer",
    "Doctor", "Dr", "Professor", "Prof", "Mr", "Mrs", "Ms", "Miss",
    "Representative", "Congressman", "Congresswoman", "Ambassador",
    "Minister", "Prime", "King", "Queen", "Prince", "Princess", "Sir",
    "Dame", "Lord", "Lady", "Reverend", "Rev", "Father", "Sister", "Pastor",
    "Bishop", "Rabbi", "Imam", "CEO", "CFO", "COO", "Coach", "Agent",
    "Detective", "Lieutenant", "Commissioner", "Superintendent",
}


def mention_patterns(name: str, other_name: str = ""):
    """Returns (mention_pattern, conflict_pattern) for finding `name` in prose.

    mention_pattern matches either the full name or just its last token
    (surname) -- real prose re-mentions someone by surname alone after the
    first full mention ("Redfield" / "Trump", never "Robert R Redfield"
    again), and requiring the exact full name every time would miss almost
    every real sentence, including the one that actually states the
    relationship (a Wikipedia article body uses surnames throughout: "He was
    appointed to the post by President Donald Trump...").

    But a bare surname is genuinely ambiguous for anyone who shares it with
    someone else notable -- scanning a WHOLE article for "Trump" also matches
    "Ivanka Trump", "Trump Tower", "Fred Trump", none of which are Donald
    Trump. conflict_pattern matches a DIFFERENT full name sharing the same
    surname (a capitalized word immediately before it that isn't this person's
    own first name AND isn't a title/honorific like "President" -- "President
    Trump" is the same Donald Trump, not a different person); text matching
    that should be treated as probably about someone else, not silently
    trusted as a mention of this person. None when the name has no first name
    to compare against (a mononym), since there's nothing to distinguish it
    from in that case.

    other_name is a counterpart being searched for alongside this one (e.g.
    name_b, when this is name_a's pattern). Its first name is excluded from
    the conflict check too -- otherwise, whenever the two people share a
    surname (spouses, siblings, parent/child), the counterpart's own full-name
    mention ("Jane Smith" while building John Smith's pattern) would misfire
    the conflict check as if it named some unrelated third Smith, dropping
    every window that states the very relationship being searched for.

    Lives here rather than in its one original caller (graph.connect) because
    subject-window narrowing needs the identical question answered -- "is this
    span of prose about this person" -- and two implementations of that would
    drift.
    """
    tokens = name.split()
    surname = tokens[-1] if tokens else name
    firstname = tokens[0] if len(tokens) > 1 else None
    other_tokens = other_name.split()
    other_firstname = other_tokens[0] if len(other_tokens) > 1 else None
    alts = sorted({re.escape(name), re.escape(surname)}, key=len, reverse=True)
    mention = re.compile(r"\b(" + "|".join(alts) + r")\b", re.IGNORECASE)
    conflict = None
    if firstname:
        excluded_words = [firstname] + list(TITLE_WORDS)
        if other_firstname:
            excluded_words.append(other_firstname)
        excluded = "|".join(re.escape(w) for w in excluded_words)
        conflict = re.compile(
            r"\b(?!(?:" + excluded + r")\b)[A-Z][a-zA-Z'-]+\s+" + re.escape(surname) + r"\b")
    return mention, conflict
