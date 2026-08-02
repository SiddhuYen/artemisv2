"""Shared roster-page machinery: is this page a list of an organization's
own people, and which names does it actually assert?

Extracted verbatim from firms.py, which built and hardened all of it for VC
team pages. Nothing here is VC-specific -- a law firm's attorney directory,
a hospital's provider list and a consultancy's leadership page are the same
problem -- so it lives here, where both the person-keyed firm lookup
(firms.FirmsProvider) and the org-keyed directory lookup
(directory.DirectoryProvider) can share ONE implementation of the guards.

That sharing is the point. These guards each cost a real bug to discover,
and a second, parallel copy written for directories would have re-earned
every one of them:

  1. A page must LOOK like a roster (`/team`, `/people`, ...), not a
     homepage -- a homepage interleaves staff with quoted customers or
     portfolio founders, and neither NER nor name-shape filtering can tell
     them apart.
  2. A page must BELONG to the organization, established by IDENTITY (the
     domain, or the name the page declares for itself) rather than keyword
     presence. "Storm Ventures" is a substring of a rival firm's domain
     "calmstorm.vc", and a search for "Homebrew team page" returns the
     package manager's cask index -- keyword presence alone attaches the
     wrong roster.

Guard 3 (the subject must actually be NAMED on the roster) is not here: it
belongs to the caller, because the two callers want different things from
it. firms.py requires it, since it is answering "who does this PERSON work
with". directory.py deliberately does not, since it is answering "who works
at this ORG" -- and it compensates by emitting weaker, org-membership-only
edges when the subject is absent (see directory.DirectoryProvider).
"""
from __future__ import annotations

import re
from typing import Iterable, List
from urllib.parse import urlparse

from ..utils.htmltext import soup_of, text_blocks
from ..utils.names import (
    is_noise_name,
    looks_like_person_name,
    normalize,
    org_norm_key,
    person_norm_key,
)
from .base import Page, fetch_page

# Path segments that mark a page as a roster of people.
#
# Deliberately EXCLUDES "/about" and "/founders": an about page interleaves
# the team with portfolio companies (proper nouns that look like people to a
# shape-only filter), so treating it as a roster invites false members.
ROSTER_HINTS = ("team", "people", "our-team", "ourteam", "partners", "staff",
                "leadership", "who-we-are", "whoweare", "members",
                "our-firm", "ourfirm", "crew", "humans")

# The same structural assertion under the words a law firm, a hospital or a
# university uses for it. Kept SEPARATE from ROSTER_HINTS, and opted into via
# is_roster_url(extra_hints=...), so that broadening what counts as a roster
# is a decision directory.py makes for itself rather than one silently
# inherited by firms.py, whose behavior this extraction leaves untouched.
DIRECTORY_HINTS = ("attorneys", "lawyers", "professionals", "physicians",
                   "providers", "doctors", "faculty", "directory", "advisors",
                   "consultants", "bankers", "agents", "clinicians")

# Never a roster, even when a hint appears elsewhere in the path.
#
# The second group is what makes this usable for technology companies at all.
# In software, "team" overwhelmingly names a PRODUCT FEATURE rather than a
# staff list, and those pages live on the company's own domain -- so Guard 2
# cannot tell them apart, because they genuinely do belong to the company.
# Measured live: /docs/cli/teams and /docs/rest-api/teams/list-team-members
# (Vercel), /get-started/account/teams (Stripe), /docs/teams (Linear),
# /integrations/microsoft-teams (Retool), /community/file/... (a Figma
# template). Every one passed both guards and yielded product copy.
NEGATIVE_HINTS = ("portfolio", "blog", "post", "news", "careers", "jobs",
                  "contact", "privacy", "terms", "press", "insights",
                  "docs", "documentation", "help", "support", "changelog",
                  "integrations", "community", "api", "reference", "guides",
                  "tutorial", "pricing", "download", "signup", "login",
                  "get-started", "getting-started", "faq", "status",
                  "dashboard", "resources", "forum", "discuss", "answers",
                  "article", "articles", "guide", "kb", "knowledge")

# The same exclusions, as a HOSTNAME prefix. A company's product docs and
# user forum usually live on their own subdomain rather than under a path,
# so a path-only check misses them: docs.zapier.com/platform/manage/add-team
# and forum.ghost.org/t/... both passed Guard 1 and Guard 2, being genuinely
# the company's own domain, and were scraped as staff rosters.
NEGATIVE_HOST_PREFIXES = ("docs.", "help.", "support.", "forum.", "community.",
                          "developer.", "developers.", "api.", "status.",
                          "blog.", "learn.", "academy.", "answers.", "faq.",
                          "kb.", "knowledge.")

# Aggregators and socials: real pages, but never the org's own roster.
BLOCKED_HOSTS = ("linkedin.com", "twitter.com", "x.com", "facebook.com",
                 "crunchbase.com", "pitchbook.com", "dealroom.co", "f6s.com",
                 "reddit.com", "wikipedia.org", "medium.com", "youtube.com")

# Tokens too generic to identify an organization on a page.
GENERIC_ORG_TOKENS = {"ventures", "capital", "partners", "fund", "funds",
                      "group", "management", "the", "and", "vc", "llc", "lp"}

# Leading role words scraped rosters glue onto a name in one text node
# ("Partner Alex Harris" -> "Alex Harris"). Stripped before judging name
# shape, not treated as disqualifying -- outright rejection silently drops
# real team members.
ROLE_PREFIXES = (
    "general partner", "managing partner", "venture partner", "founding partner",
    "operating partner", "managing director", "senior partner",
    "partner", "principal", "associate", "analyst",
    "chief executive officer", "chief operating officer", "chief financial officer",
    "co-founder", "cofounder", "founder", "president", "chairman", "chairwoman",
    "ceo", "coo", "cfo", "cto", "gp", "vp", "svp", "evp",
)

TLD_TOKENS = {"co", "com", "vc", "io", "ai", "net", "org", "fund", "capital"}


def host_of(url: str) -> str:
    # NB: not .lstrip("www."), which strips any leading 'w'/'.' characters and
    # would turn "wework.com" into "ework.com".
    host = (urlparse(url).netloc or "").lower()
    return host[4:] if host.startswith("www.") else host


# Two-label public suffixes, where the registrable domain is the THIRD label
# from the right ("firm.co.uk" -> "firm"). Not the full Public Suffix List --
# that needs a dependency and monthly updates -- just the ones a company team
# page realistically sits on.
_MULTI_PART_SUFFIXES = {
    "co.uk", "ac.uk", "org.uk", "gov.uk", "me.uk", "co.jp", "or.jp", "ne.jp",
    "co.in", "co.nz", "co.za", "co.kr", "com.au", "net.au", "org.au", "edu.au",
    "com.br", "com.mx", "com.sg", "com.hk", "com.tw", "com.cn", "edu.cn",
}


def host_labels(url: str) -> List[str]:
    """Every label of the host, cleaned ("www.gtri.gatech.edu" -> [gtri, gatech, edu])."""
    return [re.sub(r"[^a-z0-9]", "", p) for p in host_of(url).split(".") if p]


def domain_stem(url: str) -> str:
    """The REGISTRABLE domain's own label.

    "www.hustlefund.vc" -> "hustlefund"; "btv.vc" -> "btv";
    "research.gatech.edu" -> "gatech"; "team.firm.co.uk" -> "firm".

    This used to return the FIRST label, which is correct only for a bare
    two-label host and silently wrong for every subdomain: "research.gatech.
    edu" yielded "research", "af.gatech.edu" yielded "af". Guard 2 compares
    this against the org's own name tokens, so any organization serving its
    team page from a subdomain -- which large ones usually do -- could never
    verify. Live, all three of Georgia Tech's legitimate leadership pages
    were rejected this way.
    """
    parts = host_labels(url)
    if not parts:
        return ""
    if len(parts) <= 2:
        return parts[0]
    if ".".join(parts[-2:]) in _MULTI_PART_SUFFIXES:
        return parts[-3]
    return parts[-2]


def is_roster_url(url: str, extra_hints: Iterable[str] = ()) -> bool:
    """Guard 1. True when the URL path looks like a team/people page (not a
    homepage).

    `extra_hints` widens the accepted vocabulary for THIS call only -- see
    DIRECTORY_HINTS. Callers that don't pass it get exactly the VC-shaped
    behavior this function has always had.
    """
    if not url:
        return False
    host = host_of(url)
    if any(bad in host for bad in BLOCKED_HOSTS):
        return False
    if host.startswith(NEGATIVE_HOST_PREFIXES):
        return False
    path = (urlparse(url).path or "/").strip("/").lower()
    if not path:
        return False  # bare homepage
    if any(neg in path for neg in NEGATIVE_HINTS):
        return False
    hints = tuple(ROSTER_HINTS) + tuple(extra_hints)
    return any(hint in path.split("/") or hint in path for hint in hints)


# The narrow subset of roster paths that specifically assert LEADERSHIP, as
# opposed to a full staff list. Used to keep a large org's directory lookup to
# its exec page -- see config.DIRECTORY_FULL_SIZE_TIERS.
LEADERSHIP_HINTS = ("leadership", "executives", "executive-team", "executive",
                    "management", "officers", "senior-team", "our-leaders")


def is_leadership_url(url: str) -> bool:
    """True for a roster page that specifically asserts an org's leadership.

    Strictly narrower than is_roster_url: every leadership page is a roster
    page, but "/staff" or "/people" is a full directory, which is exactly
    what a large org must NOT have pulled in wholesale.
    """
    if not is_roster_url(url):
        return False
    path = (urlparse(url).path or "").strip("/").lower()
    return any(hint in path for hint in LEADERSHIP_HINTS)


def org_tokens(org_name: str) -> set:
    """Distinctive words of an org name ("Uncork Capital" -> {"uncork"})."""
    return {t for t in normalize(org_name).split()
            if t and t not in GENERIC_ORG_TOKENS and len(t) > 2}


def page_title(html: str) -> str:
    title = soup_of(html).title
    return title.get_text(" ", strip=True) if title else ""


def org_name_from_page(html: str, url: str, allow_stem_fallback: bool = True) -> str:
    """Display name of the organization behind a roster page, VERIFIED by the
    domain.

    A <title> is often "BTV | Sheel Mohnot" or "Our team | Hustle Fund".
    Taking the longest segment risks a person's name or a tagline, so only
    the title segment whose letters match the registrable domain is
    accepted. Falls back to the domain stem, which is always at least honest.
    """
    stem = domain_stem(url)
    title = page_title(html)
    for segment in re.split(r"[|\-–—:·]", title):
        segment = segment.strip()
        if not segment or looks_like_person_name(segment):
            continue
        tokens = [t for t in normalize(segment).split() if t]
        while tokens and tokens[-1] in TLD_TOKENS and len(tokens) > 1:
            tokens.pop()
        key = "".join(tokens)
        if key and stem and (key == stem or key in stem or stem in key):
            display = re.sub(r"\.(co|com|vc|io|ai|net|org)$", "", segment,
                             flags=re.I).strip(" .")
            return display or segment.strip()
    # Falling back to the domain stem is fine for display, but a CALLER
    # verifying identity must be able to opt out: "the name this page
    # declares" would otherwise just be the domain again, and comparing that
    # to the org name re-runs the domain check under another name. That
    # circularity is how org "GA" verified against doas.ga.gov even after
    # the domain branch was length-guarded against exactly that match.
    if not allow_stem_fallback:
        return ""
    return stem.title() if stem else ""


def page_belongs_to_org(url: str, html: str, org_name: str,
                        official_domain: str = "") -> bool:
    """Guard 2. The page must BE this organization's, established by identity
    -- the domain, or the name the page declares for itself -- never keyword
    presence.

    So: the domain must begin with a distinctive token of the org's name, or
    the page's own declared name must equal that org (allowing an initialism,
    since "btv.vc" declares itself "BTV" and means Better Tomorrow Ventures).

    `official_domain` is the registrable domain of the org's Wikidata-declared
    official website (P856), supplied by the caller -- this module makes no
    network calls. It settles what string comparison cannot: "gatech.edu" is
    not derivable from "Georgia Institute of Technology" by any prefix or
    initialism rule, since it contracts the state's postal abbreviation. It is
    checked FIRST and is purely additive: Wikidata can carry a stale value
    (Uncork Capital still lists softtechvc.com), so a mismatch here must never
    veto a page the name checks below would have accepted on their own.
    """
    tokens = org_tokens(org_name)
    stem = domain_stem(url)
    if official_domain and stem and stem == official_domain:
        return True
    if not tokens:
        # No distinctive tokens at all -- either a very short name ("GA") or
        # one made entirely of generic words ("The Fund", "Capital Partners").
        #
        # This used to `return True`, on the reasoning "nothing to check
        # against, fall back to Guard 1". That silently switched the identity
        # guard OFF for precisely the names most likely to collide with
        # something unrelated. Live consequence: an org extracted as "GA"
        # accepted https://doas.ga.gov/leadership-council -- the Georgia
        # state government's leadership council -- and 11 of its staff were
        # written into the graph as the subject's colleagues' employer.
        #
        # Fail closed instead: with nothing distinctive to match on, only an
        # EXACT identity match is good enough.
        # A 3-character minimum, because this branch exists for names with
        # nothing distinctive about them and a 2-character "identity" is
        # coincidence, not evidence: org "GA" exact-matches the registrable
        # label of ga.gov, which is how the Georgia state government's
        # leadership council attached to it in the first place.
        compact = normalize(org_name).replace(" ", "")
        if compact and stem and compact == stem and len(compact) >= 3:
            return True
        declared = org_name_from_page(html, url, allow_stem_fallback=False)
        return bool(declared) and org_norm_key(declared) == org_norm_key(org_name)

    # Match against the registrable domain AND every other host label, so a
    # business unit that IS the subdomain verifies too ("gtri.gatech.edu" for
    # GTRI, not only for its parent). Still bounded by `tokens` being
    # distinctive: "calmstorm.vc" does not match "Storm Ventures", since
    # neither string is a prefix of the other.
    labels = [lbl for lbl in host_labels(url) if lbl and lbl not in TLD_TOKENS]
    domain_hit = any(lbl.startswith(tok) or tok.startswith(lbl)
                     for lbl in labels for tok in tokens)
    # A bare domain match settles a single-token org ("Homebrew" -> homebrew.co).
    # It does NOT settle a multi-word one -- a business unit of a much bigger
    # company can share the parent's domain stem while being a different desk.
    #
    # And it must be the WHOLE label, not a prefix of it. A short company name
    # is a prefix of plenty of unrelated domains: "Ramp" (the fintech) matched
    # rampinteractive.com, a sports-league software company, and scraped its
    # roster. A longer domain that merely starts with the org's only
    # distinctive token is a different company far more often than it is the
    # same one.
    # The label must equal the token, or the org's whole name compacted --
    # NOT merely start with it. "Uncork Capital" -> uncorkcapital.com is the
    # ordinary name-as-domain pattern and must pass; "Ramp" -> rampinteractive
    # .com is a different company that happens to share a prefix, and did get
    # its roster scraped.
    if domain_hit and len(tokens) == 1:
        only = next(iter(tokens))
        compact_name = normalize(org_name).replace(" ", "")
        if any(lbl == only or lbl == compact_name for lbl in labels):
            return True

    declared = org_name_from_page(html, url)
    if not declared:
        return False
    if org_norm_key(declared) == org_norm_key(org_name):
        return True
    initials = "".join(word[0] for word in normalize(org_name).split() if word)
    return normalize(declared).replace(" ", "") == initials


def strip_role_prefix(text: str) -> str:
    t = text.strip()
    low = t.lower()
    for role in sorted(ROLE_PREFIXES, key=len, reverse=True):
        if low.startswith(role + " "):
            return t[len(role):].strip(" ,.-")
    return t


# Org-chart vocabulary: department names, org units, and role nouns. A
# directory page interleaves these with actual people ("Executive Team",
# "Human Resources Administration", "President's Cabinet" as section
# headings), and two capitalised words is all `looks_like_person_name` needs
# -- so they were being written into the graph as PEOPLE with employment
# edges. Live, the St. Thomas directory yielded exactly one "member":
# "President's Cabinet".
_ORG_CHART_WORDS = {
    # organizational units
    "team", "teams", "cabinet", "committee", "council", "board", "department",
    "division", "office", "bureau", "administration", "group", "staff",
    "leadership", "management", "services", "service", "operations", "affairs",
    "resources", "relations", "communications", "development", "programs",
    # functional areas
    "technology", "information", "finance", "accounting", "marketing", "sales",
    "engineering", "legal", "compliance", "human", "public", "corporate",
    # role nouns
    "executive", "senior", "deputy", "assistant", "associate", "interim",
    "acting", "chief", "officer", "director", "manager", "president", "vice",
    "commissioner", "secretary", "treasurer", "chair", "chairman", "chairwoman",
    "coordinator", "administrator", "supervisor", "analyst", "specialist",
    "representative", "member", "members", "partner", "partners", "principal",
    "counsel", "advisor", "adviser", "consultant", "dean", "provost",
    # connectives / possessive remnant
    "and", "of", "the", "for", "s",
}


# Page furniture and marketing copy that survives the name-shape filter
# because it is simply two or three capitalised words. Observed live on real
# roster pages: "GET YOUR TICKET!", "ROSTER LOADED", "Continuous Integration".
_PAGE_FURNITURE_WORDS = {
    "get", "your", "our", "ticket", "tickets", "roster", "loaded", "menu",
    "search", "login", "sign", "subscribe", "newsletter", "read", "more",
    "learn", "view", "apply", "join", "contact", "email", "share", "follow",
    "continuous", "integration", "delivery", "product", "design", "data",
    "quality", "security", "support", "success", "platform", "solutions",
    "lead", "leads", "hiring", "careers", "culture", "values", "mission",
    "us", "we", "here", "now", "all", "home", "about", "back", "next", "close",
    "project", "projects", "ops", "account", "accounts", "portfolio", "press",
    "notice", "notices", "info", "privacy", "terms", "copyright", "sitemap",
    "accessibility", "policy", "cookie", "cookies", "disclaimer", "legal",
    "bar", "admissions", "spotlight", "priority", "priorities", "attorney",
    "attorneys", "overview", "events", "insights", "publications", "awards",
    "recognition", "rankings", "practice", "practices", "industries", "offices",
    "portal", "directory", "directories", "profile", "profiles", "bio", "bios",
    # Job titles. A team page frequently labels each person with their role,
    # and some list roles INSTEAD of names (doist.com/team yields "Data
    # Engineer", "Motion Design", "Backend Development" and no people at all).
    "engineer", "engineers", "developer", "developers", "designer", "designers",
    "scientist", "scientists", "architect", "recruiter", "strategist", "writer",
    "editor", "marketer", "marketers", "frontend", "backend", "fullstack",
    "motion", "brand", "lifecycle", "growth", "content", "creative", "web",
    "mobile", "cloud", "infrastructure", "reliability", "founding",
    "customer", "customers", "experience", "android", "ios", "qa", "devops",
    "sre", "onboarding", "enablement", "partnerships", "community",
}

# A person's name does not contain these. Digits, terminal punctuation and
# separators mark a heading, a CTA, or a run-together text node. A comma is
# included ("Washington, D.C.") but NOT a bare period, which would reject
# every middle initial ("M. Joseph Sirgy").
_NON_NAME_CHARS = re.compile(r"[0-9!?:;,|/@#$%•·]")

# Office locations, which professional directories list as sibling text nodes
# right beside the people ("Demian Ahn", "Palo Alto", "Josephine Aiello
# LeBeau", ...). Only MULTI-WORD cities are listed: single-word ones like
# Austin, Dallas, Savannah or Paris are also perfectly ordinary given names,
# and rejecting those would cost real people.
_OFFICE_CITIES = {
    "palo alto", "new york", "san francisco", "los angeles", "san diego",
    "hong kong", "salt lake city", "washington dc", "district of columbia",
    "menlo park", "mountain view", "santa monica", "san jose", "las vegas",
    "buenos aires", "sao paulo", "tel aviv", "abu dhabi", "kuala lumpur",
    "new delhi", "san mateo", "redwood city", "century city", "costa mesa",
}


def is_org_chart_label(name: str) -> bool:
    """True when `name` is a section heading, role, or page furniture rather
    than a person.

    Three independent rejects:

    1. Every token is org-chart / page-furniture vocabulary. Deliberately
       requires *all* tokens, not any: plenty of real surnames are also role
       nouns (Dean, Chase, Marshall, Bishop, Steward), so an any-token rule
       would reject "Dean Martin" as a job title. Requiring the whole string
       keeps "Dean Martin" while dropping "Deputy Commissioner".
       Tokens are singularised first, so "Product Managers" and "Delivery
       Leads" match the same vocabulary as their singular forms.
    2. Contains a character no personal name carries (digit, !, ?, :, |, @...).
    3. Multi-word ALL CAPS -- "ROSTER LOADED" is a status message, and a
       roster that genuinely upper-cases its people still yields them via the
       JSON-LD path, which is preferred anyway.
    """
    raw = (name or "").strip()
    if not raw:
        return False
    if _NON_NAME_CHARS.search(raw):
        return True
    words = raw.split()
    if len(words) > 1 and raw == raw.upper() and any(c.isalpha() for c in raw):
        return True
    tokens = [t for t in normalize(raw).split() if t]
    if not tokens:
        return False
    if " ".join(tokens) in _OFFICE_CITIES:
        return True
    vocab = _ORG_CHART_WORDS | _PAGE_FURNITURE_WORDS
    singular = [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in tokens]
    return all(t in vocab or s in vocab for t, s in zip(tokens, singular))


def clean_roster_names(candidates: List[str]) -> List[str]:
    """Deterministic name-shape filter over roster text blocks; dedups on the
    person key. Never an LLM -- see utils/names.looks_like_person_name."""
    seen, out = set(), []
    for raw in candidates:
        name = strip_role_prefix((raw or "").strip())
        if is_noise_name(name) or not looks_like_person_name(name):
            continue
        if is_org_chart_label(name):
            continue
        key = person_norm_key(name)
        if key and key not in seen:
            seen.add(key)
            out.append(name)
    return out


def fetch_readable(url: str) -> Page:
    """Fetch `url`, falling back to a headless render if the plain GET returns
    a JavaScript shell (no readable text blocks). Rendering is optional: when
    the browser is unavailable this is just the plain fetch."""
    page = fetch_page(url)
    if page.status_code == 200 and text_blocks(page.content):
        return page
    from .browser import available as _browser_available
    if _browser_available():
        rendered = fetch_page(url, render=True)
        if rendered.content and text_blocks(rendered.content):
            return rendered
    return page
