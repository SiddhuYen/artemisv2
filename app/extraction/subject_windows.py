"""Narrow a scraped page to the passages actually about the subject.

Per-source Claude extraction (claude_extractor) reads a whole page -- up to
config.MAX_PAGE_CHARS, roughly 5k tokens -- to answer a question about ONE
person. On a real search result that is mostly waste: a conference speaker
roster, a news article about someone else that mentions the subject once, a
company blog with the subject in a single quote. The subject-relevant part is
usually a few sentences.

This module finds those sentences, pads each with `window` sentences either
side, merges the ones that overlap, and hands back just that. Two ways a
sentence qualifies:

  1. It names the subject -- full name or bare surname, via
     utils.names.mention_patterns (real prose re-mentions people by surname
     alone after the first full mention).

  2. It uses a gendered pronoun that COULD refer to the subject. This is the
     case a name-only scan misses entirely and it is not rare -- it is how
     English prose states most relationships across a sentence boundary
     ("Redfield became Director... He was appointed to the post by President
     Donald Trump..."), where the sentence carrying the actual relationship
     never says the subject's name at all.

Pronoun handling is a hardcoded walk, not a model call or a coreference
library -- and it deliberately answers "could this be the subject", not "who
is this". See _pronoun_could_be_subject: the first version picked a single
antecedent (the last gender-compatible name) and compared it to the subject,
which measured a 15% hit rate on two real biographies, because the name left
standing was so often an organisation ("joined Oracle. He led..."), an acronym
("at NIAID. He was awarded...") or the object of a by-phrase ("appointed by
Donald Trump. He served..."). The subject was a plausible antecedent in every
one of those; it just was not the last name standing. Since the decision here
is binary -- keep this sentence or drop it -- picking a winner was the wrong
shape for the question.

Three properties make the walk safe enough to gate spending on:

  - UNKNOWN GENDER FAILS OPEN. A candidate name whose gender we cannot
    determine is eligible for both "he" and "she". The alternative -- requiring
    a positive gender match -- would silently drop every name outside an
    English given-name lexicon, which is precisely the population this narrowing
    serves (see expansion.py's note on en_core_web_sm tagging "Molly Iyer" and
    "Prantik Chakraborty" as ORG). A wrong-but-open resolution costs a few
    hundred extra tokens; a wrong-and-closed one loses the relationship. The
    lexicon is therefore tuned for PRECISION, not coverage: an absent name
    fails open and costs nothing, while a wrong one closes the walk and loses
    a sentence, so unisex given names are deliberately left out of it.

  - Candidates come from CAPITALISATION, not spaCy NER, for the same reason.
    NER's person/org confusion on non-Anglo names would decide which pronouns
    resolve, putting that bias upstream of what we even send.

  - A LONE capitalised token that is not the subject is not treated as a
    person. Organisations, acronyms and stray capitalised nouns are exactly
    that shape ("Oracle", "NIAID", "Sciences"), and letting them count as
    antecedents is what made the walk stop on them.

The whole thing is skipped for short texts: below SUBJECT_WINDOW_MIN_CHARS the
input is a synthetic enrichment string (Wikidata evidence, a roster colleague
summary, an OpenAlex coauthor list) rather than a scraped page. Those are
already dense, already about the subject, and often never spell the subject's
name in a sentence at all -- narrowing them would be pure loss for no saving.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .. import config
from ..utils.names import (
    TITLE_WORDS,
    is_noise_name,
    looks_like_person_name,
    mention_patterns,
    normalize,
    person_norm_key,
)
from .spacy_extractor import split_sentences

MALE = "m"
FEMALE = "f"

_PRONOUN_GENDER = {
    "he": MALE, "him": MALE, "his": MALE, "himself": MALE,
    "she": FEMALE, "her": FEMALE, "hers": FEMALE, "herself": FEMALE,
}
_PRONOUN_RE = re.compile(
    r"\b(" + "|".join(sorted(_PRONOUN_GENDER, key=len, reverse=True)) + r")\b",
    re.IGNORECASE)

# Honorifics that carry gender on their own. Checked against the token
# immediately before a candidate name, so they beat the given-name lexicon
# below ("Ms. Andrea Cruz" is female regardless of how Andrea is listed).
_GENDERED_TITLES = {
    "mr": MALE, "sir": MALE, "lord": MALE, "king": MALE, "prince": MALE,
    "father": MALE, "brother": MALE, "congressman": MALE, "chairman": MALE,
    "spokesman": MALE, "businessman": MALE,
    "mrs": FEMALE, "ms": FEMALE, "miss": FEMALE, "lady": FEMALE,
    "queen": FEMALE, "princess": FEMALE, "dame": FEMALE, "sister": FEMALE,
    "congresswoman": FEMALE, "chairwoman": FEMALE, "spokeswoman": FEMALE,
    "businesswoman": FEMALE,
}

# Common English given names, used ONLY to rule a candidate OUT of being a
# pronoun's antecedent. Absence from these sets means "unknown", never "no" --
# see the module docstring on failing open. Kept deliberately small: this is a
# tiebreaker for sentences naming two people, not an identity oracle.
_MALE_NAMES = {
    "aaron", "adam", "adrian", "alan", "albert", "alexander", "alfred",
    "andrew", "anthony", "antonio", "arthur", "benjamin", "bernard", "bradley",
    "brandon", "brian", "bruce", "bryan", "carl", "carlos", "charles",
    "christopher", "clarence", "craig", "daniel", "david", "dennis", "derek",
    "donald", "douglas", "duane", "edward", "edwin", "elliot", "eric", "ernest",
    "eugene", "francis", "frank", "franklin", "frederick", "gabriel", "gary",
    "george", "gerald", "gilbert", "glenn", "gordon", "gregory", "harold",
    "harry", "henry", "herbert", "howard", "hugh", "ian", "jack", "jacob",
    "james", "jason", "jeffrey", "jeremy", "jerome", "jesse", "joel", "john",
    "jonathan", "jose", "joseph", "joshua", "juan", "keith", "kenneth", "kevin",
    "lawrence", "leonard", "louis", "luis", "marcus", "mark", "martin",
    "matthew", "maurice", "michael", "miguel", "nathan", "neil", "nicholas",
    "norman", "oliver", "oscar", "patrick", "paul", "peter", "philip", "ralph",
    "raymond", "richard", "robert", "roger", "ronald", "roy", "russell", "ryan",
    "samuel", "scott", "sean", "sergio", "seth", "stanley", "stephen", "steven",
    "stuart", "terry", "theodore", "thomas", "timothy", "todd", "victor",
    "vincent", "walter", "warren", "wayne", "wesley", "william", "zachary",
}
_FEMALE_NAMES = {
    "abigail", "alice", "alicia", "amanda", "amber", "amy", "andrea", "angela",
    "anita", "ann", "anna", "anne", "annette", "april", "audrey", "barbara",
    "beatrice", "bethany", "beverly", "bonnie", "brenda", "carol", "caroline",
    "carolyn", "catherine", "cheryl", "christina", "christine", "cindy",
    "claire", "clara", "colleen", "connie", "constance", "cynthia", "danielle",
    "dawn", "deborah", "debra", "denise", "diana", "diane", "dolores",
    "donna", "doris", "dorothy", "edith", "eileen", "elaine", "eleanor",
    "elizabeth", "ellen", "emily", "emma", "erica", "erin", "esther", "ethel",
    "evelyn", "florence", "frances", "gail", "gloria", "grace", "gwendolyn",
    "hannah", "heather", "helen", "holly", "irene", "isabel", "jacqueline",
    "jane", "janet", "janice", "jeanne", "jennifer", "jessica", "jill",
    "joan", "joanne", "josephine", "joyce", "juanita", "judith", "judy",
    "julia", "julie", "june", "karen", "katherine", "kathleen", "kathryn",
    "kimberly", "kristen", "laura", "lauren", "laurie", "lillian", "linda", "lisa", "lois", "loretta", "lori", "louise", "lucille",
    "lydia", "marcia", "margaret", "maria", "marie", "marilyn",
    "marjorie", "martha", "mary", "maureen", "megan", "melanie",
    "melissa", "michelle", "mildred", "molly", "monica", "nancy", "naomi",
    "natalie", "nicole", "nina", "norma", "olivia", "pamela", "patricia",
    "paula", "pauline", "peggy", "phyllis", "priscilla", "rachel", "rebecca",
    "regina", "renee", "rhonda", "rita", "roberta", "rosa", "rose",
    "ruby", "ruth", "sally", "samantha", "sandra", "sara", "sarah", "sharon",
    "sheila", "shirley", "sonia", "stephanie", "susan", "suzanne", "sylvia",
    "tammy", "teresa", "theresa", "tiffany", "tina", "valerie",
    "vanessa", "vera", "veronica", "vicki", "victoria", "violet", "virginia",
    "vivian", "wanda", "wendy", "yvonne",
}

# A capitalised run of 1..4 tokens (allowing &, ., -, ' so "O'Brien" and
# "D'Angelo" stay one token). Same shape heuristic.py's own candidate scan
# uses; single-token runs are kept here because a bare surname re-mention is
# exactly what a pronoun's antecedent usually is.
_CANDIDATE = re.compile(r"\b[A-Z][A-Za-z.&'\-]*(?:\s+[A-Z][A-Za-z.&'\-]*){0,3}\b")

# Capitalised words that start sentences or head clauses and are never names.
# Without this every sentence-initial "The"/"After"/"However" is a candidate
# antecedent, and since an unknown gender fails open, each one would swallow
# the walk on its first step and resolve every pronoun to nothing.
_NOT_A_NAME = {
    "the", "a", "an", "this", "that", "these", "those", "there", "their",
    "his", "her", "hers", "he", "she", "they", "them", "it", "its", "we",
    "our", "you", "your", "i", "my", "and", "but", "or", "nor", "so", "yet",
    "for", "from", "with", "without", "within", "as", "at", "by", "to", "in",
    "on", "of", "off", "up", "down", "over", "under", "into", "onto",
    "after", "before", "during", "while", "when", "where", "why", "how",
    "if", "then", "than", "because", "although", "though", "since", "until",
    "also", "however", "meanwhile", "later", "earlier", "previously",
    "following", "prior", "both", "each", "every", "all", "any", "some",
    "many", "most", "other", "another", "such", "no", "not", "now", "here",
    "one", "two", "three", "first", "second", "third", "next", "last",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "today", "yesterday", "tomorrow",
}

# Marks where text was cut out. Without it the model sees two distant passages
# glued together and can read a relationship across the seam that the page
# never stated -- the same fabrication the proximity gate exists to prevent.
_ELISION = "\n\n[...]\n\n"


@dataclass
class Focus:
    """What to send for one (subject, page), plus why.

    `text` is empty exactly when nothing on the page is about the subject --
    callers should skip the model call entirely rather than send it.
    """
    text: str
    total_sentences: int = 0
    anchors: List[int] = field(default_factory=list)
    pronoun_anchors: List[int] = field(default_factory=list)
    segments: int = 0
    reason: str = ""

    @property
    def empty(self) -> bool:
        return not self.text


# Normalised TITLE_WORDS, for stripping an honorific off the front of a
# capitalised run. _CANDIDATE swallows it ("Mr. Robert Redfield" is one match),
# which otherwise defeats BOTH gender paths at once: the honorific is no longer
# "the token before the name" for the title check, and person_norm_key sees a
# first name of "mr" so the lexicon misses too.
_TITLE_TOKENS = {normalize(t) for t in TITLE_WORDS} | set(_GENDERED_TITLES)


# Institutional words that mark a multi-word capitalised run as an org rather
# than a person. ORG_SUFFIXES (utils.names) covers legal forms -- Inc, LLC,
# University -- but an internal unit carries none of those and sails through
# looks_like_person_name: "LCI's Clinical Physiology Section" is three
# capitalised words with no suffix and no stopword, i.e. person-shaped.
_ORG_WORDS = {
    "section", "division", "department", "institute", "institutes",
    "laboratory", "laboratories", "center", "centre", "office", "bureau",
    "agency", "committee", "council", "board", "program", "programme", "unit",
    "school", "college", "hospital", "clinic", "ministry", "commission",
    "association", "society", "federation", "academy", "faculty", "service",
    "services", "administration", "authority", "trust", "fund", "press",
    "museum", "library", "branch", "directorate", "secretariat", "network",
    "networks", "group", "team", "project", "initiative", "coalition",
}


def _is_org_phrase(name: str) -> bool:
    """True for a capitalised run that reads as an institution, not a person."""
    tokens = name.split()
    # A possessive inside a MULTI-token run is an institutional construction
    # ("NIAID's Laboratory"). Single tokens are left alone: "Fauci's" is the
    # subject's own name in the possessive, not an organisation.
    if len(tokens) > 1 and any("'s" in t or "’s" in t for t in tokens):
        return True
    return any(normalize(t) in _ORG_WORDS for t in tokens)


@dataclass(frozen=True)
class _Mention:
    """One capitalised run, cleaned up and classified."""
    name: str
    gender: Optional[str]
    is_person_shaped: bool


def _strip_leading_noise(phrase: str) -> str:
    """Drop leading/trailing non-name capitalised words ("After Smith" -> "Smith")."""
    parts = phrase.split()
    while parts and normalize(parts[0]) in _NOT_A_NAME:
        parts.pop(0)
    while parts and normalize(parts[-1]) in _NOT_A_NAME:
        parts.pop()
    return " ".join(parts).strip(" .,'\"")


def _split_honorific(phrase: str):
    """('Mr. Robert Redfield') -> ('Robert Redfield', MALE).

    Returns (remaining_name, gender_from_honorific). Strips every leading title
    token, keeping the gender of the last gendered one seen -- 'President' and
    'Dr' carry none, 'Ms' does.
    """
    parts = phrase.split()
    gender = None
    while parts:
        head = normalize(parts[0])
        if head not in _TITLE_TOKENS:
            break
        gender = _GENDERED_TITLES.get(head, gender)
        parts.pop(0)
    return " ".join(parts), gender


def name_gender(name: str, sentence: str = "", at: int = -1) -> Optional[str]:
    """MALE / FEMALE when determinable, None when not.

    None means UNKNOWN, and every caller treats unknown as compatible with
    either pronoun -- see the module docstring. An honorific carried on the
    name itself, or sitting immediately before it in `sentence`, wins over the
    given-name lexicon.
    """
    stripped, honorific = _split_honorific(name)
    if honorific:
        return honorific
    if sentence and at > 0:
        tokens = sentence[:at].rstrip().replace(".", " ").split()
        if tokens and normalize(tokens[-1]) in _GENDERED_TITLES:
            return _GENDERED_TITLES[normalize(tokens[-1])]
    parts = person_norm_key(stripped or name).split()
    if parts:
        first = parts[0]
        if first in _MALE_NAMES:
            return MALE
        if first in _FEMALE_NAMES:
            return FEMALE
    return None


def _mentions(span: str) -> List[_Mention]:
    """Capitalised runs in `span`, cleaned, gendered and shape-classified.

    `is_person_shaped` marks a run that looks like somebody's name on its own
    (looks_like_person_name: 2-4 capitalised words, no org suffix, no
    stopwords). Single tokens are never person-shaped, which is the point: an
    org, an acronym or a stray capitalised noun -- 'Oracle', 'NIAID',
    'Sciences' -- is exactly a lone capitalised token, and treating those as
    people is what made the walk stop on them.
    """
    out: List[_Mention] = []
    for match in _CANDIDATE.finditer(span):
        cleaned = _strip_leading_noise(match.group(0))
        stripped, honorific = _split_honorific(cleaned)
        stripped = stripped or cleaned
        if len(stripped) < 2 or is_noise_name(stripped):
            continue
        gender = honorific or name_gender(stripped, span, match.start())
        person_shaped = (looks_like_person_name(stripped)
                         and not _is_org_phrase(stripped))
        out.append(_Mention(name=stripped, gender=gender,
                            is_person_shaped=person_shaped))
    return out


def _pronoun_could_be_subject(sentences: List[str], index: int, gender: str,
                              lookback: int, is_subject,
                              chained: Optional[dict] = None) -> bool:
    """Could this pronoun refer to the subject?

    Deliberately NOT coreference resolution. The question this stage has to
    answer is binary -- keep this sentence or drop it -- and picking exactly
    one antecedent to answer it was the wrong shape. Measured on two real
    biographies, the pick-one walk resolved 15% of eligible pronouns to the
    subject, because the losing candidate was so often an organisation
    ('joined Oracle. He led...'), an acronym ('at NIAID. He was awarded...')
    or the object of a by-phrase ('appointed by Donald Trump. He served...').
    In every one of those the subject WAS a plausible antecedent; it just was
    not the last name standing.

    So: walk back to the nearest sentence that offers any plausible antecedent
    -- a name that looks like a person, or any form of the subject's own name
    -- and answer whether the subject is among them. A sentence whose only
    candidates are lone capitalised tokens that are not the subject offers
    nothing and the walk continues past it, which is what stops 'Oracle' and
    'NIAID' from swallowing the question.

    The same sentence is searched too, on both sides of the pronoun, so a
    forward reference resolves ("In her role at Acme, Sandra Whitfield led
    sales") instead of walking backwards past its own answer.

    `chained` maps an earlier sentence index to the pronoun gender that
    resolved it, and lets a run of pronoun-only sentences hold together.
    Biographies write long stretches that never repeat the name -- "He became
    head of the section in 1974. He became director of the NIAID in 1984." --
    and without chaining the second sentence walks back over a first that
    names nobody either, runs out of lookback, and is dropped even though its
    neighbour was already established as being about the subject. The gender
    has to match the one that anchored the earlier sentence: a "she" following
    a run of "he" is a new referent, not a continuation.
    """
    chained = chained or {}
    for j in range(index, max(-1, index - lookback - 1), -1):
        # An earlier sentence already established as the subject's, by a
        # pronoun of this same gender, supplies the subject even when it names
        # nobody at all. Never the sentence being decided -- it cannot be its
        # own antecedent.
        if j != index and chained.get(j) == gender:
            return True

        plausible = [
            m for m in _mentions(sentences[j])
            if m.is_person_shaped or is_subject(m.name)
        ]
        # "the last name said WITH THAT GENDER": a candidate positively of the
        # other gender is not an antecedent for this pronoun, and a sentence
        # left with none is not an answer -- keep walking.
        compatible = [m for m in plausible
                      if m.gender is None or m.gender == gender]
        if any(is_subject(m.name) for m in compatible):
            return True

        # The pronoun's OWN sentence may CONFIRM (a forward reference) but must
        # never VETO. Names sitting beside a pronoun are usually its objects,
        # and in institutional prose usually not people at all -- "He became
        # head of the LCI's Clinical Physiology Section in 1974" would
        # otherwise answer itself, on org fragments, without ever looking back
        # at the sentence that names the subject.
        if j == index:
            continue
        if compatible:
            return False
    return False


def focus(subject: str, text: str, window: Optional[int] = None) -> Focus:
    """Narrow `text` to the passages about `subject`.

    Returns a Focus whose `.text` is what should be sent to the model. An
    empty `.text` means the page has no subject-relevant passage at all.
    """
    if not subject or not text:
        return Focus(text=text, reason="no subject or no text")
    if not config.SUBJECT_WINDOW_ENABLED:
        return Focus(text=text, reason="disabled")
    if len(text) < config.SUBJECT_WINDOW_MIN_CHARS:
        return Focus(text=text, reason="short text (enrichment string)")

    sentences = split_sentences(text)
    if len(sentences) < 2:
        return Focus(text=text, total_sentences=len(sentences),
                     reason="not segmentable into sentences")

    span = window if window is not None else config.SUBJECT_WINDOW_SENTENCES
    mention, conflict = mention_patterns(subject)
    full_name = re.compile(r"\b" + re.escape(subject) + r"\b", re.IGNORECASE)

    def _is_subject(name: str) -> bool:
        """Any accepted form of the subject's name: the full name outright, or
        a bare surname that no same-surname relative has claimed."""
        if full_name.search(name):
            return True
        return bool(mention.search(name)
                    and not (conflict and conflict.search(name)))

    # --- anchors by name ---------------------------------------------------
    # A sentence spelling the subject's FULL name is unambiguous and is kept
    # even when a same-surname relative appears in it too -- the conflict
    # pattern exists to disown a bare-surname match, not to veto a mention
    # that already named the person outright.
    direct = {i for i, sentence in enumerate(sentences) if _is_subject(sentence)}

    # --- anchors by resolved pronoun ---------------------------------------
    # Front to back, so `chained` only ever holds sentences EARLIER than the
    # one being decided -- which is the only direction the walk looks.
    pronoun_hits = set()
    chained: dict = {}
    lookback = config.SUBJECT_WINDOW_PRONOUN_LOOKBACK
    for i, sentence in enumerate(sentences):
        if i in direct:
            continue  # already anchored; nothing a pronoun could add
        for match in _PRONOUN_RE.finditer(sentence):
            gender = _PRONOUN_GENDER[match.group(1).lower()]
            if _pronoun_could_be_subject(sentences, i, gender, lookback,
                                         _is_subject, chained):
                pronoun_hits.add(i)
                chained[i] = gender
                break

    anchors = sorted(direct | pronoun_hits)
    if not anchors:
        return Focus(text="", total_sentences=len(sentences),
                     reason="subject never mentioned or referred to")

    # --- pad, then merge overlapping/adjacent windows -----------------------
    spans: List[List[int]] = []
    for i in anchors:
        lo, hi = max(0, i - span), min(len(sentences) - 1, i + span)
        # `<= last_hi + 1` merges ADJACENT windows too, not just overlapping
        # ones: leaving a one-sentence hole would insert an elision marker
        # around text we are about to include anyway. Bounded by
        # SUBJECT_WINDOW_MAX_MERGED_SENTENCES so a run of frequent, close-
        # together anchors (a repeating byline on a listing/archive page)
        # cannot chain indefinitely and swallow unrelated text between them
        # with no elision marker to flag the seam -- see its config comment.
        if (spans and lo <= spans[-1][1] + 1
                and max(spans[-1][1], hi) - spans[-1][0] + 1
                <= config.SUBJECT_WINDOW_MAX_MERGED_SENTENCES):
            spans[-1][1] = max(spans[-1][1], hi)
        else:
            spans.append([lo, hi])

    narrowed = _ELISION.join(
        " ".join(sentences[lo:hi + 1]) for lo, hi in spans)

    if len(narrowed) >= len(text):
        # Windows covered the whole page. Send the original: it is no larger,
        # and it has no elision markers to explain.
        return Focus(text=text, total_sentences=len(sentences), anchors=anchors,
                     pronoun_anchors=sorted(pronoun_hits), segments=len(spans),
                     reason="windows cover the page")

    return Focus(text=narrowed, total_sentences=len(sentences), anchors=anchors,
                 pronoun_anchors=sorted(pronoun_hits), segments=len(spans),
                 reason="narrowed")


__all__ = ["Focus", "focus", "name_gender", "MALE", "FEMALE"]
