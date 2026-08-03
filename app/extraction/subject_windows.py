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

  2. It uses a gendered pronoun whose antecedent resolves to the subject.
     This is the case a name-only scan misses entirely and it is not rare --
     it is how English prose states most relationships across a sentence
     boundary ("Redfield became Director... He was appointed to the post by
     President Donald Trump..."), where the sentence carrying the actual
     relationship never says the subject's name at all.

Antecedent resolution is deliberately a hardcoded walk, not a model call or a
coreference library: step back sentence by sentence and take the last-mentioned
name compatible with the pronoun's gender (see _antecedent). Two properties
make that safe enough to gate spending on:

  - UNKNOWN GENDER FAILS OPEN. A candidate name whose gender we cannot
    determine is eligible for both "he" and "she". The alternative -- requiring
    a positive gender match -- would silently drop every name outside an
    English given-name lexicon, which is precisely the population this narrowing
    serves (see expansion.py's note on en_core_web_sm tagging "Molly Iyer" and
    "Prantik Chakraborty" as ORG). A wrong-but-open resolution costs a few
    hundred extra tokens; a wrong-and-closed one loses the relationship.

  - Candidates come from CAPITALISATION, not spaCy NER, for the same reason.
    NER's person/org confusion on non-Anglo names would decide which pronouns
    resolve, putting that bias upstream of what we even send.

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
from ..utils.names import is_noise_name, mention_patterns, normalize, person_norm_key
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
    "aaron", "adam", "adrian", "alan", "albert", "alex", "alexander", "alfred",
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
    "jane", "janet", "janice", "jean", "jeanne", "jennifer", "jessica", "jill",
    "joan", "joanne", "josephine", "joyce", "juanita", "judith", "judy",
    "julia", "julie", "june", "karen", "katherine", "kathleen", "kathryn",
    "kelly", "kimberly", "kristen", "laura", "lauren", "laurie", "leslie",
    "lillian", "linda", "lisa", "lois", "loretta", "lori", "louise", "lucille",
    "lydia", "lynn", "marcia", "margaret", "maria", "marie", "marilyn",
    "marion", "marjorie", "martha", "mary", "maureen", "megan", "melanie",
    "melissa", "michelle", "mildred", "molly", "monica", "nancy", "naomi",
    "natalie", "nicole", "nina", "norma", "olivia", "pamela", "patricia",
    "paula", "pauline", "peggy", "phyllis", "priscilla", "rachel", "rebecca",
    "regina", "renee", "rhonda", "rita", "roberta", "robin", "rosa", "rose",
    "ruby", "ruth", "sally", "samantha", "sandra", "sara", "sarah", "sharon",
    "sheila", "shirley", "sonia", "stephanie", "susan", "suzanne", "sylvia",
    "tammy", "teresa", "theresa", "tiffany", "tina", "tracy", "valerie",
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


def _strip_leading_noise(phrase: str) -> str:
    """Drop leading non-name capitalised words ("After Smith" -> "Smith")."""
    parts = phrase.split()
    while parts and normalize(parts[0]) in _NOT_A_NAME:
        parts.pop(0)
    while parts and normalize(parts[-1]) in _NOT_A_NAME:
        parts.pop()
    return " ".join(parts).strip(" .,'\"")


def name_gender(name: str, sentence: str = "", at: int = -1) -> Optional[str]:
    """MALE / FEMALE when determinable, None when not.

    None means UNKNOWN, and every caller treats unknown as compatible with
    either pronoun -- see the module docstring. An honorific immediately
    before the mention wins over the given-name lexicon.
    """
    if sentence and at > 0:
        before = sentence[:at].rstrip()
        tokens = before.replace(".", " ").split()
        if tokens:
            title = normalize(tokens[-1])
            if title in _GENDERED_TITLES:
                return _GENDERED_TITLES[title]
    parts = person_norm_key(name).split()
    if parts:
        first = parts[0]
        if first in _MALE_NAMES:
            return MALE
        if first in _FEMALE_NAMES:
            return FEMALE
    return None


def _candidates(span: str) -> List[str]:
    """Capitalised name candidates in `span`, left to right."""
    out = []
    for match in _CANDIDATE.finditer(span):
        cleaned = _strip_leading_noise(match.group(0))
        if not cleaned or len(cleaned) < 2 or is_noise_name(cleaned):
            continue
        out.append((cleaned, match.start()))
    return out


def _antecedent(sentences: List[str], index: int, before: int,
                gender: str, lookback: int) -> Optional[str]:
    """The last name said before this pronoun that is compatible with `gender`.

    Walks back one sentence at a time, exactly as far as it has to: within a
    sentence the RIGHTMOST compatible candidate wins ("last said"), and a
    sentence with names but none compatible is not an answer -- the walk
    continues past it. Returns None if the walk runs out, which leaves the
    pronoun unresolved and the sentence unanchored.
    """
    for j in range(index, max(-1, index - lookback - 1), -1):
        span = sentences[j][:before] if j == index else sentences[j]
        best = None
        for cand, at in _candidates(span):
            cand_gender = name_gender(cand, span, at)
            if cand_gender is not None and cand_gender != gender:
                continue  # positively the other gender -- not this pronoun's
            best = cand   # keep going; the rightmost compatible one wins
        if best:
            return best
    return None


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

    # --- anchors by name ---------------------------------------------------
    # A sentence spelling the subject's FULL name is unambiguous and is kept
    # even when a same-surname relative appears in it too -- the conflict
    # pattern exists to disown a bare-surname match, not to veto a mention
    # that already named the person outright.
    direct = set()
    for i, sentence in enumerate(sentences):
        if full_name.search(sentence):
            direct.add(i)
        elif mention.search(sentence) and not (conflict and conflict.search(sentence)):
            direct.add(i)

    # --- anchors by resolved pronoun ---------------------------------------
    pronoun_hits = set()
    lookback = config.SUBJECT_WINDOW_PRONOUN_LOOKBACK
    for i, sentence in enumerate(sentences):
        if i in direct:
            continue  # already anchored; nothing a pronoun could add
        for match in _PRONOUN_RE.finditer(sentence):
            gender = _PRONOUN_GENDER[match.group(1).lower()]
            antecedent = _antecedent(sentences, i, match.start(), gender, lookback)
            if not antecedent:
                continue
            if full_name.search(antecedent) or (
                    mention.search(antecedent)
                    and not (conflict and conflict.search(antecedent))):
                pronoun_hits.add(i)
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
        # around text we are about to include anyway.
        if spans and lo <= spans[-1][1] + 1:
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
