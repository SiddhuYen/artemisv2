"""Subject-window narrowing decides what a paid extraction call even sees.

Two failure directions, and they are not symmetric. Sending too much is a bill;
dropping the sentence that states the relationship loses the answer outright and
looks exactly like "there was no connection". These tests pin the recall side
hardest: surname re-mentions, resolved pronouns, and unknown-gender names all
have to survive.
"""
from app import config
from app.extraction import subject_windows
from app.extraction.subject_windows import FEMALE, MALE, focus, name_gender


def _page(*sentences: str) -> str:
    """A body long enough to clear SUBJECT_WINDOW_MIN_CHARS."""
    filler = ("Unrelated background copy about the venue and the schedule. "
              * 40)
    return " ".join(sentences) + " " + filler


# --- gender detection ------------------------------------------------------
def test_unknown_first_names_have_no_gender():
    """The whole design rests on this: unknown must be UNKNOWN, not a guess.
    A lexicon-of-English-names verdict on 'Prantik' would decide which pronouns
    resolve, and that bias lands on exactly the people this feature serves."""
    assert name_gender("Prantik Chakraborty") is None
    assert name_gender("Xiaoyu Zhang") is None


def test_common_first_names_are_recognised():
    assert name_gender("Michael Reeves") == MALE
    assert name_gender("Molly Chakraborty") == FEMALE


def test_an_honorific_beats_the_lexicon():
    """Goes through _mentions, not name_gender directly. The first version of
    this test computed the offset by hand, which is not how the caller reaches
    it -- _CANDIDATE swallows the honorific into the run ("Ms. Robin Vance" is
    ONE match), so hand-passing an offset tested a path nothing uses and hid
    the fact that the honorific defeated both gender routes at once."""
    found = subject_windows._mentions("Ms. Robin Vance chaired the meeting.")
    assert [(m.name, m.gender) for m in found] == [("Robin Vance", FEMALE)]


def test_a_leading_honorific_is_stripped_off_the_name():
    found = subject_windows._mentions("Mr. Robert Redfield chaired the panel.")
    assert [(m.name, m.gender) for m in found] == [("Robert Redfield", MALE)]


def test_unisex_given_names_are_left_unknown():
    """A WRONG gender closes the walk and loses a sentence; an absent one fails
    open and costs nothing. So the lexicon is tuned for precision, not size."""
    for ambiguous in ("Robin Vance", "Jean Marchand", "Marion Cole"):
        assert name_gender(ambiguous) is None


def test_diminutives_resolve_through_the_shared_name_key():
    assert name_gender("Tim Cook") == MALE


# --- name anchors ----------------------------------------------------------
def test_a_bare_surname_mention_anchors_a_window():
    """Prose re-mentions people by surname alone; requiring the full name every
    time would miss almost every sentence that states a relationship."""
    page = _page(
        "The conference opened on Monday.",
        "Redfield joined the agency in 2018.",
        "Attendance was up on last year.",
    )
    out = focus("Robert Redfield", page)
    assert not out.empty
    assert "Redfield joined the agency" in out.text


def test_a_same_surname_relative_does_not_anchor_on_its_own():
    page = _page(
        "The gala raised a record sum.",
        "Ivanka Trump spoke briefly about the initiative.",
        "Dessert was served at nine.",
    )
    out = focus("Donald Trump", page)
    assert out.empty


def test_a_full_name_survives_a_same_surname_relative_in_the_sentence():
    """The conflict pattern exists to disown a BARE surname, not to veto a
    sentence that named the person outright."""
    page = _page(
        "The programme was announced last spring.",
        "Donald Trump appeared alongside Ivanka Trump at the ceremony.",
        "The venue was full.",
    )
    out = focus("Donald Trump", page)
    assert not out.empty
    assert "Donald Trump appeared alongside" in out.text


# --- pronoun anchors -------------------------------------------------------
def test_a_pronoun_sentence_that_never_names_the_subject_is_kept():
    """The documented case: the sentence carrying the relationship says only
    'He was appointed...' and never repeats the subject's name."""
    page = _page(
        "The agency restructured in 2018.",
        "Redfield became Director that March.",
        "He was appointed to the post by President Donald Trump.",
        "The budget was unchanged.",
    )
    out = focus("Robert Redfield", page)
    assert "He was appointed to the post by President Donald Trump." in out.text


def test_the_walk_skips_a_name_of_the_wrong_gender():
    page = _page(
        "The board met in June.",
        "Molly Iyer presented the quarterly results.",
        "She thanked the team for its work.",
        "The meeting closed early.",
    )
    male = focus("Gregory Fowler", page)
    female = focus("Molly Iyer", page)
    # 'She' resolves to Molly, not to a male subject who is never in the text.
    assert male.empty
    assert "She thanked the team" in female.text


def test_the_walk_continues_past_a_sentence_with_no_compatible_name():
    """'if no name, go back one more sentence' -- across two hops here."""
    page = _page(
        "Sandra Whitfield opened the session.",
        "The agenda was circulated in advance.",
        "The room was at capacity.",
        "She closed with a summary of next steps.",
    )
    out = focus("Sandra Whitfield", page)
    assert "She closed with a summary" in out.text


def test_an_unknown_gender_name_can_still_be_an_antecedent():
    """Unknown gender fails OPEN. Requiring a positive match would drop every
    name outside an English given-name lexicon."""
    page = _page(
        "The round closed in April.",
        "Prantik Chakraborty led the sales organisation.",
        "He reported to the chief executive.",
        "The company moved offices that summer.",
    )
    out = focus("Prantik Chakraborty", page)
    assert "He reported to the chief executive." in out.text


def test_an_organisation_does_not_displace_the_subject():
    """'Person joined Org. He led...' is the most common shape in this corpus,
    and the next sentence is the relationship-bearing one. Picking a single
    antecedent gave that sentence to 'Oracle' and dropped it."""
    page = _page(
        "Prantik Chakraborty joined Oracle.",
        "He led the sales organisation.",
    )
    out = focus("Prantik Chakraborty", page)
    assert "He led the sales organisation." in out.text


def test_an_acronym_does_not_displace_the_subject():
    page = _page("Fauci worked at NIAID.", "He was awarded the medal.")
    out = focus("Anthony Fauci", page)
    assert "He was awarded the medal." in out.text


def test_the_object_of_a_by_phrase_does_not_displace_the_subject():
    """The codebase's own motivating example. Recency picked Trump; both are
    plausible antecedents, and the subject being among them is the answer."""
    page = _page(
        "Redfield was appointed by Donald Trump.",
        "He served until 2021.",
    )
    out = focus("Robert Redfield", page)
    assert "He served until 2021." in out.text


def test_a_capitalised_non_name_does_not_displace_the_subject():
    """_NOT_A_NAME is a blocklist, so unlisted sentence-initial adverbs leak
    through as candidates. They must not be able to swallow the walk."""
    page = _page(
        "Despite the setback, Sandra Whitfield resigned.",
        "She later joined Acme.",
    )
    out = focus("Sandra Whitfield", page)
    assert "She later joined Acme." in out.text


def test_a_forward_reference_resolves():
    """Cataphora: the pronoun precedes its antecedent in the same sentence, so
    a backwards-only walk marched straight past the answer."""
    page = _page(
        "The company grew steadily.",
        "In her role at Acme, Sandra Whitfield led sales.",
        "She reported to the board.",
    )
    out = focus("Sandra Whitfield", page)
    assert "In her role at Acme" in out.text


def test_a_run_of_pronoun_only_sentences_holds_together():
    """Biographies write long stretches that never repeat the name. Without
    chaining, the second sentence walks back over a first that names nobody
    either, runs out of lookback, and is dropped."""
    page = _page(
        "Fauci joined the institute in 1968.",
        "He became head of the section in 1974.",
        "He became director in 1984.",
        "He held the post for decades.",
    )
    out = focus("Anthony Fauci", page)
    assert "He held the post for decades." in out.text


def test_an_org_unit_in_the_pronoun_sentence_does_not_veto():
    """'LCI's Clinical Physiology Section' is three capitalised words with no
    legal suffix, so looks_like_person_name calls it person-shaped. Letting the
    pronoun's own sentence answer on that fragment stopped the walk before it
    ever reached the sentence naming the subject."""
    page = _page(
        "Fauci joined the institute in 1968.",
        "He became head of the LCI's Clinical Physiology Section in 1974.",
    )
    out = focus("Anthony Fauci", page)
    assert "Clinical Physiology Section" in out.text


def test_institutional_phrases_are_not_person_shaped():
    for org in ("LCI's Clinical Physiology Section", "Weill Cornell Medical Center",
                "National Cancer Institute", "Human Rights Committee"):
        assert not subject_windows._mentions(org)[0].is_person_shaped, org


def test_a_possessive_surname_is_still_the_subject():
    """The possessive test must not demote a single-token name: "Fauci's" is
    the subject in the possessive, not an organisation."""
    assert not subject_windows._is_org_phrase("Fauci's")


def test_a_secondary_figure_does_not_collect_another_persons_pronouns():
    """The over-inclusion guard. A page about someone else, where the subject
    appears once, must not have every 'he' handed to the subject."""
    page = _page(
        "Gregory Fowler joined as an adviser.",
        *[f"Filler sentence number {i} about the industry." for i in range(8)],
        "Thomas Baker founded the studio in 2004.",
        "He sold it in 2010.",
        "He retired to Lisbon.",
    )
    out = focus("Gregory Fowler", page)
    # Far enough from the subject's own anchor that the context window cannot
    # explain their presence -- so if they appear, a pronoun claimed them.
    assert "He sold it in 2010." not in out.text
    assert "He retired to Lisbon." not in out.text


def test_a_pronoun_beyond_the_lookback_is_not_resolved(monkeypatch):
    monkeypatch.setattr(config, "SUBJECT_WINDOW_PRONOUN_LOOKBACK", 1)
    page = _page(
        "Sandra Whitfield opened the session.",
        "The agenda was circulated in advance.",
        "The room was at capacity.",
        "The catering arrived late.",
        "She closed with a summary of next steps.",
    )
    out = focus("Sandra Whitfield", page)
    assert "She closed with a summary" not in out.text


# --- windowing and merging -------------------------------------------------
def test_context_sentences_are_kept_either_side(monkeypatch):
    monkeypatch.setattr(config, "SUBJECT_WINDOW_SENTENCES", 1)
    page = _page(
        "Alpha sentence one.",
        "Beta sentence two.",
        "Sandra Whitfield signed the agreement.",
        "Delta sentence four.",
        "Epsilon sentence five.",
    )
    out = focus("Sandra Whitfield", page)
    assert "Beta sentence two." in out.text
    assert "Delta sentence four." in out.text
    assert "Alpha sentence one." not in out.text


def test_overlapping_windows_merge_into_one_segment(monkeypatch):
    monkeypatch.setattr(config, "SUBJECT_WINDOW_SENTENCES", 2)
    page = _page(
        "Sandra Whitfield joined in 2019.",
        "The team grew quickly.",
        "Sandra Whitfield was promoted in 2021.",
    )
    out = focus("Sandra Whitfield", page)
    assert out.segments == 1
    assert subject_windows._ELISION not in out.text


def test_distant_windows_stay_separate_and_are_marked_as_elided(monkeypatch):
    monkeypatch.setattr(config, "SUBJECT_WINDOW_SENTENCES", 1)
    middle = " ".join(f"Filler sentence number {i}." for i in range(20))
    page = _page("Sandra Whitfield joined in 2019.", middle,
                 "Sandra Whitfield left in 2024.")
    out = focus("Sandra Whitfield", page)
    assert out.segments >= 2
    # Without the marker the model sees two distant passages glued together and
    # can state a relationship the page never did.
    assert subject_windows._ELISION in out.text


# --- the gate and the escape hatches ---------------------------------------
def test_a_page_that_never_mentions_the_subject_is_dropped_entirely():
    page = _page(
        "Thomas Baker founded the studio in 2004.",
        "The studio moved to Berlin in 2010.",
    )
    out = focus("Sandra Whitfield", page)
    assert out.empty


def test_short_enrichment_strings_are_never_narrowed():
    """Wikidata evidence text and roster summaries are dense, already about the
    subject, and often never spell the name in a sentence."""
    text = "Educated at: Stanford University. Employer: Acme Corp."
    out = focus("Sandra Whitfield", text)
    assert out.text == text
    assert not out.empty


def test_narrowing_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(config, "SUBJECT_WINDOW_ENABLED", False)
    page = _page("Thomas Baker founded the studio in 2004.")
    out = focus("Sandra Whitfield", page)
    assert out.text == page


def test_narrowing_actually_shrinks_a_mostly_irrelevant_page():
    page = _page(
        "Sandra Whitfield signed the agreement.",
        " ".join(f"Unrelated sentence number {i} about other people."
                 for i in range(60)),
    )
    out = focus("Sandra Whitfield", page)
    assert len(out.text) < len(page) / 2


def test_the_original_is_returned_when_windows_cover_the_page():
    """No point paying the elision-marker explanation for zero saving."""
    page = " ".join(f"Sandra Whitfield did thing number {i} at the company."
                    for i in range(60))
    assert len(page) >= config.SUBJECT_WINDOW_MIN_CHARS
    out = focus("Sandra Whitfield", page)
    assert out.text == page
    assert out.reason == "windows cover the page"
