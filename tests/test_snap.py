"""Clip edges must land on sentence boundaries; segment edges are only a fallback."""
import clipper

SEGS = [(0.0, 1.6, "So, you"), (1.6, 4.9, "do what you do."), (5.0, 7.0, "You know, I always tell the guys,"),
        (7.0, 9.5, "I'm like, yo, when extraordinary is possible,"), (9.5, 12.0, "good just ain't good enough."),
        (12.2, 14.0, "We gotta be the best."), (14.1, 17.0, "Let us fly above the radar."),
        (17.2, 20.0, "All this hard work, working on a road"), (20.0, 22.5, "for the past few years,"),
        (22.5, 25.0, "This is about to pay off."), (25.3, 28.0, "Ladies and gentlemen, put your hands together"),
        (28.0, 31.0, "and welcome to the stage, your host."), (31.2, 33.0, "What's happening?")]
WIN_END = 33.5


def sentence_ends(segs):
    return {round(b, 2) for _, b, t in segs if clipper.SENTENCE_END.search(t.strip())}


def sentence_starts(segs):
    return {round(segs[0][0], 2)} | {round(segs[k][0], 2) for k in range(1, len(segs))
                                       if clipper.SENTENCE_END.search(segs[k - 1][2].strip())}


def on_end(e, ends):
    """e is a sentence end plus at most 0.5 s of padding."""
    return any(b - 1e-6 <= e <= b + 0.5 + 1e-6 for b in ends)


def on_start(s, starts):
    """s is a sentence start minus at most 0.25 s of padding."""
    return any(a - 0.25 - 1e-6 <= s <= a + 1e-6 for a in starts)


def test_edges_land_on_sentences():
    ends, starts = sentence_ends(SEGS), sentence_starts(SEGS)
    for p in [x / 2 for x in range(30, 41)]:                 # 15.0 .. 20.0 step 0.5
        s, e = clipper.snap(SEGS, p - 13, p + 8, WIN_END)
        assert e == WIN_END or on_end(e, ends), (p, e)
        assert s == 0 or on_start(s, starts), (p, s)


def test_short_pick_grows_to_min_length_on_sentences():
    # a 12.5 s judge pick that sentence snapping would shrink to 11.65 s (then clamped mid-sentence)
    s, e = clipper.snap(SEGS, 14.0, 26.5, WIN_END, min_len=12.0, max_len=45.0)
    assert e - s >= 12.0
    assert on_end(e, sentence_ends(SEGS))
    assert on_start(s, sentence_starts(SEGS))


def test_long_pick_shrinks_to_max_length_on_sentences():
    segs = [(t, t + 3.0, f"Sentence {n}.") for n, t in enumerate(range(0, 90, 3))]
    s, e = clipper.snap(segs, 10.0, 54.8, 90.0, min_len=12.0, max_len=45.0)   # nearest edges: 45.75 s
    assert e - s <= 45.0
    assert on_end(e, sentence_ends(segs))
    assert on_start(s, sentence_starts(segs))


def test_falls_back_to_segment_edge():
    assert clipper.snap([(0, 30, "no punctuation"), (30, 60, "still none")], 5, 28, 60) == (0.0, 30.0)


def test_padding_stops_before_the_next_word():
    # real transcript: the comedian starts "We gotta..." 0.3 s after "...good enough."
    W = clipper.Word
    words = [W(10.95, 11.05, "I"), W(11.05, 11.25, "always"), W(11.25, 11.89, "tell them."),
             W(12.0, 13.81, "it is possible."), W(14.43, 15.59, "Good just ain't good"), W(15.59, 15.83, "enough."),
             W(16.13, 16.25, "We"), W(16.25, 16.39, "gotta"), W(16.39, 17.09, "be the best.")]
    s, e = clipper.snap(clipper.sentences(words), 0.0, 15.5, 34.0)
    assert 15.83 <= e <= 16.13
    s, e = clipper.snap(clipper.sentences(words), 12.1, 25.0, 34.0)
    assert 11.89 <= s <= 12.0


def test_sentences_split_on_word_punctuation():
    W = clipper.Word
    words = [W(10.0, 10.4, "Good"), W(10.5, 10.9, "just"), W(11.0, 11.4, "ain't"), W(11.5, 11.9, "good"),
             W(12.0, 12.5, "enough."), W(13.0, 13.3, "We"), W(13.4, 13.8, "gotta"), W(13.9, 14.1, "be"),
             W(14.2, 14.4, "the"), W(14.5, 15.0, "best."), W(15.5, 15.9, "And")]
    assert clipper.sentences(words) == [(10.0, 12.5, "Good just ain't good enough."),
                                        (13.0, 15.0, "We gotta be the best."), (15.5, 15.9, "And")]


def test_sentence_end_pattern():
    for t in ["Thank you.", 'right?"', "wow!)", "हाँ।"]:
        assert clipper.SENTENCE_END.search(t), t
    for t in ["years,", "hands together"]:
        assert not clipper.SENTENCE_END.search(t), t


def test_start_never_before_floor():
    segs = [(t, t + 3.0, f"Sentence {n}.") for n, t in enumerate(range(0, 90, 3))]
    s, e = clipper.snap(segs, 20.0, 52.0, 90.0, floor=24.0)
    assert s >= 24.0
