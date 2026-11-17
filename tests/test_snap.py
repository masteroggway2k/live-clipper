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


def test_edges_land_on_sentences():
    ends, starts = sentence_ends(SEGS), sentence_starts(SEGS)
    for p in [x / 2 for x in range(30, 41)]:                 # 15.0 .. 20.0 step 0.5
        s, e = clipper.snap(SEGS, p - 13, p + 8, WIN_END)
        assert e == WIN_END or round(e - 0.5, 2) in ends, (p, e)
        assert s == 0 or round(s + 0.25, 2) in starts, (p, s)


def test_short_pick_grows_to_min_length_on_sentences():
    # a 12.5 s judge pick that sentence snapping would shrink to 11.65 s (then clamped mid-sentence)
    s, e = clipper.snap(SEGS, 14.0, 26.5, WIN_END)
    assert e - s >= clipper.MIN_CLIP
    assert round(e - 0.5, 2) in sentence_ends(SEGS)
    assert round(s + 0.25, 2) in sentence_starts(SEGS)


def test_long_pick_shrinks_to_max_length_on_sentences():
    segs = [(t, t + 3.0, f"Sentence {n}.") for n, t in enumerate(range(0, 90, 3))]
    s, e = clipper.snap(segs, 10.0, 54.8, 90.0)              # nearest sentence edges would give 45.75 s
    assert e - s <= clipper.MAX_CLIP
    assert round(e - 0.5, 2) in sentence_ends(segs)
    assert round(s + 0.25, 2) in sentence_starts(segs)


def test_falls_back_to_segment_edge():
    assert clipper.snap([(0, 30, "no punctuation"), (30, 60, "still none")], 5, 28, 60) == (0.0, 30.5)


def test_sentence_end_pattern():
    for t in ["Thank you.", 'right?"', "wow!)", "हाँ।"]:
        assert clipper.SENTENCE_END.search(t), t
    for t in ["years,", "hands together"]:
        assert not clipper.SENTENCE_END.search(t), t
