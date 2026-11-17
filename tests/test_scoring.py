"""A loud burst between two lines only counts as funny when it is laughter. Applause and music
(an intro, a walk-on) no longer make clips. Without the sound model the old rule still works."""
from pathlib import Path

import numpy as np

import clipper
from conftest import make_clipper
from test_snap import on_end, on_start

DUR = 60.0
NB = int(DUR / clipper.GRID)
BURST = (30.0, 34.0)                                       # loud, nobody talking


def make_chunk(laugh=0.0, applause=0.0, music=0.0, with_sounds=True) -> clipper.Chunk:
    c = clipper.Chunk(Path("s0_00000.ts"), 0, 0, pts=0.0, dur=DUR)
    t = 0.0
    while t < DUR - 3:
        if BURST[0] - 3 < t < BURST[1]:
            t = BURST[1]
            continue
        words = [clipper.Word(t + k * 0.5, t + k * 0.5 + 0.45, w) for k, w in enumerate("this is a line of talk.".split())]
        c.words += words
        c.segments.append((t, t + 3.0, "This is a line of talk."))
        t += 3.0
    c.db = np.full(NB, -25.0)
    b0, b1 = int(BURST[0] / clipper.GRID), int(BURST[1] / clipper.GRID)
    c.db[b0:b1] = -10.0
    if with_sounds:
        c.sounds = {k: np.zeros(NB) for k in ("laugh", "applause", "music", "speech")}
        c.sounds["laugh"][b0:b1] = laugh
        c.sounds["applause"][b0:b1] = applause
        c.sounds["music"][b0:b1] = music
    return c


def moments(tmp_path, chunk) -> list:
    c = make_clipper(tmp_path, tmp_path, sensitivity="medium")
    made = []
    c._make_clip = lambda chs, s, e, words, m: made.append((s, e, m))
    c._analyse(None, chunk, final=True)
    return made


def test_laughter_after_a_line_is_a_moment(tmp_path):
    made = moments(tmp_path, make_chunk(laugh=0.3))
    assert len(made) == 1
    s, e, m = made[0]
    assert s < BURST[0] < e
    assert "laughter" in m["reason"]


def test_applause_is_not_a_moment(tmp_path):
    assert moments(tmp_path, make_chunk(applause=0.95)) == []


def test_music_is_not_a_moment(tmp_path):
    assert moments(tmp_path, make_chunk(music=0.7)) == []


def test_clip_ends_on_a_sentence_inside_a_long_segment(tmp_path):
    # the speech model often runs several sentences into one segment that ends mid-sentence
    chunk = make_chunk(laugh=0.3)
    chunk.segments = [(0.0, 30.0, "This is a line of talk. This is a line of"),
                      (34.0, DUR, "talk. This is a line of talk. This is a line of")]
    made = moments(tmp_path, chunk)
    assert len(made) == 1
    s, e, _ = made[0]
    ends = [w.end for w in chunk.words if w.text.endswith(".")]
    starts = [chunk.words[0].start] + [b.start for a, b in zip(chunk.words, chunk.words[1:]) if a.text.endswith(".")]
    assert e == DUR or on_end(e, ends)
    assert s == 0 or on_start(s, starts)


def test_without_sound_model_old_rule_still_works(tmp_path):
    assert len(moments(tmp_path, make_chunk(with_sounds=False))) == 1
