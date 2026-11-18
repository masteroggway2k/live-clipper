"""A loud burst between two lines only counts as funny when it is laughter. Applause and music
(an intro, a walk-on) no longer make clips. Without the sound model the old rule still works."""
import time
from pathlib import Path

import numpy as np

import clipper
from conftest import make_clipper
from test_snap import on_end, on_start

DUR = 60.0
NB = int(DUR / clipper.GRID)
BURST = (30.0, 34.0)                                       # loud, nobody talking


def make_chunk(laugh=0.0, applause=0.0, music=0.0, with_sounds=True, big=0.0, burst=BURST,
               burst_db=-10.0) -> clipper.Chunk:
    c = clipper.Chunk(Path("s0_00000.ts"), 0, 0, pts=0.0, dur=DUR)
    t = 0.0
    while t < DUR - 3:
        if burst[0] - 3 < t < burst[1]:
            t = burst[1]
            continue
        words = [clipper.Word(t + k * 0.5, t + k * 0.5 + 0.45, w) for k, w in enumerate("this is a line of talk.".split())]
        c.words += words
        c.segments.append((t, t + 3.0, "This is a line of talk."))
        t += 3.0
    c.db = np.full(NB, -25.0)
    b0, b1 = int(burst[0] / clipper.GRID), int(burst[1] / clipper.GRID)
    c.db[b0:b1] = burst_db
    if with_sounds:
        c.sounds = {k: np.zeros(NB) for k in ("laugh", "big", "applause", "music", "speech")}
        c.sounds["laugh"][b0:b1] = laugh
        c.sounds["big"][b0:b1] = big
        c.sounds["applause"][b0:b1] = applause
        c.sounds["music"][b0:b1] = music
    return c


def moments(tmp_path, chunk, sensitivity="medium", setup=None) -> list:
    c = make_clipper(tmp_path, tmp_path, sensitivity=sensitivity)
    if setup:
        setup(c)
    made = []
    c._make_clip = lambda chs, s, e, words, m: made.append((s, e, m))
    c._analyse(None, chunk, final=True)
    return made


def test_laughter_after_a_line_is_a_moment(tmp_path):
    made = moments(tmp_path, make_chunk(laugh=0.3))
    assert len(made) == 1
    s, e, m = made[0]
    assert s < BURST[0] < e
    assert "laugh" in m["reason"]


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


def test_big_laugh_is_labelled(tmp_path):
    made = moments(tmp_path, make_chunk(laugh=0.3, big=0.4))
    assert len(made) == 1
    assert made[0][2]["reason"].startswith("big laugh")


def test_short_quiet_chuckle_only_on_generous(tmp_path):
    chuckle = dict(laugh=0.12, burst=(30.0, 31.0), burst_db=-25.0)
    assert moments(tmp_path, make_chunk(**chuckle), sensitivity="medium") == []
    made = moments(tmp_path, make_chunk(**chuckle), sensitivity="high")
    assert len(made) == 1
    assert made[0][2]["reason"].startswith("chuckle")


def test_normal_laugh_not_clipped_on_only_the_best(tmp_path):
    assert moments(tmp_path, make_chunk(laugh=0.2, burst=(30.0, 32.0), burst_db=-25.0), sensitivity="low") == []


QUIET = dict(burst=(30.0, 31.0), burst_db=-25.0)     # nothing in the audio: only chat can make the moment


def test_replay_chat_burst_makes_a_moment(tmp_path):
    def setup(c):
        c.is_live = False
        c.first_pts = 0.0
        c.chat_started = time.time() - 60
        c.chat.extend((0.0, 1.0, t) for t in (3, 12, 20, 45, 50, 57))          # normal background chat
        c.chat.extend((0.0, 3.0, 36.0 + k * 0.4) for k in range(10))        # strong burst ~6 s after the joke
    made = moments(tmp_path, make_chunk(**QUIET), setup=setup)
    assert len(made) == 1
    s, e, m = made[0]
    assert s < 31.0 < e
    assert "chat" in m["reason"]


def test_live_chat_burst_makes_a_moment(tmp_path):
    def setup(c):
        c.is_live = True
        c.chat_started = time.time() - 600
        c.chat.extend((1000.0 + t, 1.0, None) for t in (3, 12, 20, 45, 50, 57))
        c.chat.extend((1036.0 + k * 0.4, 3.0, None) for k in range(10))
    chunk = make_chunk(**QUIET)
    chunk.wall = 1000.0
    made = moments(tmp_path, chunk, setup=setup)
    assert len(made) == 1
    assert "chat" in made[0][2]["reason"]


def test_mild_chat_alone_is_not_a_moment(tmp_path):
    def setup(c):
        c.is_live = False
        c.first_pts = 0.0
        c.chat_started = time.time() - 60
        c.chat.extend((0.0, 1.0, float(t)) for t in range(0, 60, 4))         # steady lol every 4 s
    assert moments(tmp_path, make_chunk(**QUIET), setup=setup) == []


def test_clips_are_30_to_35_seconds(tmp_path):
    made = moments(tmp_path, make_chunk(laugh=0.3))
    assert len(made) == 1
    s, e, _ = made[0]
    assert 30.0 <= e - s <= 35.0


def test_clip_does_not_repeat_the_previous_clip(tmp_path):
    def setup(c):
        c.last_clip_end = 22.0                       # a clip already covered the stream up to 22 s
    made = moments(tmp_path, make_chunk(laugh=0.3), setup=setup)
    assert len(made) == 1
    s, e, _ = made[0]
    assert s >= 22.0
    assert 30.0 <= e - s <= 35.0


def test_moment_inside_the_previous_clip_is_skipped(tmp_path):
    def setup(c):
        c.last_clip_end = 40.0                       # the laugh at 30-34 s is already in the last clip
    assert moments(tmp_path, make_chunk(laugh=0.3), setup=setup) == []


def test_laugh_heavy_stream_raises_the_bar(tmp_path):
    def setup(c):
        c.history = [9.0, 9.5, 8.8, 9.2]               # every minute so far had a big laugh
    assert moments(tmp_path, make_chunk(laugh=0.2, burst=(30.0, 32.0), burst_db=-25.0), setup=setup) == []


def test_quiet_stream_lowers_the_bar(tmp_path):
    def setup(c):
        c.history = [1.0, 1.2, 0.8, 1.1]               # nothing much happens on this stream
    chuckle = dict(laugh=0.12, burst=(30.0, 31.0), burst_db=-25.0)
    assert len(moments(tmp_path, make_chunk(**chuckle), setup=setup)) == 1


def test_short_laugh_label(tmp_path):
    made = moments(tmp_path, make_chunk(laugh=0.3, burst=(30.0, 30.5), burst_db=-10.0))
    assert len(made) == 1
    assert "(0 s)" not in made[0][2]["reason"]
