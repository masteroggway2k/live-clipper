"""Captions and picture must stay within 40 ms of the sound, in every layout, including a chunk
whose audio starts late and a clip that crosses a chunk boundary."""
import pytest

import clipper
from conftest import make_clipper, measure


def _cut(tmp_path, chunks_dir, layout, title="sync"):
    c = make_clipper(tmp_path, chunks_dir, layout)
    files = sorted(chunks_dir.glob("s0_*.ts"))[:2]
    a, b = (c._prepare(f, (0, i)) for i, f in enumerate(files))
    words, _ = clipper.timeline([(a, 0.0), (b, b.pts - a.pts)])
    s, e = 50.37, 72.37                                      # crosses the chunk boundary at ~60 s
    c._make_clip([a, b], s, e, [w for w in words if w.end > s and w.start < e],
                 {"title": title, "score": 0, "reason": "test"})
    return c, a


@pytest.mark.parametrize("layout", ["vertical", "crop", "horizontal"])
def test_sync(tmp_path, synthetic_chunks, layout):
    c, first = _cut(tmp_path, synthetic_chunks, layout)
    assert first.aoff == pytest.approx(0.48, abs=0.05)      # the late-audio case is really exercised
    m = measure(next(c.clips_dir.glob("clip_001_*.mp4")), layout)
    assert len(m["beeps"]) >= 18 and m["captions"] and m["flashes"]
    near = lambda xs: max(min(abs(x - y) for y in m["beeps"]) for x in xs[:18] if x < m["beeps"][-1])
    assert near(m["captions"]) <= 0.040
    assert near(m["flashes"]) <= 0.040


def test_clip_with_emoji_title(tmp_path, synthetic_chunks):
    c, _ = _cut(tmp_path, synthetic_chunks, "vertical", title="\U0001F534 LIVE हाहा \U0001F602")
    made = list(c.clips_dir.glob("clip_001_*.mp4"))
    assert len(made) == 1
    assert "LIVE" in made[0].name
