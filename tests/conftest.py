"""Shared test helpers: synthetic live chunks with known beeps/flashes, a fake speech model,
and a meter that reads beep, flash and caption onsets back out of a rendered clip."""
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import clipper  # noqa: E402


def build_synthetic_chunks(folder: Path) -> Path:
    """150 s source split like the recorder does: a flash and a beep on every whole second,
    and the audio stream starting 0.5 s after the video while staying in phase."""
    folder.mkdir(parents=True, exist_ok=True)

    def ff(*args):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], cwd=folder, check=True)

    ff("-f", "lavfi", "-i",
       "nullsrc=s=640x360:r=30:d=150,geq=lum='if(lt(mod(T\\,1)\\,0.1)\\,235\\,40)':cb=128:cr=128,format=yuv420p",
       "-itsoffset", "0.5", "-f", "lavfi", "-i",
       "aevalsrc='if(lt(mod(t+0.5\\,1)\\,0.1)\\,0.8*sin(2*PI*1000*t)\\,0)':s=48000:d=149.5",
       "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-g", "60", "-keyint_min", "60", "-sc_threshold", "0",
       "-preset", "ultrafast", "-c:a", "aac", "-f", "mpegts", "src.ts")
    ff("-i", "src.ts", "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", "-f", "segment", "-segment_time", "60",
       "-segment_format", "mpegts", "s0_%05d.ts")
    (folder / "src.ts").unlink()
    return folder


@pytest.fixture(scope="session")
def synthetic_chunks(tmp_path_factory) -> Path:
    return build_synthetic_chunks(tmp_path_factory.mktemp("chunks"))


class FakeWhisper:
    """Reports the word BEEP at every beep, timed from the audio itself."""

    def transcribe(self, pcm, **kw):
        hop = 160                                            # 10 ms at 16 kHz
        e = np.sqrt(np.mean(pcm[: len(pcm) // hop * hop].reshape(-1, hop) ** 2, axis=1))
        on = e > 0.05
        segs = []
        for i in np.flatnonzero(on[1:] & ~on[:-1]) + 1:
            j = i
            while j < len(on) and on[j]:
                j += 1
            a, b = i * 0.01, j * 0.01
            segs.append(NS(start=a, end=b, text="Beep!", words=[NS(start=a, end=b, word="BEEP")]))
        return segs, None


def make_clipper(tmp_path: Path, work: Path, layout: str = "vertical",
                 sensitivity: str = "medium") -> clipper.LiveClipper:
    c = clipper.LiveClipper(clipper.Settings(url="test", layout=layout, sensitivity=sensitivity, use_judge=False),
                            tmp_path / "out")
    c.work = work
    c.whisper = FakeWhisper()
    c.encoder = c._pick_encoder()
    return c


BANDS = {  # (flash band above the captions, caption band) as ffmpeg crop arguments
    "vertical": ("1080:120:0:560", "1080:110:0:1385"),
    "crop": ("1080:200:0:100", "1080:110:0:1385"),
    "horizontal": ("1280:150:0:20", "1280:70:0:595"),
}


def _stat(clip: Path, crop: str, key: str):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(clip), "-vf",
                          f"crop={crop},signalstats,metadata=print:key=lavfi.signalstats.{key}", "-f", "null", "-"],
                         capture_output=True, text=True, errors="replace").stderr
    t, rows = None, []
    for line in out.splitlines():
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            t = float(m.group(1))
        m = re.search(rf"{key}=([\d.]+)", line)
        if m and t is not None:
            rows.append((t, float(m.group(1))))
    return rows


def _onsets(rows, test):
    res, prev = [], False
    for t, v in rows:
        on = test(v)
        if on and not prev:
            res.append(t)
        prev = on
    return res


def measure(clip: Path, layout: str) -> dict:
    flash, cap = BANDS[layout]
    a = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(clip), "-af", "silencedetect=n=-30dB:d=0.3",
                        "-f", "null", "-"], capture_output=True, text=True, errors="replace").stderr
    return {
        "beeps": [float(x) for x in re.findall(r"silence_end: ([\d.]+)", a)],
        "flashes": _onsets(_stat(clip, flash, "YAVG"), lambda v: v > 140),
        "captions": _onsets(_stat(clip, cap, "UMIN"), lambda v: v < 70),   # yellow highlighted word
    }
