"""Replays recorded chunks through Live Clipper's real transcription and scoring, and prints every
step: the best peaks per stretch with their parts, the transcript around them, and the picks.

usage: python tools/replay.py <fixture dir> [--sensitivity low|medium|high] [--judge] [--render]
Transcripts are cached in <fixture dir>/chunks.pkl, so only the first run needs the speech model.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clipper  # noqa: E402


def ctx(words, a, b):
    return " ".join(w.text for w in words if a <= w.start < b)


def make_trace():
    def trace(d):
        W, sc, G = d["words"], d["score"], clipper.GRID
        print(f"\n=== stretch {d['lo']:.0f}-{d['hi']:.0f}s of window 0-{d['win_end']:.0f}s"
              + (f" | peak {d['peak']:.2f} need {d['need']:.2f}" if d["peak"] is not None else " | judged by Claude"))
        z = sc.copy()
        z[: int(d["lo"] / G)] = -1
        z[int(d["hi"] / G):] = -1
        for _ in range(5):
            i = int(np.argmax(z))
            if z[i] <= 0:
                break
            t = i * G
            print(f"  peak t={t:6.1f} score={sc[i]:.2f}  loud={d['loud'][i]:.2f} laugh={d['laugh'][i]:.2f} "
                  f"text={d['text'][i]:.2f} chat={d['chat'][i]:.2f}")
            print(f"      ...{ctx(W, t - 10, t)} || {ctx(W, t, t + 6)}...")
            z[max(0, i - 20): i + 20] = -1
        for m in d["moments"]:
            s, e = m["start"], m["end"]
            print(f"  MOMENT {s:.1f}-{e:.1f} ({e - s:.1f}s) score {m['score']} [{m['reason']}] {m.get('title', '')}")
            print(f"      starts: {ctx(W, s, s + 6)}...")
            print(f"      ends:   ...{ctx(W, e - 6, e)}   | next: {ctx(W, e, e + 4)}")
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture", type=Path)
    ap.add_argument("--sensitivity", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--judge", action="store_true", help="let Claude pick (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--render", action="store_true", help="also cut the clips to <fixture>/out_*")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    fx = args.fixture
    c = clipper.LiveClipper(clipper.Settings(url="replay", sensitivity=args.sensitivity, use_judge=args.judge),
                            fx / f"out_{args.sensitivity}_{int(time.time())}")
    c.work = fx
    c.log = lambda m: print("LOG", m)
    if args.judge:
        import judge
        c.judge = judge.make_judge()
        if not c.judge:
            sys.exit("No Anthropic credentials found: set ANTHROPIC_API_KEY, then run again.")

    cache = fx / "chunks.pkl"
    chunks = pickle.loads(cache.read_bytes()) if cache.exists() else {}
    files = sorted(fx.glob("s0_*.ts"))
    if not files:
        sys.exit(f"No s0_*.ts chunks in {fx}")
    if any(f.name not in chunks for f in files):
        c.s.use_judge = False                                 # model loading only; the judge is set above
        judge_client, c.judge = c.judge, None
        c._load_models()
        c.judge = judge_client
        for i, f in enumerate(files):
            if f.name not in chunks:
                chunks[f.name] = c._prepare(f, (0, i))
        cache.write_bytes(pickle.dumps(chunks))
    else:
        c.encoder = c._pick_encoder()

    made = []
    if not args.render:
        c._make_clip = lambda chs, s, e, words, m: made.append((s, e, words, m))
    c.trace = make_trace()
    names = [f.name for f in files]
    for i, n in enumerate(names):
        if chunks.get(n):
            c._analyse(chunks.get(names[i - 1]) if i else None, chunks[n], final=(i == len(names) - 1))

    print("\nCLIPS:", len(c.clips) if args.render else len(made))
    for s, e, words, m in made:
        print(f"  {s:.1f}-{e:.1f}: {' '.join(w.text for w in words)[:300]}")


if __name__ == "__main__":
    main()
