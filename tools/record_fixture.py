"""Records part of a live stream into 60 s chunks exactly like the app's recorder does,
so tools/replay.py can run the same stretch through the scoring again and again.

usage: python tools/record_fixture.py <youtube url> <out dir> <seconds> [--cookies chrome|edge|firefox]
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clipper  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out", type=Path)
    ap.add_argument("seconds", type=int)
    ap.add_argument("--cookies", default="", help="browser whose YouTube login to use")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # stream titles often contain emoji

    args.out.mkdir(parents=True, exist_ok=True)
    c = clipper.LiveClipper(clipper.Settings(url=args.url, cookies_browser=args.cookies), args.out / "_unused")
    info = c._resolve()
    parts = info.get("requested_formats") or [info]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for p in parts[:2]:
        cmd += ["-rw_timeout", "20000000", "-i", p["url"]]
    cmd += ["-map", "0:v:0", "-map", "1:a:0" if len(parts) > 1 else "0:a:0", "-c", "copy", "-t", str(args.seconds),
            "-f", "segment", "-segment_time", str(clipper.CHUNK_SECONDS), "-segment_format", "mpegts", "s0_%05d.ts"]
    print("recording:", info.get("title"))
    subprocess.run(cmd, cwd=args.out)
    for f in sorted(args.out.glob("s0_*.ts")):
        print(" ", f.name, f"{f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
