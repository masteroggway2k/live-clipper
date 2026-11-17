"""Live Clipper pipeline.

Records a YouTube live stream in ~60 s chunks, transcribes each chunk, scores
every moment for "funny" (loud bursts such as laughter or shouting, laugh words
in the speech, laughing in live chat, and optionally a Claude judge), then cuts
the best moments into captioned clips. Runs until stopped or the stream ends.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CHUNK_SECONDS = 60
GRID = 0.5            # seconds per scoring bin
HOLD_BACK = 25.0      # seconds of "future" kept so a clip never runs off the recorded end
MIN_CLIP, MAX_CLIP = 12.0, 45.0
THRESHOLDS = {"low": 6.0, "medium": 4.0, "high": 2.8}       # heuristic score needed
QUANTILE = {"low": 0.85, "medium": 0.65, "high": 0.4}           # share of recent stretches a clip must beat
FLOOR = 1.8                                                  # never clip below this, however quiet the stream
JUDGE_MIN_SCORE = {"low": 8, "medium": 6, "high": 5}        # Claude score (1-10) needed
NO_WINDOW = 0x08000000 if os.name == "nt" else 0
BOT_CHECK_HELP = ('YouTube is asking this PC to prove it is not a bot. Pick your browser under '
                  '"If YouTube asks to sign in" and press Start again.')

LAUGH_WORDS = re.compile(
    r"^(a?(ha){2,}h?|(he){2,}|lol+|lmf?ao+|rofl|hilarious|funny|laugh\w*|joke\w*|omg|bruh|wtf|"
    r"crazy|insane|ridiculous|stupid|idiot|dead|dying|हाहा\w*|मज़ाक|मजाक|पागल)$",
    re.IGNORECASE,
)
CHAT_LAUGH = re.compile(
    r"(lol|lmf?ao|rofl|haha|hehe|xd+\b|kek|dead|funny|hilarious|clip (it|that)|"
    r"😂|🤣|💀|😆|😹|:rolling_on_the_floor_laughing:|:face_with_tears_of_joy:|:skull:|हाहा)",
    re.IGNORECASE,
)


def run(cmd, cwd=None, timeout=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace",
                          timeout=timeout, creationflags=NO_WINDOW)


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Chunk:
    path: Path
    session: int
    index: int
    pts: float = 0.0          # stream timestamp of the chunk start
    aoff: float = 0.0         # audio (and so the wav and every word time) starts this much after pts
    dur: float = 0.0
    wall: float = 0.0         # wall-clock time the chunk started being written
    words: list = field(default_factory=list)      # chunk-local times
    segments: list = field(default_factory=list)   # (start, end, text)
    db: np.ndarray | None = None                   # loudness per GRID bin


@dataclass
class Settings:
    url: str
    layout: str = "vertical"       # vertical | crop | horizontal
    sensitivity: str = "medium"    # low | medium | high
    language: str = "auto"         # whisper language code or "auto"
    translate: bool = False        # captions translated to English
    model: str = "small"           # faster-whisper model size
    cookies_browser: str = ""      # e.g. "chrome" if YouTube asks to sign in
    use_judge: bool = True         # use Claude when credentials are available


class LiveClipper:
    def __init__(self, settings: Settings, out_root: Path):
        self.s = settings
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.job_dir = out_root / stamp
        self.work = self.job_dir / "work"
        self.clips_dir = self.job_dir
        self.work.mkdir(parents=True, exist_ok=True)

        self.state = "starting"
        self.title = ""
        self.error = ""
        self.clips: list[dict] = []
        self.logs: deque[str] = deque(maxlen=300)
        self.stats = {"recorded": 0.0, "analysed": 0.0}
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.rec_done = threading.Event()
        self.ffmpeg: subprocess.Popen | None = None
        self.is_live = True
        self.chat: deque[float] = deque(maxlen=20000)   # wall times of laughing chat messages
        self.chat_all = 0
        self.chat_started = 0.0
        self.history: list[float] = []
        self.trace = None                               # optional callback: sees every scoring step
        self.warned_slow = False
        self.whisper = None
        self.device = "?"
        self.encoder = None
        self.judge = None
        self.threads: list[threading.Thread] = []

    # ------------------------------------------------------------------ utils
    def log(self, msg: str):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        with self.lock:
            self.logs.append(line)
        try:
            print(line, flush=True)
        except Exception:                               # console that cannot show emoji, or no console
            pass

    def status(self) -> dict:
        with self.lock:
            return {
                "state": self.state, "title": self.title, "error": self.error,
                "clips": list(reversed(self.clips)), "logs": list(self.logs)[-60:],
                "stats": dict(self.stats), "folder": str(self.clips_dir),
                "device": self.device, "judge": bool(self.judge), "chat": self.chat_all,
            }

    def start(self):
        for target in (self._record_loop, self._chat_loop, self._process_loop):
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self.threads.append(t)

    def stop(self):
        self.stop_flag.set()

    # -------------------------------------------------------------- recording
    def _resolve(self) -> dict:
        import yt_dlp
        opts = {
            "quiet": True, "no_warnings": True, "noplaylist": True,
            "format": "bv*[height<=720][vcodec^=avc]+ba/bv*[height<=720]+ba/b[height<=720]/b",
        }
        if shutil.which("node") and not shutil.which("deno"):
            opts["js_runtimes"] = {"node": {}}
        if self.s.cookies_browser:
            opts["cookiesfrombrowser"] = (self.s.cookies_browser,)
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(self.s.url, download=False)

    def _record_loop(self):
        session, failures = 0, 0
        try:
            while not self.stop_flag.is_set():
                try:
                    info = self._resolve()
                except Exception as e:
                    failures += 1
                    msg = str(e).replace("ERROR: ", "")
                    if "confirm you" in msg and "bot" in msg:   # the apostrophe in YouTube's text varies
                        if session == 0:
                            self._fail(BOT_CHECK_HELP)
                        else:
                            self.log(BOT_CHECK_HELP)
                        return
                    if session == 0 or failures >= 3:
                        if session == 0:
                            self._fail(f"Could not open that link: {msg}")
                        else:
                            self.log(f"Stream is no longer available ({msg[:120]}).")
                        return
                    time.sleep(10)
                    continue
                self.is_live = bool(info.get("is_live"))
                with self.lock:
                    self.title = info.get("title") or self.s.url
                    self.state = "recording"
                if session == 0:
                    kind = "live stream" if self.is_live else "video (not live, so it is processed as fast as possible)"
                    self.log(f"Recording {kind}: {self.title}  [{info.get('height', '?')}p]")
                headers = info.get("http_headers") or {}
                cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
                parts = info.get("requested_formats") or [info]
                for part in parts[:2]:
                    if headers.get("User-Agent"):
                        cmd += ["-user_agent", headers["User-Agent"]]
                    cmd += ["-rw_timeout", "20000000", "-i", part["url"]]   # give up on a dead connection after 20 s
                audio = "1:a:0" if len(parts) > 1 else "0:a:0"
                cmd += ["-map", "0:v:0", "-map", audio, "-c", "copy",
                        "-f", "segment", "-segment_time", str(CHUNK_SECONDS),
                        "-segment_format", "mpegts", f"s{session}_%05d.ts"]
                started = time.time()
                errlog = self.work / f"ffmpeg_s{session}.log"
                with open(errlog, "wb") as errf:
                    self.ffmpeg = subprocess.Popen(cmd, cwd=self.work, stdin=subprocess.PIPE,
                                                   stdout=subprocess.DEVNULL, stderr=errf,
                                                   creationflags=NO_WINDOW)
                    asked = False
                    while self.ffmpeg.poll() is None:
                        if self.stop_flag.is_set() and not asked:
                            asked = True
                            try:                        # "q" lets ffmpeg close the chunk cleanly
                                self.ffmpeg.stdin.write(b"q")
                                self.ffmpeg.stdin.flush()
                                self.ffmpeg.wait(timeout=8)
                            except Exception:
                                self.ffmpeg.kill()
                        time.sleep(0.5)
                if self.stop_flag.is_set():
                    return
                ran = time.time() - started
                err = errlog.read_bytes()
                if not self.is_live:
                    self.log("Reached the end of the video.")
                    return
                err_text = (err or b"").decode(errors="replace").strip().splitlines()[-1:] or [""]
                if ran < 45:
                    failures += 1
                    if failures >= 5:
                        self.log(f"Recording keeps failing ({err_text[0][:160]}). The stream has probably ended.")
                        return
                else:
                    failures = 0
                self.log(f"Recording was interrupted after {ran:.0f}s ({err_text[0][:120] or 'no error message'}), reconnecting...")
                session += 1
                time.sleep(5)
        finally:
            self.rec_done.set()

    def _fail(self, msg: str):
        with self.lock:
            self.state, self.error = "error", msg
        self.log(msg)
        self.stop_flag.set()

    # ------------------------------------------------------------------- chat
    def _chat_loop(self):
        """Reads live chat by letting yt-dlp write it to a file and following that file."""
        chat_dir = self.job_dir / "chat"
        chat_dir.mkdir(exist_ok=True)
        cmd = [sys.executable, "-m", "yt_dlp", "--quiet", "--no-warnings", "--skip-download",
               "--write-subs", "--sub-langs", "live_chat", "-o", "chat", self.s.url]
        if shutil.which("node") and not shutil.which("deno"):
            cmd[3:3] = ["--js-runtimes", "node"]
        if self.s.cookies_browser:
            cmd[3:3] = ["--cookies-from-browser", self.s.cookies_browser]
        proc = None
        try:
            proc = subprocess.Popen(cmd, cwd=chat_dir, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
            self.chat_started = time.time()
            pos, tail = 0, b""
            while not self.stop_flag.is_set() and not self.rec_done.is_set():
                files = list(chat_dir.glob("chat.live_chat.json*"))
                if files:
                    with open(files[0], "rb") as fh:
                        fh.seek(pos)
                        data = fh.read()
                        pos = fh.tell()
                    *lines, tail = (tail + data).split(b"\n")
                    for line in lines:
                        self._chat_line(line)
                elif proc.poll() is not None:
                    break                               # stream has no chat
                time.sleep(2)
        except Exception:
            pass
        finally:
            if proc and proc.poll() is None:
                if os.name == "nt":
                    run(["taskkill", "/PID", str(proc.pid), "/T", "/F"])
                else:
                    proc.kill()
            if not self.chat_all:
                self.chat_started = 0.0
            time.sleep(0.5)
            shutil.rmtree(chat_dir, ignore_errors=True)

    def _chat_line(self, line: bytes):
        try:
            actions = json.loads(line)["replayChatItemAction"]["actions"]
        except Exception:
            return
        for act in actions:
            item = (act.get("addChatItemAction") or {}).get("item") or {}
            msg = item.get("liveChatTextMessageRenderer") or item.get("liveChatPaidMessageRenderer")
            if not msg:
                continue
            text = ""
            for r in (msg.get("message") or {}).get("runs", []):
                emoji = r.get("emoji") or {}
                text += r.get("text") or (emoji.get("emojiId", "") + " ".join(emoji.get("shortcuts", [])))
            self.chat_all += 1
            if CHAT_LAUGH.search(text):
                try:
                    self.chat.append(int(msg["timestampUsec"]) / 1e6)
                except Exception:
                    self.chat.append(time.time())

    # ------------------------------------------------------------- processing
    def _load_models(self):
        from faster_whisper import WhisperModel
        self._add_cuda_dlls()
        silence = np.zeros(16000, dtype=np.float32)
        from faster_whisper import download_model
        model_dir = download_model(self.s.model)        # first use downloads it (75 MB to 1.5 GB)
        options = [("cpu", "int8")]
        if self._gpu_works(model_dir):
            options.insert(0, ("cuda", "float16"))
        for device, ctype in options:
            try:
                model = WhisperModel(model_dir, device=device, compute_type=ctype)
                list(model.transcribe(silence, language="en")[0])
                self.whisper, self.device = model, device
                break
            except Exception as e:
                self.log(f"Speech model could not start on {device} ({str(e)[:100]}).")
        if not self.whisper:
            raise RuntimeError("The speech-to-text model could not be loaded.")
        self.log(f"Speech-to-text ready: whisper '{self.s.model}' on {'GPU' if self.device == 'cuda' else 'CPU'}.")
        self.encoder = self._pick_encoder()
        if self.s.use_judge:
            try:
                import judge
                self.judge = judge.make_judge()
                if self.judge:
                    self.log("Claude is judging which moments are funny.")
            except Exception as e:
                self.log(f"Claude judge is off ({str(e)[:100]}).")
        if not self.judge:
            self.log("Funny moments are picked from loudness, laugh words and live chat.")

    def _gpu_works(self, model_dir: str) -> bool:
        """Tries the model on the GPU in a throwaway process, with a time limit."""
        marker = Path.home() / ".live-clipper" / "gpu_failed"
        if marker.exists() and time.time() - marker.stat().st_mtime < 7 * 86400:
            return False
        code = ("import clipper, numpy as np; clipper.LiveClipper._add_cuda_dlls(); "
                "import ctranslate2; assert ctranslate2.get_cuda_device_count() > 0; "
                "from faster_whisper import WhisperModel; "
                f"m = WhisperModel({model_dir!r}, device='cuda', compute_type='float16'); "
                "list(m.transcribe(np.zeros(16000, dtype=np.float32), language='en')[0]); print('GPU_OK')")
        proc = subprocess.Popen([sys.executable, "-c", code], cwd=Path(__file__).parent,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
        try:
            out, _ = proc.communicate(timeout=75)
            ok = b"GPU_OK" in out
        except subprocess.TimeoutExpired:
            ok = False
            if os.name == "nt":
                run(["taskkill", "/PID", str(proc.pid), "/T", "/F"])
            else:
                proc.kill()
        if not ok:
            try:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("GPU speech-to-text did not work; delete this file to try again.")
            except OSError:
                pass
            self.log("The graphics card could not be used for speech-to-text, using the CPU instead.")
        return ok

    @staticmethod
    def _add_cuda_dlls():
        if os.name != "nt":
            return
        try:
            import nvidia
            for base in nvidia.__path__:
                for d in Path(base).glob("*/bin"):
                    os.add_dll_directory(str(d))
                    os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]
        except Exception:
            pass

    def _pick_encoder(self):
        test = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                "color=black:s=256x256:d=0.2", "-c:v", "h264_nvenc", "-f", "null", "-"]
        try:
            if run(test, timeout=30).returncode == 0:
                return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "23", "-b:v", "0"]
        except Exception:
            pass
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"]

    def _process_loop(self):
        try:
            self._load_models()
        except Exception as e:
            self._fail(str(e))
            return
        done: dict[tuple[int, int], Chunk] = {}
        while True:
            files = sorted(self.work.glob("s*_*.ts"), key=self._key)
            keys = [self._key(f) for f in files]
            finished = self.rec_done.is_set()          # the recorder always ends, also on Stop
            progressed = False
            for i, f in enumerate(files):
                k = keys[i]
                if k in done:
                    continue
                later_exists = i + 1 < len(files)
                if not later_exists and not finished:
                    break                               # still being written
                last_of_session = not (later_exists and keys[i + 1][0] == k[0])
                try:
                    chunk = self._prepare(f, k)
                    if chunk:
                        prev = done.get((k[0], k[1] - 1))
                        self._analyse(prev, chunk, final=last_of_session)
                        done[k] = chunk
                    else:
                        done[k] = None
                except Exception as e:
                    self.log(f"Skipped a chunk because of an error: {str(e)[:200]}")
                    done[k] = None
                progressed = True
                if i >= 1:
                    self._remove(files[i - 1])          # only the newest chunk is needed as context
            if finished and not progressed:
                break
            if not progressed:
                time.sleep(2)
        shutil.rmtree(self.work, ignore_errors=True)
        if not self.clips:
            try:
                self.job_dir.rmdir()                    # nothing was saved, leave no empty folder
            except OSError:
                pass
        with self.lock:
            if self.state != "error":
                self.state = "stopped"
        self.log(f"Finished. {len(self.clips)} clip(s) saved in {self.clips_dir}")

    @staticmethod
    def _key(f: Path):
        m = re.match(r"s(\d+)_(\d+)", f.stem)
        return int(m.group(1)), int(m.group(2))

    def _remove(self, f: Path):
        for p in (f, f.with_suffix(".wav")):
            try:
                p.unlink()
            except OSError:
                pass

    def _prepare(self, f: Path, key) -> Chunk | None:
        p = run(["ffprobe", "-v", "error", "-show_entries",
                 "format=start_time,duration:stream=codec_type,start_time", "-of", "json", f.name],
                cwd=self.work, timeout=60)
        try:
            info = json.loads(p.stdout)
            fmt = info["format"]
            c = Chunk(f, key[0], key[1], pts=float(fmt["start_time"]), dur=float(fmt["duration"]))
            astart = [float(st["start_time"]) for st in info.get("streams", [])
                      if st.get("codec_type") == "audio" and "start_time" in st]
            if astart:
                c.aoff = max(0.0, astart[0] - c.pts)
        except Exception:
            return None
        if c.dur < 2:
            return None
        c.wall = f.stat().st_ctime
        wav = f.with_suffix(".wav")
        r = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", f.name, "-vn",
                 "-ac", "1", "-ar", "16000", wav.name], cwd=self.work, timeout=300)
        if r.returncode != 0 or not wav.exists():
            return None
        with wave.open(str(wav), "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        hop = int(16000 * GRID)
        n = len(pcm) // hop
        if n == 0:
            return None
        rms = np.sqrt(np.mean(pcm[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-10)
        c.db = 20 * np.log10(rms + 1e-6)

        t0 = time.time()
        lang = None if self.s.language == "auto" else self.s.language
        segs, _ = self.whisper.transcribe(
            pcm, language=lang, task="translate" if self.s.translate else "transcribe",
            word_timestamps=True, vad_filter=True, beam_size=2, condition_on_previous_text=False)
        for sg in segs:
            text = sg.text.strip()
            if not text:
                continue
            c.segments.append((sg.start, sg.end, text))
            for w in sg.words or []:
                if w.word.strip():
                    c.words.append(Word(w.start, w.end, w.word.strip()))
        with self.lock:
            self.stats["recorded"] += c.dur
        self.log(f"Chunk {key[1] + 1}: {c.dur:.0f}s of video, {len(c.words)} words heard "
                 f"(transcribed in {time.time() - t0:.0f}s).")
        if time.time() - t0 > c.dur and self.is_live and not self.warned_slow:
            self.warned_slow = True
            self.log("This PC is too busy to keep up with the stream right now. Close games or other heavy "
                     "apps, or pick 'Fast' caption accuracy; until then clips arrive later and later.")
        return c

    # ---------------------------------------------------------------- scoring
    def _analyse(self, prev: Chunk | None, cur: Chunk, final: bool):
        off = (cur.pts - prev.pts) if prev else 0.0
        if prev and not (0 < off < prev.dur + 5):
            prev, off = None, 0.0
        win_end = off + cur.dur
        lo = max(0.0, off - HOLD_BACK) if prev else 0.0
        hi = win_end if final else win_end - HOLD_BACK
        chunks = [(prev, 0.0), (cur, off)] if prev else [(cur, 0.0)]

        words, segs = timeline(chunks)
        nb = int(win_end / GRID) + 1
        db = np.full(nb, np.nan)
        for c, o in chunks:
            i0 = int((o + c.aoff) / GRID)
            seg = c.db[: max(0, nb - i0)]
            db[i0:i0 + len(seg)] = seg
        db = np.where(np.isnan(db), np.nanmedian(db), db)

        med = np.median(db)
        mad = np.median(np.abs(db - med)) * 1.4826 + 1.0
        loud = smooth(np.clip((db - med) / mad, 0, 6), 2.0)

        speaking = np.zeros(nb, dtype=bool)
        for w in words:
            speaking[max(0, int((w.start - 0.15) / GRID)): int((w.end + 0.15) / GRID) + 1] = True
        laugh = np.where(~speaking & (db > med - 3), 2.0 + np.clip((db - med) / mad, 0, 3), 0.0)
        laugh = smooth(laugh, 2.0)                      # audience or streamer laughing, cheering

        text = np.zeros(nb)
        hits = []
        for w in words:
            token = re.sub(r"[^\w]", "", w.text)
            if LAUGH_WORDS.match(token):
                bump(text, w.start, 2.0, 2.0)
                hits.append((w.start, token.lower()))
        for a, b, t in segs:
            if "!" in t:
                bump(text, b, 0.6, 2.0)

        chat = np.zeros(nb)
        if self.chat_started and self.is_live:
            lag = 6.0                                   # viewers react a few seconds after the moment
            times = np.array(self.chat, dtype=float)
            base_span = max(60.0, time.time() - self.chat_started)
            base = len(times) / base_span * 8.0         # expected laugh messages per 8 s
            for c, o in chunks:
                local = times - c.wall - lag
                for t in local[(local >= 0) & (local < c.dur)]:
                    bump(chat, t + o, 1.0, 4.0, flat=True)
            chat = np.clip((chat - base) / (np.sqrt(base) + 1.0), 0, 6)

        score = smooth(loud + 1.2 * laugh + 1.5 * text + 1.5 * chat, 3.0)
        a, b = int(lo / GRID), max(int(lo / GRID) + 1, int(hi / GRID))

        moments = []
        if self.judge and words:
            moments = self._ask_judge(words, segs, lo, hi, win_end, loud, chat)
        if moments is None or (not self.judge):
            moments = []
            zone = score.copy()
            zone[:a] = -1
            zone[b:] = -1
            peak = float(zone.max())
            need = THRESHOLDS[self.s.sensitivity]
            if len(self.history) >= 3:                  # judge each stream against its own recent minutes
                need = min(need, max(FLOOR, float(np.quantile(self.history[-30:], QUANTILE[self.s.sensitivity]))))
            self.history.append(peak)
            self.log(f"   best moment in this stretch scored {peak:.1f} (a clip needs {need:.1f}).")
            for _ in range(2):
                i = int(np.argmax(zone))
                if zone[i] < need:
                    break
                p = i * GRID
                why = []
                if loud[i] >= 1.5:
                    why.append("loud burst")
                if laugh[i] >= 1.2:
                    why.append("laughter or cheering")
                if text[i] >= 0.8:
                    near = sorted({h for t, h in hits if abs(t - p) < 6})
                    why.append("laugh words" + (f" ({', '.join(near[:3])})" if near else ""))
                if chat[i] >= 0.8:
                    why.append("chat laughing")
                s, e = snap(segs, p - 13.0, p + 8.0, win_end)
                moments.append({"start": s, "end": e, "score": round(float(zone[i]), 1),
                                "title": "", "reason": " + ".join(why) or "energy spike"})
                zone[max(0, i - int(30 / GRID)): i + int(30 / GRID)] = -1

        if self.trace:
            self.trace(dict(lo=lo, hi=hi, win_end=win_end, words=words, segs=segs, db=db, loud=loud,
                            laugh=laugh, text=text, chat=chat, score=score, moments=[dict(m) for m in moments],
                            need=locals().get("need"), peak=locals().get("peak")))
        for m in moments:
            s, e = max(0.0, m["start"]), min(win_end, m["end"])
            if e - s < MIN_CLIP:
                s = max(0.0, e - MIN_CLIP)
            if e - s > MAX_CLIP:
                s = e - MAX_CLIP
            clip_words = [w for w in words if w.end > s and w.start < e]
            if len(clip_words) < 5:
                continue                                # nothing to caption (music, silence)
            self._make_clip([c for c, _ in chunks], s, e, clip_words, m)
        with self.lock:
            self.stats["analysed"] += cur.dur

    def _ask_judge(self, words, segs, lo, hi, win_end, loud, chat):
        lines = [f"[{a:6.1f}-{b:6.1f}] {t}" for a, b, t in segs]
        spikes = [f"{i * GRID:.0f}s" for i in peaks(loud, 2.5)]
        chat_sp = [f"{i * GRID:.0f}s" for i in peaks(chat, 1.5)]
        try:
            import judge
            found = judge.find_moments(self.judge, self.title, lines, lo, hi, win_end, spikes, chat_sp)
        except Exception as e:
            self.log(f"Claude judge failed, using the built-in scoring for this chunk ({str(e)[:120]}).")
            return None
        out = []
        need = JUDGE_MIN_SCORE[self.s.sensitivity]
        for m in found:
            if m["score"] >= need and lo <= (m["start"] + m["end"]) / 2 < hi + 5:
                s, e = snap(segs, m["start"], m["end"], win_end)
                out.append({"start": s, "end": e, "score": m["score"], "title": m["title"],
                            "reason": m.get("why", "picked by Claude")})
        return out

    # ---------------------------------------------------------------- cutting
    def _make_clip(self, chunks: list[Chunk], s: float, e: float, words: list[Word], m: dict):
        n = len(self.clips) + 1
        title = m["title"] or " ".join(w.text for w in words if w.start >= max(s, e - 14))[:60]
        slug = re.sub(r"[^\w\- ]", "", title).strip().replace(" ", "_")[:40] or "clip"
        out = self.clips_dir / f"clip_{n:03d}_{slug}.mp4"
        vertical = self.s.layout != "horizontal"
        W, H = (1080, 1920) if vertical else (1280, 720)
        ass = self.work / f"cap_{n}.ass"
        ass.write_text(build_ass(words, s, e, W, H, vertical), encoding="utf-8")

        # No seeking before the input: trim/atrim cut video and audio at exactly the same
        # timestamps, and the captions are drawn after the cut so their clock is the clip's clock.
        head = f"[0:v:0]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS,fps=30"
        if self.s.layout == "vertical":
            vf = (f"{head},split=2[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,"
                  f"crop={W}:{H},boxblur=24:4,eq=brightness=-0.12[bg];[b]scale={W}:-2[fg];"
                  f"[bg][fg]overlay=(W-w)/2:(H-h)/2-120,ass={ass.name}[v]")
        elif self.s.layout == "crop":
            vf = (f"{head},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
                  f"ass={ass.name}[v]")
        else:
            vf = (f"{head},scale={W}:{H}:force_original_aspect_ratio=decrease,"
                  f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,ass={ass.name}[v]")
        vf += f";[0:a:0]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a]"
        src = "concat:" + "|".join(c.path.name for c in chunks) if len(chunks) > 1 else chunks[0].path.name
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
               "-filter_complex", vf, "-map", "[v]", "-map", "[a]", *self.encoder,
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)]
        r = run(cmd, cwd=self.work, timeout=600)
        ass.unlink(missing_ok=True)
        if r.returncode != 0 or not out.exists():
            self.log(f"Could not render a clip: {r.stderr.strip()[-200:]}")
            return
        info = {"file": out.name, "title": title, "score": m["score"], "reason": m["reason"],
                "seconds": round(e - s, 1), "made": time.strftime("%H:%M:%S"),
                "text": " ".join(w.text for w in words)}
        with self.lock:
            self.clips.append(info)
        (self.clips_dir / "clips.json").write_text(json.dumps(self.clips, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
        self.log(f"NEW CLIP  {out.name}  ({e - s:.0f}s, {m['reason']})")


# ------------------------------------------------------------------- helpers
def timeline(chunks) -> tuple[list[Word], list[tuple]]:
    """Words and sentences of consecutive chunks on one clock: seconds since the first chunk's
    start, which is also the clock of the filter graph when those chunk files are cut."""
    words, segs = [], []
    for c, o in chunks:
        shift = o + c.aoff
        words += [Word(w.start + shift, w.end + shift, w.text) for w in c.words]
        segs += [(a + shift, b + shift, t) for a, b, t in c.segments]
    return words, segs


def smooth(x: np.ndarray, seconds: float) -> np.ndarray:
    k = max(1, int(seconds / GRID))
    return np.convolve(x, np.ones(k) / k, mode="same")


def bump(arr: np.ndarray, t: float, amount: float, spread: float, flat: bool = False):
    i0, i1 = int((t - spread) / GRID), int((t + spread) / GRID) + 1
    for i in range(max(0, i0), min(len(arr), i1)):
        arr[i] += amount if flat else amount * (1 - abs(i * GRID - t) / (spread + GRID))


def peaks(x: np.ndarray, level: float) -> list[int]:
    out, x = [], x.copy()
    for _ in range(6):
        i = int(np.argmax(x))
        if x[i] < level:
            break
        out.append(i)
        x[max(0, i - 16): i + 16] = -1
    return sorted(out)


SENTENCE_END = re.compile(r"[.?!…。।][\"')\]]*$")


def snap(segs, s: float, e: float, win_end: float):
    """Move clip edges to sentence boundaries so clips do not start or end mid-sentence.
    Speech segments are not sentences (a third end on a comma or mid-phrase), so prefer
    boundaries where a sentence really ends; use any segment edge only when none is near."""
    s, e = max(0.0, s), min(win_end, e)
    segs = sorted(segs)
    is_end = [bool(SENTENCE_END.search(t.strip())) for _, _, t in segs]
    sent_ends = [b for (_, b, _), ok in zip(segs, is_end) if ok and b <= win_end]
    sent_starts = [segs[k][0] for k in range(len(segs)) if k == 0 or is_end[k - 1]]
    ends = ([b for b in sent_ends if e - 4 <= b <= e + 8]
            or [b for _, b, _ in segs if e - 2 <= b <= e + 6 and b <= win_end])
    starts = ([a for a in sent_starts if s - 8 <= a <= s + 4]
              or [a for a, _, _ in segs if s - 5 <= a <= s + 3])
    if starts:
        s = max(0.0, min(starts, key=lambda a: abs(a - s)) - 0.25)
    if ends:
        e = min(win_end, min(ends, key=lambda b: abs(b - e)) + 0.5)
    return s, e


def ass_time(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{sec:05.2f}"


def build_ass(words: list[Word], s: float, e: float, W: int, H: int, vertical: bool) -> str:
    """Word-by-word captions: three words on screen, the spoken one highlighted."""
    clean = []
    for w in words:
        t = re.sub(r"[{}\\]", "", w.text)
        t = re.sub(r"^[^\w]+|[^\w'’%]+$", "", t)
        if t:
            clean.append((max(0.0, w.start - s), min(e - s, w.end - s), t.upper(), w.text[-1:] in ".?!"))
    latin = all(ord(ch) < 0x250 for _, _, t, _ in clean for ch in t)
    font = "Arial Black" if latin else "Nirmala UI"
    size, margin = (88, 430) if vertical else (54, 60)
    head = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {W}\nPlayResY: {H}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, "
        "Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Cap,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,"
        f"{7 if vertical else 4},2,2,60,60,{margin},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    groups, cur = [], []
    for item in clean:
        if cur and (len(cur) >= 3 or sum(len(x[2]) for x in cur) + len(item[2]) > 18
                    or item[0] - cur[-1][1] > 0.6 or cur[-1][3]):
            groups.append(cur)
            cur = []
        cur.append(item)
    if cur:
        groups.append(cur)
    lines = []
    for gi, g in enumerate(groups):
        next_start = groups[gi + 1][0][0] if gi + 1 < len(groups) else e - s
        for k, (a, b, t, _) in enumerate(g):
            end = g[k + 1][0] if k + 1 < len(g) else min(max(b, a + 0.25) + 0.35, max(next_start, a + 0.1))
            if end <= a:
                end = a + 0.1
            parts = []
            for j, (_, _, tj, _) in enumerate(g):
                parts.append(r"{\c&H00E5FF&\fscx108\fscy108}" + tj + r"{\c&HFFFFFF&\fscx100\fscy100}"
                             if j == k else tj)
            lines.append(f"Dialogue: 0,{ass_time(a)},{ass_time(end)},Cap,,0,0,0,,{' '.join(parts)}")
    return head + "\n".join(lines) + "\n"
