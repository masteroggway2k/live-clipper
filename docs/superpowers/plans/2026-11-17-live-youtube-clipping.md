# Live YouTube Clipping: Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the working Live Clipper prototype to a dependable tool. Claude picks the funny moments, every fix found so far is pinned by an automated test, and the code is published on GitHub.

**Architecture:** Keep the current single-process design: `app.py` (Flask page) → `clipper.py` (record → transcribe → score → cut) → `judge.py` (Claude picks moments). Tests drive `LiveClipper` methods directly on synthetic `.ts` chunks built with ffmpeg, with a fake speech model, so no network, GPU or API key is needed for the suite. Tasks 2–4 are tests-first. Task 1 ports regression checks that already pass on the current code (those fixes were proven failing-first earlier). Task 3's live check is manual, because it needs the user's key and YouTube.

**Tech Stack:** Python 3.10+, Flask, yt-dlp, faster-whisper, ffmpeg/ffprobe 7+, anthropic SDK 2.x, pytest.

**Spec:** No separate spec document. Requirements come from the user's request ("paste a YouTube live link; it keeps creating videos of the best funny moments with captions") and the 2026-11-17 debugging session. Its findings are summarised under Background. `README.md` describes the user-facing behaviour.

## Background (findings this plan acts on)

- Caption delay is fixed in `clipper.py` (trim/atrim cutting plus the per-chunk `Chunk.aoff`). It was verified only by scratch scripts that are not in the repo.
- Mid-sentence clip edges are fixed (`snap()` plus `SENTENCE_END`), proven test-first by a scratch script. That fix is **uncommitted** in `clipper.py`, along with a read-only `self.trace` hook in `_analyse`.
- The built-in score treats applause and music as laughter: the "laughter" signal is just loud audio with no speech. The user chose the Claude judge as the fix. `judge.py` has never been called against the API.
- Why stand-up jokes get missed is undiagnosed: only 85 s of stream could be recorded before YouTube answered "Sign in to confirm you're not a bot".
- The repo has one local commit (`1151da0`) and is not pushed. The GitHub CLI is signed in as `masteroggway2k`.

## Global Constraints

- Windows 11 first: the app runs from `run.bat` with its venv at `%USERPROFILE%\.live-clipper\venv`, and ffmpeg/ffprobe must be on PATH.
- Run every test with: `"$USERPROFILE/.live-clipper/venv/Scripts/python" -m pytest tests -v` from `live-clipper/`.
- Captions and picture stay within **40 ms** of the audio in all three layouts: `vertical`, `crop`, `horizontal`.
- Clips are **12–45 s** (`MIN_CLIP`, `MAX_CLIP`) and start and end on sentence boundaries whenever one lies within the snap window.
- Judge model defaults to `claude-opus-5-5`, and `CLIPPER_JUDGE_MODEL` overrides it **at call time**. Requests carry `betas=["server-side-fallback-2026-07-01"]`, `fallbacks="default"`, and `output_config={"effort": "low", "format": {"type": "json_schema", ...}}`.
- The judge never crashes a run: any judge failure falls back to built-in scoring for that stretch.
- No API keys, personal emails or local user paths in the repo. Commits use `150432289+masteroggway2k@users.noreply.github.com`.
- Recorded fixtures (real stream video) live outside the repo in `%USERPROFILE%\.live-clipper\fixtures\<name>\` and are never committed.

## Review Focus

1. **Stream titles with emoji or non-Latin text** (`🔴 LIVE`, Hindi): clip files are still written and logging never raises. Pinned by `test_clip_with_emoji_title` (Task 1).
2. **A chunk whose audio starts later than its video** (seen live: 0.48 s): captions stay in sync. Pinned by `test_sync[...]` with `audio_delay=0.5` (Task 1).
3. **A clip that crosses a chunk boundary**: same 40 ms sync. Pinned by `test_sync[...]` using window 50.37–72.37 s (Task 1).
4. **The judge returns a refusal, cut-off output or broken JSON**: the stretch falls back to built-in scoring, with no crash and no empty run. Pinned by `test_max_tokens_raises`, `test_bad_json_raises` and `test_ask_judge_returns_none_on_failure` (Task 2).
5. **YouTube's bot check at start**: the page tells the user exactly which setting to change, not a raw yt-dlp error. Pinned by `test_bot_check_message` (Task 4).

---

### Task 1: Regression test suite for sync, sentence edges and titles

**Files:**
- Create: `tests/conftest.py`, `tests/test_sync.py`, `tests/test_snap.py`, `requirements-dev.txt` (`pytest`)
- Modify: none (the snap fix and trace hook already in `clipper.py` are committed with this task)

**Interfaces:**
- Consumes: `clipper.LiveClipper`, `Settings`, `timeline(chunks)`, `snap(segs, s, e, win_end)`, `SENTENCE_END`, `LiveClipper._prepare(f, key)`, `LiveClipper._make_clip(chunks, s, e, words, m)`.
- Produces (in `tests/conftest.py`):
  - fixture `synthetic_chunks(tmp_path) -> Path`: a folder holding `s0_00000.ts`… built as below
  - `class FakeWhisper` with `transcribe(pcm, **kw) -> tuple[list, None]`, which returns one segment `"Beep!"` with one word `"BEEP"` per beep found in `pcm` (energy > 0.05 per 10 ms frame)
  - `make_clipper(tmp_path: Path, work: Path, layout: str = "vertical", sensitivity: str = "medium") -> clipper.LiveClipper`, with `work`, `whisper=FakeWhisper()` and `encoder=_pick_encoder()` set
  - `measure(clip: Path, layout: str) -> dict[str, list[float]]` with keys `beeps` (from `silencedetect`), `flashes` (YAVG > 140 in a band above the captions) and `captions` (UMIN < 70 in the caption band), all onset times in seconds

`synthetic_chunks` builds the source like this. Keep these exact commands: flashes and beeps land on every whole second, and the audio stream starts 0.5 s after the video while staying in phase.

```bash
ffmpeg -y -loglevel error -f lavfi -i "nullsrc=s=640x360:r=30:d=150,geq=lum='if(lt(mod(T\,1)\,0.1)\,235\,40)':cb=128:cr=128,format=yuv420p" \
  -itsoffset 0.5 -f lavfi -i "aevalsrc='if(lt(mod(t+0.5\,1)\,0.1)\,0.8*sin(2*PI*1000*t)\,0)':s=48000:d=149.5" \
  -map 0:v -map 1:a -c:v libx264 -g 60 -keyint_min 60 -sc_threshold 0 -preset ultrafast -c:a aac -f mpegts src.ts
ffmpeg -y -loglevel error -i src.ts -map 0:v:0 -map 0:a:0 -c copy -f segment -segment_time 60 -segment_format mpegts s0_%05d.ts
```

Crop bands for `measure` (flash band, caption band): vertical `1080:120:0:560` / `1080:110:0:1385`; crop `1080:200:0:100` / `1080:110:0:1385`; horizontal `1280:150:0:20` / `1280:70:0:595`.

- [ ] **Step 1: Write the tests**

```python
# tests/test_sync.py
@pytest.mark.parametrize("layout", ["vertical", "crop", "horizontal"])
def test_sync(tmp_path, synthetic_chunks, layout):
    c = make_clipper(tmp_path, synthetic_chunks, layout)
    a, b = (c._prepare(f, (0, i)) for i, f in enumerate(sorted(synthetic_chunks.glob("s0_*.ts"))[:2]))
    assert a.aoff == pytest.approx(0.48, abs=0.05)          # the late-audio case is really exercised
    words, _ = clipper.timeline([(a, 0.0), (b, b.pts - a.pts)])
    s, e = 50.37, 72.37                                     # crosses the chunk boundary at ~60 s
    c._make_clip([a, b], s, e, [w for w in words if w.end > s and w.start < e],
                 {"title": "sync", "score": 0, "reason": "test"})
    m = measure(next(c.clips_dir.glob("clip_001_*.mp4")), layout)
    near = lambda xs: max(min(abs(x - y) for y in m["beeps"]) for x in xs[:18] if x < m["beeps"][-1])
    assert near(m["captions"]) <= 0.040
    assert near(m["flashes"]) <= 0.040

def test_clip_with_emoji_title(tmp_path, synthetic_chunks):
    # same setup as test_sync (vertical), title "🔴 LIVE हाहा 😂"
    # assert exactly one clip_001_*.mp4 exists and "LIVE" is in its name
```

```python
# tests/test_snap.py — segments mimic the recorded comedy transcript
SEGS = [(0.0, 1.6, "So, you"), (1.6, 4.9, "do what you do."), (5.0, 7.0, "You know, I always tell the guys,"),
        (7.0, 9.5, "I'm like, yo, when extraordinary is possible,"), (9.5, 12.0, "good just ain't good enough."),
        (12.2, 14.0, "We gotta be the best."), (14.1, 17.0, "Let us fly above the radar."),
        (17.2, 20.0, "All this hard work, working on a road"), (20.0, 22.5, "for the past few years,"),
        (22.5, 25.0, "This is about to pay off."), (25.3, 28.0, "Ladies and gentlemen, put your hands together"),
        (28.0, 31.0, "and welcome to the stage, your host."), (31.2, 33.0, "What's happening?")]
# test_edges_land_on_sentences: for p in 15.0..20.0 step 0.5: s, e = snap(SEGS, p-13, p+8, 33.5)
#   assert e == 33.5 or round(e - 0.5, 2) is the end of a segment whose text matches SENTENCE_END
#   assert s == 0 or round(s + 0.25, 2) is the start of a segment whose previous segment matches SENTENCE_END
# test_falls_back_to_segment_edge: snap([(0, 30, "no punctuation"), (30, 60, "still none")], 5, 28, 60) == (0.0, 30.5)
# test_sentence_end_pattern: matches "Thank you.", 'right?"', "wow!)", "हाँ।"; not "years,", "hands together"
```

- [ ] **Step 2: Run the tests.** Run: `"$USERPROFILE/.live-clipper/venv/Scripts/python" -m pytest tests -v`. Expected: all PASS. These pin fixes that already exist; if `test_sync` fails, the fix regressed, so stop and debug.
- [ ] **Step 3: Prove the sync test can fail.** Temporarily set `aoff` to `0.0` in `LiveClipper._prepare` and run `test_sync`. Expected: FAIL on `aoff == approx(0.48)`. Restore the line and re-run (PASS).
- [ ] **Step 4: Commit**

```bash
git add clipper.py tests requirements-dev.txt
git commit -m "test: pin caption sync, sentence edges and emoji titles; snap clip edges to sentences"
```

---

### Task 2: Make the Claude judge safe to call

**Files:**
- Modify: `judge.py` (whole file, 79 lines), `clipper.py:609-627` (`_ask_judge`)
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: `clipper.snap`, `JUDGE_MIN_SCORE`.
- Produces:
  - `judge.model_name() -> str`, which reads `CLIPPER_JUDGE_MODEL` at call time (default `"claude-opus-5-5"`). This replaces the module constant `MODEL`.
  - `class judge.JudgeError(Exception)`
  - `judge.find_moments(client, title, lines, lo, hi, win_end, loud_spikes, chat_spikes) -> list[dict]` (unchanged signature). It returns `[]` on `stop_reason == "refusal"` and raises `JudgeError` on `stop_reason == "max_tokens"` or JSON that fails to parse.
  - `LiveClipper._ask_judge(...) -> list[dict] | None`. `None` means "judge failed; use built-in scoring for this stretch" (the caller already handles `None`).
  - `judge.SYSTEM` gains this sentence: "Applause, cheering, music and introductions are not funny by themselves; only clip them when a joke or reaction inside them is."

Tests use `FakeClient`: an object whose `.beta.messages.create(**kw)` records `kw` and returns `SimpleNamespace(stop_reason=..., content=[SimpleNamespace(type="text", text=...)])`.

- [ ] **Step 1: Write the failing tests**

```python
def test_request_shape(monkeypatch):            # CLIPPER_JUDGE_MODEL=claude-haiku-5-5 set before the call
    # assert kw["model"] == "claude-haiku-5-5"; kw["betas"] == ["server-side-fallback-2026-07-01"]
    # kw["fallbacks"] == "default"; kw["output_config"]["effort"] == "low"
    # kw["output_config"]["format"]["type"] == "json_schema"
def test_refusal_returns_empty():               # stop_reason "refusal" -> []
def test_max_tokens_raises():                   # stop_reason "max_tokens" -> pytest.raises(judge.JudgeError)
def test_bad_json_raises():                     # text "{not json" -> pytest.raises(judge.JudgeError)
def test_prompt_rejects_applause():             # "applause" in judge.SYSTEM.lower()
def test_ask_judge_returns_none_on_failure(monkeypatch):   # find_moments raises JudgeError -> _ask_judge(...) is None
def test_ask_judge_filters_and_snaps(monkeypatch):
    # find_moments returns [{start 14, end 30, score 9}, {start 14, end 30, score 3}, {start 200, end 215, score 9}]
    # with lo=0, hi=60, sensitivity "medium" and segments = test_snap.SEGS:
    # exactly one moment comes back, and its start/end sit on sentence boundaries (same rule as test_snap)
```

- [ ] **Step 2: Run them to see them fail.** Run: `... -m pytest tests/test_judge.py -v`. Expected: FAIL (`model_name`/`JudgeError` not defined, request model still `claude-opus-5-5`).
- [ ] **Step 3: Implement** `model_name()`, `JudgeError` and the stop-reason/JSON handling in `judge.py`. Then make `_ask_judge` catch `Exception`, log once per failure as it does now, and return `None`.
- [ ] **Step 4: Run the whole suite.** Run: `... -m pytest tests -v`. Expected: all PASS.
- [ ] **Step 5: Commit** `git commit -am "fix: judge reads model at call time, rejects applause, fails safe"` (add `tests/test_judge.py`).

---

### Task 3: Replay tools and a live judge check (manual, needs the user's API key)

**Files:**
- Create: `tools/record_fixture.py`, `tools/replay.py` (port from the session scratch scripts), `README.md` section "Checking how it picks"
- Modify: `.gitignore`: add `fixtures/`

**Interfaces:**
- Consumes: `LiveClipper._resolve`, `_prepare`, `_analyse`, the `self.trace` callback dict (`lo, hi, win_end, words, segs, db, loud, laugh, text, chat, score, moments, need, peak`), `judge.make_judge()`.
- Produces: the CLIs
  - `python tools/record_fixture.py <url> <out_dir> <seconds>`, which records 60 s `s0_%05d.ts` chunks with the recorder's exact ffmpeg arguments and prints with UTF-8-safe output
  - `python tools/replay.py <fixture_dir> --sensitivity {low,medium,high} [--judge] [--render]`, which caches transcripts in `<fixture_dir>/chunks.pkl`, prints the five best peaks per stretch with their signal parts and the surrounding transcript, then `MOMENT` lines and a final `CLIPS: <n>`

- [ ] **Step 1: Port the scripts**, add `--judge` (sets `use_judge=True` and `c.judge = judge.make_judge()`; exit with a clear message if that returns `None`), and run `replay.py` on the synthetic chunks from Task 1. Expected: it runs to `CLIPS:` without errors.
- [ ] **Step 2: Move the 85 s sample** from the session scratchpad to `%USERPROFILE%\.live-clipper\fixtures\comedy-intro\`. Run `replay.py ... --sensitivity high` and expect `CLIPS: 1` (the applause intro: this is the bug).
- [ ] **Step 3 (needs key): run the same replay with `--judge`.** Expected: the intro moment around 44.7–69.2 s is **not** picked. Cost is under $0.05.
- [ ] **Step 4 (when YouTube stops the bot check): record ≥ 8 min** of a live stand-up stream to `fixtures/comedy-long/`, then replay with and without `--judge` at `medium`. Accept when the judge picks at least 1 clip per 3 minutes of material with crowd laughter, and no picked clip is only intro, applause or music. Write the per-mode clip counts into the README section.
- [ ] **Step 5: Commit** `tools/`, `.gitignore` and `README.md`: `git commit -m "feat: fixture recorder and replay tool for checking picks"`.

---

### Task 4: Clear message when YouTube asks for a sign-in

**Files:**
- Modify: `clipper.py:159-230` (`_record_loop`)
- Test: `tests/test_recorder.py`

**Interfaces:**
- Produces: the constant `clipper.BOT_CHECK_HELP = 'YouTube is asking this PC to prove it is not a bot. Pick your browser under "If YouTube asks to sign in" and press Start again.'`

- [ ] **Step 1: Write the failing tests**

```python
def test_bot_check_message(tmp_path, monkeypatch):
    # c._resolve raises yt_dlp.utils.DownloadError("ERROR: [youtube] x: Sign in to confirm you’re not a bot. Use --cookies-from-browser")
    # c._record_loop() (called directly) -> c.state == "error", c.error == clipper.BOT_CHECK_HELP, c.rec_done.is_set()
def test_other_open_errors_unchanged(tmp_path, monkeypatch):
    # DownloadError("ERROR: Video unavailable") -> c.error starts with "Could not open that link:"
```

- [ ] **Step 2: Run them to see them fail.** Run: `... -m pytest tests/test_recorder.py -v`. Expected: `test_bot_check_message` FAILS (raw yt-dlp text).
- [ ] **Step 3: Implement.** In `_record_loop`, detect `"confirm you" in msg and "bot" in msg` (the message's apostrophe varies). At session 0, call `_fail(BOT_CHECK_HELP)`. Mid-stream, log `BOT_CHECK_HELP` and return.
- [ ] **Step 4: Run the whole suite.** Expected: all PASS.
- [ ] **Step 5: Commit** `git commit -am "fix: explain YouTube's bot check instead of showing the raw error"` (add the test file).

---

### Task 5: Publish to GitHub

**Files:** none changed.

- [ ] **Step 1: Check the suite and the tree.** Run: `... -m pytest tests -v`, then `git status --short`. Expected: all PASS and a clean tree.
- [ ] **Step 2: Check for anything private.** Run `git grep -nIiE "sk-ant|gmail|AppData|Users\\\\" -- . ':!docs'`. Expected: no output.
- [ ] **Step 3: Create the repo and push.** Run: `"/c/Program Files/GitHub CLI/gh.exe" repo create live-clipper --public --source . --push`
- [ ] **Step 4: Verify.** Run: `gh repo view masteroggway2k/live-clipper --json url,visibility,defaultBranchRef`. Expected: `"visibility":"PUBLIC"`, branch `main`, and the URL to give the user.
