# Live Clipper

Paste a YouTube live link and it keeps cutting the funny moments into captioned clips until the stream ends or you press Stop.

## Use it

1. Double-click `run.bat`. The first run installs what it needs (a few minutes), then a page opens in your browser.
2. Paste the live stream link and press **Start clipping**.
3. Clips appear on the page and are saved in `Videos\LiveClipper\<date and time>\`.

A clip shows up roughly two to three minutes after the moment happened: the tool records in one-minute chunks and needs a little of what comes next so clips do not end mid-sentence.

Needs Python 3.10+ and FFmpeg (`winget install Gyan.FFmpeg`). An NVIDIA GPU is used automatically when there is one.

## How it decides what is funny

For every stretch of the stream it combines:

- **Loud bursts** in the audio, which is what laughter and shouting look like
- **Laugh words** in the speech ("haha", "lol", "funny", ...)
- **Live chat**: spikes of viewers typing lol, 😂, 💀 and similar

**How picky** sets how high that combined score must be. These signals find loud, lively moments; they cannot understand a joke.

### Better picks with Claude (optional)

If an Anthropic API key is set, Claude reads the transcript of each stretch, picks the moments that are actually funny, chooses where each clip starts and ends, and writes a title. To turn it on, run this once in a terminal and restart `run.bat`:

```
setx ANTHROPIC_API_KEY "your-key"
```

This uses `claude-opus-5-5` and costs roughly one to two US dollars per hour of stream. Set `CLIPPER_JUDGE_MODEL` (for example to `claude-haiku-5-5`) to use a cheaper model.

## Checking how it picks

Two tools let you record a stretch of a stream once and replay it through the scoring as often as you like:

```
python tools/record_fixture.py <youtube url> %USERPROFILE%\.live-clipper\fixtures\my-stream 600
python tools/replay.py %USERPROFILE%\.live-clipper\fixtures\my-stream --sensitivity medium
python tools/replay.py %USERPROFILE%\.live-clipper\fixtures\my-stream --sensitivity medium --judge
```

Use the app's Python (`%USERPROFILE%\.live-clipper\venv\Scripts\python`). The replay prints the best moments of each minute, what they scored on loudness, laughter, laugh words and chat, the words around them, and the clips it would make. `--judge` lets Claude pick instead (needs the API key), and `--render` also cuts the clips. Recordings stay out of git.

## Settings

| Setting | What it does |
|---|---|
| Clip shape | Vertical 9:16 for Shorts/Reels/TikTok (blurred background or zoomed in), or the original 16:9 |
| How picky | Fewer, stronger clips or more, weaker ones |
| Spoken language | Leave on automatic unless captions come out in the wrong language |
| Caption accuracy | Bigger speech models are more accurate and slower |
| Captions in English | Translates the speech so captions are English whatever is spoken |
| If YouTube asks to sign in | Lets the downloader use the YouTube login of your browser |

Normal (not live) YouTube videos work too; they are processed as fast as your PC allows.

## Before you post

Clips contain someone else's stream. Post them only if you have the streamer's permission or their clipping rules allow it.
