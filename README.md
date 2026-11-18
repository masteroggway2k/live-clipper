# Live Clipper

Paste a YouTube live link and it keeps cutting the funny moments into captioned clips until the stream ends or you press Stop.

## Use it

1. Double-click `run.bat`. The first run installs what it needs (a few minutes), then a page opens in your browser.
2. Paste the live stream link and press **Start clipping**.
3. Clips appear on the page and are saved in `Videos\LiveClipper\<date and time>\`.

Each clip is 30 to 35 seconds: the setup, the punchline and the reaction, cut on whole sentences, and never repeating the previous clip. A clip shows up roughly two to three minutes after the moment happened: the tool records in one-minute chunks and needs a little of what comes next so clips do not end mid-sentence.

Finished streams and ordinary videos work too: paste their link the same way. They are processed as fast as your internet connection allows, and stop by themselves at the end. If the connection drops during a live stream, the app keeps trying for about five minutes before giving up.

Needs Python 3.10+ and FFmpeg (`winget install Gyan.FFmpeg`). An NVIDIA GPU is used automatically when there is one.

## How it decides what is funny

For every stretch of the stream it combines:

- **Laughter and how hard it is**, heard by a small sound-recognition model (YAMNet) that tells laughter apart from applause, cheering and music. Each laugh is graded by strength, length and loudness:
  - **big laugh**: a belly laugh, three seconds or more of clear laughter, or laughter far louder than the stream usually is
  - **laugh**: an ordinary laugh
  - **chuckle**: short and faint
- **Live chat reactions and how strong they are.** A mild reaction ("lol", "haha", one 😂) counts 1; a strong one ("LMAO", "ROFL", "dead", "hahahaha", 🤣, 💀, "हाहाहा", "clip it") counts 3; writing it in capitals adds 1 and each extra laugh emoji adds 0.5, up to 6. A spike well above the stream's usual chat counts. For finished streams the chat replay is used, placed where each message appeared in the video.
- **Laugh words** spoken in the stream: haha, hehe, lol, lmao, rofl, hilarious, funny, laugh/laughing, joke, omg, bruh, wtf, crazy, insane, ridiculous, stupid, idiot, dead, dying, हाहा, मज़ाक, पागल
- **Loud bursts** in the audio, ignored while the model hears applause or music

**How picky** sets which laughs are enough: *Only the best* clips big laughs, *Balanced* clips laughs, *Generous* also clips chuckles. After the first few minutes the bar follows the stream itself, stricter on a stream full of laughs and more lenient on a quiet one, so you get its best moments rather than all of them. Each clip on the page says why it was picked, for example "big laugh (4 s) + chat laughing".

These signals find moments people laugh at; they cannot understand a joke nobody laughs at.

The laughter model (about 14 MB, MIT licence, from Qualcomm AI Hub) downloads the first time you start clipping and is checked against a fixed checksum before it is used. It runs on the CPU and takes well under a second per minute of stream. If it cannot be downloaded, the app falls back to treating any loud stretch without talking as laughter, and says so in the activity log.

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
