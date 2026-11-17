"""Live Clipper web app. Run with run.bat, then paste a YouTube live link in the page."""
from __future__ import annotations

import os
import shutil
import sys
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from clipper import LiveClipper, Settings

PORT = int(os.environ.get("CLIPPER_PORT") or os.environ.get("PORT") or 5177)
OUT_ROOT = Path(os.environ.get("CLIPPER_OUTPUT", Path.home() / "Videos" / "LiveClipper"))
HERE = Path(__file__).parent

import logging

logging.getLogger("werkzeug").setLevel(logging.ERROR)
app = Flask(__name__, static_folder=None)
job: LiveClipper | None = None
job_lock = threading.Lock()


@app.get("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.post("/api/start")
def start():
    global job
    data = request.get_json(force=True)
    url = (data.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        return jsonify(error="Paste a full YouTube link, starting with https://"), 400
    with job_lock:
        if job and job.state in ("starting", "recording"):
            return jsonify(error="A stream is already being clipped. Stop it first."), 409
        job = LiveClipper(Settings(
            url=url,
            layout=data.get("layout", "vertical"),
            sensitivity=data.get("sensitivity", "medium"),
            language=data.get("language", "auto"),
            translate=bool(data.get("translate")),
            model=data.get("model", "small"),
            cookies_browser=data.get("cookies", ""),
        ), OUT_ROOT)
        job.start()
    return jsonify(ok=True)


@app.post("/api/stop")
def stop():
    if job:
        threading.Thread(target=job.stop, daemon=True).start()
    return jsonify(ok=True)


@app.get("/api/status")
def status():
    if not job:
        return jsonify(state="idle", clips=[], logs=[], folder=str(OUT_ROOT))
    return jsonify(job.status())


@app.get("/clip/<path:name>")
def clip(name):
    if not job:
        return "", 404
    return send_from_directory(job.clips_dir, name, conditional=True)


@app.post("/api/open-folder")
def open_folder():
    folder = job.clips_dir if job else OUT_ROOT
    folder.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(folder)  # noqa: S606 - opens the clips folder in Explorer
    return jsonify(ok=True)


if __name__ == "__main__":
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} was not found. Install FFmpeg and make sure it is on PATH.")
    if "--open" in sys.argv:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    print(f"Live Clipper is running at http://127.0.0.1:{PORT}  (close this window to quit)")
    app.run(host="127.0.0.1", port=PORT, threaded=True)
