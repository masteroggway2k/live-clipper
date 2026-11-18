"""What the user sees when YouTube refuses to hand over the stream."""
import yt_dlp

import clipper
from conftest import make_clipper


def _open_fails(tmp_path, monkeypatch, message):
    c = make_clipper(tmp_path, tmp_path)

    def refuse():
        raise yt_dlp.utils.DownloadError(message)

    monkeypatch.setattr(c, "_resolve", refuse)
    c._record_loop()
    return c


def test_bot_check_message(tmp_path, monkeypatch):
    c = _open_fails(tmp_path, monkeypatch, "ERROR: [youtube] x: Sign in to confirm you’re not a bot. "
                                           "Use --cookies-from-browser or --cookies for the authentication.")
    assert c.state == "error"
    assert c.error == clipper.BOT_CHECK_HELP
    assert c.rec_done.is_set()


def test_other_open_errors_unchanged(tmp_path, monkeypatch):
    c = _open_fails(tmp_path, monkeypatch, "ERROR: Video unavailable")
    assert c.state == "error"
    assert c.error.startswith("Could not open that link:")


def test_internet_drop_mid_stream_is_waited_out(tmp_path, monkeypatch):
    c = make_clipper(tmp_path, tmp_path)
    c.retry_wait = 0.0
    calls = {"n": 0}

    def resolve():
        calls["n"] += 1
        if calls["n"] == 1:
            return {"is_live": True, "title": "t", "url": "http://127.0.0.1:9/never", "http_headers": {}}
        if calls["n"] <= 12:                            # about two minutes of failed lookups
            raise yt_dlp.utils.DownloadError("ERROR: Unable to download API page: Failed to resolve")
        c.stop_flag.set()                               # the test ends here
        raise yt_dlp.utils.DownloadError("ERROR: stop")

    monkeypatch.setattr(c, "_resolve", resolve)
    c._record_loop()
    assert calls["n"] == 13
    assert c.state != "error"
    assert not any("no longer available" in line for line in c.logs)
