"""Chat reactions: how strongly a message laughs, and when it happened in the video."""
import json

import pytest

import clipper
from conftest import make_clipper


@pytest.mark.parametrize("text, weight", [
    ("nice stream", 0.0),
    ("hello everyone", 0.0),
    ("lol", 1.0),
    ("haha", 1.0),
    ("\U0001F602", 1.0),                                   # one 😂
    ("\U0001F602\U0001F602\U0001F602\U0001F602", 2.5),     # 😂😂😂😂: repeats add up
    ("hahahaha", 3.0),
    ("HAHAHAHA", 4.0),                                     # all caps shouts it
    ("LMAO \U0001F602\U0001F602\U0001F602", 5.0),
    ("im dead \U0001F480", 3.0),                           # 💀
    ("clip it", 3.0),
    ("हाहाहा", 3.0),         # हाहाहा
    ("LOL" + " \U0001F923" * 9, 6.0),                      # capped
])
def test_chat_weight(text, weight):
    assert clipper.chat_weight(text) == pytest.approx(weight)


def _line(text, offset_ms=None, usec=1_700_000_000_000_000):
    item = {"liveChatTextMessageRenderer": {"message": {"runs": [{"text": text}]}, "timestampUsec": str(usec)}}
    rec = {"replayChatItemAction": {"actions": [{"addChatItemAction": {"item": item}}]}}
    if offset_ms is not None:
        rec["replayChatItemAction"]["videoOffsetTimeMsec"] = str(offset_ms)
    return json.dumps(rec).encode()


def test_chat_line_keeps_weight_time_and_video_offset(tmp_path):
    c = make_clipper(tmp_path, tmp_path)
    c._chat_line(_line("LMAO \U0001F602\U0001F602\U0001F602", offset_ms=65_000))
    c._chat_line(_line("good morning", offset_ms=66_000))
    assert c.chat_all == 2
    assert list(c.chat) == [(1_700_000_000.0, 5.0, 65.0)]


def test_live_chat_line_without_offset(tmp_path):
    c = make_clipper(tmp_path, tmp_path)
    c._chat_line(_line("lol"))
    assert list(c.chat) == [(1_700_000_000.0, 1.0, None)]
