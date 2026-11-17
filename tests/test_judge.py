"""The Claude judge: request shape, safe failure, and how its picks are filtered and snapped."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

import clipper
import judge
from conftest import make_clipper
from test_snap import SEGS, WIN_END, on_end, on_start, sentence_ends, sentence_starts


class FakeClient:
    """Stands in for anthropic.Anthropic(): records the request, returns a canned response."""

    def __init__(self, stop_reason="end_turn", text='{"moments": []}'):
        self.kw = None
        self._resp = NS(stop_reason=stop_reason, content=[NS(type="text", text=text)])
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.kw = kw
        return self._resp


def _call(client):
    return judge.find_moments(client, "title", ["[  0.0-  2.0] hi."], 0, 60, 85, [], [])


def test_request_shape(monkeypatch):
    monkeypatch.setenv("CLIPPER_JUDGE_MODEL", "claude-haiku-5-5")
    client = FakeClient()
    _call(client)
    kw = client.kw
    assert kw["model"] == "claude-haiku-5-5"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["fallbacks"] == "default"
    assert kw["output_config"]["effort"] == "low"
    assert kw["output_config"]["format"]["type"] == "json_schema"


def test_default_model(monkeypatch):
    monkeypatch.delenv("CLIPPER_JUDGE_MODEL", raising=False)
    assert judge.model_name() == "claude-opus-5-5"


def test_refusal_raises():
    # a declined stretch must fall back to built-in scoring, not silently produce no clips
    with pytest.raises(judge.JudgeError):
        _call(FakeClient(stop_reason="refusal", text=""))


def test_max_tokens_raises():
    with pytest.raises(judge.JudgeError):
        _call(FakeClient(stop_reason="max_tokens", text='{"moments": [{"start": 1'))


def test_bad_json_raises():
    with pytest.raises(judge.JudgeError):
        _call(FakeClient(text="{not json"))


def test_prompt_rejects_applause():
    assert "applause" in judge.SYSTEM.lower()


def _ask(tmp_path, monkeypatch, behaviour):
    c = make_clipper(tmp_path, tmp_path, sensitivity="medium")
    c.judge = FakeClient()
    monkeypatch.setattr(judge, "find_moments", behaviour)
    z = np.zeros(int(WIN_END / clipper.GRID) + 1)
    return c._ask_judge([], SEGS, 0.0, 60.0, WIN_END, z, z)


def test_ask_judge_returns_none_on_failure(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise judge.JudgeError("cut off")
    assert _ask(tmp_path, monkeypatch, boom) is None


def test_ask_judge_filters_and_snaps(tmp_path, monkeypatch):
    picks = [{"start": 14, "end": 30, "score": 9, "title": "keep", "why": "x"},
             {"start": 14, "end": 30, "score": 3, "title": "weak", "why": "x"},
             {"start": 200, "end": 215, "score": 9, "title": "outside", "why": "x"}]
    out = _ask(tmp_path, monkeypatch, lambda *a, **k: json.loads(json.dumps(picks)))
    assert [m["title"] for m in out] == ["keep"]
    s, e = out[0]["start"], out[0]["end"]
    assert e == WIN_END or on_end(e, sentence_ends(SEGS))
    assert s == 0 or on_start(s, sentence_starts(SEGS))
