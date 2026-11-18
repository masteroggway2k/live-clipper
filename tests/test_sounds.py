"""The laughter detector: real crowd laughter and applause are told apart, silence is nothing,
and a damaged model file is refused instead of being loaded."""
import subprocess
from pathlib import Path

import numpy as np
import pytest

import sounds

DATA = Path(__file__).parent / "data"


def load(name: str) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(DATA / name), "-ac", "1", "-ar", "16000",
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


@pytest.fixture(scope="module")
def tagger():
    try:
        return sounds.SoundTagger(sounds.ensure_model())
    except sounds.SoundModelError as e:
        pytest.skip(f"laughter model unavailable: {e}")


def test_laughter_is_tagged_as_laughter(tagger):
    t = tagger.tag(load("laughter-only.mp3"))
    assert t["laugh"].max() >= 0.15
    assert t["applause"].max() < 0.1


def test_applause_is_tagged_as_applause(tagger):
    t = tagger.tag(load("laughter-clapping.mp3"))
    assert t["applause"][-8:].mean() >= 0.8                  # the last 4 s are a crowd clapping
    assert t["laugh"][-8:].max() < 0.05


def test_silence_tags_nothing(tagger):
    t = tagger.tag(np.zeros(16000 * 5, np.float32))
    assert all(v.max() < 0.05 for v in t.values())


def test_one_value_per_grid_bin(tagger):
    t = tagger.tag(np.zeros(16000 * 10, np.float32), grid=0.5)
    assert set(t) == {"laugh", "big", "applause", "music", "speech"}
    assert all(len(v) == 20 for v in t.values())


def test_missing_model_is_refused_without_download(tmp_path):
    with pytest.raises(sounds.SoundModelError):
        sounds.ensure_model(tmp_path, download=False)


def test_damaged_model_is_refused(tmp_path):
    folder = tmp_path / sounds.MODEL_FOLDER
    folder.mkdir()
    for name in sounds.FILES_SHA256:
        (folder / name).write_bytes(b"not a model")
    with pytest.raises(sounds.SoundModelError):
        sounds.ensure_model(tmp_path, download=False)
