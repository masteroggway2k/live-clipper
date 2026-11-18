"""Laughter detector: YAMNet (Google's AudioSet sound tagger, MobileNet-sized) via onnxruntime.

Tells laughter apart from applause, music and speech, so a cheering intro or walk-on music
is not mistaken for a joke landing. Runs on the CPU in well under a second per minute of audio.
The model is Qualcomm AI Hub's MIT-licensed ONNX export; it is downloaded on first use and
checked against pinned SHA-256 sums before it is loaded.
"""
from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

MODEL_URL = ("https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/yamnet/"
             "releases/v0.66.0/yamnet-onnx-float.zip")
MODEL_FOLDER = "yamnet-onnx-float"
FILES_SHA256 = {
    "yamnet.onnx": "9261fbe1f196c8bb695933845af28df0a155845c9ec5e37ec96295e82c96f096",
    "yamnet.data": "0b4de0b14bf2b528d51e3cd84e2d3e2ae826b942fbec6001c654605f1f2ceb60",
}
DEFAULT_DIR = Path.home() / ".live-clipper" / "models" / "yamnet"

# AudioSet class indices in YAMNet's 521-class output
CLASSES = {
    "laugh": [13, 14, 15, 16, 17, 18],      # Laughter, Baby laughter, Giggle, Snicker, Belly laugh, Chuckle
    "big": [17],                            # Belly laugh: someone laughing really hard
    "applause": [58, 61, 62],               # Clapping, Cheering, Applause
    "music": [132],                         # Music
    "speech": [0],                          # Speech
}
SR = 16000
PATCH = 96                                  # frames of 10 ms: one 0.96 s patch per prediction


class SoundModelError(Exception):
    """The sound model is missing, damaged or could not be downloaded."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _valid(folder: Path) -> bool:
    return all((folder / n).is_file() and _sha256(folder / n) == s for n, s in FILES_SHA256.items())


def ensure_model(model_dir: Path = DEFAULT_DIR, download: bool = True) -> Path:
    """Path to a verified yamnet.onnx, downloading it (about 14 MB) when it is missing or damaged."""
    folder = Path(model_dir) / MODEL_FOLDER
    if _valid(folder):
        return folder / "yamnet.onnx"
    if not download:
        raise SoundModelError(f"no verified model in {folder}")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=120) as r:
            data = r.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in FILES_SHA256:
                member = f"{MODEL_FOLDER}/{name}"
                folder.mkdir(parents=True, exist_ok=True)
                (folder / name).write_bytes(z.read(member))
    except Exception as e:                                    # network, disk or archive problems
        raise SoundModelError(f"could not download the laughter model ({e})") from e
    if not _valid(folder):
        raise SoundModelError("the downloaded laughter model failed its checksum")
    return folder / "yamnet.onnx"


def _mel_matrix(n_fft: int = 512, n_mels: int = 64, lo: float = 125.0, hi: float = 7500.0) -> np.ndarray:
    """HTK-scale triangular filters without the DC bin, as YAMNet's TensorFlow front end builds them."""
    hz2mel = lambda f: 1127.0 * np.log1p(np.asarray(f) / 700.0)
    mels = hz2mel(np.linspace(0, SR / 2, n_fft // 2 + 1)[1:])
    edges = np.linspace(hz2mel(lo), hz2mel(hi), n_mels + 2)
    w = np.zeros((n_fft // 2 + 1, n_mels), np.float32)
    for k in range(n_mels):
        left, centre, right = edges[k], edges[k + 1], edges[k + 2]
        w[1:, k] = np.maximum(0, np.minimum((mels - left) / (centre - left), (right - mels) / (right - centre)))
    return w


_MEL = _mel_matrix()
_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)).astype(np.float32)   # periodic Hann, 25 ms


def log_mel(pcm: np.ndarray) -> np.ndarray:
    """[frames, 64] log-mel spectrogram: 25 ms windows every 10 ms, log(mel magnitude + 0.001)."""
    if len(pcm) < 400:
        pcm = np.pad(pcm, (0, 400 - len(pcm)))
    n = 1 + (len(pcm) - 400) // 160
    idx = np.arange(400)[None, :] + 160 * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(pcm[idx] * _WINDOW, 512))
    return np.log(spec @ _MEL + 0.001).astype(np.float32)


class SoundTagger:
    def __init__(self, onnx_path: Path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def tag(self, pcm: np.ndarray, grid: float = 0.5) -> dict[str, np.ndarray]:
        """Probability of laughter, applause, music and speech for every `grid` seconds of 16 kHz audio,
        each read from the 0.96 s patch centred on that stretch."""
        bins = int(len(pcm) / (SR * grid))
        out = {k: np.zeros(bins, np.float32) for k in CLASSES}
        if bins == 0:
            return out
        mel = log_mel(pcm)
        if len(mel) < PATCH:
            mel = np.pad(mel, ((0, PATCH - len(mel)), (0, 0)), constant_values=np.log(0.001))
        for i in range(bins):
            start = int(round(((i + 0.5) * grid - 0.48) * 100))
            start = min(max(0, start), len(mel) - PATCH)
            logits = self.session.run(None, {self.input: mel[start:start + PATCH][None, None]})[0][0]
            p = np.exp(-np.logaddexp(0.0, -logits))         # sigmoid without overflow on silence
            for k, ids in CLASSES.items():
                out[k][i] = p[ids].max()
        return out


def load_tagger(log=print) -> SoundTagger | None:
    """The tagger, or None (with a log line) when the model cannot be made available."""
    try:
        return SoundTagger(ensure_model())
    except Exception as e:
        log(f"Laughter detector is off ({str(e)[:120]}).")
        return None
