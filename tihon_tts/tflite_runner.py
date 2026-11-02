"""Run TiHon-TTS from the two TFLite graphs (no PyTorch needed): numpy glue + harmonic-plus-noise synthesis.

    encoder.tflite: syllables [1, 64, 7] int32, num_syllables [1] int32, speaker [1] int32
                    → token_states [1, 321, 56], durations [1, 321]
    decoder.tflite: token_states, durations, frame_offset [1] int32, num_frames [1] int32, speaker [1] int32
                    → f0_hz [1, 96], voiced [1, 96], log_harmonic [1, 96, 40], log_noise [1, 96, 24]

The decoder sees 192 frames (96 output frames + 48 context frames on each side) per call; `frame_offset` is the
index of its first input frame (output frame − 48).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .text import BOUNDARIES, text_to_ids

SR, HOP, N_FFT = 24_000, 256, 1024
N_MAX, CENTER, CONTEXT = 64, 96, 48


def _mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _band_matrix(bins: int, bands: int, nyq: float) -> np.ndarray:
    pos = np.clip(_mel(np.linspace(0, nyq, bins)) / _mel(nyq), 0, 1) * (bands - 1)
    i0 = np.minimum(np.floor(pos), bands - 2).astype(int)
    fr = pos - i0
    m = np.zeros((bins, bands))
    m[np.arange(bins), i0] = 1 - fr
    m[np.arange(bins), i0 + 1] = fr
    return m.T


def _frames_to_samples(x: np.ndarray, length: int) -> np.ndarray:
    """Linear interpolation from T frames to `length` ≤ (T − 1)·hop samples (frame t at sample t·hop)."""
    fr = (np.arange(HOP) / HOP).astype(x.dtype).reshape(1, HOP, *([1] * (x.ndim - 1)))
    a = x[:-1, None]
    return (a + (x[1:, None] - a) * fr).reshape(-1, *x.shape[1:])[:length]


def _stft(x: np.ndarray, window: np.ndarray) -> np.ndarray:
    x = np.pad(x, N_FFT // 2, mode="reflect")
    n = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    return np.fft.rfft(x[idx] * window, axis=-1)  # (frames, bins)


def _istft(spec: np.ndarray, window: np.ndarray, length: int) -> np.ndarray:
    frames = np.fft.irfft(spec, n=N_FFT, axis=-1) * window
    n = len(frames)
    y, norm = np.zeros(N_FFT + HOP * (n - 1)), np.zeros(N_FFT + HOP * (n - 1))
    for j in range(N_FFT // HOP):  # overlap-add, one hop-sized quarter of every frame at a time
        y[j * HOP:(j + n) * HOP] += frames[:, j * HOP:(j + 1) * HOP].reshape(-1)
        norm[j * HOP:(j + n) * HOP] += np.tile(window[j * HOP:(j + 1) * HOP] ** 2, n)
    y = y / np.where(norm > 1e-11, norm, 1.0)
    return y[N_FFT // 2:N_FFT // 2 + length]


class HarmonicNoiseSynth:
    """numpy port of the PyTorch synthesiser (same maths; the noise RNG differs)."""

    def __init__(self, harmonic_phase: np.ndarray, fir: np.ndarray, kh: int = 40, kn: int = 24) -> None:
        self.theta, self.fir, self.kh = harmonic_phase.astype(np.float64), fir.astype(np.float64), kh
        bins, nyq = N_FFT // 2 + 1, SR / 2
        self.k = np.arange(1, len(harmonic_phase) + 1, dtype=np.float64)
        self.noise_interp, self.harm_interp = _band_matrix(bins, kn, nyq), _band_matrix(bins, kh, nyq)
        self.bin_hz = np.linspace(0, nyq, bins)
        self.window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)  # periodic Hann
        self.mvf_hz, self.mp_low_hz, self.mp_lifter, self.eq = 0.0, 0.0, 60, []

    def _mvf(self, hz):
        r = np.clip((hz - self.mvf_hz + 1000) / 2000, 0, 1)
        return 0.5 - 0.5 * np.cos(np.pi * r)

    def _eq(self, hz):
        p = np.asarray(self.eq, dtype=np.float64)
        return 10 ** (np.interp(np.log(np.maximum(hz, 1.0)), np.log(p[:, 0]), p[:, 1]) / 20)

    def _min_phase(self, log_harm, freqs):
        cep = np.fft.irfft(log_harm @ self.harm_interp, n=N_FFT, axis=-1)
        fold = np.zeros(N_FFT)
        fold[0], fold[1:self.mp_lifter] = 1.0, 2.0
        ph = np.fft.rfft(cep * fold, axis=-1).imag
        pos = np.clip(freqs / (SR / N_FFT), 0, N_FFT // 2 - 1.001)
        i0 = np.floor(pos).astype(int)
        fr = pos - i0
        pk = np.take_along_axis(ph, i0, -1) * (1 - fr) + np.take_along_axis(ph, i0 + 1, -1) * fr
        pk = pk * np.clip((freqs - self.mp_low_hz) / self.mp_low_hz, 0, 1)
        dt = np.diff(pk, axis=0)
        return np.concatenate([pk[:1], pk[:1] + np.cumsum(dt - 2 * np.pi * np.round(dt / (2 * np.pi)), 0)], 0)

    def __call__(self, f0, voiced, log_harm, log_noise, rng: np.random.Generator) -> np.ndarray:
        """f0 (T,) Hz, voiced (T,), log_harm (T, 40), log_noise (T, 24) → waveform ((T − 1) · 256,)."""
        f0, voiced = f0.astype(np.float64), voiced.astype(np.float64)
        log_harm, log_noise = log_harm.astype(np.float64), log_noise.astype(np.float64)
        length = (len(f0) - 1) * HOP
        freqs = f0[:, None] * self.k
        pos = np.clip(_mel(freqs) / _mel(SR / 2), 0, 1) * (self.kh - 1)
        i0 = np.minimum(np.floor(pos), self.kh - 2).astype(int)
        fr = pos - i0
        env = np.take_along_axis(log_harm, i0, -1) * (1 - fr) + np.take_along_axis(log_harm, i0 + 1, -1) * fr
        amp = np.exp(env) * (freqs < 0.95 * SR / 2) * voiced[:, None]
        if self.mvf_hz > 0:
            amp = amp * np.sqrt(1 - self._mvf(freqs))
        if self.eq:
            amp = amp * self._eq(freqs)
        # sample-rate harmonic sum in float32 (as in PyTorch), only over harmonics that sound somewhere in the chunk;
        # the running phase is accumulated in float64
        on = np.flatnonzero(amp.any(0))
        phase = np.mod(np.cumsum(_frames_to_samples(f0, length) * (2 * np.pi / SR)), 2 * np.pi).astype(np.float32)
        ang = phase[:, None] * self.k[on].astype(np.float32) + self.theta[on].astype(np.float32)
        if self.mp_low_hz > 0:
            ang += _frames_to_samples(self._min_phase(log_harm, freqs)[:, on].astype(np.float32), length)
        np.sin(ang, out=ang)
        ang *= _frames_to_samples(amp[:, on].astype(np.float32), length)
        harm = ang.sum(-1, dtype=np.float64)
        mag = np.exp(log_noise @ self.noise_interp)
        if self.mvf_hz > 0:
            w = self._mvf(self.bin_hz) * (self.bin_hz < 0.95 * SR / 2)
            hp = np.exp(2 * (log_harm @ self.harm_interp)) * voiced[:, None] ** 2
            mag = np.sqrt(mag ** 2 + w * hp * (SR / (4 * np.maximum(f0, 50.0)))[:, None])
        if self.eq:
            mag = mag * self._eq(self.bin_hz)
        noise = _istft(_stft(rng.standard_normal(length), self.window) * mag, self.window, length)
        y = np.pad(harm + noise, len(self.fir) // 2)
        return np.correlate(y, self.fir, mode="valid").astype(np.float32)


def _split(ids: list[list[int]]) -> list[list[list[int]]]:
    """Cut a chunk into pieces of ≤ 64 syllables, preferably after a phrase or sentence boundary."""
    pieces = []
    while len(ids) > N_MAX:
        cut = max((i + 1 for i in range(N_MAX) if ids[i][5] >= BOUNDARIES.index("PHRASE")), default=N_MAX)
        pieces.append(ids[:cut])
        ids = ids[cut:]
    return pieces + ([ids] if ids else [])


class TiHonTFLite:
    """Text → speech with the two TFLite graphs (`encoder.tflite`, `decoder.tflite`) + numpy synthesis."""

    def __init__(self, folder: str | Path, tflite_dir: str | Path | None = None) -> None:
        """`folder`: this repository (config.json, model.safetensors). `tflite_dir`: folder with encoder.tflite and
        decoder.tflite (default `folder/tflite`); the graphs are distributed on request in ngocthien/tihon-tts-int16-npu,
        folder `tflite-fp32/`."""
        from safetensors.numpy import load_file

        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:  # TensorFlow's bundled interpreter works too
            from tensorflow.lite.python.interpreter import Interpreter
        folder = Path(folder)
        tdir = Path(tflite_dir) if tflite_dir else folder / "tflite"
        if not (tdir / "encoder.tflite").is_file() or not (tdir / "decoder.tflite").is_file():
            raise FileNotFoundError(f"encoder.tflite / decoder.tflite not found in {tdir}. The TFLite graphs are in "
                                    "ngocthien/tihon-tts-int16-npu (folder tflite-fp32/, access on request).")
        self.cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        self.voices = [v["name"] for v in self.cfg["voices"]]
        self.enc = self._runner(Interpreter(model_path=str(tdir / "encoder.tflite")))
        self.dec = self._runner(Interpreter(model_path=str(tdir / "decoder.tflite")))
        w = load_file(str(folder / "model.safetensors"))
        self.synth = HarmonicNoiseSynth(w["synth.harmonic_phase"], w["synth.fir"])

    @staticmethod
    def _runner(it):
        it.allocate_tensors()
        sig = it.get_signature_runner()
        spec = it.get_signature_list()["serving_default"]
        ins, outs = sorted(spec["inputs"]), sorted(spec["outputs"])
        return lambda *args: [sig(**dict(zip(ins, args, strict=True)))[k] for k in outs]

    def _frame_params(self, ids: list[list[int]], voice: int):
        i32 = lambda v: np.asarray(v, dtype=np.int32)  # noqa: E731
        syl = np.zeros((1, N_MAX, 7), dtype=np.int32)
        syl[0, :len(ids)] = ids
        states, dur = self.enc(syl, i32([len(ids)]), i32([voice]))
        frames = int(dur.sum()) + 1
        outs = [self.dec(states, dur, i32([s - CONTEXT]), i32([frames]), i32([voice]))
                for s in range(0, frames, CENTER)]
        return [np.concatenate([o[i][0] for o in outs], 0)[:frames] for i in range(4)]

    def synthesize(self, text: str, voice: str | int, seed: int = 0) -> np.ndarray:
        """Text → float32 waveform at 24 kHz (0.3 s pause after every chunk)."""
        v = self.voices.index(voice) if isinstance(voice, str) else int(voice)
        dsp = self.cfg["voices"][v].get("dsp", {})
        self.synth.mvf_hz, self.synth.mp_low_hz = float(dsp.get("mvf_hz", 0)), float(dsp.get("mp_low_hz", 0))
        self.synth.eq = dsp.get("eq", [])
        rng, pause, parts = np.random.default_rng(seed), np.zeros(int(0.3 * SR), dtype=np.float32), []
        for chunk in text_to_ids(text):
            for piece in _split(chunk):
                parts += [self.synth(*self._frame_params(piece, v), rng=rng), pause]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
