"""TiHon-TTS acoustic model + harmonic-plus-noise synthesiser (PyTorch reference implementation).

syllable ids (N, 7) → 1 + 5N tokens → looped conv encoder → BiGRU → token durations and pitch
→ Gaussian upsampling → dilated conv decoder (speaker FiLM) → F0, voicing, harmonic and noise envelopes
→ harmonic-plus-noise synthesis → 24 kHz waveform.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE, HOP = 24_000, 256


# ── building blocks ────────────────────────────────────────────────────────────────────────────────────────────────
class ChannelNorm(nn.Module):
    """LayerNorm over channels of (B, C, T)."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.weight, self.bias = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c))

    def forward(self, x):
        return F.layer_norm(x.transpose(1, 2), (x.shape[1],), self.weight, self.bias).transpose(1, 2)


class FFN(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.up, self.down = nn.Conv1d(c, 2 * c, 1), nn.Conv1d(2 * c, c, 1)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class LoopedEncoder(nn.Module):
    """One conv-GLU + FFN block applied `loops` times with shared weights and per-iteration norms."""

    def __init__(self, c: int, k: int, loops: int) -> None:
        super().__init__()
        self.loops = loops
        self.depthwise = nn.Conv1d(c, c, k, padding=k // 2, groups=c)
        self.glu_proj = nn.Conv1d(c, 2 * c, 1)
        self.ffn = FFN(c)
        self.norm_conv = nn.ModuleList(ChannelNorm(c) for _ in range(loops))
        self.norm_ffn = nn.ModuleList(ChannelNorm(c) for _ in range(loops))

    def forward(self, x, mask):
        for i in range(self.loops):
            x = x + F.glu(self.glu_proj(self.depthwise(self.norm_conv[i](x) * mask)), dim=1)
            x = (x + self.ffn(self.norm_ffn[i](x) * mask)) * mask
        return x


class Context(nn.Module):
    """Residual bidirectional GRU over the token sequence."""

    def __init__(self, c: int, h: int) -> None:
        super().__init__()
        self.norm = ChannelNorm(c)
        self.gru = nn.GRU(c, h, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.gru(self.norm(x).transpose(1, 2))
        return x + out.transpose(1, 2)


class Predictor(nn.Module):
    def __init__(self, c: int, h: int, k: int = 5) -> None:
        super().__init__()
        self.norm = ChannelNorm(c)
        self.depthwise = nn.Conv1d(c, c, k, padding=k // 2, groups=c)
        self.proj1, self.proj2, self.out = nn.Conv1d(c, h, 1), nn.Conv1d(h, h, 1), nn.Conv1d(h, 1, 1)

    def forward(self, x, mask):
        y = self.depthwise(self.norm(x) * mask)
        return self.out(F.relu(self.proj2(F.relu(self.proj1(y))))).squeeze(1)


class Decoder(nn.Module):
    """Dilated depthwise conv blocks sharing a few FFNs; FiLM (1 + γ, β) per voice after every block."""

    def __init__(self, c: int, k: int, layers: int, dilations: list[int], ffns: int) -> None:
        super().__init__()
        self.layers, self.ffns_n = layers, ffns
        self.input_proj = nn.Conv1d(c + 2, c, 1)
        dil = [dilations[i % len(dilations)] for i in range(layers)]
        self.depthwise = nn.ModuleList(nn.Conv1d(c, c, k, padding=d * (k - 1) // 2, dilation=d, groups=c) for d in dil)
        self.norms = nn.ModuleList(ChannelNorm(c) for _ in range(layers))
        self.ffns = nn.ModuleList(FFN(c) for _ in range(ffns))
        self.out_norm = ChannelNorm(c)

    def forward(self, x, mask, film):
        x = self.input_proj(x) * mask
        for i in range(self.layers):
            x = (x + self.ffns[i % self.ffns_n](self.norms[i](self.depthwise[i](x)))) * mask
            x = (x * (1 + film[:, 0, i, :, None]) + film[:, 1, i, :, None]) * mask
        x = self.out_norm(x)
        return x * (1 + film[:, 0, self.layers, :, None]) + film[:, 1, self.layers, :, None]


def gaussian_upsample(h, dur, sigma, n_frames, token_mask):
    """Soft expansion of tokens (B, C, N) with durations (B, N) to frames (B, C, T) + 2 position features."""
    h, dur = h.float(), dur.float()
    centers = dur.cumsum(1) - dur / 2
    t = torch.arange(n_frames, device=h.device).float() + 0.5
    diff = t[None, :, None] - centers[:, None, :]
    w = (-(diff ** 2) / (2 * sigma ** 2)).masked_fill(~token_mask[:, None, :], -1e9).softmax(-1)
    safe = dur.clamp(min=1.0)
    pos = (w * (diff / safe[:, None, :])).sum(-1)
    logd = (w * torch.log(safe)[:, None, :]).sum(-1)
    return torch.bmm(h, w.transpose(1, 2)), torch.stack([pos, logd], 1)


def token_masks(syl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, N, 7) → masks (B, 1 + 5N): tokens that may own frames / must own at least one frame."""
    b, n = syl.shape[:2]
    on = torch.ones(b, n, dtype=torch.bool, device=syl.device)
    off = torch.zeros_like(on)
    onset, medial, coda = syl[..., 0] != 0, syl[..., 1] != 0, syl[..., 3] != 0
    lead = torch.ones(b, 1, dtype=torch.bool, device=syl.device)
    active = torch.cat([lead, torch.stack([onset, medial, on, coda, on], -1).reshape(b, 5 * n)], 1)
    phonetic = torch.cat([~lead, torch.stack([onset, medial, on, coda, off], -1).reshape(b, 5 * n)], 1)
    return active, phonetic


# ── harmonic-plus-noise synthesiser ────────────────────────────────────────────────────────────────────────────────
def _mel(f):
    return 2595.0 * torch.log10(1.0 + f / 700.0)


def _band_matrix(bins: int, bands: int, nyq: float) -> torch.Tensor:
    pos = (_mel(torch.linspace(0, nyq, bins)) / float(_mel(torch.tensor(nyq)))).clamp(0, 1) * (bands - 1)
    i0 = pos.floor().clamp(max=bands - 2).long()
    fr = pos - i0
    m = torch.zeros(bins, bands)
    m[torch.arange(bins), i0] = 1 - fr
    m[torch.arange(bins), i0 + 1] = fr
    return m.t().contiguous()


class Synthesizer(nn.Module):
    """Σ_k a_k sin(k φ + θ_k) + filtered noise, then a learned FIR. Optional parameter-free post-processing:
    `mvf_hz` (harmonics above it become noise), `mp_low_hz` (minimum-phase dispersion), `eq` [[Hz, dB], ...]."""

    def __init__(self, n_fft: int, kh: int, kn: int, n_harm: int, fir_taps: int) -> None:
        super().__init__()
        self.n_fft, self.kh, self.fir_taps = n_fft, kh, fir_taps
        nyq, bins = SAMPLE_RATE / 2, n_fft // 2 + 1
        self.nyq_mel = float(2595 * math.log10(1 + nyq / 700))
        self.register_buffer("k", torch.arange(1, n_harm + 1).float(), persistent=False)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer("noise_interp", _band_matrix(bins, kn, nyq), persistent=False)
        self.register_buffer("harm_interp", _band_matrix(bins, kh, nyq), persistent=False)
        self.register_buffer("bin_hz", torch.linspace(0, nyq, bins), persistent=False)
        self.harmonic_phase = nn.Parameter(torch.zeros(n_harm))
        self.fir = nn.Parameter(torch.zeros(fir_taps))
        self.mvf_hz, self.mp_low_hz, self.mp_lifter, self.eq = 0.0, 0.0, 60, []

    def forward(self, f0, voiced, log_harm, log_noise, generator=None):
        """f0 (B, T) Hz, voiced (B, T), log_harm (B, T, kh), log_noise (B, T, kn) → (B, (T − 1) · 256)."""
        b, frames = f0.shape
        length = (frames - 1) * HOP
        freqs = f0.unsqueeze(-1) * self.k
        pos = (_mel(freqs) / self.nyq_mel).clamp(0, 1) * (self.kh - 1)
        i0 = pos.floor().clamp(max=self.kh - 2)
        fr, i0 = pos - i0, i0.long()
        env = log_harm.gather(-1, i0) * (1 - fr) + log_harm.gather(-1, i0 + 1) * fr
        amp = env.exp() * (freqs < 0.95 * SAMPLE_RATE / 2).float() * voiced.unsqueeze(-1)
        if self.mvf_hz > 0:
            amp = amp * torch.sqrt(1 - self._mvf(freqs))
        if self.eq:
            amp = amp * self._eq(freqs)
        amp = F.interpolate(amp.transpose(1, 2), size=length + 1, mode="linear", align_corners=True)[..., :length]
        f0s = F.interpolate(f0.unsqueeze(1), size=length + 1, mode="linear", align_corners=True)[:, 0, :length]
        acc = torch.float64 if length > 32768 else torch.float32
        phase = torch.cumsum(f0s.to(acc) * (2 * math.pi / SAMPLE_RATE), 1).remainder(2 * math.pi).float()
        ang = phase.unsqueeze(1) * self.k.view(1, -1, 1) + self.harmonic_phase.view(1, -1, 1)
        if self.mp_low_hz > 0:
            ang = ang + F.interpolate(self._min_phase(log_harm, freqs), size=length + 1, mode="linear",
                                      align_corners=True)[..., :length]
        harm = (amp * torch.sin(ang)).sum(1)
        white = torch.randn(b, length, device=f0.device, generator=generator)
        spec = torch.stft(white, self.n_fft, HOP, window=self.window, center=True, return_complex=True)
        mag = (log_noise @ self.noise_interp).exp()
        if self.mvf_hz > 0:  # re-inject removed harmonic power as noise of equal density A²·sr / (4·F0)
            w = self._mvf(self.bin_hz) * (self.bin_hz < 0.95 * SAMPLE_RATE / 2)
            hp = (2 * (log_harm @ self.harm_interp)).exp() * voiced.unsqueeze(-1) ** 2
            mag = torch.sqrt(mag ** 2 + w * hp * (SAMPLE_RATE / (4 * f0.clamp(min=50.0))).unsqueeze(-1))
        if self.eq:
            mag = mag * self._eq(self.bin_hz)
        noise = torch.istft(spec * mag.transpose(1, 2), self.n_fft, HOP, window=self.window, center=True,
                            length=length)
        return F.conv1d((harm + noise).unsqueeze(1), self.fir.view(1, 1, -1), padding=self.fir_taps // 2).squeeze(1)

    def _mvf(self, hz):
        r = ((hz - self.mvf_hz + 1000) / 2000).clamp(0, 1)
        return 0.5 - 0.5 * torch.cos(math.pi * r)

    def _eq(self, hz):
        p = torch.tensor(self.eq, dtype=hz.dtype, device=hz.device)
        xp, yp = p[:, 0].log(), p[:, 1]
        x = hz.clamp(min=1.0).log().clamp(xp[0], xp[-1])
        i = torch.bucketize(x, xp).clamp(1, len(xp) - 1)
        return 10 ** ((yp[i - 1] + (yp[i] - yp[i - 1]) * (x - xp[i - 1]) / (xp[i] - xp[i - 1])) / 20)

    def _min_phase(self, log_harm, freqs):
        cep = torch.fft.irfft(log_harm @ self.harm_interp, n=self.n_fft)
        fold = torch.zeros(self.n_fft, dtype=cep.dtype, device=cep.device)
        fold[0], fold[1:self.mp_lifter] = 1.0, 2.0
        ph = torch.fft.rfft(cep * fold).imag
        pos = (freqs / (SAMPLE_RATE / self.n_fft)).clamp(0, self.n_fft // 2 - 1.001)
        i0 = pos.floor().long()
        fr = pos - i0
        pk = ph.gather(-1, i0) * (1 - fr) + ph.gather(-1, i0 + 1) * fr
        pk = (pk * ((freqs - self.mp_low_hz) / self.mp_low_hz).clamp(0, 1)).transpose(1, 2)
        dt = torch.diff(pk, dim=-1)
        two_pi = 2 * math.pi
        return torch.cat([pk[..., :1], pk[..., :1] + torch.cumsum(dt - two_pi * torch.round(dt / two_pi), -1)], -1)


# ── full model ─────────────────────────────────────────────────────────────────────────────────────────────────────
class TiHonTTS(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        a = cfg["architecture"]
        d, s = a["d_model"], len(cfg["voices"])
        self.cfg, self.voices = cfg, [v["name"] for v in cfg["voices"]]
        self.components = nn.ModuleList(nn.Embedding(n, d) for n in a["vocab_sizes"])
        self.roles = nn.Embedding(6, d)
        self.encoder = LoopedEncoder(d, a["encoder_kernel"], a["encoder_loops"])
        self.context = Context(d, a["gru_hidden"])
        self.duration = Predictor(d, a["predictor_channels"])
        self.pitch = Predictor(d, a["predictor_channels"])
        self.pitch_embedding = nn.Conv1d(1, d, 1)
        self.sigma_raw = nn.Parameter(torch.tensor(0.0))
        self.decoder = Decoder(d, a["decoder_kernel"], a["decoder_layers"], a["decoder_dilations"],
                               a["decoder_shared_ffns"])
        self.head = nn.Conv1d(d, 2 + a["harmonic_bands"] + a["noise_bands"], 1)
        self.synth = Synthesizer(a["noise_fft"], a["harmonic_bands"], a["noise_bands"], a["n_harmonics"],
                                 a["fir_taps"])
        self.speaker_embedding = nn.Embedding(s, d)
        self.speaker_film = nn.Embedding(s, 2 * (a["decoder_layers"] + 1) * d)
        self.register_buffer("f0_mean", torch.zeros(s))
        self.register_buffer("f0_std", torch.ones(s))

    @classmethod
    def from_pretrained(cls, folder: str | Path) -> TiHonTTS:
        """Load `config.json` + `model.safetensors` from a local folder."""
        from safetensors.torch import load_file

        folder = Path(folder)
        model = cls(json.loads((folder / "config.json").read_text(encoding="utf-8")))
        model.load_state_dict(load_file(str(folder / "model.safetensors")))
        return model.eval()

    def voice_id(self, voice: str | int) -> int:
        return self.voices.index(voice) if isinstance(voice, str) else int(voice)

    def set_postprocessing(self, voice: str | int) -> None:
        """Apply the recommended DSP post-processing of a voice (from config.json)."""
        dsp = self.cfg["voices"][self.voice_id(voice)].get("dsp", {})
        self.synth.mvf_hz = float(dsp.get("mvf_hz", 0.0))
        self.synth.mp_low_hz = float(dsp.get("mp_low_hz", 0.0))
        self.synth.eq = [list(map(float, p)) for p in dsp.get("eq", [])]

    def tokens(self, syl: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        """(B, N, 7) → (B, C, 1 + 5N) token embeddings + speaker offset."""
        b, n = syl.shape[:2]
        tone = self.components[4](syl[..., 4])
        slots = [self.components[i](syl[..., i]) + tone for i in range(4)]
        slots.append(self.components[5](syl[..., 5]) + self.components[6](syl[..., 6]))
        x = (torch.stack(slots, 2) + self.roles.weight[1:6].view(1, 1, 5, -1)).reshape(b, 5 * n, -1)
        x = torch.cat([self.roles.weight[0].view(1, 1, -1).expand(b, 1, -1), x], 1).transpose(1, 2)
        return x + self.speaker_embedding(spk).unsqueeze(-1)

    @torch.no_grad()
    def forward(self, syl: torch.Tensor, voice: str | int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Syllable ids (1, N, 7) → waveform (L,) at 24 kHz."""
        spk = torch.tensor([self.voice_id(voice)], device=syl.device)
        active, phonetic = token_masks(syl)
        mask = active.unsqueeze(1).float()
        x = self.encoder(self.tokens(syl, spk) * mask, mask)
        x = self.context(x) * mask
        logd, pitch = self.duration(x, mask), self.pitch(x, mask)
        dur = (logd.exp() - 1).round().clamp(min=0) * active
        dur = torch.where(phonetic, dur.clamp(min=1), dur)
        n_frames = int(dur.sum()) + 1
        x = x + self.pitch_embedding(pitch.unsqueeze(1)) * mask
        up, posf = gaussian_upsample(x, dur, F.softplus(self.sigma_raw), n_frames, active & (dur > 0))
        fmask = torch.ones(1, 1, n_frames, device=syl.device)
        film = self.speaker_film(spk).view(1, 2, self.decoder.layers + 1, -1)
        out = self.head(self.decoder(torch.cat([up, posf], 1) * fmask, fmask, film)).float()
        kh = self.cfg["architecture"]["harmonic_bands"]
        f0 = torch.exp(out[:, 0] * self.f0_std[spk].unsqueeze(1) + self.f0_mean[spk].unsqueeze(1)).clamp(60, 600)
        voiced = (torch.sigmoid(out[:, 1]) > 0.5).float()
        log_harm = out[:, 2:2 + kh].transpose(1, 2).clamp(-14, 2)
        log_noise = out[:, 2 + kh:].transpose(1, 2).clamp(-14, 2)
        return self.synth(f0, voiced, log_harm, log_noise, generator)[0]
