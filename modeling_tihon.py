"""Transformers wrapper of TiHon-TTS (loaded with `trust_remote_code=True`).

    from transformers import AutoModel
    model = AutoModel.from_pretrained("ngocthien/tihon-tts", trust_remote_code=True)
    wav = model.synthesize("Xin chào, hôm nay trời đẹp không?", voice="trucly")   # float32, 24 kHz

The network and the text front end live in the `tihon_tts/` package of the repository; this file only adapts them to
`PreTrainedModel`. Parameter names are the same as in `model.safetensors`.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedModel

from .configuration_tihon import TiHonTTSConfig


def _load_package(config):
    """Import `tihon_tts` from the local folder or the Hub snapshot the model was loaded from."""
    root = Path(str(getattr(config, "_name_or_path", "") or ""))
    if not (root / "tihon_tts" / "model.py").is_file():
        from huggingface_hub import snapshot_download

        repo = str(getattr(config, "_name_or_path", "") or "ngocthien/tihon-tts")
        root = Path(snapshot_download(repo, revision=getattr(config, "_commit_hash", None),
                                      allow_patterns=["tihon_tts/*.py"]))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module("tihon_tts.model"), importlib.import_module("tihon_tts.text")


class TiHonTTSModel(PreTrainedModel):
    config_class = TiHonTTSConfig
    main_input_name = "syllables"

    def __init__(self, config: TiHonTTSConfig) -> None:
        super().__init__(config)
        model_mod, self._text = _load_package(config)
        # Built on the CPU even when Transformers instantiates the model on the meta device: the synthesiser computes
        # fixed tables (band interpolation, window) that are not stored in the checkpoint.
        with torch.device("cpu"):
            core = model_mod.TiHonTTS({"sample_rate": config.sample_rate, "hop_length": config.hop_length,
                                       "architecture": config.architecture, "voices": config.voices})
        self._core = type(core)
        for name, module in core.named_children():
            self.add_module(name, module)
        for name, param in core.named_parameters(recurse=False):
            self.register_parameter(name, param)
        for name, buf in core.named_buffers(recurse=False):
            self.register_buffer(name, buf)
        self.cfg, self.voices = core.cfg, core.voices
        self.post_init()

    def _init_weights(self, module):
        """Weights always come from the checkpoint; nothing to initialise."""

    def _restore_tables(self) -> None:
        """Recompute the fixed tables of the synthesiser once (Transformers may leave non-persistent buffers empty)."""
        if getattr(self, "_tables_ready", False):
            return
        a = self.config.architecture
        with torch.device("cpu"):
            fresh = type(self.synth)(a["noise_fft"], a["harmonic_bands"], a["noise_bands"], a["n_harmonics"],
                                     a["fir_taps"])
        device = self.synth.harmonic_phase.device
        for name, buf in fresh.named_buffers():
            self.synth._buffers[name] = buf.to(device)
        self._tables_ready = True

    def voice_id(self, voice):
        return self._core.voice_id(self, voice)

    def set_postprocessing(self, voice) -> None:
        """Recommended DSP post-processing of a voice (from config.json)."""
        self._core.set_postprocessing(self, voice)

    def tokens(self, syl, spk):
        return self._core.tokens(self, syl, spk)

    def forward(self, syllables: torch.Tensor, voice="trucly", generator=None) -> torch.Tensor:
        """Syllable ids (1, N, 7) of one chunk → waveform (L,) at 24 kHz."""
        self._restore_tables()
        return self._core.forward(self, syllables, voice, generator)

    @torch.no_grad()
    def synthesize(self, text: str, voice="trucly", seed: int = 0, pause: float = 0.3) -> np.ndarray:
        """Text → float32 waveform at 24 kHz, with `pause` seconds of silence after every chunk."""
        self.set_postprocessing(voice)
        device = next(self.parameters()).device
        generator = torch.Generator(device=device).manual_seed(int(seed))
        gap = np.zeros(int(pause * self.config.sample_rate), dtype=np.float32)
        parts = []
        for ids in self._text.text_to_ids(text):
            wav = self(torch.tensor([ids], device=device), voice, generator)
            parts += [wav.float().cpu().numpy(), gap]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
