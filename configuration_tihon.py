"""Transformers configuration of TiHon-TTS (loaded with `trust_remote_code=True`)."""

from transformers import PretrainedConfig


class TiHonTTSConfig(PretrainedConfig):
    model_type = "tihon-tts"

    def __init__(self, architecture=None, voices=None, sample_rate=24_000, hop_length=256, **kwargs):
        self.architecture = architecture or {}
        self.voices = voices or []
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        super().__init__(**kwargs)
