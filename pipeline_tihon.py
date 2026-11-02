"""Transformers `text-to-speech` pipeline of TiHon-TTS (loaded with `trust_remote_code=True`).

    from transformers import pipeline
    tts = pipeline("text-to-speech", model="ngocthien/tihon-tts", trust_remote_code=True)
    out = tts("Xin chào, hôm nay trời đẹp không?", voice="haidang")   # {"audio": float32 array, "sampling_rate": 24000}
"""

from transformers import Pipeline


class TiHonTTSPipeline(Pipeline):
    _load_tokenizer = False
    _load_feature_extractor = False
    _load_image_processor = False
    _load_processor = False

    def _sanitize_parameters(self, voice=None, seed=None, **kwargs):
        forward = {}
        if voice is not None:
            forward["voice"] = voice
        if seed is not None:
            forward["seed"] = seed
        return {}, forward, {}

    def preprocess(self, text):
        return {"text": text}

    def _forward(self, inputs, voice="trucly", seed=0):
        return {"audio": self.model.synthesize(inputs["text"], voice=voice, seed=seed)}

    def postprocess(self, outputs):
        return {"audio": outputs["audio"], "sampling_rate": self.model.config.sample_rate}
