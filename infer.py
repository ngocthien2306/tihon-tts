"""Text → WAV with TiHon-TTS.

    python infer.py --text "Xin chào, hôm nay trời đẹp không?" --voice trucly --out hello.wav
    python infer.py --text "..." --voice haidang --out hello.wav --backend tflite --tflite-dir <folder with the graphs>

The TFLite graphs are in ngocthien/tihon-tts-int16-npu (folder tflite-fp32/, access on request).
"""

import argparse
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def synthesize_torch(text: str, voice: str, seed: int) -> np.ndarray:
    import torch

    from tihon_tts.model import TiHonTTS
    from tihon_tts.text import text_to_ids

    model = TiHonTTS.from_pretrained(HERE)
    model.set_postprocessing(voice)
    gen = torch.Generator().manual_seed(seed)
    pause = np.zeros(int(0.3 * 24_000), dtype=np.float32)
    parts = []
    for ids in text_to_ids(text):
        parts += [model(torch.tensor([ids]), voice, gen).numpy(), pause]
    return np.concatenate(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", required=True)
    ap.add_argument("--voice", default="trucly", choices=["ngochuyen", "infore", "trucly", "haidang"])
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--backend", default="torch", choices=["torch", "tflite"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tflite-dir", default=None, help="folder with encoder.tflite and decoder.tflite")
    args = ap.parse_args()
    if args.backend == "torch":
        wav = synthesize_torch(args.text, args.voice, args.seed)
    else:
        from tihon_tts.tflite_runner import TiHonTFLite

        wav = TiHonTFLite(HERE, args.tflite_dir).synthesize(args.text, args.voice, args.seed)
    pcm = (np.clip(wav, -1, 1) * 32767).round().astype("<i2")
    with wave.open(args.out, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24_000)
        f.writeframes(pcm.tobytes())
    print(f"{args.out}: {len(wav) / 24_000:.2f} s")


if __name__ == "__main__":
    main()
