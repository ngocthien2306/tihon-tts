# TiHon-TTS

**Vietnamese text-to-speech small enough for a microcontroller.**

TiHon-TTS is a Vietnamese text-to-speech model with **100,118 parameters** and four voices in one network. It turns
text into 24 kHz audio without a vocoder network: the model predicts pitch, voicing and spectral envelopes, and a
small harmonic-plus-noise synthesiser renders the waveform. It is built to run on small edge devices, on CPU or NPU.
*Tí hon* means "tiny" in Vietnamese.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-ngocthien%2Ftihon--tts-yellow)](https://huggingface.co/ngocthien/tihon-tts) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://huggingface.co/ngocthien/tihon-tts/blob/main/notebook.ipynb) ![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)

Weights, samples you can play in the browser and the model card: [https://huggingface.co/ngocthien/tihon-tts](https://huggingface.co/ngocthien/tihon-tts).

| | |
|---|---|
| Parameters | 100,118 (95,414 shared + 1,176 per voice) |
| Size | 402 KB in float32 |
| Output | 24 kHz mono, frames of 10.7 ms |
| Voices | 4 (three female, one male) |
| Intelligibility | 1.0–2.3 % character error rate (ASR, 36 held-out sentences per voice) |
| Speed, measured | 40× real time on one x86 core, 83× on one Apple M4 core (TFLite backend) |
| Microcontroller, simulated | Cortex-M55 + Ethos-U55: 6.6 M CPU instructions + 4.7 M NPU cycles per second of audio |
| License | CC BY-NC 4.0, non-commercial |

Try it in the browser: open [`notebook.ipynb`](https://huggingface.co/ngocthien/tihon-tts/blob/main/notebook.ipynb) on
Hugging Face and click "Open in Colab".

## 1. Voices

| Voice | `voice` | Gender | Source | Sample |
|---|---|---|---|---|
| Ngọc Huyền | `ngochuyen` | female | [`pnnbao-ump/ngochuyen_voice`](https://huggingface.co/datasets/pnnbao-ump/ngochuyen_voice) | [ngochuyen.wav](samples/ngochuyen.wav) |
| InfoRe | `infore` | female | [InfoRe Technology 25-hour corpus](https://huggingface.co/datasets/doof-ferb/infore1_25hours) | [infore.wav](samples/infore.wav) |
| Trúc Ly | `trucly` | female | [VieNeu-TTS v3 Turbo](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo) preset | [trucly.wav](samples/trucly.wav) |
| Hải Đăng | `haidang` | male | VieNeu-TTS v3 Turbo preset | [haidang.wav](samples/haidang.wav) |

All samples read *"Xin chào, tôi là giọng đọc tiếng Việt tí hon, chỉ có khoảng một trăm nghìn tham số. Hôm nay bạn
có khoẻ không?"*

## 2. Usage

```bash
git clone https://github.com/ngocthien2306/tihon-tts && cd tihon-tts
pip install -r requirements.txt
hf download ngocthien/tihon-tts model.safetensors --local-dir .     # weights (402 KB) from Hugging Face
python infer.py --text "Xin chào, hôm nay trời đẹp không?" --voice trucly --out hello.wav

# TFLite backend, no PyTorch needed: the graphs are in ngocthien/tihon-tts-int16-npu (access on request)
pip install numpy safetensors ai-edge-litert
hf download ngocthien/tihon-tts-int16-npu --include "tflite-fp32/*" --local-dir npu
python infer.py --text "Xin chào, hôm nay trời đẹp không?" --voice haidang --out hello.wav \
    --backend tflite --tflite-dir npu/tflite-fp32
```

With Transformers (`trust_remote_code=True` loads the small wrapper in this repository):

```python
from transformers import pipeline

tts = pipeline("text-to-speech", model="ngocthien/tihon-tts", trust_remote_code=True)
out = tts("Xin chào, hôm nay trời đẹp không?", voice="haidang")    # {"audio": float32 array, "sampling_rate": 24000}

from transformers import AutoModel

model = AutoModel.from_pretrained("ngocthien/tihon-tts", trust_remote_code=True)
wav = model.synthesize("Bây giờ là 9 giờ 36 phút.", voice="trucly")
```

Without Transformers, from this repository (after downloading `model.safetensors`):

```python
import torch
from tihon_tts.model import TiHonTTS
from tihon_tts.text import text_to_ids

model = TiHonTTS.from_pretrained(".")
model.set_postprocessing("trucly")                       # recommended DSP of the voice (config.json)
for ids in text_to_ids("Bây giờ là 9 giờ 36 phút."):      # one chunk of syllable ids at a time
    wav = model(torch.tensor([ids]), "trucly")            # float32 waveform, 24 kHz
```

The text front end (`tihon_tts/text.py`) splits text into chunks and syllables. Numbers, dates, times, money,
percentages, units, phone numbers and e-mail addresses are read by built-in rules; other non-Vietnamese words are
spelled letter by letter.

## 3. Evaluation

### 3.1 Intelligibility

36 sentences per voice that are not in the training text (12 statements, 12 questions, 12 short phrases) are
synthesised and transcribed by the [Zipformer-30M](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h) Vietnamese
ASR model. The table gives the character error rate (CER); spelling variants of the ASR count as errors.

| Voice | PyTorch | TFLite |
|---|---:|---:|
| Ngọc Huyền | 2.2 % | 2.3 % |
| InfoRe | 1.5 % | 1.4 % |
| Trúc Ly | 1.0 % | 1.1 % |
| Hải Đăng | 1.1 % | 1.1 % |

With 36 sentences, one misrecognised sentence moves the CER by about 0.5 points, and so does the noise seed, so
smaller differences are not meaningful. The CER measures intelligibility only; no listening test (MOS) has been run.
The int16 NPU graphs reach the same CER as float32 (see [`tihon-tts-int16-npu`](https://huggingface.co/ngocthien/tihon-tts-int16-npu)).

### 3.2 Speed on application processors

Real-time factor (compute time / audio time) over the 144 sentences above (355 s of audio), including the text front
end and the waveform synthesis. Lower is faster. The TFLite graphs are available on request (section 6).

| Processor | Backend | 1 thread | 4 threads |
|---|---|---:|---:|
| AMD EPYC 7K62 (x86-64) | PyTorch | 0.047 (21×) | 0.025 (40×) |
| AMD EPYC 7K62 (x86-64) | TFLite + numpy | 0.025 (40×) | 0.033 (30×) |
| Apple M4 (arm64, Linux VM) | TFLite + numpy | 0.012 (83×) | 0.017 (61×) |

### 3.3 Microcontroller with NPU

The model is split for a Cortex-M CPU with an Ethos-U NPU:

| Part | Runs on | Per |
|---|---|---|
| Token embedding (table lookup, + voice embedding) | CPU | chunk |
| `enc`: looped conv-GLU encoder | NPU | chunk of ≤ 64 syllables |
| BiGRU context (the NPU has no GRU) | CPU | chunk |
| `heads`: duration and pitch predictors | NPU | chunk |
| Durations, pitch embedding, Gaussian upsampling | CPU | frame |
| `dec`: dilated decoder with the voice FiLM, 96 frames + 48 context frames on each side | NPU | 96 frames (1.02 s) |
| Harmonic-plus-noise synthesis (FFT from CMSIS-DSP) | CPU | frame |

The three NPU graphs use int8 weights and int16 activations (int8-activation versions, with half the NPU work and
lower quality, are also available). Vela 5.2 reports no CPU operators for any Ethos-U55, U65 or U85 configuration. The graphs, their inputs and outputs, their quality and their NPU cost for each
configuration are in [`ngocthien/tihon-tts-int16-npu`](https://huggingface.co/ngocthien/tihon-tts-int16-npu) (access on request). The C99
host code is not released.

**Measured on Arm Corstone fixed virtual platforms (FVP 11.32).** Sentence of 15 syllables, four voices, 13.8 s of
audio. Both columns are per second of audio. The synthesised PCM matches the Python reference in length to the
sample.

| Platform | CPU | NPU | CPU instructions | NPU cycles (PMU) |
|---|---|---|---:|---:|
| Corstone-300 | Cortex-M55 | Ethos-U55-128 | 6.6 M | 4.7 M |
| Corstone-310 | Cortex-M85 | Ethos-U55-128 | 6.6 M | 4.7 M |
| Corstone-310 | Cortex-M85 | Ethos-U55-256 | 6.6 M | 3.0 M |
| Corstone-315 | Cortex-M85 | Ethos-U65-256 | 6.6 M | 2.9 M |
| Corstone-315 | Cortex-M85 | Ethos-U65-512 | 6.6 M | 1.8 M* |

\* The Ethos-U65-512 run stopped after the first voice; the value is computed from its cycles per chunk and per
decoder window, the same way as the other rows (which this method reproduces to within 0.01 M).

Of the CPU work, the synthesiser takes 97 %, the BiGRU 1.7 % and the upsampling 1.5 %. Of the NPU work, the
decoder takes 76 %.

How to read these numbers:
* They are simulator counts, not silicon measurements. The FVP executes the CPU instruction by instruction, so the
  CPU column counts instructions. It is the same for Cortex-M55 and Cortex-M85, and real cycle counts are higher
  (a Helium vector instruction takes about two cycles on a Cortex-M55).
* NPU cycles come from the performance model of the NPU in the FVP; memory latency is not modelled. Vela estimates
  for the same graphs are 30–90 % higher.
* The Corstone-320 FVP (Cortex-M85 + Ethos-U85) did not boot the Arm platform firmware in our setup, so Ethos-U85
  has Vela estimates only.

## 4. Model

The network design is adapted from [AuroraTTS-100K](https://huggingface.co/AuroraAI-Research/AuroraTTS-100K)
(Apache-2.0): a looped conv-GLU encoder, a BiGRU, duration and pitch predictors, Gaussian upsampling, a dilated
depthwise decoder, and a harmonic-plus-noise synthesiser in place of a vocoder network. Changes in TiHon-TTS:

* **Vietnamese input.** Each syllable becomes five tokens: onset, medial, nucleus and coda (each carrying the tone)
  and a prosody token for pauses and punctuation.
* **Four voices in one network.** Each voice has its own token embedding and FiLM scale and shift after every
  decoder block (1,176 parameters), and its own F0 mean and standard deviation.
* **Training from text only**, by distillation (section 5), instead of training on recordings.
* **Per-voice post-processing** without parameters (in `config.json`): harmonics above 5 kHz become noise of the
  same power, a minimum-phase response is added to harmonics above 1.5 kHz, and Ngọc Huyền has a static EQ.

| Component | Parameters |
|---|---:|
| Token embedding (7 component tables + 6 roles, d = 56) | 4,200 |
| Looped conv-GLU encoder (1 block × 4 iterations, per-iteration norms) | 20,328 |
| BiGRU context (2 × 28) | 14,560 |
| Duration predictor / pitch predictor | 3,361 / 3,361 |
| Pitch embedding + Gaussian width σ | 113 |
| Dilated depthwise decoder (9 blocks, dilations 1-2-4, 3 shared FFNs) | 45,584 |
| Output head (2 + 40 + 24 channels) | 3,762 |
| Synthesiser (80 harmonic phases + 65-tap FIR) | 145 |
| Per voice: token embedding 56 + FiLM 2 × 10 × 56 | 4 × 1,176 |
| **Total** | **100,118** |

## 5. Training

The model never heard a recording. It was distilled from text only: at every step a frozen teacher of one voice (a
17.1M-parameter acoustic model with the Vocos vocoder
[`charactr/vocos-mel-24khz`](https://huggingface.co/charactr/vocos-mel-24khz), MIT) synthesises the targets on the
fly, and the student learns the teacher's durations, pitch and audio. The text corpus has 78.7K Vietnamese sentence
chunks. Training ran 80K steps (20K per voice) and ended with an adversarial phase; the released weights are an
exponential moving average. The training code is not released yet.

## 6. Files

| File | Content |
|---|---|
| `config.json` | configuration, including the DSP settings of each voice; the weights `model.safetensors` (402 KB) are on [Hugging Face](https://huggingface.co/ngocthien/tihon-tts) |
| `infer.py`, `tihon_tts/` | inference code: text front end, PyTorch model, TFLite runner with a numpy synthesiser |
| `configuration_tihon.py`, `modeling_tihon.py`, `pipeline_tihon.py` | Transformers wrapper (`AutoModel`, `text-to-speech` pipeline), loaded from the Hugging Face repository with `trust_remote_code=True` |
| `notebook.ipynb` | quick test in Google Colab |
| `requirements.txt`, `LICENSE`, `NOTICE` | dependencies, licence, third-party credits |
| `samples/` | one sample per voice |

Float32 TFLite graphs of the whole network (for the TFLite backend and for conversion to other runtimes) and int16
and int8 graphs for Ethos-U NPUs are in [`ngocthien/tihon-tts-int16-npu`](https://huggingface.co/ngocthien/tihon-tts-int16-npu), access on request.

## 7. Limitations

* The audio quality is below that of large TTS models: a slightly buzzy timbre and occasional voicing flicker.
* The model learned from teachers, not from recordings, so it inherits their errors and their timbre.
* Words outside the built-in rules are spelled letter by letter.
* No listening test (MOS) has been run; the evaluation covers intelligibility (CER) and speed only.

## 8. License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): **non-commercial use only**. The Ngọc Huyền voice
data are CC BY-NC 4.0, so the model cannot be licensed for commercial use.

Do not use the voices to impersonate a person or to deceive. Tell listeners that the speech is synthetic.

The architecture is adapted from AuroraTTS-100K (Apache-2.0) and training used Vocos (MIT); see [`NOTICE`](NOTICE).

Voice sources: Ngọc Huyền, [`pnnbao-ump/ngochuyen_voice`](https://huggingface.co/datasets/pnnbao-ump/ngochuyen_voice)
(CC BY-NC 4.0); InfoRe, InfoRe Technology 25-hour Vietnamese speech corpus
([`doof-ferb/infore1_25hours`](https://huggingface.co/datasets/doof-ferb/infore1_25hours), CC BY 4.0); Trúc Ly and Hải Đăng, preset
voices of [VieNeu-TTS v3 Turbo](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo) (Apache-2.0).

## Citation

```bibtex
@misc{tihontts2026,
  author       = {Nguyen Ngoc Thien},
  title        = {TiHon-TTS: Tiny Vietnamese Text-to-Speech for Edge Device},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/ngocthien/tihon-tts}}
}
```
