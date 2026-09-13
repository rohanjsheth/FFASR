"""FFASR custom evaluator for Tiro v2 -- paste verbatim into the Space's editor.

The leaderboard's default backends cannot drive a native-HF Qwen3ASR model: the
generic pipeline hands mel features in as `input_ids`, and the universal
`_call_processor` supplies the waveform positionally as `text`
(`results/ffasr_loading_probe/`). Only `apply_transcription_request` builds the
transcription prompt correctly, so the custom evaluator exists to use it.

Decoding matches `eval_utils/transcribe.py` exactly -- greedy, one beam, 256 new
tokens, bfloat16, the same prompt language -- so every WER this campaign recorded
predicts what this file will produce. `scripts/verify_ffasr_evaluator.py` checks
that claim against already-scored audio.
"""

from pathlib import Path

import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

MODEL_ID = "rohansheth/tiro-qwen3-asr-1.7b-v2"
REVISION = "d81d6f28a554e5af8f03218b442918fc1456f31a"
SAMPLE_RATE = 16_000
LANGUAGE = "English"
MAX_NEW_TOKENS = 256

processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
model = Qwen3ASRForConditionalGeneration.from_pretrained(
    MODEL_ID, revision=REVISION, dtype=torch.bfloat16, device_map="auto"
).eval()


def evaluate(file: Path) -> str:
    audio, rate = sf.read(str(file), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, rate).astype("float32")

    inputs = processor.apply_transcription_request(
        audio=[audio], language=LANGUAGE,
        processor_kwargs={"sampling_rate": SAMPLE_RATE},
    )
    prompt_length = inputs["input_ids"].shape[1]
    inputs = inputs.to(model.device, model.dtype)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, do_sample=False, num_beams=1,
            max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
        )

    decoded = processor.batch_decode(
        [output_ids[0][prompt_length:]], skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return str(processor.extract_transcription(decoded))
