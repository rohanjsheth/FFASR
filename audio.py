from collections.abc import Iterable
from typing import Any

from datasets import Audio, Value, load_dataset
from datasets.utils.file_utils import xopen
from scipy.signal import resample_poly, spectrogram
import matplotlib.pyplot as plt
import numpy as np
import io, soundfile as sf
from audio_utils.audio_types import FloatArray, Speech, Noise, SceneConfig
from audio_utils.make_scene import make_scene


rng_seed = 0
noise_offsets_ms = [0.0, 500.0]

scene_config = SceneConfig(sr=16000, pink_db=20.0, target_snr_db=10.0)

sr = scene_config.sr


def read_audio(audio_record: dict[str, Any] | str) -> tuple[FloatArray, int]:
    """Read either embedded audio bytes or a local/remote audio path."""
    if isinstance(audio_record, dict) and audio_record.get("bytes") is not None:
        return sf.read(io.BytesIO(audio_record["bytes"]), dtype="float64")

    path = audio_record if isinstance(audio_record, str) else audio_record.get("path")
    if not path:
        raise ValueError("Audio record has neither bytes nor a path")

    with xopen(path, "rb") as audio_file:
        return sf.read(audio_file, dtype="float64")


def resample(audio: FloatArray, input_sr: int) -> FloatArray:
    """Resample one waveform to the scene sample rate."""
    if input_sr == sr:
        return audio
    return resample_poly(audio, sr, input_sr)


def take_records(
    dataset: Iterable[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    """Take records from a streaming dataset and close its iterator promptly."""
    iterator = iter(dataset)
    try:
        return [next(iterator) for _ in range(count)]
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


print("Loading files...")
# 1. Load one LibriSpeech sample
speech_ds = load_dataset(
    "openslr/librispeech_asr",
    "clean",
    split="test",
    cache_dir="./.hf_cache",
    streaming=True,
)
speech_ds = speech_ds.cast_column("audio", Audio(decode=False))
speech_rec = take_records(speech_ds, 1)[0]
speech, speech_sr = read_audio(speech_rec["audio"])

# 2. Load one speech RIR and two noise RIRs
rir_ds = load_dataset(
    "treble-technologies/Treble10-RIR", split="rir_mono", streaming=True
)
rir_ds = rir_ds.cast_column("audio", Audio(decode=False))
rir_records = take_records(rir_ds, 3)
rir_audio = [read_audio(record["audio"]) for record in rir_records]

# 3. Load two noise clips
noise_ds = load_dataset("bilguun/musan-noise", split="train", streaming=True)
# SoundFolder yields paths. Keeping this as a string avoids Audio.encode_example,
# which imports TorchCodec even when decoding is disabled.
noise_ds = noise_ds.cast_column("audio", Value("string"))
noise_records = take_records(noise_ds, 2)
noise_audio = [read_audio(record["audio"]) for record in noise_records]

print("Downsampling...")
speech = resample(speech, speech_sr)
rirs = [resample(waveform, input_sr) for waveform, input_sr in rir_audio]
noises = [resample(waveform, input_sr) for waveform, input_sr in noise_audio]

speech_source = Speech(
    stem=speech, rir=rirs[0], distance=float(rir_records[0]["Direct Path Length [m]"])
)

noise_sources = [
    Noise(
        stem=stem,
        rir=rir,
        distance=float(record["Direct Path Length [m]"]),
        offset_ms=offset_ms,
    )
    for stem, rir, record, offset_ms in zip(
        noises, rirs[1:], rir_records[1:], noise_offsets_ms, strict=True
    )
]

rng = np.random.default_rng(rng_seed)

print("Simulating...")

rev, metadata = make_scene(
    speech=speech_source,
    noises=noise_sources,
    config=scene_config,
    rng=rng,
    rng_seed=rng_seed,
)
print("Metadata:", metadata)

print("Plot Generating...")
f, t, Sxx = spectrogram(rev, fs=sr, nperseg=512, noverlap=256)
plt.figure(figsize=(10, 4))
plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="auto")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.title("Spectrogram of Simulated Noisy Reverberant Speech")
plt.colorbar(label="Power [dB]")
plt.tight_layout()
plt.show()

# 4. Save the mixture and diagnostic inputs
print("Saving...")
sf.write("audio_reverb.wav", rev, sr, subtype="FLOAT")
print("Saved: audio_reverb.wav")
