from collections.abc import Iterator
from contextlib import closing
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
num_scenes = 50
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
    iterator: Iterator[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    """Pull the next `count` records off an open streaming iterator."""
    return [next(iterator) for _ in range(count)]


# --- Dataset setup ---------------------------------------------------------

print("Loading files...")
speech_ds = load_dataset(
    "openslr/librispeech_asr",
    "clean",
    split="test",
    cache_dir="./.hf_cache",
    streaming=True,
)

noise_ds = load_dataset("bilguun/musan-noise", split="train", streaming=True)

rir_ds = load_dataset(
    "treble-technologies/Treble10-RIR", split="rir_mono", streaming=True
)

# decode=False leaves the raw bytes in place for read_audio to open.
speech_ds = speech_ds.cast_column("audio", Audio(decode=False))
rir_ds = rir_ds.cast_column("audio", Audio(decode=False))

# MUSAN is a SoundFolder, so it yields paths. Keeping the column a plain string
# avoids Audio.encode_example, which imports TorchCodec even with decoding off.
noise_ds = noise_ds.cast_column("audio", Value("string"))


# --- Scene generation ------------------------------------------------------

# One iterator per stream, held open across the whole run. Calling iter() per
# scene would rewind each stream and rebuild the same scene every time.
with (
    closing(iter(speech_ds)) as speech_iter,
    closing(iter(rir_ds)) as rir_iter,
    closing(iter(noise_ds)) as noise_iter,
):
    for i in range(num_scenes):
        print(f"Working on Waveform {i}...")

        # One speech clip, three RIRs (one for speech, two for the noises),
        # and two noise clips per scene.
        speech_rec = take_records(speech_iter, 1)[0]
        speech, speech_sr = read_audio(speech_rec["audio"])

        rir_records = take_records(rir_iter, 3)
        rir_audio = [read_audio(record["audio"]) for record in rir_records]

        noise_records = take_records(noise_iter, 2)
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

        # Derive per-scene so the run stays reproducible but scenes differ.
        scene_seed = rng_seed + i
        rng = np.random.default_rng(scene_seed)

        print("Simulating...")

        rev, metadata = make_scene(
            speech=speech_source,
            noises=noise_sources,
            config=scene_config,
            rng=rng,
            rng_seed=scene_seed,
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
        plt.savefig(f"./sample_plots/{i}_audio_reverb.png")

        # Save the mixture alongside its spectrogram.
        print("Saving...")
        sf.write(f"./sample_audios/{i}_audio_reverb.wav", rev, sr, subtype="FLOAT")
        print(f"Saved: ./sample_audios/{i}_audio_reverb.wav")
