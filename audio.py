from datasets import Audio, Value, load_dataset
from datasets.utils.file_utils import xopen
from scipy.signal import fftconvolve, resample_poly, spectrogram
import matplotlib.pyplot as plt
import numpy as np
import io, soundfile as sf
from audio_utils.make_scene import make_scene

sr = 16000


def read_audio(audio_record):
    """Read either embedded audio bytes or a local/remote audio path."""
    if isinstance(audio_record, dict) and audio_record.get("bytes") is not None:
        return sf.read(io.BytesIO(audio_record["bytes"]), dtype="float64")

    path = audio_record if isinstance(audio_record, str) else audio_record.get("path")
    if not path:
        raise ValueError("Audio record has neither bytes nor a path")

    with xopen(path, "rb") as audio_file:
        return sf.read(audio_file, dtype="float64")

print("Loading files...")
# 1. Load one LibriSpeech sample
speech_ds = load_dataset("openslr/librispeech_asr", "clean", split="test", cache_dir="./.hf_cache", streaming=True)
speech_ds = speech_ds.cast_column("audio", Audio(decode=False))
speech_rec = next(iter(speech_ds))
speech, speech_sr = read_audio(speech_rec["audio"])

# 2. Load one Treble10 RIR
rir_ds = load_dataset("treble-technologies/Treble10-RIR", split="rir_mono", streaming=True)
rir_ds = rir_ds.cast_column("audio", Audio(decode=False))
rir_rec = next(iter(rir_ds))
rir, rir_sr = read_audio(rir_rec["audio"])

# 3. Load one Noise
noise_ds = load_dataset("bilguun/musan-noise", split="train", streaming=True)
# SoundFolder yields paths. Keeping this as a string avoids Audio.encode_example,
# which imports TorchCodec even when decoding is disabled.
noise_ds = noise_ds.cast_column("audio", Value("string"))
noise_rec = next(iter(noise_ds))
noise, noise_sr = read_audio(noise_rec["audio"])

print("Downsampling...")
# 3. Downsample
if rir_sr != sr:
    rir = resample_poly(rir, sr, rir_sr)

if noise_sr != sr:
    noise = resample_poly(noise, sr, noise_sr)

if speech_sr != sr:
    speech = resample_poly(speech, sr, speech_sr)

print("Simulating...")
rir_distance = float(rir_rec["Direct Path Length [m]"])
rev, metadata = make_scene(speech, noise, rir, rir_distance, rir, rir_distance, sr, 10.0)

print("Plot Generating...")
f, t, Sxx = spectrogram(rev, fs=sr, nperseg=512, noverlap=256)
plt.pcolormesh(t, f, 10*np.log10(Sxx+1e-12), shading="auto")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.title("Spectrogram of Reverberated Speech")
plt.tight_layout()
plt.show()

# 6. Save
print("Saving...")
sf.write("audio_reverb.wav", rev, sr, subtype="FLOAT")
sf.write("speech.wav", speech, sr, subtype="FLOAT")
sf.write("noise.wav", noise, sr, subtype="FLOAT")
sf.write("rir.wav", rir, sr, subtype="FLOAT")
print("✅ Saved: audio_reverb.wav, speech.wav, noise.wav, rir.wav")
