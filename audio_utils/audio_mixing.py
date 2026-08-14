# Audio simulation utilities.
#
# Assumptions / caveats:
# - Waveforms are floating-point arrays and RIRs have already been resampled to `sr`.
# - `distance` is source-to-receiver direct-path length in meters; direct arrival is
#   estimated using c = 343 m/s and refined by a local peak search.
# - Trimming each RIR near its direct arrival removes most absolute propagation delay,
#   so relative source-to-microphone delays are not preserved between sources.
# - DRR uses a 2.5 ms total direct-path window; DRR values depend on this convention.
# - The speech-activity mask (25 ms frames, 1% of peak frame energy) is a heuristic
#   used for SNR measurement and may not exactly match the benchmark's implementation.
# - Continuous noise is tiled when too short, which can create periodic repetition.
# - RMS normalization should only be used before scene-level SNR scaling; do not
#   independently normalize speech/noise after setting the target SNR.
# - Anti-clipping is applied as one shared gain to mixture, speech, and noise so SNR
#   is preserved.

import numpy as np
from scipy.signal import fftconvolve

EPS = 1e-12
c = 343


def direct_index(rir, distance, sr, window_ms=1.0):
    """Find the direct arrival near the physically expected sample."""
    expected_idx = round(distance / c * sr)
    window_samples = round(sr * window_ms / 1000)

    start = max(0, expected_idx - window_samples)
    end = min(len(rir), expected_idx + window_samples + 1)

    if start >= end:
        raise ValueError("Invalid direct-path search window")

    return int(np.argmax(np.abs(rir[start:end]))) + start


def drr_db(rir, sr, distance, half_window_ms=1.25):
    """Compute DRR using a 2.5 ms total window around the direct arrival."""
    direct_idx = direct_index(rir, distance, sr)

    half_window_samples = round(sr * half_window_ms / 1000)

    direct_start = max(0, direct_idx - half_window_samples)
    direct_end = min(len(rir), direct_idx + half_window_samples + 1)

    direct_energy = float(np.sum(rir[direct_start:direct_end] ** 2))
    total_energy = float(np.sum(rir ** 2))
    reverberant_energy = max(total_energy - direct_energy, 0.0)

    return 10 * np.log10(
        (direct_energy + EPS) / (reverberant_energy + EPS)
    )


def trim_rir(rir, distance, sr, pre_ms=1.0):
    direct_idx = direct_index(rir, distance, sr)
    lead_samples = round(sr * pre_ms / 1000)

    start = max(0, direct_idx - lead_samples)
    return rir[start:]


def convolve(audio, rir, distance, sr):
    rir = trim_rir(rir, distance, sr)
    return fftconvolve(audio, rir, mode="full")


def loop_to_length(audio, target_length, offset_ms, sr):
    offset = round(offset_ms * sr / 1000)

    if len(audio) == 0:
        raise ValueError("Cannot loop empty audio")

    repeats = int(np.ceil((offset + target_length) / len(audio)))
    return np.tile(audio, repeats)[offset:offset+target_length]


def rms(audio):
    return np.sqrt(np.mean(audio ** 2))


def normalize_rms(audio):
    return audio / (rms(audio) + EPS)


def active_speech_mask(speech, sr=16000, frame_ms=25, threshold=0.01):
    frame_len = int(sr * frame_ms / 1000)

    mask = np.zeros(len(speech), dtype=bool)
    energies = []

    starts = list(range(0, len(speech), frame_len))

    for start in starts:
        frame = speech[start:start + frame_len]
        energies.append(np.mean(frame ** 2))

    peak = max(energies) + EPS
    cutoff = threshold * peak

    for start, energy in zip(starts, energies):
        if energy >= cutoff:
            mask[start:start + frame_len] = True

    return mask


def signal_power(signal, mask):
    """Return mean-square power over masked samples."""
    if not np.any(mask):
        print("Activity mask contains no active samples")

    return np.mean(signal[mask] ** 2)


def measure_snr_db(speech, noise, mask):
    p_speech = signal_power(speech, mask)
    p_noise = signal_power(noise, mask) + EPS

    return 10 * np.log10(p_speech / p_noise)


def noise_gain_for_snr(speech, noise, target_snr_db, mask):
    snr_diff_db = measure_snr_db(speech, noise, mask) - target_snr_db

    # snr_diff_db = 20 * log10(g)
    return 10 ** (snr_diff_db / 20)


def prevent_clipping(mixture, speech, noise, max_peak=0.99):
    peak = np.max(np.abs(mixture))
    scale = 1.0

    over = peak > max_peak
    if over:
        scale = max_peak / peak
        mixture *= scale
        speech *= scale
        noise *= scale

    return mixture, speech, noise, scale, over
