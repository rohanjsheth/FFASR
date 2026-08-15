from collections.abc import Sequence
from typing import Any

from . import audio_mixing as am
from .audio_mixing import FloatArray
import numpy as np


def make_scene(
    speech: FloatArray,
    noise_stems: Sequence[FloatArray],
    speech_rir: FloatArray,
    speech_distance: float,
    noise_rirs: Sequence[FloatArray],
    noise_distances: Sequence[float],
    noise_offsets_ms: Sequence[float],
    pink_db: float,
    rng: np.random.Generator,
    rng_seed: int,
    sr: int,
    target_snr_db: float,
) -> tuple[FloatArray, dict[str, Any]]:
    num_noises = len(noise_stems)
    num_rirs = len(noise_rirs)
    num_distances = len(noise_distances)
    num_offsets = len(noise_offsets_ms)

    if num_noises == 0:
        raise ValueError("At least one noise source is required")

    if not (num_noises == num_rirs == num_distances == num_offsets):
        raise ValueError(
            "noise_stems, noise_rirs, noise_distances, and noise_offsets_ms "
            "must have equal lengths; got "
            f"{num_noises}, {num_rirs}, {num_distances}, and {num_offsets}"
        )

    speech_drr_db = am.drr_db(speech_rir, sr, speech_distance)
    room_speech = am.convolve(speech, speech_rir, speech_distance, sr)

    room_noises = []
    noise_drrs_db = []

    for stem, rir, distance, offset_ms in zip(
        noise_stems,
        noise_rirs,
        noise_distances,
        noise_offsets_ms,
    ):
        noise_drrs_db.append(am.drr_db(rir, sr, distance))
        stem = am.normalize_rms(stem)
        warmup = len(rir) - 1
        resized_noise = am.loop_to_length(
            stem,
            warmup + len(room_speech),
            offset_ms,
            sr,
        )
        room_noise = am.convolve(resized_noise, rir, distance, sr)
        room_noise = room_noise[warmup : warmup + len(room_speech)]

        room_noises.append(room_noise)

    total_room_noise = np.sum(room_noises, axis=0)

    speech_mask = am.active_speech_mask(room_speech, sr)

    g = am.noise_gain_for_snr(
        room_speech,
        total_room_noise,
        target_snr_db,
        speech_mask,
    )

    scaled_noise = total_room_noise * g

    g_p = np.sqrt(am.signal_power(room_speech, speech_mask)) * (
        10 ** (-pink_db / 20)
    )
    pink = g_p * am.pink_noise(len(room_speech), rng)

    noise_component = scaled_noise + pink
    mixture = room_speech + noise_component

    final_mix, final_speech, final_noise, scale, clipped = am.prevent_clipping(
        mixture,
        room_speech,
        noise_component,
    )

    final_snr_db = am.measure_snr_db(final_speech, final_noise, speech_mask)

    return final_mix, {
        "target_snr_db": target_snr_db,
        "final_snr_db": float(final_snr_db),
        "pink_db": pink_db,
        "rng_seed": rng_seed,
        "speech_drr_db": float(speech_drr_db),
        "noise_drrs_db": [float(drr) for drr in noise_drrs_db],
        "clipping_scale": float(scale),
        "clipped": bool(clipped),
        "offsets": noise_offsets_ms,
    }
