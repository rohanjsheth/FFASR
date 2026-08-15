from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from . import audio_mixing as am
from .audio_types import FloatArray, Noise, Speech, SceneConfig
import numpy as np


def make_scene(
    speech: Speech,
    noises: Sequence[Noise],
    config: SceneConfig,
    rng: np.random.Generator,
    rng_seed: int,
) -> tuple[FloatArray, dict[str, Any]]:
    if not noises:
        raise ValueError("At least one noise source is required")

    sr = config.sr

    speech_drr_db = am.drr_db(speech.rir, sr, speech.distance)
    room_speech = am.convolve(speech.stem, speech.rir, speech.distance, sr)

    room_noises = []
    noise_drrs_db = []

    for noise in noises:
        noise_drrs_db.append(am.drr_db(noise.rir, sr, noise.distance))
        stem = am.normalize_rms(noise.stem)
        warmup = len(noise.rir) - 1
        resized_noise = am.loop_to_length(
            stem,
            warmup + len(room_speech),
            noise.offset_ms,
            sr,
        )
        room_noise = am.convolve(resized_noise, noise.rir, noise.distance, sr)
        room_noise = room_noise[warmup : warmup + len(room_speech)]

        room_noises.append(room_noise)

    total_room_noise = np.sum(room_noises, axis=0)

    speech_mask = am.active_speech_mask(room_speech, sr)

    g = am.noise_gain_for_snr(
        room_speech,
        total_room_noise,
        config.target_snr_db,
        speech_mask,
    )

    scaled_noise = total_room_noise * g

    g_p = np.sqrt(am.signal_power(room_speech, speech_mask)) * (
        10 ** (-config.pink_db / 20)
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
        **asdict(config),
        "rng_seed": rng_seed,
        "final_snr_db": float(final_snr_db),
        "speech_drr_db": float(speech_drr_db),
        "noises": [
            {"drr_db": float(drr), "offset_ms": float(noise.offset_ms)}
            for drr, noise in zip(noise_drrs_db, noises)
        ],
        "clipping_scale": float(scale),
        "clipped": bool(clipped),
    }
