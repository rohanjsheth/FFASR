from . import audio_mixing as am
import numpy as np
def make_scene(
    speech,
    noise_stems,
    speech_rir,
    speech_distance,
    noise_rirs,
    noise_distances,
    noise_offsets_ms,
    sr,
    target_snr_db
):
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

    room_speech = am.convolve(speech, speech_rir, speech_distance, sr)

    room_noises = []

    for stem, rir, distance, offset_ms in zip(
        noise_stems,
        noise_rirs,
        noise_distances,
        noise_offsets_ms,
    ):
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

    g = am.noise_gain_for_snr(room_speech, total_room_noise, target_snr_db, speech_mask)

    scaled_noise  = total_room_noise * g
    mixture = room_speech + scaled_noise

    final_mix, final_speech, final_noise, scale, clipped = am.prevent_clipping(mixture, room_speech, scaled_noise)

    final_snr_db = am.measure_snr_db(
    final_speech,
    final_noise,
    speech_mask
    )

    return final_mix, {
        "target_snr_db": target_snr_db,
        "final_snr_db": final_snr_db,
        "clipping_scale": scale,
        "clipped": clipped,
        "offsets": noise_offsets_ms
    }
