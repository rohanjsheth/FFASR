from . import audio_mixing as am
import numpy as np
def make_scene(
    speech,
    noise,
    speech_rir,
    speech_distance,
    noise_rir,
    noise_distance,
    sr,
    target_snr_db
):
    #Get room sounds of both
    room_speech = am.convolve(speech, speech_rir, speech_distance, sr)

    resized_noise = am.loop_to_length(noise, len(room_speech))
    room_noise = am.convolve(resized_noise, noise_rir, noise_distance, sr)
    room_noise = room_noise[:len(room_speech)]

    speech_mask = am.active_speech_mask(room_speech, sr)

    g = am.noise_gain_for_snr(room_speech, room_noise, target_snr_db, speech_mask)

    scaled_noise  = room_noise * g
    mixture = room_speech + scaled_noise

    final_mix, final_speech, final_noise = am.prevent_clipping(mixture, room_speech, scaled_noise)

    final_snr_db = am.measure_snr_db(
    final_speech,
    final_noise,
    speech_mask
    )

    snr_matches = np.isclose(
        final_snr_db,
        target_snr_db,
        atol=1e-3
    )

    return final_mix, {
        "target_snr_db": target_snr_db,
        "final_snr_db": final_snr_db,
        "snr_matches": snr_matches,
    }
