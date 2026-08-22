from dataclasses import dataclass
from typing import Any, TypedDict

from datasets import Dataset
from datasets.utils.file_utils import xopen
from scipy.signal import resample_poly
import numpy as np
import numpy.typing as npt
import io, soundfile as sf
from audio_utils.audio_types import FloatArray, Speech, Noise, SceneConfig
from audio_utils.make_scene import make_scene


RIRIndex = dict[str, dict[str, npt.NDArray[np.intp]]]


class RenderedScene(TypedDict):
    audio: FloatArray
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SceneRecipe:
    speech_index: int
    room: str
    receiver: str
    speech_rir_index: int
    noise_indices: tuple[int, ...]
    noise_rir_indices: tuple[int, ...]
    offsets_ms: tuple[float, ...]
    number_of_noises: int
    target_snr_db: float
    pink_db: float
    rng_seed: int

def read_audio(audio_record: dict[str, Any] | str) -> tuple[FloatArray, int]:
    if isinstance(audio_record, dict) and audio_record.get("bytes") is not None:
        return sf.read(io.BytesIO(audio_record["bytes"]), dtype="float64")

    path = audio_record if isinstance(audio_record, str) else audio_record.get("path")
    if not path:
        raise ValueError("Audio record has neither bytes nor a path")

    with xopen(path, "rb") as audio_file:
        return sf.read(audio_file, dtype="float64")

def audio_duration_seconds(audio_record: dict[str, Any] | str) -> float:
    if isinstance(audio_record, dict) and audio_record.get("bytes") is not None:
        info = sf.info(io.BytesIO(audio_record["bytes"]))
    else:
        path = audio_record if isinstance(audio_record, str) else audio_record.get("path")
        if not path:
            raise ValueError("Audio record has neither bytes nor a path")
        info = sf.info(path)
    return info.frames / info.samplerate

def resample(audio: FloatArray, input_sr: int, global_sr : int) -> FloatArray:
    if input_sr == global_sr:
        return audio
    return resample_poly(audio, global_sr, input_sr)

def read_and_resample(audio_record: dict[str, Any] | str, global_sr : int) -> FloatArray:
    audio, audio_sr = read_audio(audio_record)
    resampled = resample(audio, audio_sr, global_sr)
    return resampled

def sample_key(mapping, rng):
    keys = tuple(mapping)
    return keys[rng.integers(len(keys))]

def get_groups(rir_ds: Dataset) -> RIRIndex: 
    metadata_df = rir_ds.select_columns(
        ["Room", "Receiver Label"]
    ).to_pandas()

    flat_groups =  metadata_df.groupby(
      ["Room", "Receiver Label"]
    ).indices

    groups = {}

    for (room, receiver), row_indices in flat_groups.items():
        groups.setdefault(
            room, {}
        )[receiver] = row_indices

    return groups


def sample_scene_recipe(
    speech_index: int,
    noise_ds: Dataset,
    groups: RIRIndex,
    rng: np.random.Generator,
    number_of_noises: int,
) -> SceneRecipe:
    room = sample_key(groups, rng)
    receiver = sample_key(groups[room], rng)

    rirs = groups[room][receiver]

    rir_indices = rng.choice(
        rirs,
        size=1 + number_of_noises,
        replace = False
    )

    speech_rir_index = int(rir_indices[0])

    noise_rir_indices = tuple(int(index) for index in rir_indices[1:])

    noise_indices = rng.choice(
        len(noise_ds),
        size=number_of_noises,
        replace=False
    )

    noise_indices = tuple(int(index) for index in noise_indices)

    target_snr_db = float(rng.uniform(0, 24))
    offsets_ms = tuple(
        float(offset)
        for offset in rng.uniform(0, 3000, size=number_of_noises)
    )
    pink_db = float(rng.uniform(35, 55))
    rng_seed = int(rng.integers(0, np.iinfo(np.int64).max))

    return SceneRecipe(
        speech_index=speech_index,
        room=room,
        receiver=receiver,
        speech_rir_index=speech_rir_index,
        noise_indices=noise_indices,
        noise_rir_indices=noise_rir_indices,
        offsets_ms=offsets_ms,
        number_of_noises=number_of_noises,
        target_snr_db=target_snr_db,
        pink_db=pink_db,
        rng_seed=rng_seed,
    )

def render_scene_from_recipe(
    recipe: SceneRecipe,
    speech_ds: Dataset,
    noise_ds: Dataset,
    rir_ds: Dataset,
    sr: int,
) -> RenderedScene:
    speech_rec = speech_ds[recipe.speech_index]
    speech_rir_rec = rir_ds[recipe.speech_rir_index]
    noise_recs = [noise_ds[index] for index in recipe.noise_indices]
    noise_rir_recs = [rir_ds[index] for index in recipe.noise_rir_indices]

    speech = read_and_resample(speech_rec["audio"], sr)
    speech_rir = read_and_resample(speech_rir_rec["audio"], sr)
    noise_rirs = [read_and_resample(rec["audio"], sr) for rec in noise_rir_recs]
    noises = [read_and_resample(rec["audio"], sr) for rec in noise_recs]

    speech_source = Speech(
                stem=speech, rir=speech_rir, distance=float(speech_rir_rec["Direct Path Length [m]"])
            )

    noise_sources = [
                Noise(
                    stem=stem,
                    rir=rir,
                    distance=float(record["Direct Path Length [m]"]),
                    offset_ms=offset_ms,
                )
                for stem, rir, record, offset_ms in zip(
                    noises, noise_rirs, noise_rir_recs, recipe.offsets_ms, strict=True
                )
            ]

    config = SceneConfig(
        sr = sr,
        target_snr_db=recipe.target_snr_db,
        pink_db=recipe.pink_db
    )

    text = speech_rec["text"]

    # make_scene uses this RNG only to generate reproducible pink noise.
    rng = np.random.default_rng(recipe.rng_seed)

    scene, metadata = make_scene(
                speech=speech_source,
                noises=noise_sources,
                config=config,
                rng=rng,
                rng_seed=recipe.rng_seed,
            )

    return {
        "audio": scene, 
        "text": text, 
        "metadata": metadata 
    }

def render_clean_scene(
    speech_index: int,
    speech_ds: Dataset,
    sr: int,
) -> RenderedScene:
    speech_rec = speech_ds[speech_index]
    audio = read_and_resample(speech_rec["audio"], sr)
    return {
        "audio" : audio,
        "text" : speech_rec["text"],
        "metadata" : {
            "info": "clean"
        }
    }


    
