# Shared type vocabulary and the per-source data types for one scene.
#
# These classes are plain data holders; all DSP stays in audio_mixing.
# eq=False throughout: a generated __eq__ would compare the array fields
# elementwise and raise "truth value is ambiguous" instead of returning a bool.

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True, eq=False)
class Speech:
    stem: FloatArray
    rir: FloatArray
    distance: float


@dataclass(frozen=True, slots=True, eq=False)
class Noise:
    stem: FloatArray
    rir: FloatArray
    distance: float
    offset_ms: float = 0.0


@dataclass(frozen=True, slots=True, eq=False)
class SceneConfig:
    sr: int
    target_snr_db: float
    pink_db: float
