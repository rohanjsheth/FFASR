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
