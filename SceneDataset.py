from datasets import Dataset as HFDataset
from data_utils import (
    RIRIndex,
    RenderedScene,
    get_groups,
    render_scene_from_recipe,
    render_clean_scene,
    sample_scene_recipe,
)
from torch.utils.data import Dataset
import numpy as np


class SceneDataset(Dataset):
    def __init__(
        self,
        speech_ds: HFDataset,
        noise_ds: HFDataset,
        rir_ds: HFDataset,
        base_seed: int,
        sample_rate: int,
        number_of_noises: int = 2,
        clean_probability: float = 0.25,
    ) -> None:
        self._speech_ds = speech_ds
        self._noise_ds = noise_ds
        self._rir_ds = rir_ds
        self._groups: RIRIndex = get_groups(rir_ds)
        self._base_seed = base_seed
        self._sample_rate = sample_rate
        self._number_of_noises = number_of_noises
        self._epoch = 0
        self._clean_probability = clean_probability


    def __len__(self) -> int:
        return len(self._speech_ds)

    def __getitem__(self, idx: int) -> RenderedScene:
        seed = np.random.SeedSequence([self._base_seed, self._epoch, idx])
        rng = np.random.default_rng(seed)

        if rng.random() < self._clean_probability:
            return render_clean_scene(idx, self._speech_ds, self._sample_rate)

        recipe = sample_scene_recipe(idx, self._noise_ds, self._groups, rng, self._number_of_noises)
        return render_scene_from_recipe(recipe, self._speech_ds, self._noise_ds, self._rir_ds, self._sample_rate)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
