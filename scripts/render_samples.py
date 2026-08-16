"""Render a small collection of scenes for manual audio inspection."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render noisy, reverberant speech examples and spectrograms."
    )
    parser.add_argument("--num-scenes", type=int, default=50)
    parser.add_argument("--number-of-noises", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("samples"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf
    from datasets import Audio, Value, load_dataset
    from scipy.signal import spectrogram

    from data_utils import (
        get_groups,
        render_scene_from_recipe,
        sample_scene_recipe,
    )

    print("Loading datasets...")
    cache_dir = str(args.cache_dir)

    speech_ds = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="test",
        cache_dir=cache_dir,
    )
    noise_ds = load_dataset(
        "bilguun/musan-noise",
        split="train",
        cache_dir=cache_dir,
    )
    rir_ds = load_dataset(
        "treble-technologies/Treble10-RIR",
        split="rir_mono",
        cache_dir=cache_dir,
    )

    speech_ds = speech_ds.cast_column("audio", Audio(decode=False))
    rir_ds = rir_ds.cast_column("audio", Audio(decode=False))

    # MUSAN is a SoundFolder, so retaining paths avoids automatic decoding.
    noise_ds = noise_ds.cast_column("audio", Value("string"))

    groups = get_groups(rir_ds)
    recipe_rng = np.random.default_rng(args.seed)

    audio_dir = args.output_dir / "sample_audios"
    plot_dir = args.output_dir / "sample_plots"
    audio_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if args.num_scenes > len(speech_ds):
        raise ValueError(
            f"Requested {args.num_scenes} scenes from {len(speech_ds)} speech records"
        )

    for speech_index in range(args.num_scenes):
        recipe = sample_scene_recipe(
            speech_index=speech_index,
            noise_ds=noise_ds,
            groups=groups,
            rng=recipe_rng,
            number_of_noises=args.number_of_noises,
        )
        rendered_scene = render_scene_from_recipe(
            recipe=recipe,
            speech_ds=speech_ds,
            noise_ds=noise_ds,
            rir_ds=rir_ds,
            sr=args.sample_rate,
        )
        scene = rendered_scene["audio"]
        text = rendered_scene["text"]
        metadata = rendered_scene["metadata"]

        stem = f"{speech_index:04d}_audio_reverb"
        sf.write(audio_dir / f"{stem}.wav", scene, args.sample_rate, subtype="FLOAT")

        frequencies, times, power = spectrogram(
            scene,
            fs=args.sample_rate,
            nperseg=512,
            noverlap=256,
        )
        figure, axes = plt.subplots(figsize=(10, 4))
        image = axes.pcolormesh(
            times,
            frequencies,
            10 * np.log10(power + 1e-12),
            shading="auto",
        )
        axes.set_xlabel("Time [s]")
        axes.set_ylabel("Frequency [Hz]")
        axes.set_title("Simulated noisy reverberant speech")
        figure.colorbar(image, ax=axes, label="Power [dB]")
        figure.tight_layout()
        figure.savefig(plot_dir / f"{stem}.png")
        plt.close(figure)

        print(f"Rendered {speech_index + 1}/{args.num_scenes}: {text!r}")
        print(f"Recipe: {recipe}")
        print(f"Metadata: {metadata}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
