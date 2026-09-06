"""Offline checks for AID assembly and paired, materialized evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import soundfile as sf
from datasets import Dataset

from scripts import evaluate_snr_wer as evaluate
from scripts import prepare_aid_noise as aid
from scripts import prepare_noise_eval as prepare
from scripts.noise_eval_io import sha256, wav_bytes, write_parquet

SR = 8000


def audio_record(signal: np.ndarray) -> dict:
    return {"bytes": wav_bytes(signal, SR), "path": "fixture.wav"}


def test_wav_encoding_has_no_timestamped_peak_chunk() -> None:
    import io
    signal = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
    encoded = wav_bytes(signal, SR)
    assert b"PEAK" not in encoded
    decoded, rate = sf.read(io.BytesIO(encoded), dtype="float32")
    assert rate == SR
    np.testing.assert_array_equal(decoded, signal)


def clips() -> list[aid.Clip]:
    return [aid.Clip(f"clapping_{i}_NT1.wav", "clapping", str(i), np.ones(400) * (i + 1))
            for i in range(3)]


@pytest.mark.parametrize("kind", ["continuous", "transient"])
def test_assembly_is_deterministic_and_does_not_mutate_sources(kind: str) -> None:
    sources = clips()
    original = [clip.audio.copy() for clip in sources]
    first, events = aid.assemble_stem(sources, SR * 2, SR, np.random.default_rng(42), kind)
    second, repeated = aid.assemble_stem(sources, SR * 2, SR, np.random.default_rng(42), kind)
    np.testing.assert_array_equal(first, second)
    assert events == repeated
    assert len(first) == SR * 2 and np.isfinite(first).all() and np.any(first)
    for clip, before in zip(sources, original, strict=True):
        np.testing.assert_array_equal(clip.audio, before)
    for left, right in zip(events, events[1:]):
        assert left["source"] != right["source"]
        if kind == "transient":
            gap = right["start_sample"] - left["end_sample"]
            assert round(0.25 * SR) <= gap <= round(1.5 * SR)
            assert not first[left["end_sample"]:right["start_sample"]].any()
        else:
            assert right["start_sample"] < left["end_sample"]


def make_aid_input(root: Path) -> None:
    root.mkdir()
    for category in ("blender", "hairdryer"):
        for index in (1, 2):
            wave = np.random.default_rng(index).normal(0, 0.1, SR // 10)
            sf.write(root / f"{category}_{index}_RHODE_NT1.wav", wave, SR, subtype="FLOAT")
            sf.write(root / f"{category}_{index}_NT5.wav", wave, SR, subtype="FLOAT")


def test_aid_cli_balanced_categories_one_mic_and_provenance(tmp_path: Path) -> None:
    source = tmp_path / "aid"
    make_aid_input(source)
    output = tmp_path / "noise.parquet"
    options = ["--input-dir", str(source), "--sample-rate", str(SR), "--count", "4",
               "--duration-seconds", "4", "--output", str(output)]
    assert aid.main(options) == 0
    rows = pq.read_table(output).to_pylist()
    assert [row["category"] for row in rows] == ["blender", "hairdryer"] * 2
    for row in rows:
        assert row["license"] == "CC-BY-NC-SA-4.0"
        assert row["microphone"] == "NT1" and row["duration"] == 4
        assert row["sha256"] == sha256(row["audio"]["bytes"])
        assert all(event["source"].endswith("_NT1.wav") for event in json.loads(row["events_json"]))
    with pytest.raises(FileExistsError):
        aid.main(options)


@pytest.mark.parametrize("options", [["--count", "0"], ["--duration-seconds", "nan"],
                                    ["--min-gap-seconds", "2", "--max-gap-seconds", "1"]])
def test_invalid_aid_options(options: list[str]) -> None:
    with pytest.raises(SystemExit):
        aid.parse_args(["--input-dir", "unused", *options])


def test_pairing_does_not_depend_on_noise_pool_size_or_order() -> None:
    groups = {"room": {"receiver": np.arange(8)}}
    recipes = prepare.paired_recipes(7, groups, {"musan": 2, "aid": 10000}, 42, 2)
    changed = prepare.paired_recipes(7, groups, {"aid": 3, "musan": 20}, 42, 2)
    controls = []
    for recipe in [*recipes.values(), *changed.values()]:
        values = asdict(recipe)
        indices = values.pop("noise_indices")
        assert len(set(indices)) == 2
        controls.append(values)
    assert all(control == controls[0] for control in controls)
    with pytest.raises(ValueError, match="fewer"):
        prepare.paired_recipes(7, groups, {"aid": 1}, 42, 2)


def fixture_datasets() -> tuple[Dataset, Dataset, Dataset]:
    rng = np.random.default_rng(9)
    speech = Dataset.from_list([
        {"audio": audio_record(rng.normal(0, 0.1, SR // 2)), "id": str(i), "text": "Hello world"}
        for i in range(2)
    ])
    noises = Dataset.from_list([
        {"audio": audio_record(rng.normal(0, 0.1, SR * 4)), "id": str(i)} for i in range(3)
    ])
    rirs = Dataset.from_list([
        {"audio": audio_record(np.r_[1., np.zeros(19), 0.2 + 0.1 * i, np.zeros(19)]),
         "Room": "room", "Receiver Label": "receiver", "Direct Path Length [m]": 0.0}
        for i in range(4)
    ])
    return speech, noises, rirs


def test_short_stems_fail_in_preparation_without_changing_dsp() -> None:
    speech, noises, rirs = fixture_datasets()
    recipe = prepare.paired_recipes(0, prepare.get_groups(rirs), {"aid": 3}, 0, 2)["aid"]
    prepare.require_unlooped_stems(recipe, speech, noises, rirs, SR)
    short = Dataset.from_list([{"audio": audio_record(np.ones(10))} for _ in range(3)])
    with pytest.raises(ValueError, match="Reassemble"):
        prepare.require_unlooped_stems(recipe, speech, short, rirs, SR)


@pytest.mark.parametrize("gap", [0.1, 0.3])
def test_pair_tolerance_is_enforced_without_changing_input_targets(gap: float, monkeypatch: pytest.MonkeyPatch) -> None:
    recipes = prepare.paired_recipes(0, {"room": {"receiver": np.arange(4)}},
                                     {"musan": 3, "aid": 3}, 0, 2)
    targets = []

    def render(recipe, speech_ds, noise_ds, rir_ds, sr):
        targets.append(recipe.target_snr_db)
        return {"metadata": {"final_snr_db": recipe.target_snr_db + (gap if noise_ds == "aid" else 0)}}

    monkeypatch.setattr(prepare, "render_scene_from_recipe", render)
    args = (recipes, "high", np.random.default_rng(0), None,
            {"musan": "musan", "aid": "aid"}, None, SR, 2, 0.25)
    if gap < 0.25:
        _, rejected = prepare.render_pair_in_band(*args)
        assert rejected == 0
    else:
        with pytest.raises(RuntimeError, match="jointly render"):
            prepare.render_pair_in_band(*args)
    assert all(left == right for left, right in zip(targets[::2], targets[1::2]))


def test_full_offline_preparation_and_frozen_scoring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    speech, noises, rirs = fixture_datasets()
    for name, dataset in (("speech", speech), ("musan", noises), ("rirs", rirs)):
        dataset.to_parquet(tmp_path / f"{name}.parquet")
    source = tmp_path / "aid"
    make_aid_input(source)
    aid_path = tmp_path / "aid.parquet"
    aid.main(["--input-dir", str(source), "--output", str(aid_path), "--count", "4",
              "--sample-rate", str(SR), "--duration-seconds", "4"])
    corpus = tmp_path / "paired.parquet"
    options = ["--aid-noise-parquet", str(aid_path), "--musan-noise-parquet", str(tmp_path / "musan.parquet"),
               "--speech-parquet", str(tmp_path / "speech.parquet"), "--rir-parquet", str(tmp_path / "rirs.parquet"),
               "--samples-per-band", "2", "--sample-rate", str(SR), "--output", str(corpus),
               "--cache-dir", str(tmp_path / "cache")]
    assert prepare.main(options) == 0
    rows = pq.read_table(corpus).to_pylist()
    assert len(rows) == 14  # Two utterances, one clean plus 2 noise pools x 3 bands.
    from scripts.audit_noise_eval import audit
    report = audit(corpus)
    assert report["status"] == "passed" and report["utterances"] == 2
    assert report["scenes"] == 14
    for index in range(2):
        for band in prepare.BAND_ORDER:
            pair = [row for row in rows if row["speech_index"] == index and row["condition"] == band]
            assert {row["noise_source"] for row in pair} == {"musan", "aid"}
            controls = []
            for row in pair:
                control = json.loads(row["recipe_json"])
                control.pop("noise_indices")
                controls.append(control)
                assert evaluate.classify_snr(json.loads(row["metadata_json"])["final_snr_db"]) == band
            assert controls[0] == controls[1]
            snrs = [json.loads(row["metadata_json"])["final_snr_db"] for row in pair]
            assert abs(snrs[0] - snrs[1]) <= 0.05
    second = tmp_path / "repeated.parquet"
    options[options.index(str(corpus))] = str(second)
    prepare.main(options)
    assert [row["sha256"] for row in pq.read_table(second).to_pylist()] == [row["sha256"] for row in rows]

    seen = []
    monkeypatch.setattr(evaluate, "load_model", lambda args: (None, None, None, None))

    def transcribe(**kwargs):
        seen.extend(scene["audio"].copy() for scene in kwargs["scenes"])
        return [scene["text"] for scene in kwargs["scenes"]]

    monkeypatch.setattr(evaluate, "transcribe_batch", transcribe)
    for model in ("stock", "tiro"):
        assert evaluate.main(["--rendered-parquet", str(corpus), "--model-id", model,
                              "--sample-rate", str(SR), "--batch-size", "3",
                              "--cache-dir", str(tmp_path / "cache"),
                              "--output-dir", str(tmp_path / model)]) == 0
        summary = json.loads((tmp_path / model / "summary.json").read_text())
        assert len(summary["conditions"]) == 7
        assert all(score["wer"] == 0 and score["examples"] == 2 for score in summary["conditions"].values())
    for first, second_audio in zip(seen[:14], seen[14:], strict=True):
        np.testing.assert_array_equal(first, second_audio)

    broken = {**rows[0], "sha256": "bad"}
    with pytest.raises(ValueError, match="checksum"):
        evaluate.frozen_batch_scenes([broken], SR)
    with pytest.raises(ValueError, match="rate mismatch"):
        evaluate.frozen_batch_scenes([rows[0]], SR * 2)


def test_failed_parquet_is_not_published(tmp_path: Path) -> None:
    output = tmp_path / "broken.parquet"

    def broken_rows():
        for _ in range(16):
            yield {"value": 1}
        raise RuntimeError("broken source")

    with pytest.raises(RuntimeError, match="broken source"):
        write_parquet(output, broken_rows())
    assert not output.exists()
    assert not list(tmp_path.iterdir())
