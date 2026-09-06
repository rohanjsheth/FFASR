"""Check frozen paired audio on CPU and optionally export listening examples."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from eval_utils.text_norm import normalize_for_wer
from scripts.evaluate_snr_wer import BAND_ORDER, classify_snr, frozen_batch_scenes


def audit(path: Path, samples_dir: Path | None = None) -> dict:
    groups = defaultdict(dict)
    counts = Counter()
    snrs = defaultdict(list)
    kinds = Counter()
    categories = Counter()
    total_seconds = 0.0
    maximum_peak = 0.0
    seen_ids = set()
    first_speech = None
    experiment = None
    expected = {("none", "clean"), *((source, band) for source in ("musan", "aid") for band in BAND_ORDER)}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
        for row in batch.to_pylist():
            key = (row["noise_source"], row["condition"])
            speech_index = row["speech_index"]
            if key not in expected or row["id"] in seen_ids or key in groups[speech_index]:
                raise ValueError(f"Unexpected/duplicate scene: {row['id']}")
            seen_ids.add(row["id"])
            current_experiment = json.loads(row["experiment_json"])
            if experiment is None:
                experiment = current_experiment
            elif current_experiment != experiment:
                raise ValueError("Mixed experiment provenance")
            if not normalize_for_wer(row["text"]):
                raise ValueError(f"Empty normalized reference: {row['id']}")
            scene = frozen_batch_scenes([row], row["sample_rate"])[0]
            if row["sample_rate"] != experiment["args"]["sample_rate"]:
                raise ValueError("Sample rate differs from experiment")
            audio = scene["audio"]
            peak = float(np.max(np.abs(audio)))
            maximum_peak = max(maximum_peak, peak)
            if not np.any(audio):
                raise ValueError(f"Silent scene: {row['id']}")
            total_seconds += len(audio) / row["sample_rate"]
            label = "clean" if key[1] == "clean" else "/".join(key)
            counts[label] += 1
            recipe = json.loads(row["recipe_json"])
            if key[1] != "clean":
                if peak > 0.990001:
                    raise ValueError(f"Peak guard exceeded: {row['id']}")
                snr = float(scene["metadata"]["final_snr_db"])
                if classify_snr(snr) != key[1]:
                    raise ValueError(f"Out-of-band scene: {row['id']}")
                snrs[label].append(snr)
                noise_records = json.loads(row["noise_records_json"])
                if key == ("aid", "high"):
                    types = {noise["noise_kind"] for noise in noise_records}
                    kinds["mixed" if len(types) > 1 else next(iter(types))] += 1
                    categories.update(noise["category"] for noise in noise_records)
            groups[speech_index][key] = {
                "text": row["text"], "speech_id": row["speech_id"], "recipe": recipe,
                "snr": scene["metadata"].get("final_snr_db"),
            }
            if first_speech is None:
                first_speech = speech_index
            if samples_dir is not None and speech_index == first_speech:
                samples_dir.mkdir(parents=True, exist_ok=True)
                target = samples_dir / f"{int(speech_index)}_{key[0]}_{key[1]}.wav"
                with target.open("xb") as output:
                    output.write(row["audio"]["bytes"])
    if not groups:
        raise ValueError("Empty corpus")
    max_snr_difference = 0.0
    for index, conditions in groups.items():
        if set(conditions) != expected:
            raise ValueError(f"Incomplete conditions for speech {index}")
        if len({(row["text"], row["speech_id"]) for row in conditions.values()}) != 1:
            raise ValueError(f"Mismatched paired reference for speech {index}")
        for band in BAND_ORDER:
            musan, aid = (conditions[(source, band)] for source in ("musan", "aid"))
            controls = [{k: v for k, v in row["recipe"].items() if k != "noise_indices"}
                        for row in (musan, aid)]
            if controls[0] != controls[1]:
                raise ValueError(f"Non-noise parameters differ for speech {index}, {band}")
            max_snr_difference = max(max_snr_difference, abs(musan["snr"] - aid["snr"]))
    if max_snr_difference > experiment["args"]["snr_tolerance_db"]:
        raise ValueError("Paired realized SNRs exceed tolerance")
    if len(groups) != experiment["args"]["samples_per_band"]:
        raise ValueError("Utterance count differs from experiment")
    return {
        "status": "passed", "corpus": str(path), "bytes": path.stat().st_size,
        "utterances": len(groups), "scenes": len(seen_ids), "counts": dict(sorted(counts.items())),
        "audio_hours_per_model": total_seconds / 3600, "maximum_peak": maximum_peak,
        "max_paired_snr_difference_db": max_snr_difference,
        "snr_db": {key: {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}
                   for key, values in sorted(snrs.items())},
        "aid_scene_kinds": dict(kinds), "aid_source_categories": dict(sorted(categories.items())),
        "checks": ["WAV SHA256", "finite mono audio", "nonempty references", "complete pairing",
                   "shared non-noise recipe fields", "SNR bands and paired tolerance", "peak guard"],
        "experiment": experiment,
        "license_note": "AID archive LICENSE says CC-BY-NC-SA-4.0; Zenodo metadata says CC-BY-4.0. No commercial-use or redistribution clearance inferred.",
        "subjective_listening": "not performed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    report = audit(args.input, args.samples_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
