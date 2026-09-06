"""Run the pinned upstream OpenASR English suite inside its official HF Job image."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request


UPSTREAM_REVISION = "48219c6028db0517d704600d92f31edfc96e8c23"
MODEL_ID = "rohansheth/tiro-qwen3-asr-1.7b-ffasr-v1"
MODEL_REVISION = "92a967532f9b10abdb898a701dab175583a8cfdd"
DEFAULT_DATASET = "hf-audio/open-asr-leaderboard"
# Same eight public English splits as transformers/submit_jobs_qwen3asr.sh.
# Order the smaller sets first so a timeout preserves useful completed results.
DATASETS = [
    ("librispeech-clean", DEFAULT_DATASET, "librispeech", "test.clean"),
    ("librispeech-other", DEFAULT_DATASET, "librispeech", "test.other"),
    ("voxpopuli", DEFAULT_DATASET, "voxpopuli_cleaned_aa", "test"),
    ("ami", DEFAULT_DATASET, "ami_cleaned", "test"),
    ("gigaspeech", DEFAULT_DATASET, "gigaspeech_cleaned", "test"),
    ("earnings22", "ArtificialAnalysis/Earnings22-Cleaned-AA-chunked", "", "test"),
    ("spgispeech", DEFAULT_DATASET, "spgispeech", "test"),
    ("monsoon", "VoiceArena/Monsoon_en_IN_test", "", "test"),
]


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def run_logged(command: list[str], cwd: Path, log_path: Path, seconds: float) -> int:
    """Stream output without buffering whole datasets or printing credentials."""
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            ["timeout", "--signal=TERM", "--kill-after=15", str(max(1, int(seconds))), *command],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = re.sub(r"hf_[A-Za-z0-9]{20,}", "[REDACTED]", line)
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return proc.wait()


def main() -> int:
    started = time.monotonic()
    deadline = started + int(os.environ.get("OPENASR_SOFT_TIMEOUT", "6100"))
    output = Path("/results") / os.environ["OPENASR_RUN_ID"]
    output.mkdir(parents=True, exist_ok=False)
    work = Path(tempfile.mkdtemp(prefix="tiro-openasr-"))
    archive = work / "upstream.tar.gz"
    urllib.request.urlretrieve(
        f"https://codeload.github.com/huggingface/open_asr_leaderboard/tar.gz/{UPSTREAM_REVISION}",
        archive,
    )
    with tarfile.open(archive) as source:
        source.extractall(work, filter="data")
    upstream = work / f"open_asr_leaderboard-{UPSTREAM_REVISION}"
    shutil.copy2(archive, output / "upstream.tar.gz")
    runner = upstream / "transformers" / "run_eval.py"
    os.environ["PYTHONPATH"] = str(upstream)
    os.environ["PYTHONUNBUFFERED"] = "1"

    versions = {}
    for package in ("torch", "transformers", "datasets", "huggingface_hub", "kaldialign"):
        versions[package] = importlib.metadata.version(package)
    import torch
    from transformers import AutoConfig, AutoProcessor, MODEL_FOR_MULTIMODAL_LM_MAPPING

    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if type(config) not in MODEL_FOR_MULTIMODAL_LM_MAPPING:
        raise RuntimeError("Official image cannot load Tiro through native Transformers")
    if not hasattr(processor, "apply_transcription_request"):
        raise RuntimeError("Official image lacks the native Qwen transcription processor")

    provenance = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_runner": "transformers/run_eval.py (unmodified)",
        "image": "hf.co/spaces/hf-audio/open-asr-leaderboard-transformers",
        "versions": versions,
        "gpu": torch.cuda.get_device_name(0),
        "batch_size": 256,
        "warmup_steps": 1,
        "max_new_tokens": 512,
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "datasets": DATASETS,
        "hardware_timeout_seconds": int(os.environ.get("OPENASR_HARD_TIMEOUT", "6300")),
        "soft_timeout_seconds": int(os.environ.get("OPENASR_SOFT_TIMEOUT", "6100")),
        "scoring": "upstream normalizer.eval_utils.score_results (including chunk merge and compound handling)",
    }
    write_json(output / "provenance.json", provenance)
    print("PROVENANCE", json.dumps(provenance), flush=True)
    status: dict = {"stage": "running", "completed": [], "failed": [], "current": None}
    manifests = output / "manifests"
    manifests.mkdir()

    def save_status() -> None:
        status["elapsed_seconds"] = round(time.monotonic() - started, 1)
        write_json(output / "status.json", status)

    for label, dataset_path, dataset_config, split in DATASETS:
        if deadline - time.monotonic() < 120:
            status["stage"] = "budget_timeout"
            break
        status["current"] = label
        save_status()
        dataset_work = work / label
        dataset_work.mkdir()
        command = [
            sys.executable, "-u", str(runner),
            "--model_id", MODEL_ID, "--revision", MODEL_REVISION,
            "--dataset_path", dataset_path, "--dataset", dataset_config,
            "--split", split, "--device", "0", "--batch_size", "256",
            "--max_eval_samples", "-1", "--max_new_tokens", "512",
            "--warmup_steps", "1",
        ]
        print("START_DATASET", label, flush=True)
        rc = run_logged(command, dataset_work, output / f"{label}.log", deadline - time.monotonic())
        files = list((dataset_work / "results").glob("*.jsonl"))
        for result in files:
            shutil.copy2(result, manifests / result.name)
        if rc or not files:
            status["failed"].append({"dataset": label, "returncode": rc, "manifests": len(files)})
            save_status()
            if rc in (124, 137):
                status["stage"] = "budget_timeout"
                break
            # A loader/config failure is likely shared across datasets. Stop early.
            if not status["completed"]:
                status["stage"] = "failed"
                break
            continue
        status["completed"].append(label)
        save_status()
        print("COMPLETE_DATASET", label, flush=True)

    if list(manifests.glob("*.jsonl")) and deadline - time.monotonic() > 10:
        scoring = (
            "import json, sys; from normalizer.eval_utils import score_results; "
            "_, datasets = score_results(sys.argv[1], sys.argv[2], families=['public']); "
            "means = {sys.argv[2]: sum(v['wer'] for v in datasets.values()) / len(datasets)}; "
            "open(sys.argv[3], 'w').write(json.dumps({'mean_wer':dict(means), 'datasets':datasets}, indent=2))"
        )
        status["scoring_returncode"] = run_logged(
            [sys.executable, "-u", "-c", scoring, str(manifests), MODEL_ID, str(output / "scores.json")],
            upstream, output / "scores.log", max(10, deadline - time.monotonic()),
        )
    if status["stage"] == "running":
        status["stage"] = "completed" if len(status["completed"]) == len(DATASETS) else "partial"
    status["current"] = None
    save_status()
    print("FINAL_STATUS", json.dumps(status), flush=True)
    return 0 if status["stage"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
