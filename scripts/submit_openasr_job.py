"""Submit one budget-limited Tiro evaluation; defaults to a read-only preflight."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from huggingface_hub import HfApi, Volume, get_token


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--bucket", default="rohansheth/tiro-open-asr")
    parser.add_argument("--timeout-minutes", type=int, default=105)
    args = parser.parse_args()
    api = HfApi()
    identity = api.whoami()
    if identity["name"] != "rohansheth":
        raise RuntimeError("Expected rohansheth's account")
    api.bucket_info(args.bucket)
    hardware = next(h for h in api.list_jobs_hardware() if h.name == "h200")
    if hardware.unit_label != "minute":
        raise RuntimeError("Unexpected hardware billing unit")
    if not 5 <= args.timeout_minutes <= 105:
        raise ValueError("Timeout must be between 5 and 105 minutes")
    max_compute_usd = hardware.unit_cost_usd * args.timeout_minutes
    if max_compute_usd > 8.76:
        raise RuntimeError(f"H200 price exceeds planned compute limit: ${max_compute_usd:.2f}")
    prior_jobs = list(api.list_jobs(labels={"task": "tiro-openasr"}))
    active = [
        j for j in prior_jobs
        if j.status.stage not in {"COMPLETED", "ERROR", "CANCELED", "DELETED"}
    ]
    if active:
        raise RuntimeError(f"An evaluation already exists: {[j.id for j in active]}")
    # Include scheduling time and round up to conservatively reserve prior costs.
    prior_seconds = sum(
        max(0, ((j.finished_at or datetime.now(timezone.utc)) - j.created_at).total_seconds())
        for j in prior_jobs
    )
    prior_cost_reserve = math.ceil(prior_seconds / 60) * hardware.unit_cost_usd
    if prior_cost_reserve + max_compute_usd > 9.0:
        raise RuntimeError("Combined prior and planned compute exceeds the $9 reserve limit")
    script = Path(__file__).with_name("run_openasr_job.py")
    payload = base64.b64encode(script.read_bytes()).decode("ascii")
    run_id = datetime.now(timezone.utc).strftime("tiro-%Y%m%dT%H%M%SZ")
    plan = {
        "run_id": run_id,
        "bucket": args.bucket,
        "image": "hf.co/spaces/hf-audio/open-asr-leaderboard-transformers",
        "flavor": "h200",
        "timeout_seconds": args.timeout_minutes * 60,
        "max_compute_usd": round(max_compute_usd, 4),
        "prior_compute_reserve_usd": round(prior_cost_reserve, 4),
        "total_compute_reserve_usd": round(prior_cost_reserve + max_compute_usd, 4),
        "model_id": "rohansheth/tiro-qwen3-asr-1.7b-ffasr-v1",
        "model_revision": "92a967532f9b10abdb898a701dab175583a8cfdd",
        "upstream_revision": "48219c6028db0517d704600d92f31edfc96e8c23",
        "official_image_space_revision_at_submission": api.space_info(
            "hf-audio/open-asr-leaderboard-transformers"
        ).sha,
    }
    print(json.dumps(plan, indent=2))
    if not args.launch:
        return
    receipt_dir = Path(__file__).resolve().parents[1] / "results" / "openasr" / run_id
    receipt_dir.mkdir(parents=True, exist_ok=False)
    (receipt_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    job = api.run_job(
        image=plan["image"],
        command=["python", "-u", "-c", f"import base64; exec(compile(base64.b64decode('{payload}'), 'run_openasr_job.py', 'exec'))"],
        env={
            "OPENASR_RUN_ID": run_id,
            "OPENASR_SOFT_TIMEOUT": str(plan["timeout_seconds"] - 200),
            "OPENASR_HARD_TIMEOUT": str(plan["timeout_seconds"]),
            "PYTHONUNBUFFERED": "1",
        },
        secrets={"HF_TOKEN": get_token()},
        flavor="h200",
        timeout=plan["timeout_seconds"],
        name="tiro-openasr-english",
        labels={"task": "tiro-openasr", "run": run_id},
        volumes=[Volume(type="bucket", source=args.bucket, mount_path="/results", read_only=False)],
        namespace="rohansheth",
    )
    receipt = {**plan, "job_id": job.id, "job_url": job.url, "status": job.status.stage}
    (receipt_dir / "job.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    print("Receipt:", receipt_dir / "job.json")


if __name__ == "__main__":
    main()
