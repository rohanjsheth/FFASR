"""Inspect a submitted OpenASR job and optionally download its result artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--tail", type=int, default=12)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    api = HfApi()
    job = api.inspect_job(job_id=receipt["job_id"], namespace="rohansheth")
    print(json.dumps({"job_url": job.url, "stage": job.status.stage, "message": job.status.message}), flush=True)
    if args.tail:
        for line in api.fetch_job_logs(job_id=job.id, namespace="rohansheth", follow=False, tail=args.tail):
            line = re.sub(r"hf_[A-Za-z0-9]{20,}", "[REDACTED]", line)
            print(line, flush=True)
    if args.download:
        files = []
        prefix = receipt["run_id"] + "/"
        for entry in api.list_bucket_tree(receipt["bucket"], prefix=prefix, recursive=True):
            if entry.type != "file":
                continue
            relative = Path(entry.path.removeprefix(prefix))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unexpected artifact path: {entry.path}")
            destination = args.receipt.parent / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            files.append((entry.path, destination))
        if files:
            api.download_bucket_files(receipt["bucket"], files=files, raise_on_missing_files=True)
        print("Downloaded artifacts:", len(files), flush=True)
        for name in ("status.json", "scores.json"):
            path = args.receipt.parent / name
            if path.exists():
                print(name, path.read_text(), flush=True)


if __name__ == "__main__":
    main()
