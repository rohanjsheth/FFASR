"""Compare two VOiCES prediction files, split by distractor, room and mic."""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics as st
from collections.abc import Sequence
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    return {r["query_name"]: r for r in (json.loads(l) for l in path.open())}


def wer(rows: Sequence[dict]) -> float:
    words = sum(r["reference_words"] for r in rows)
    return 100 * sum(r["word_errors"] for r in rows) / max(1, words)


def paired_ci(base: dict, tuned: dict, keys: Sequence[str], draws: int = 2000) -> tuple[float, float]:
    """Bootstrap the delta, clustering on source clip so rooms/mics move together."""
    by_source = collections.defaultdict(list)
    for k in keys:
        by_source[base[k]["source"]].append(k)
    sources = list(by_source)
    rng = random.Random(0)
    deltas = []
    for _ in range(draws):
        sample = [rng.choice(sources) for _ in sources]
        ks = [k for s in sample for k in by_source[s]]
        w = sum(base[k]["reference_words"] for k in ks)
        be = sum(base[k]["word_errors"] for k in ks)
        te = sum(tuned[k]["word_errors"] for k in ks)
        deltas.append(100 * (te - be) / max(1, w))
    deltas.sort()
    return deltas[int(0.025 * draws)], deltas[int(0.975 * draws)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, required=True)
    args = parser.parse_args(argv)

    base, tuned = load(args.base), load(args.tuned)
    keys = sorted(set(base) & set(tuned))
    print(f"paired clips: {len(keys)}\n")

    for field in ("distractor", "room", "mic"):
        groups = collections.defaultdict(list)
        for k in keys:
            groups[base[k][field]].append(k)
        print(f"{field:<12}{'n':>6}{'base':>9}{'tuned':>9}{'delta':>9}   95% CI")
        for name, ks in sorted(groups.items(), key=lambda kv: str(kv[0])):
            b = wer([base[k] for k in ks])
            t = wer([tuned[k] for k in ks])
            lo, hi = paired_ci(base, tuned, ks)
            flag = "  *" if (hi < 0 or lo > 0) else ""
            print(f"  {str(name):<10}{len(ks):>6}{b:>8.2f}%{t:>8.2f}%{t - b:>+9.2f}   [{lo:+.2f}, {hi:+.2f}]{flag}")
        print()

    b, t = wer([base[k] for k in keys]), wer([tuned[k] for k in keys])
    lo, hi = paired_ci(base, tuned, keys)
    print(f"{'overall':<12}{len(keys):>6}{b:>8.2f}%{t:>8.2f}%{t - b:>+9.2f}   [{lo:+.2f}, {hi:+.2f}]")

    be = st.mean(base[k]["mean_entropy"] for k in keys)
    te = st.mean(tuned[k]["mean_entropy"] for k in keys)
    bl = sum(base[k]["hypothesis_words"] for k in keys) / sum(base[k]["reference_words"] for k in keys)
    tl = sum(tuned[k]["hypothesis_words"] for k in keys) / sum(tuned[k]["reference_words"] for k in keys)
    print(f"\nentropy   base {be:.3f} -> tuned {te:.3f}")
    print(f"len/ref   base {bl:.3f} -> tuned {tl:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
