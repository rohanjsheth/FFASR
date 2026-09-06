"""Small, CPU-only file helpers for materialized noise experiments."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.io import wavfile


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("Audio must be finite, nonempty and mono")
    buffer = io.BytesIO()
    # libsndfile's FLOAT WAV writer adds a wall-clock timestamp in a PEAK chunk.
    # SciPy writes the same float32 samples without time-dependent metadata.
    samples = audio.astype(np.float32)
    if not np.isfinite(samples).all():
        raise ValueError("Audio cannot be represented as finite float32 samples")
    wavfile.write(buffer, sample_rate, samples)
    return buffer.getvalue()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_parquet(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Stream small row groups; publish only a complete file, never overwrite."""
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with TemporaryDirectory(prefix="noise-prep-", dir=path.parent) as staging:
        temporary = Path(staging) / "audio.parquet"
        writer = None
        pending = []
        try:
            for row in rows:
                pending.append(row)
                count += 1
                if len(pending) == 16:
                    table = pa.Table.from_pylist(pending)
                    if writer is None:
                        writer = pq.ParquetWriter(temporary, table.schema)
                    writer.write_table(table)
                    pending.clear()
            if pending:
                table = pa.Table.from_pylist(pending)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        if not count:
            raise ValueError("No audio records were generated")
        # A same-filesystem hard link publishes atomically and fails if another
        # process created the destination while we were generating the file.
        path.hardlink_to(temporary)
    return count
