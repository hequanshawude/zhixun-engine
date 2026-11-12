"""Prepare a group-disjoint FD001 RUL replay from the official C-MAPSS ZIP.

The official test split is deliberately never opened. Each replay batch contains
complete trajectories from distinct engines, so feedback from past batches
cannot contain observations from an engine in a future batch.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path("data/raw/cmapss/cmapss.zip")
TARGET = Path("data/processed/cmapss-fd001")
COLUMNS = ["id", "cycle", *(f"setting_{i}" for i in range(1, 4)),
           *(f"sensor_{i}" for i in range(1, 22))]
SEED = 42
TRAIN_ENGINES = 70
ENGINES_PER_BATCH = 3


def prepare(source: Path = SOURCE, target: Path = TARGET) -> dict:
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with zipfile.ZipFile(source) as archive:
        damaged = archive.testzip()
        if damaged is not None:
            raise ValueError(f"C-MAPSS ZIP CRC failed: {damaged}")
        raw = archive.read("train_FD001.txt")

    frame = pd.read_csv(io.BytesIO(raw), sep=r"\s+", header=None, names=COLUMNS)
    if frame.shape != (20631, 26) or frame.isna().any().any():
        raise ValueError(f"Unexpected FD001 training shape or missing values: {frame.shape}")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("FD001 training data contains non-finite values")
    if not np.all(frame[["id", "cycle"]].to_numpy() ==
                  frame[["id", "cycle"]].to_numpy(dtype=np.int64)):
        raise ValueError("Engine IDs and cycles must be integers")
    frame[["id", "cycle"]] = frame[["id", "cycle"]].astype(np.int64)
    engines = np.sort(frame.id.unique())
    if len(engines) != 100 or not np.array_equal(engines, np.arange(1, 101)):
        raise ValueError("Unexpected FD001 training engine IDs")
    for engine_id, group in frame.groupby("id", sort=True):
        if not np.array_equal(group.cycle.to_numpy(), np.arange(1, len(group) + 1)):
            raise ValueError(f"Engine {engine_id} has missing or unordered cycles")

    # Full-life length is used only to construct the retrospective target.
    # Neither engine ID nor the terminal cycle is included in model features.
    frame["y"] = frame.groupby("id").cycle.transform("max") - frame.cycle
    if (frame.y < 0).any() or frame.groupby("id").y.min().ne(0).any():
        raise ValueError("Invalid RUL target")
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(engines)
    train_ids = np.sort(shuffled[:TRAIN_ENGINES])
    stream_ids = shuffled[TRAIN_ENGINES:]
    if set(train_ids) & set(stream_ids):
        raise ValueError("Bootstrap and replay engines overlap")

    target.mkdir(parents=True, exist_ok=True)
    stream_dir = target / "stream"
    stream_dir.mkdir(exist_ok=True)
    bootstrap = frame[frame.id.isin(train_ids)].sort_values(["id", "cycle"])
    bootstrap.to_csv(target / "train.csv", index=False, float_format="%.9g")
    batches = []
    for k, offset in enumerate(range(0, len(stream_ids), ENGINES_PER_BATCH), 1):
        batch_ids = stream_ids[offset:offset + ENGINES_PER_BATCH]
        batch = frame[frame.id.isin(batch_ids)].sort_values(["id", "cycle"])
        filename = f"batch_{k:03d}.csv"
        batch.to_csv(stream_dir / filename, index=False, float_format="%.9g")
        batches.append({"file": filename, "engine_ids": [int(x) for x in batch_ids],
                        "rows": int(len(batch))})
    if len(batches) != 10 or len(bootstrap) + sum(b["rows"] for b in batches) != len(frame):
        raise ValueError("FD001 split does not cover every training row exactly once")
    if {engine for batch in batches for engine in batch["engine_ids"]} != set(stream_ids):
        raise ValueError("Replay engine coverage is invalid")

    manifest = {
        "source": "https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data",
        "source_file": str(source), "source_sha256": source_hash,
        "official_subset": "train_FD001.txt only; official test_FD001.txt and RUL_FD001.txt reserved",
        "split_seed": SEED, "split_unit": "engine id", "target": "uncapped RUL = terminal cycle - current cycle",
        "feedback": "retrospective; labels are revealed to the simulator after each complete-engine batch",
        "feature_columns": [col for col in COLUMNS if col != "id"],
        "excluded_from_features": ["id", "y", "terminal cycle"],
        "bootstrap": {"file": "train.csv", "engine_ids": [int(x) for x in train_ids],
                      "rows": int(len(bootstrap))},
        "batches": batches,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = prepare()
    print(f"FD001 ready: {result['bootstrap']['rows']} bootstrap rows, "
          f"{sum(batch['rows'] for batch in result['batches'])} replay rows, "
          f"{len(result['batches'])} engine-disjoint batches")
