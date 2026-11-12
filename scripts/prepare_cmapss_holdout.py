"""Reserve ten FD001 training engines for a fixed candidate-promotion gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path("data/processed/cmapss-fd001")
TARGET = Path("data/processed/cmapss-fd001-holdout")
SEED = 2026


def prepare(source: Path = SOURCE, target: Path = TARGET) -> dict:
    parent = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(source / "train.csv")
    engines = np.sort(frame.id.unique())
    if len(engines) != 70 or len(frame) != parent["bootstrap"]["rows"]:
        raise ValueError("FD001 parent bootstrap does not match its manifest")
    gate_ids = np.sort(np.random.default_rng(SEED).choice(engines, size=10, replace=False))
    train_ids = np.setdiff1d(engines, gate_ids)
    stream_ids = {x for batch in parent["batches"] for x in batch["engine_ids"]}
    if set(engines) & stream_ids or len(train_ids) != 60:
        raise ValueError("Engine groups overlap or have incorrect size")
    train = frame[frame.id.isin(train_ids)].sort_values(["id", "cycle"])
    gate = frame[frame.id.isin(gate_ids)].sort_values(["id", "cycle"])
    if len(train) + len(gate) != len(frame):
        raise ValueError("Holdout split lost rows")
    target.mkdir(parents=True, exist_ok=True)
    train.to_csv(target / "train.csv", index=False, float_format="%.9g")
    gate.to_csv(target / "gate.csv", index=False, float_format="%.9g")
    manifest = {
        "parent_manifest": str(source / "manifest.json"),
        "source_sha256": parent["source_sha256"],
        "split_seed": SEED,
        "train": {"engine_ids": train_ids.tolist(), "rows": int(len(train))},
        "gate": {"engine_ids": gate_ids.tolist(), "rows": int(len(gate)),
                 "usage": "candidate acceptance only; never added to training history"},
        "replay": {"directory": str(source / "stream"), "engine_ids": sorted(stream_ids),
                   "usage": "external engine-disjoint evaluation; unchanged from first FD001 run"},
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = prepare()
    print(f"FD001 holdout ready: {result['train']['rows']} train rows, "
          f"{result['gate']['rows']} gate rows, 30 unchanged replay engines")
