"""Optional checkpoint-focused B/C training helpers for a local candidate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def balanced_b_train(source: Path, destination: Path) -> dict:
    """Sample public B train rows equally by bearing, keeping unique row IDs."""
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = np.asarray([row["id"].rsplit("_s", 1)[0] for row in rows])
    rng = np.random.default_rng(20261114)
    selected = np.concatenate([
        rng.choice(np.flatnonzero(groups == group), size=800, replace=True)
        for group in np.unique(groups)
    ])
    with destination.open("w", encoding="utf-8") as stream:
        for ordinal, index in enumerate(selected):
            row = dict(rows[int(index)])
            row["id"] = f"{row['id']}__balanced_{ordinal}"
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return {"method": "public_train_device_balancing", "seed": 20261114,
            "devices": len(np.unique(groups)), "rows_per_device": 800,
            "sampled_rows": len(selected)}


def b_device_metrics(package: Path, checkpoint: Path) -> dict:
    """Measure pooled and per-device validation RMSE for a B checkpoint."""
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    from baselines.data import read_dataset
    from baselines.model import Model

    model = Model.load(checkpoint)
    data = read_dataset(package / "public/validation.jsonl", "regression", [], model.meta["paths"])
    groups = np.asarray([item.rsplit("_s", 1)[0] for item in data["ids"]])
    error = model.predict(data["x"]) - data["y"]
    per_device = {str(group): float(np.sqrt(np.mean(error[groups == group] ** 2)))
                  for group in np.unique(groups)}
    return {"pooled_rmse": float(np.sqrt(np.mean(error ** 2))),
            "device_mean_rmse": float(np.mean(list(per_device.values()))),
            "per_device": per_device}
