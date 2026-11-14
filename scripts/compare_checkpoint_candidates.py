"""Compare CPU checkpoint candidates using only official public validation."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/raw/cpu/a"))
from baselines.data import read_dataset  # noqa: E402
from baselines.model import Model  # noqa: E402


CANDIDATES = {
    "A": {
        "direct_27b": ROOT / "out/candidate-27b-direct-20261114/a/final-checkpoint.npz",
    },
    "B": {
        "direct_27b": ROOT / "out/candidate-27b-direct-20261114/b/final-checkpoint.npz",
        "balanced_27b": ROOT / "out/b-balanced-qwen27b-20261114/final-checkpoint.npz",
    },
    "C": {
        "direct_27b": ROOT / "out/candidate-27b-direct-20261114/c/final-checkpoint.npz",
        "readout_27b": ROOT / "out/c-qwen27b-distilled-20261114/checkpoint.npz",
    },
}


result = {}
for profile, candidates in CANDIDATES.items():
    package = ROOT / "data/raw/cpu" / profile.lower()
    first = Model.load(next(iter(candidates.values())))
    data_path = package / "public/validation.jsonl"
    raw = data_path.read_bytes()
    data = read_dataset(data_path, first.meta["task"], first.meta["classes"], first.meta["paths"])
    ids = np.asarray(data["ids"])
    if profile == "B":
        groups = np.asarray([item.rsplit("_s", 1)[0] for item in ids])
    elif profile == "C":
        groups = np.asarray([item.split("_")[0] for item in ids])
    else:
        groups = None
    entries = {}
    for name, checkpoint in candidates.items():
        model = Model.load(checkpoint)
        if model.meta["profile"] != profile or model.meta["paths"] != data["paths"]:
            raise ValueError(f"{name}: checkpoint schema mismatch")
        predictions = model.predict(data["x"])
        if profile == "A":
            chosen = predictions.argmax(axis=1)
            labels = sorted(set(data["y"].tolist()) | set(chosen.tolist()))
            score = float(np.mean([
                2 * np.sum((chosen == label) & (data["y"] == label)) /
                (np.sum(chosen == label) + np.sum(data["y"] == label))
                for label in labels
            ]))
            metric = {"macro_f1": score}
        else:
            error = predictions - data["y"]
            metric = {
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "group_rmse": {str(group): float(np.sqrt(np.mean(error[groups == group] ** 2)))
                               for group in np.unique(groups)},
            }
            metric["group_mean_rmse"] = float(np.mean(list(metric["group_rmse"].values())))
        entries[name] = {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "epochs": model.meta["epochs"], "steps": model.meta["steps"],
            **metric,
        }
    result[profile] = {
        "validation_rows": len(ids),
        "validation_sha256": hashlib.sha256(raw).hexdigest(),
        "candidates": entries,
    }
    print(profile, json.dumps(entries, ensure_ascii=False), flush=True)

destination = ROOT / "out/checkpoint-candidate-compare-20261114.json"
destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
