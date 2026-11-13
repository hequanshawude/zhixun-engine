"""Batch JSONL inference for a versioned B/C blend bundle."""
import hashlib
import json
import sys
from itertools import islice
from pathlib import Path

import numpy as np

from baselines.data import flatten, json_value
from baselines.model import Model
from rul_tree_correction import CorrectedModel


def load_bundle(folder: Path):
    spec = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in spec["sha256"].items():
        actual = hashlib.sha256((folder / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Weight digest mismatch: {relative}")
    models = []
    for name in ("reference", "guided"):
        checkpoint = folder / name / "checkpoint.npz"
        base = Model.load(checkpoint)
        models.append(CorrectedModel(base, checkpoint, folder / name / "correction.npz"))
    reference, guided = models
    if reference.meta["paths"] != guided.meta["paths"] or reference.meta["task"] != guided.meta["task"]:
        raise ValueError("The two predictors use different input schemas")
    share = float(spec["guided_share"])
    strength = float(spec["guided_correction_strength"])
    if not 0 <= share <= 1 or not 0 <= strength <= 1.5:
        raise ValueError("Invalid blend weights")
    guided.tree["blend"] = np.asarray(strength)
    return reference, guided, share


def main():
    folder = Path(__file__).resolve().parent
    reference, guided, share = load_bundle(folder)
    meta = reference.meta
    lines = (line for line in sys.stdin if line.strip())
    while batch := list(islice(lines, 256)):
        vectors = []
        for line in batch:
            request = json_value(line)
            if (not isinstance(request, dict) or "inputs" not in request or
                    set(request) - {"inputs", "task"} or request.get("task", meta["task"]) != meta["task"]):
                raise ValueError("Expected an input-only regression record")
            paths, vector = flatten(request["inputs"])
            if paths != meta["paths"]:
                raise ValueError("Unexpected input feature schema")
            vectors.append(vector)
        x = np.asarray(vectors, dtype=np.float64)
        predictions = (1 - share) * reference.predict(x) + share * guided.predict(x)
        for value in predictions:
            print(json.dumps({"outputs": {"value": float(value)}}, allow_nan=False))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
