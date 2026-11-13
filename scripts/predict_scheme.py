"""Predict from an exported scheme-v2 bundle using the official JSONL schema."""
import json
import sys
from pathlib import Path

from baselines.data import json_value
from baselines.model import Model
from baselines.predict import predict


def main():
    folder = Path(__file__).resolve().parent
    spec = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
    checkpoint = folder / "checkpoint.npz"
    model = Model.load(checkpoint)
    if spec["accepted"]:
        correction = folder / "correction.npz"
        if spec["profile"] == "A":
            from gas_pair_gate import GasPairGate
            model = GasPairGate(checkpoint, correction)
        else:
            from rul_tree_correction import CorrectedModel
            model = CorrectedModel(model, checkpoint, correction)
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json_value(line)
        if not isinstance(request, dict) or set(request) - {"inputs", "task"} or "inputs" not in request:
            raise ValueError("Expected an input-only record")
        if request.get("task", model.meta["task"]) != model.meta["task"]:
            raise ValueError("Task mismatch")
        print(json.dumps(predict(model, request["inputs"]), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
