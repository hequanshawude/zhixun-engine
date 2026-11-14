"""Check that public-train forest corrections can be loaded as one CPU model."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selftrain.hybrid_checkpoint import embed_correction, patch_model_source  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=("B", "C"))
    args = parser.parse_args()
    profile = args.profile
    source = ROOT / "data/raw/cpu" / profile.lower()
    predictor = ROOT / "out/candidate-27b-direct-20261114" / profile.lower() / "predictor"
    output = ROOT / "out" / f"hybrid-probe-{profile.lower()}-20261114"
    if output.exists():
        raise FileExistsError(output)
    package = output / "baselines"
    package.mkdir(parents=True)
    for name in ("__init__.py", "model.py", "data.py", "predict.py"):
        shutil.copyfile(source / "baselines" / name, package / name)
    patch_model_source(package / "model.py")
    base = predictor / "checkpoint.npz"
    correction = predictor / "correction.npz"
    artifact = output / "checkpoint.npz"
    lineage = embed_correction(base, correction, artifact)

    sys.path.insert(0, str(output))
    sys.path.insert(0, str(ROOT / "scripts"))
    from baselines.data import read_dataset
    from baselines.model import Model
    from rul_tree_correction import CorrectedModel

    original = Model.load(base)
    hybrid = Model.load(artifact)
    data = read_dataset(source / "public/validation.jsonl", "regression", [], hybrid.meta["paths"])
    expected = CorrectedModel(original, base, correction).predict(data["x"])
    actual = hybrid.predict(data["x"])
    largest = float(np.max(np.abs(actual - expected)))
    if largest > 1e-9:
        raise ValueError(f"Hybrid prediction drift: {largest}")
    rmse = float(np.sqrt(np.mean((actual - data["y"]) ** 2)))
    request = json.dumps({"inputs": json.loads((source / "public/validation.jsonl").open(encoding="utf-8").readline())["inputs"]}) + "\n"
    run = subprocess.run([sys.executable, "-m", "baselines.predict", "--checkpoint", str(artifact)],
                         cwd=output, input=request, capture_output=True, text=True, timeout=60)
    if run.returncode:
        raise RuntimeError(run.stderr)
    cli_value = json.loads(run.stdout)["outputs"]["value"]
    if abs(cli_value - actual[0]) > 1e-9:
        raise ValueError("Official prediction adapter disagrees")
    report = {"profile": profile, "validation_rows": len(actual), "rmse": rmse,
              "max_prediction_drift": largest, "official_predict_adapter": True,
              "lineage": lineage, "checkpoint": str(artifact)}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
