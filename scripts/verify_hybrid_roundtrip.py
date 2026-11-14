"""Verify a generated hybrid checkpoint through the official prediction adapter."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("profile", choices=("B", "C"))
args = parser.parse_args()
tag = args.profile.lower()
run = ROOT / "out/hybrid-27b-full-20261114" / tag
package = run / "_package"
sys.path.insert(0, str(package))
from baselines.data import read_dataset  # noqa: E402
from baselines.model import Model  # noqa: E402

checkpoint = run / "final-checkpoint.npz"
copy = run / "roundtrip-checkpoint.npz"
if copy.exists():
    raise FileExistsError(copy)
model = Model.load(checkpoint)
if not hasattr(model, "forest"):
    raise RuntimeError("Forest was not embedded")
model.save(copy)
again = Model.load(copy)
data = read_dataset(ROOT / "data/raw/cpu" / tag / "public/validation.jsonl",
                    "regression", [], model.meta["paths"])
original = model.predict(data["x"])
reloaded = again.predict(data["x"])
drift = float(np.max(np.abs(original - reloaded)))
if drift != 0:
    raise RuntimeError(f"Roundtrip changed predictions: {drift}")
rmse = float(np.sqrt(np.mean((reloaded - data["y"]) ** 2)))
line = (ROOT / "data/raw/cpu" / tag / "public/validation.jsonl").open(encoding="utf-8").readline()
request = json.dumps({"inputs": json.loads(line)["inputs"]}) + "\n"
proc = subprocess.run([sys.executable, "-m", "baselines.predict", "--checkpoint", str(copy)],
                      cwd=package, input=request, capture_output=True, text=True, timeout=60)
if proc.returncode:
    raise RuntimeError(proc.stderr)
if abs(json.loads(proc.stdout)["outputs"]["value"] - reloaded[0]) > 1e-9:
    raise RuntimeError("Official prediction adapter differs")
print(json.dumps({"profile": args.profile, "rmse": rmse, "roundtrip_max_drift": drift,
                  "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  "roundtrip_sha256": hashlib.sha256(copy.read_bytes()).hexdigest(),
                  "bytes": checkpoint.stat().st_size}), flush=True)
