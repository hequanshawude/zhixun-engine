"""Recheck the two 2026-11-13 local predictor images on public validation."""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RECORD = json.loads((ROOT / "out/27b-docker-replay-20261113.json").read_text(encoding="utf-8"))
RESULT = {}

for group in ("B", "C"):
    spec = RECORD[group]
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "-f", "{{.Id}}", spec["image"]], text=True
    ).strip()
    if image_id != spec["image_id"]:
        raise RuntimeError(f"{group}: image ID has changed")

    path = ROOT / "data/raw/cpu" / group.lower() / "public/validation.jsonl"
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    requests = "".join(json.dumps({"inputs": row["inputs"]}) + "\n" for row in rows)
    command = ["docker", "run", "--rm", "--network", "none", "--cpus", "4", "--memory", "2g", "-i", image_id]
    run = subprocess.run(command, input=requests, text=True, capture_output=True, timeout=300)
    if run.returncode:
        raise RuntimeError(f"{group}: {run.stderr[-1000:]}")
    outputs = [json.loads(line)["outputs"]["value"] for line in run.stdout.splitlines() if line.strip()]
    if len(outputs) != len(rows):
        raise RuntimeError(f"{group}: prediction count mismatch")
    predicted = np.asarray(outputs, dtype=np.float64)
    actual = np.asarray([row["answer"] for row in rows], dtype=np.float64)
    if not np.isfinite(predicted).all() or (predicted < 0).any():
        raise RuntimeError(f"{group}: invalid predictions")
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    if abs(rmse - spec["rmse"]) > 1e-6:
        raise RuntimeError(f"{group}: RMSE changed: {rmse} != {spec['rmse']}")
    RESULT[group] = {
        "image": spec["image"], "image_id": image_id,
        "rows": len(rows), "validation_sha256": hashlib.sha256(raw).hexdigest(),
        "rmse": rmse, "matches_20261113": True,
    }
    print(group, json.dumps(RESULT[group]), flush=True)

destination = ROOT / "out/27b-local-images-recheck-20261114.json"
destination.write_text(json.dumps(RESULT, indent=2) + "\n", encoding="utf-8")
