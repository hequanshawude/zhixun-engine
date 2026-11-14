"""Explore C blends of checkpoints already created within one ten-round run."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "out/hybrid-27b-full-20261114/c"
PACKAGE = RUN / "_package"
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(ROOT / "scripts"))

from baselines.data import read_dataset
from baselines.model import Model
from scheme_v2 import fit_correction


train_path = ROOT / "data/raw/cpu/c/public/train.jsonl"
validation_path = ROOT / "data/raw/cpu/c/public/validation.jsonl"
result_dir = ROOT / "out/single-trajectory-blend-20261114"
result_dir.mkdir(exist_ok=True)

predictions = {}
for number in (1, 2, 6):
    checkpoint = RUN / f"rounds/r{number:02d}/checkpoint.npz"
    model = Model.load(checkpoint)
    train = read_dataset(train_path, model.meta["task"], model.meta["classes"], model.meta["paths"])
    val = read_dataset(validation_path, model.meta["task"], model.meta["classes"], model.meta["paths"])
    correction = result_dir / f"correction-r{number:02d}.npz"
    if correction.exists():
        correction.unlink()
    candidate = fit_correction("C", model, checkpoint, train, correction)
    predictions[number] = candidate.predict(val["x"])
    print("round", number, "sha256", hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
          "rmse", float(np.sqrt(np.mean((predictions[number] - val["y"])**2))), flush=True)

groups = np.array([sample.split("_")[0] for sample in val["ids"]])
rows = []
for earlier in (1, 2):
    for share in (0.2, 0.4, 0.6, 0.8):
        prediction = (1 - share) * predictions[earlier] + share * predictions[6]
        group_scores = {str(group): float(np.sqrt(np.mean((prediction[groups == group] - val["y"][groups == group])**2)))
                        for group in np.unique(groups)}
        row = {"earlier_round": earlier, "round6_share": share,
               "rmse": float(np.sqrt(np.mean((prediction - val["y"])**2))),
               "group_rmse": group_scores}
        rows.append(row)
        print(json.dumps(row), flush=True)
(result_dir / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
