"""Check that a runtime-fitted A gate survives checkpoint load and save."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from selftrain.hybrid_checkpoint import embed_gas_gate, patch_gas_model_source

RUN = ROOT / "out/hybrid-27b-full-20261114/a"
PACKAGE = RUN / "_package"
DEST = ROOT / "out/gas-checkpoint-probe-20261114"
DEST.mkdir(exist_ok=True)
source = PACKAGE / "baselines/model.py"
patch_gas_model_source(source)
checkpoint = DEST / "final-checkpoint.npz"
if checkpoint.exists():
    checkpoint.unlink()
embed_gas_gate(RUN / "predictor/checkpoint.npz", RUN / "predictor/correction.npz", checkpoint)
spec = importlib.util.spec_from_file_location("gas_final_model", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.path.insert(0, str(PACKAGE))
sys.path.insert(1, str(ROOT / "scripts"))
from baselines.data import read_dataset
from baselines.train import metrics
from gas_pair_gate import GasPairGate

model = module.Model.load(checkpoint)
validation = ROOT / "data/raw/cpu/a/public/validation.jsonl"
data = read_dataset(validation, "classification", model.meta["classes"], model.meta["paths"])
expected = GasPairGate(RUN / "predictor/checkpoint.npz", RUN / "predictor/correction.npz").predict(data["x"])
actual = model.predict(data["x"])
drift = float(np.max(np.abs(actual - expected)))
if drift > 1e-12:
    raise RuntimeError(f"Gas checkpoint changed prediction: {drift}")
score = metrics(model, data)["macro_f1"]
if abs(score - 0.9116961994431878) > 1e-12:
    raise RuntimeError(f"Unexpected gas score: {score}")
roundtrip = DEST / "roundtrip.npz"
if roundtrip.exists():
    roundtrip.unlink()
model.save(roundtrip)
reloaded = module.Model.load(roundtrip)
if not np.array_equal(actual, reloaded.predict(data["x"])):
    raise RuntimeError("Gas save/load changed prediction")
sample = json.loads(validation.open(encoding="utf-8").readline())
proc = subprocess.run([sys.executable, "-m", "baselines.predict", "--checkpoint", str(checkpoint)],
                      cwd=PACKAGE, input=json.dumps({"inputs": sample["inputs"]}) + "\n",
                      text=True, capture_output=True, timeout=60)
if proc.returncode or "outputs" not in json.loads(proc.stdout):
    raise RuntimeError(f"Official prediction adapter failed: {proc.stderr}")
print(json.dumps({"macro_f1": score, "drift": drift, "roundtrip": True,
                  "official_predict_adapter": True, "checkpoint_bytes": checkpoint.stat().st_size}))
