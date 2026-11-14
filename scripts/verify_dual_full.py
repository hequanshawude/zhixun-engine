"""Independently reload and score the final A/B/C checkpoints."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "out/dual-full-27b-20261114"
summary = json.loads((RUN / "all-summary.json").read_text(encoding="utf-8"))
result = {}
for tag in ("a", "b", "c"):
    package = RUN / tag / "_package"
    checkpoint = RUN / tag / "final-checkpoint.npz"
    validation = ROOT / "data/raw/cpu" / tag / "public/validation.jsonl"
    code = r'''
import hashlib,json,sys
from pathlib import Path
import numpy as np
from baselines.data import read_dataset
from baselines.model import Model
package,checkpoint,validation=map(Path,sys.argv[1:])
model=Model.load(checkpoint)
data=read_dataset(validation,model.meta["task"],model.meta["classes"],model.meta["paths"])
prediction=model.predict(data["x"])
if model.meta["task"]=="classification":
    chosen=prediction.argmax(axis=1)
    labels=sorted(set(data["y"].tolist())|set(chosen.tolist()))
    score=float(np.mean([2*np.sum((chosen==k)&(data["y"]==k))/(np.sum(chosen==k)+np.sum(data["y"]==k)) for k in labels]))
else:
    score=float(np.sqrt(np.mean((prediction-data["y"])**2)))
copy=checkpoint.parent/"final-roundtrip-verify.npz"
model.save(copy)
other=Model.load(copy)
drift=float(np.max(np.abs(prediction-other.predict(data["x"]))))
if drift>1e-9: raise RuntimeError("roundtrip changed output")
print(json.dumps({"score":score,"rows":len(data["y"]),"max_roundtrip_drift":drift,
                  "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  "checkpoint_bytes":checkpoint.stat().st_size,
                  "hybrid":hasattr(model,"forest"),"dual":hasattr(model,"reference_model")}))
'''
    run = subprocess.run([sys.executable, "-c", code, str(package), str(checkpoint), str(validation)],
                         cwd=package, text=True, capture_output=True, timeout=120)
    if run.returncode:
        raise RuntimeError(f"{tag}: {run.stderr}")
    row = json.loads(run.stdout)
    if abs(row["score"] - summary["results"][tag]["final"]) > 1e-8:
        raise RuntimeError(f"{tag}: summary and independent score differ")
    sample = json.loads(validation.open(encoding="utf-8").readline())
    predict = subprocess.run([sys.executable, "-m", "baselines.predict", "--checkpoint", str(checkpoint)],
                             cwd=package, input=json.dumps({"inputs": sample["inputs"]}) + "\n",
                             text=True, capture_output=True, timeout=60)
    if predict.returncode or "outputs" not in json.loads(predict.stdout):
        raise RuntimeError(f"{tag}: official prediction adapter failed: {predict.stderr}")
    row["official_predict_adapter"] = True
    result[tag] = row
    print(tag, json.dumps(row), flush=True)
(RUN / "independent-verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
