"""Test last night's C share/strength as one industrial checkpoint."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selftrain.hybrid_checkpoint import embed_dual_c, patch_dual_model_source  # noqa: E402

source = ROOT / "data/raw/cpu/c"
bundle = ROOT / "out/27b-blend-c-20261113-share60-strength90"
output = ROOT / "out/dual-c-probe-20261114"
if output.exists():
    raise FileExistsError(output)
package = output / "baselines"
package.mkdir(parents=True)
for name in ("__init__.py", "model.py", "data.py", "predict.py"):
    shutil.copyfile(source / "baselines" / name, package / name)
patch_dual_model_source(package / "model.py")
checkpoint = output / "checkpoint.npz"
lineage = embed_dual_c(
    bundle / "reference/checkpoint.npz", bundle / "reference/correction.npz",
    bundle / "guided/checkpoint.npz", bundle / "guided/correction.npz",
    checkpoint)
sys.path.insert(0, str(output))
sys.path.insert(0, str(ROOT / "scripts"))
from baselines.data import read_dataset  # noqa: E402
from baselines.model import Model  # noqa: E402
from predict_blend import load_bundle  # noqa: E402

model = Model.load(checkpoint)
data = read_dataset(source / "public/validation.jsonl", "regression", [], model.meta["paths"])
actual = model.predict(data["x"])
reference, guided, share = load_bundle(bundle)
expected = (1 - share) * reference.predict(data["x"]) + share * guided.predict(data["x"])
drift = float(np.max(np.abs(actual - expected)))
rmse = float(np.sqrt(np.mean((actual - data["y"]) ** 2)))
if drift > 1e-9 or abs(rmse - 49.953207474026016) > 1e-6:
    raise RuntimeError(f"Dual checkpoint mismatch: drift={drift}, rmse={rmse}")
first = json.loads((source / "public/validation.jsonl").open(encoding="utf-8").readline())
run = subprocess.run([sys.executable, "-m", "baselines.predict", "--checkpoint", str(checkpoint)],
                     cwd=output, input=json.dumps({"inputs": first["inputs"]}) + "\n",
                     capture_output=True, text=True, timeout=60)
if run.returncode:
    raise RuntimeError(run.stderr)
if abs(json.loads(run.stdout)["outputs"]["value"] - actual[0]) > 1e-9:
    raise RuntimeError("Official prediction adapter differs")
result = {"rmse": rmse, "rows": len(actual), "max_prediction_drift": drift,
          "checkpoint_bytes": checkpoint.stat().st_size,
          "official_predict_adapter": True, "lineage": lineage}
(output / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result), flush=True)
