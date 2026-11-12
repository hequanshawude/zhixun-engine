"""扫描 L2 权重衰减的合理量级：每个值跑一轮，比较验证集指标。

只用于选超参，不产生正式检查点。
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:\智训引擎")
PKG = ROOT / "data" / "raw" / "cpu"
TMP = ROOT / "runs" / "_wd_sweep"

PROFILE = sys.argv[1] if len(sys.argv) > 1 else "A"
CKPT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "out/cpu-models/best/a.npz"
PROFILE_JSON = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "out/cpu-models/profiles/a-wide.json"
PROFILE_JSON = PROFILE_JSON.resolve()
VALUES = [0.0, 1e-4, 5e-4, 1e-3, 5e-3]

tag = PROFILE.lower()
sys.path.insert(0, str(PKG / tag))
from baselines.model import Model  # noqa: E402

metric = "macro_f1" if PROFILE == "A" else "rmse"
shutil.rmtree(TMP, ignore_errors=True)
TMP.mkdir(parents=True, exist_ok=True)

base = Model.load(CKPT)
print(f"{PROFILE} 起点: {metric}={json.loads((PKG/tag/'training.json').read_text(encoding='utf-8'))['validation_after'][metric]:.4f}")

for wd in VALUES:
    m = Model.load(CKPT)
    m.meta["weight_decay"] = wd
    ck = TMP / f"wd{wd:g}.npz"
    if ck.exists():
        ck.unlink()
    m.save(ck)
    out = TMP / f"out{wd:g}"
    cmd = [sys.executable, "-m", "baselines.train", "--profile", str(PROFILE_JSON),
           "--train", "public/train.jsonl", "--validation", "public/validation.jsonl",
           "--resume", str(ck), "--epochs", "5", "--learning-rate", "0.001",
           "--output", str(out), "--stage-id", "sweep"]
    proc = subprocess.run(cmd, cwd=PKG / tag, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{")), None)
    if not line:
        print(f"  wd={wd:<8g} 失败: {(proc.stderr or '')[-120:]}")
        continue
    rep = json.loads(line)
    v = rep["validation_after"][metric]
    print(f"  wd={wd:<8g} -> {metric}={v:.4f}")
