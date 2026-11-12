"""用最终选定的检查点跑一遍预测协议，确认输出格式与数值合理。"""

import json
import subprocess
from pathlib import Path

ROOT = Path(r"D:\智训引擎")
PY = ROOT / ".venv" / "Scripts" / "python.exe"

FINAL = {
    "a": ("A", ROOT / "runs/cpu/a/round-2/checkpoint.npz"),
    "b": ("B", ROOT / "runs/cpu/b/round-1/checkpoint.npz"),
    "c": ("C", ROOT / "runs/cpu/c/round-2/checkpoint.npz"),
}

for tag, (profile, ckpt) in FINAL.items():
    pkg = ROOT / "data/raw/cpu" / tag
    val = (pkg / "public/validation.jsonl").open(encoding="utf-8")
    rows = [json.loads(next(val)) for _ in range(3)]
    payload = "".join(json.dumps({"task": "classification" if profile == "A" else "regression",
                                  "inputs": r["inputs"]}) + "\n" for r in rows)
    proc = subprocess.run([str(PY), "-m", "baselines.predict", "--checkpoint", str(ckpt)],
                          cwd=pkg, input=payload, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    print(f"=== {profile} ({ckpt.parent.name}) exit={proc.returncode} ===")
    out = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    for line, row in zip(out, rows):
        got = json.loads(line)
        print(f"  真值={row['answer']!r:>12}  预测={json.dumps(got.get('outputs'), ensure_ascii=False)[:90]}")
    if proc.returncode != 0:
        print(proc.stderr[-400:])
