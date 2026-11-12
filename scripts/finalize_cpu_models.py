"""固化三份最终 CPU 检查点的元信息：指标、相对基线、来源链路、sha256。"""

import hashlib
import json
from pathlib import Path

ROOT = Path(r"D:\智训引擎")
OUT = ROOT / "out/cpu-models"

BASELINE = {"A": 0.7140297220788709, "B": 681.1598326043113, "C": 52.86857256611151}
FINAL = {
    "A": dict(metric="macro_f1", higher=True, value=0.7291564, hidden=64,
              method="Net2Net 加宽 32→64 + 10 轮 Qwen 驱动续训",
              source="runs/cpu/a/r02（wide 一轮）"),
    "B": dict(metric="rmse", higher=False, value=602.5488, hidden=128,
              method="Net2Net 逐档加宽 24→64→96→128 + 各 10 轮 Qwen 驱动续训",
              source="runs/cpu/b（b128 一轮）"),
    "C": dict(metric="rmse", higher=False, value=50.9766, hidden=128,
              method="两次 Net2Net 加宽 32→64→128 + 20 轮 Qwen 驱动续训",
              source="runs/cpu/c/r09（c2 一轮）"),
}

models = []
for profile, info in FINAL.items():
    ckpt = OUT / "best" / f"{profile.lower()}.npz"
    digest = "sha256:" + hashlib.sha256(ckpt.read_bytes()).hexdigest()
    base = BASELINE[profile]
    delta = (info["value"] - base) if info["higher"] else (base - info["value"])
    rel = (delta / abs(base)) * 100
    models.append({
        "profile": profile,
        "task": "classification" if profile == "A" else "regression",
        "checkpoint": str(ckpt.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": digest,
        "hidden": info["hidden"],
        "method": info["method"],
        "lineage": info["source"],
        "metric": info["metric"],
        "baseline": base,
        "final": info["value"],
        "delta": delta,
        "delta_pct": rel,
    })

payload = {
    "source_packages": "cpu-{a,b,c}-*-20261110-v1.zip",
    "protocol": "每数据集 10 轮；Qwen 逐轮定超参，验证集闸门决定采纳；必要时用 Net2Net 显式迁移权重加宽结构",
    "selection": "每个数据集取公开验证集上最好的版本，隐藏测试集未参与任何选择",
    "models": models,
}
(OUT / "final.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
for m in models:
    print(f'{m["profile"]}: {m["metric"]}={m["final"]:.4f}  基线 {m["baseline"]:.4f}  '
          f'{m["delta"]:+.4f} ({m["delta_pct"]:+.1f}%)  hidden={m["hidden"]}')
print("\n->", OUT / "final.json")
