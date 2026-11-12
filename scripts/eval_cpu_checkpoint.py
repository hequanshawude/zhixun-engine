"""在公开验证集上评测一个或多个 CPU 检查点，并与基线/最佳值对比。

直接复用包内 `baselines.model.Model` / `baselines.data.read_dataset` / `baselines.train.metrics`，
保证评测口径与训练脚本完全一致（同一份标准化参数、同一目标缩放）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(r"D:\智训引擎")
PKG = ROOT / "data" / "raw" / "cpu"
BASELINE = {"A": 0.7140297220788709, "B": 681.1598326043113, "C": 52.86857256611151}
METRIC = {"A": "macro_f1", "B": "rmse", "C": "rmse"}


def evaluate(profile: str, ckpt: Path) -> dict:
    tag = profile.lower()
    pkg = PKG / tag
    sys.path.insert(0, str(pkg))
    for name in [m for m in list(sys.modules) if m.startswith("baselines")]:
        del sys.modules[name]
    from baselines.data import read_dataset
    from baselines.model import Model
    from baselines.train import metrics

    model = Model.load(ckpt)
    data = read_dataset(pkg / "public/validation.jsonl", model.meta["task"],
                        classes=model.meta["classes"], expected_paths=model.meta["paths"])
    return metrics(model, data)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", help="形如 A:path/to/ckpt.npz")
    args = ap.parse_args()
    rows = []
    for item in args.checkpoints:
        profile, _, path = item.partition(":")
        m = evaluate(profile, Path(path))
        key = METRIC[profile]
        base = BASELINE[profile]
        delta = (m[key] - base) if key == "macro_f1" else (base - m[key])
        print(f"{profile}  {key}={m[key]:.4f}  基线={base:.4f}  差值={delta:+.4f}   {path}")
        rows.append({"profile": profile, "checkpoint": path, **m, "baseline": base, "delta": delta})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
