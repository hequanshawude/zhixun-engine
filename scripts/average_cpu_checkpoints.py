"""把多个续训检查点的权重平均成一个（SWA 思路），用于降低单轮抖动。

续训过程中每一轮都从上一轮继续，权重轨迹在最优解附近来回走。把这些点的参数平均，
通常比"只取最后一轮"更稳。标准化参数（mean/scale）在续训中不变，直接沿用。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(r"D:\智训引擎")
PKG = ROOT / "data" / "raw" / "cpu"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, choices=["A", "B", "C"])
    ap.add_argument("--inputs", nargs="+", required=True, help="参与平均的 checkpoint 列表")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tag = args.profile.lower()
    sys.path.insert(0, str(PKG / tag))
    from baselines.model import Model

    models = [Model.load(Path(p)) for p in args.inputs]
    base = models[-1]
    ref = models[0]
    for m in models:
        if m.meta["hidden"] != ref.meta["hidden"] or m.meta["paths"] != ref.meta["paths"]:
            raise SystemExit("架构或特征路径不一致，无法平均")

    key = "params"
    averaged = {k: sum(getattr(m, key)[k] for m in models) / len(models) for k in ref.params}
    base.params = averaged
    # Adam 状态不参与推理，留着上一轮的值即可；不平均会让后续 resume 语义混乱，这里清空重算
    base.first = {k: v * 0 for k, v in base.first.items()}
    base.second = {k: v * 0 for k, v in base.second.items()}
    base.meta["averaged_from"] = [str(Path(p).name) for p in args.inputs]
    base.meta["averaged_count"] = len(models)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    base.save(out)
    print(f"{args.profile}: 平均 {len(models)} 个检查点 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
