"""把检查点的隐藏层加宽，并把旧权重复制到新槽位（Net2Net 式函数保持迁移）。

为什么需要：
    C 的训练 loss 停在 0.42（A/B 是 0.08），是典型的欠拟合——隐藏层只有 32 个单元、
    而训练集有 113,089 行。赛规允许改结构，但要求"显式实现兼容的权重迁移，
    不能将一个新模型声明为原模型续训"，所以这里保留原函数、成对拆分输出权重。

做法：
    - 旧的 hidden 个单元原样保留；
    - 新增的每个单元复制一个随机旧单元（w1/b1 相同），并把该旧单元的输出权重按复制数均分，
      保证加宽前后前向输出**完全一致**；
    - 复制时加极小噪声打破对称，否则两个单元梯度永远相同、学不到新东西。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\智训引擎")
PKG = ROOT / "data" / "raw" / "cpu"


def forward(params: dict, x: np.ndarray) -> np.ndarray:
    hidden = np.tanh(x @ params["w1"] + params["b1"])
    return hidden @ params["w2"] + params["b2"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, choices=["A", "B", "C"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--hidden", type=int, required=True, help="加宽后的隐藏层宽度（上限 128）")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not 1 <= args.hidden <= 128:
        raise SystemExit("hidden 必须在 1..128")

    tag = args.profile.lower()
    sys.path.insert(0, str(PKG / tag))
    from baselines.model import Model

    model = Model.load(Path(args.checkpoint))
    old = model.meta["hidden"]
    new = args.hidden
    if new <= old:
        raise SystemExit(f"新宽度 {new} 必须大于原宽度 {old}")

    old_params = {k: v.copy() for k, v in model.params.items()}
    rng = np.random.default_rng(args.seed)
    extra = new - old
    w1, b1, w2, b2 = (old_params[k] for k in ("w1", "b1", "w2", "b2"))

    src = rng.integers(0, old, size=extra)
    counts = np.bincount(src, minlength=old) + 1          # 每个源单元被复制的次数 + 1

    new_w1 = np.zeros((w1.shape[0], new), dtype=w1.dtype)
    new_b1 = np.zeros(new, dtype=b1.dtype)
    new_w2 = np.zeros((new, w2.shape[1]), dtype=w2.dtype)

    new_w1[:, :old] = w1
    new_b1[:old] = b1
    new_w2[:old, :] = w2 / counts[:, None]                 # 源单元的输出权重按复制数均分

    for j, i in enumerate(src, start=old):
        new_w1[:, j] = w1[:, i] + rng.normal(0, 1e-3, size=w1.shape[0])
        new_b1[j] = b1[i]
        new_w2[j, :] = w2[i, :] / counts[i]

    # 函数保持校验：随机输入上加宽前后输出应几乎一致
    probe = rng.normal(0, 1, size=(64, w1.shape[0]))
    before = forward(old_params, probe)
    after = forward({"w1": new_w1, "b1": new_b1, "w2": new_w2, "b2": b2}, probe)
    drift = float(np.max(np.abs(before - after)))
    scale = float(np.max(np.abs(before)) + 1e-9)
    if drift / scale > 0.02:
        raise SystemExit(f"迁移后输出漂移过大: {drift:.4f} ({drift / scale:.1%})")

    model.params = {"w1": new_w1, "b1": new_b1, "w2": new_w2, "b2": b2.copy()}
    model.first = {k: np.zeros_like(v) for k, v in model.params.items()}
    model.second = {k: np.zeros_like(v) for k, v in model.params.items()}
    model.meta["hidden"] = new
    model.meta["parameter_count"] = int(new_w1.size + new_b1.size + new_w2.size + b2.size)
    model.meta["widened_from"] = {"source": Path(args.checkpoint).name, "old_hidden": old,
                                  "new_hidden": new, "method": "net2net-duplicate-split",
                                  "forward_drift_ratio": round(drift / scale, 6)}
    model.validate()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    model.save(out)
    print(f"{args.profile}: hidden {old}→{new}，参数 {model.meta['parameter_count']}，"
          f"前向漂移 {drift / scale:.2%}，-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
