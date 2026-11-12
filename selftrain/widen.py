"""在模型包内执行 Net2Net 加宽（子进程入口）。

赛规允许改结构，但要求"显式实现兼容的权重迁移，不能将一个新模型声明为原模型续训"。
做法：旧单元原样保留，新单元复制随机旧单元并把该单元的输出权重按复制数均分，
保证加宽前后前向输出一致；复制时加极小噪声打破对称。迁移后校验前向漂移。

用法（由 selftrain.cpu 调用，也可单独跑）：

    python -m selftrain.widen --package <包目录> --hidden 128 \
        --checkpoint <包内 checkpoint.npz> --output <新检查点> --profile-out <新 profile.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _forward(params: dict, x: np.ndarray) -> np.ndarray:
    return np.tanh(x @ params["w1"] + params["b1"]) @ params["w2"] + params["b2"]


def widen(package: Path, checkpoint: Path, hidden: int, output: Path,
          profile_out: Path | None = None, seed: int = 42) -> dict:
    package = Path(package).resolve()
    if not 1 <= hidden <= 128:
        raise ValueError("hidden 必须在 1..128")

    # 借用包内的 Model 实现，保证检查点格式完全一致
    sys.path.insert(0, str(package))
    from baselines.model import Model  # noqa: WPS433

    model = Model.load(Path(checkpoint))
    old = model.meta["hidden"]
    if hidden <= old:
        raise ValueError(f"目标宽度 {hidden} 必须大于当前 {old}")

    old_params = {k: v.copy() for k, v in model.params.items()}
    rng = np.random.default_rng(seed)
    extra = hidden - old
    w1, b1, w2, b2 = (old_params[k] for k in ("w1", "b1", "w2", "b2"))
    src = rng.integers(0, old, size=extra)
    counts = np.bincount(src, minlength=old) + 1

    new_w1 = np.zeros((w1.shape[0], hidden), dtype=w1.dtype)
    new_b1 = np.zeros(hidden, dtype=b1.dtype)
    new_w2 = np.zeros((hidden, w2.shape[1]), dtype=w2.dtype)
    new_w1[:, :old] = w1
    new_b1[:old] = b1
    new_w2[:old, :] = w2 / counts[:, None]
    for j, i in enumerate(src, start=old):
        new_w1[:, j] = w1[:, i] + rng.normal(0, 1e-3, size=w1.shape[0])
        new_b1[j] = b1[i]
        new_w2[j, :] = w2[i, :] / counts[i]

    probe = rng.normal(0, 1, size=(64, w1.shape[0]))
    before = _forward(old_params, probe)
    after = _forward({"w1": new_w1, "b1": new_b1, "w2": new_w2, "b2": b2}, probe)
    drift = float(np.max(np.abs(before - after)))
    scale = float(np.max(np.abs(before)) + 1e-9)
    if drift / scale > 0.02:
        raise ValueError(f"迁移后输出漂移过大: {drift / scale:.2%}")

    model.params = {"w1": new_w1, "b1": new_b1, "w2": new_w2, "b2": b2.copy()}
    model.first = {k: np.zeros_like(v) for k, v in model.params.items()}
    model.second = {k: np.zeros_like(v) for k, v in model.params.items()}
    model.meta["hidden"] = hidden
    model.meta["parameter_count"] = int(new_w1.size + new_b1.size + new_w2.size + b2.size)
    model.meta["widened_from"] = {"source": Path(checkpoint).name, "old_hidden": old,
                                  "new_hidden": hidden, "method": "net2net-duplicate-split",
                                  "forward_drift_ratio": round(drift / scale, 6)}
    model.validate()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    model.save(output)

    if profile_out is not None:
        profile_name = model.meta.get("profile") or _guess_profile(package)
        cfg_path = _profile_source(package, profile_name)
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["hidden"] = hidden
        Path(profile_out).parent.mkdir(parents=True, exist_ok=True)
        Path(profile_out).write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                                     encoding="utf-8")

    return {"old_hidden": old, "new_hidden": hidden, "parameter_count":
            model.meta["parameter_count"], "forward_drift_ratio": round(drift / scale, 6)}


def _guess_profile(package: Path) -> str:
    report = package / "training.json"
    if report.exists():
        return str(json.loads(report.read_text(encoding="utf-8")).get("profile", "A")).upper()
    return "A"


def _profile_source(package: Path, profile: str) -> Path:
    candidates = list((package / "baselines" / "configs").glob("*.json"))
    for p in candidates:
        try:
            if str(json.loads(p.read_text(encoding="utf-8")).get("profile", "")).upper() == profile:
                return p
        except Exception:
            continue
    if not candidates:
        raise FileNotFoundError(f"{package}/baselines/configs 下没有 profile JSON")
    return candidates[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--profile-out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    info = widen(Path(args.package), Path(args.checkpoint), args.hidden,
                 Path(args.output), Path(args.profile_out) if args.profile_out else None,
                 args.seed)
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
