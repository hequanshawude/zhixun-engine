"""从真实回放里导出"状态卡 → 动作"样本，用于 PyTRiO 上的策略模型 SFT。

为什么不用人造状态：
    现有 train_pytrio_policy.py 的样本是随机拼的指纹，模型学到的是分布外的空壳。
    这里直接跑真实数据回放，把主循环每一步的 state_log 拿出来——输入分布、
    预算字段、历史动作都与线上完全一致。

标签来源：
    默认用规则策略的动作（行为克隆）。规则在两项真实数据上胜率约 58.9%，
    是一个"稳定但不强"的参照，适合当冷启动示范。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selftrain.config import SystemConfig
from selftrain.data import DatasetSpec
from selftrain.experiment import NullLogger
from selftrain.harness import build_messages
from selftrain.loop import SelfEvolvingSystem
from selftrain.types import Batch, TaskType

OUT = Path("out/pytrio-training")
DATA = Path("data/processed")


# --------------------------------------------------------------------------
# 两个数据集的切分
# --------------------------------------------------------------------------


def _xjtu_frame() -> pd.DataFrame:
    parts = [pd.read_csv(DATA / "xjtu-sy/train.csv")]
    parts += [pd.read_csv(p) for p in sorted((DATA / "xjtu-sy/stream").glob("batch_*.csv"))]
    return (pd.concat(parts, ignore_index=True)
            .drop_duplicates(subset=["id", "cycle"]).sort_values(["id", "cycle"]))


def _cmapss_frame() -> pd.DataFrame:
    parts = [pd.read_csv(DATA / "cmapss-fd001/train.csv")]
    parts += [pd.read_csv(p) for p in sorted((DATA / "cmapss-fd001/stream").glob("batch_*.csv"))]
    return (pd.concat(parts, ignore_index=True)
            .drop_duplicates(subset=["id", "cycle"]).sort_values(["id", "cycle"]))


def _spec(frame: pd.DataFrame, name: str) -> DatasetSpec:
    feats = [c for c in frame.columns if c not in ("id", "y")]
    return DatasetSpec(name=name, task=TaskType.REGRESSION, feature_names=feats, primary_metric="mae")


def _batches(frame: pd.DataFrame, spec: DatasetSpec, groups: list[list]) -> list[Batch]:
    out = []
    for k, ids in enumerate(groups):
        part = frame[frame.id.isin(ids)].sort_values(["id", "cycle"])
        out.append(Batch(index=k, X=part[spec.feature_names].to_numpy(dtype=np.float32),
                         y=part["y"].to_numpy(dtype=float),
                         feature_names=spec.feature_names, meta={"ids": ids}))
    return out


def _xy(frame: pd.DataFrame, spec: DatasetSpec, ids):
    part = frame[frame.id.isin(ids)]
    return (part[spec.feature_names].to_numpy(dtype=np.float32),
            part["y"].to_numpy(dtype=float))


# --------------------------------------------------------------------------
# 采集
# --------------------------------------------------------------------------


def collect_xjtu(seeds: int, policy: str) -> list[dict]:
    frame = _xjtu_frame()
    spec = _spec(frame, "xjtu-sy")
    rows = []
    ids = list(frame.id.unique())
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        shuffled = list(rng.permutation(ids))
        train_ids, stream = shuffled[:9], shuffled[9:]
        groups = [stream[i:i + 2] for i in range(0, len(stream), 2)]
        cfg = SystemConfig()
        cfg.seed = seed
        system = SelfEvolvingSystem(spec, cfg, logger=NullLogger(), policy_mode=policy)
        system.run(iter(_batches(frame, spec, groups)), bootstrap_data=_xy(frame, spec, train_ids))
        rows.extend(system.state_log)
    return rows


def collect_cmapss(seeds: int, policy: str) -> list[dict]:
    frame = _cmapss_frame()
    spec = _spec(frame, "cmapss-fd001")
    rows = []
    ids = list(frame.id.unique())
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        shuffled = list(rng.permutation(ids))
        train_ids, gate_ids = shuffled[:60], shuffled[60:70]
        stream = shuffled[70:]
        groups = [stream[i:i + 3] for i in range(0, len(stream), 3)]
        cfg = SystemConfig()
        cfg.seed = seed
        system = SelfEvolvingSystem(spec, cfg, logger=NullLogger(), policy_mode=policy)
        system.run(iter(_batches(frame, spec, groups)),
                   bootstrap_data=_xy(frame, spec, train_ids),
                   gate_data=_xy(frame, spec, gate_ids))
        rows.extend(system.state_log)
    return rows


def collect_synthetic(runs: int, policy: str) -> list[dict]:
    """合成流补齐动作覆盖。

    两个真实数据集上漂移都很强，规则几乎总是选 finetune/rollback，样本里没有一个 noop。
    但"什么时候不该动"恰恰是最省成本、也是闸门最该学的能力，所以补一批
    弱漂移/无漂移/标签漂移的合成场景，覆盖 noop 与 calibrate。
    """
    from selftrain.data import SyntheticConfig, SyntheticStream

    rows = []
    scenarios = ["none", "label", "covariate", "concept", "mixed"]
    tasks = [TaskType.BINARY, TaskType.MULTICLASS, TaskType.REGRESSION]
    for i in range(runs):
        scenario = scenarios[i % len(scenarios)]
        task = tasks[(i // len(scenarios)) % len(tasks)]
        cfg = SystemConfig()
        cfg.seed = i
        stream = SyntheticStream(SyntheticConfig(
            task=task, n_batches=10, batch_size=150, n_features=8,
            scenario=scenario, drift_start=3, drift_strength=0.6, seed=i))
        system = SelfEvolvingSystem(stream.spec, cfg, logger=NullLogger(), policy_mode=policy)
        system.run(iter(stream))
        rows.extend(system.state_log)
    return rows


# --------------------------------------------------------------------------


def to_example(rec: dict) -> dict:
    state = rec["state"]
    answer = json.dumps({"action": rec["action"], "params": {}, "reason": "示范"},
                        ensure_ascii=False)
    key = hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return {"messages": build_messages(state), "answer": answer, "key": key,
            "dataset": rec.get("dataset", ""), "action": rec["action"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xjtu-seeds", type=int, default=20)
    ap.add_argument("--cmapss-seeds", type=int, default=10)
    ap.add_argument("--synth-runs", type=int, default=15, help="合成流轮数，用于补齐 noop/calibrate 动作")
    ap.add_argument("--policy", default="rule", choices=["rule", "bandit"])
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    print(f"采集 XJTU-SY（{args.xjtu_seeds} 个切分）...", flush=True)
    xj = collect_xjtu(args.xjtu_seeds, args.policy)
    print(f"采集 C-MAPSS（{args.cmapss_seeds} 个切分）...", flush=True)
    cm = collect_cmapss(args.cmapss_seeds, args.policy)
    print(f"采集合成流（{args.synth_runs} 轮）...", flush=True)
    sy = collect_synthetic(args.synth_runs, args.policy) if args.synth_runs else []
    raw = xj + cm + sy

    seen, uniq = set(), []
    for rec in raw:
        ex = to_example(rec)
        if ex["key"] in seen:
            continue
        seen.add(ex["key"])
        uniq.append(ex)

    rng = np.random.default_rng(42)
    order = rng.permutation(len(uniq))
    uniq = [uniq[i] for i in order]
    n_hold = max(1, int(len(uniq) * args.holdout))
    holdout, train = uniq[:n_hold], uniq[n_hold:]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("holdout", holdout)):
        (out / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    dist: dict[str, int] = {}
    for r in uniq:
        dist[r["action"]] = dist.get(r["action"], 0) + 1
    meta = {"policy": args.policy, "raw": len(raw), "unique": len(uniq),
            "train": len(train), "holdout": len(holdout),
            "xjtu_states": len(xj), "cmapss_states": len(cm), "synth_states": len(sy),
            "action_distribution": dist}
    (out / "dataset-meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
