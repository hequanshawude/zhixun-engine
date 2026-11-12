"""XJTU-SY 的配对验证：重复随机轴承切分，而不是只看一次切分。

为什么需要它：
    15 个轴承里只有 6 个能回放，且寿命跨度 42～2538 个快照（60 倍）。
    单次切分的"批均 MAE"几乎完全由抽到哪几个轴承决定，无法区分策略好坏。

做法：
    对每个随机种子重新切分（9 训练 / 6 回放），在同一切分上同时跑
    系统与静态基线，得到**逐批配对差值**；再跨种子汇总胜率、均值与置信区间。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selftrain.config import SystemConfig
from selftrain.data import DatasetSpec
from selftrain.experiment import ExperimentLogger, NullLogger
from selftrain.loop import SelfEvolvingSystem
from selftrain.types import Batch, TaskType
from selftrain.validation import print_summary, summarize

DATA = Path("data/processed/xjtu-sy")
OUT = Path("out/xjtu-validation")


def load_frame() -> pd.DataFrame:
    """把已处理好的 train.csv 与 stream/*.csv 合并回全量（15 个轴承都在里面）。"""
    parts = [pd.read_csv(DATA / "train.csv")]
    parts += [pd.read_csv(p) for p in sorted((DATA / "stream").glob("batch_*.csv"))]
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.drop_duplicates(subset=["id", "cycle"]).sort_values(["id", "cycle"])
    if frame.id.nunique() != 15:
        raise ValueError(f"期望 15 个轴承，实际 {frame.id.nunique()}")
    return frame


def build_spec(frame: pd.DataFrame) -> DatasetSpec:
    feats = [c for c in frame.columns if c not in ("id", "y")]
    return DatasetSpec(name="xjtu-sy", task=TaskType.REGRESSION,
                       feature_names=feats, primary_metric="mae")


def split_bearings(ids: list[str], seed: int, n_train: int = 9, per_batch: int = 2):
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(ids))
    train_ids = sorted(shuffled[:n_train])
    stream_ids = shuffled[n_train:]
    batches = [stream_ids[i:i + per_batch] for i in range(0, len(stream_ids), per_batch)]
    return train_ids, batches


def run_trial(frame: pd.DataFrame, spec: DatasetSpec, cfg: SystemConfig,
              seed: int, policy: str) -> list[dict]:
    train_ids, batch_groups = split_bearings(list(frame.id.unique()), seed)
    train = frame[frame.id.isin(train_ids)]
    batches = []
    for k, ids in enumerate(batch_groups):
        part = frame[frame.id.isin(ids)].sort_values(["id", "cycle"])
        batches.append(Batch(index=k, X=part[spec.feature_names].to_numpy(dtype=np.float32),
                             y=part["y"].to_numpy(dtype=float),
                             feature_names=spec.feature_names,
                             meta={"ids": ids}))

    trial_cfg = SystemConfig.from_dict(cfg.to_dict())
    trial_cfg.seed = seed
    system = SelfEvolvingSystem(spec, trial_cfg, logger=NullLogger(), policy_mode=policy)
    system.run(iter(batches),
               bootstrap_data=(train[spec.feature_names].to_numpy(dtype=np.float32),
                               train["y"].to_numpy(dtype=float)))
    return [dict(s, seed=seed, bearings=b.meta["ids"])
            for s, b in zip(system.scores, batches)]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--policy", default="rule", choices=["rule", "bandit"])
    p.add_argument("--config", default="configs/xjtu-sy-rule.yaml")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--cloud", action="store_true", help="把整轮 sweep 记录到 SwanLab 云端")
    p.add_argument("--workspace", default="hequan")
    p.add_argument("--project", default="zhixunyinqing")
    args = p.parse_args()

    frame = load_frame()
    spec = build_spec(frame)
    cfg = SystemConfig.load(args.config)
    if args.cloud:
        cfg.log.backend = "swanlab"
        cfg.log.swanlab_mode = "online"
        cfg.log.workspace = args.workspace
        cfg.log.project = args.project
        cfg.log.experiment = f"xjtu-validate-{args.policy}"

    logger = ExperimentLogger(cfg.log, run_name=f"xjtu-validate-{args.policy}")
    logger.start(config={**cfg.to_dict(), "seeds": args.seeds, "protocol": "repeated-bearing-split"},
                 spec={**spec.to_dict(), "n_bearings": int(frame.id.nunique())})

    rows, per_trial = [], []
    for seed in range(args.seeds):
        got = run_trial(frame, spec, cfg, seed, args.policy)
        rows.extend(got)
        delta = [r["static"] - r["system"] for r in got]
        per_trial.append({"seed": seed,
                          "bearings": [b for r in got for b in r["bearings"]],
                          "mean_delta": float(np.mean(delta))})
        logger.log(seed, {
            "seed_mean_delta_mae": float(np.mean(delta)),
            "seed_system_mae": float(np.mean([r["system"] for r in got])),
            "seed_static_mae": float(np.mean([r["static"] for r in got])),
            "cum_mean_delta_mae": float(np.mean([r["static"] - r["system"] for r in rows])),
        })
        print(f"  seed {seed:02d}: " + "  ".join(
            f"sys={r['system']:7.1f} sta={r['static']:7.1f}" for r in got), flush=True)

    stats = summarize(rows, args.policy, label="xjtu-sy")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"stats": stats, "per_batch": rows, "per_seed": per_trial}
    (out_dir / f"validation-{args.policy}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log_artifact("validation.json", payload)
    logger.finish(stats)

    if getattr(logger, "cloud_url", None):
        print(f"\n云端实验: {logger.cloud_url}")
    print_summary(stats)
    print(f"\n明细: {out_dir / f'validation-{args.policy}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
