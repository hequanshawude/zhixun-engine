"""C-MAPSS FD001 的配对验证：重复随机发动机切分。

背景：FD001 现在的结论（规则 26.5943 vs 静态 26.3993）只有单次切分，缺显著性。
和 XJTU-SY 一样，用"同切分配对 + 多个随机切分"来量化这个差值到底是策略效果还是抽样噪声。

每个种子从 100 台发动机里重新抽 60 训练 / 10 独立闸门 / 30 回放（10 批 × 3 台），
三组互不重叠；闸门样本只用于候选晋级判定，不进训练历史。
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

DATA = Path("data/processed/cmapss-fd001")
OUT = Path("out/cmapss-validation")


def load_frame() -> pd.DataFrame:
    parts = [pd.read_csv(DATA / "train.csv")]
    parts += [pd.read_csv(p) for p in sorted((DATA / "stream").glob("batch_*.csv"))]
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.drop_duplicates(subset=["id", "cycle"]).sort_values(["id", "cycle"])
    if frame.id.nunique() != 100:
        raise ValueError(f"期望 100 台发动机，实际 {frame.id.nunique()}")
    return frame


def build_spec(frame: pd.DataFrame) -> DatasetSpec:
    feats = [c for c in frame.columns if c not in ("id", "y")]
    return DatasetSpec(name="cmapss-fd001", task=TaskType.REGRESSION,
                       feature_names=feats, primary_metric="mae")


def split_engines(ids, seed: int, n_train: int = 60, n_gate: int = 10, per_batch: int = 3):
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(list(ids)))
    train_ids = sorted(shuffled[:n_train])
    gate_ids = shuffled[n_train:n_train + n_gate]
    stream_ids = shuffled[n_train + n_gate:]
    batches = [stream_ids[i:i + per_batch] for i in range(0, len(stream_ids), per_batch)]
    return train_ids, gate_ids, batches


def run_trial(frame: pd.DataFrame, spec: DatasetSpec, cfg: SystemConfig,
              seed: int, policy: str) -> list[dict]:
    train_ids, gate_ids, batch_groups = split_engines(list(frame.id.unique()), seed)
    train = frame[frame.id.isin(train_ids)]
    gate = frame[frame.id.isin(gate_ids)]

    batches = []
    for k, ids in enumerate(batch_groups):
        part = frame[frame.id.isin(ids)].sort_values(["id", "cycle"])
        batches.append(Batch(index=k, X=part[spec.feature_names].to_numpy(dtype=np.float32),
                             y=part["y"].to_numpy(dtype=float),
                             feature_names=spec.feature_names,
                             meta={"ids": [int(i) for i in ids]}))

    trial_cfg = SystemConfig.from_dict(cfg.to_dict())
    trial_cfg.seed = seed
    system = SelfEvolvingSystem(spec, trial_cfg, logger=NullLogger(), policy_mode=policy)
    system.run(iter(batches),
               bootstrap_data=(train[spec.feature_names].to_numpy(dtype=np.float32),
                               train["y"].to_numpy(dtype=float)),
               gate_data=(gate[spec.feature_names].to_numpy(dtype=np.float32),
                          gate["y"].to_numpy(dtype=float)))
    return [dict(s, seed=seed, engines=b.meta["ids"])
            for s, b in zip(system.scores, batches)]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--policy", default="rule", choices=["rule", "bandit"])
    p.add_argument("--config", default="configs/cmapss-fd001-holdout-rule.yaml")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--cloud", action="store_true")
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
        cfg.log.experiment = f"cmapss-validate-{args.policy}"

    logger = ExperimentLogger(cfg.log, run_name=f"cmapss-validate-{args.policy}")
    logger.start(config={**cfg.to_dict(), "seeds": args.seeds, "protocol": "repeated-engine-split"},
                 spec={**spec.to_dict(), "n_engines": int(frame.id.nunique())})

    rows, per_trial = [], []
    for seed in range(args.seeds):
        got = run_trial(frame, spec, cfg, seed, args.policy)
        rows.extend(got)
        delta = [r["static"] - r["system"] for r in got]
        per_trial.append({"seed": seed,
                          "engines": [e for r in got for e in r["engines"]],
                          "mean_delta": float(np.mean(delta))})
        logger.log(seed, {
            "seed_mean_delta_mae": float(np.mean(delta)),
            "seed_system_mae": float(np.mean([r["system"] for r in got])),
            "seed_static_mae": float(np.mean([r["static"] for r in got])),
            "cum_mean_delta_mae": float(np.mean([r["static"] - r["system"] for r in rows])),
        })
        print(f"  seed {seed:02d}: " + "  ".join(
            f"sys={r['system']:6.2f} sta={r['static']:6.2f}" for r in got), flush=True)

    stats = summarize(rows, args.policy, label="cmapss-fd001")
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
