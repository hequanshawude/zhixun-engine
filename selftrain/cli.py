"""命令行入口。

    python -m selftrain demo                       # 合成数据端到端跑通
    python -m selftrain demo --scenario concept --task regression
    python -m selftrain run --data-dir ./data/A/gas --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


from .config import SystemConfig, apply_env_overrides
from .data import CSVBatchStream, SyntheticConfig, SyntheticStream, infer_spec
from .experiment import ExperimentLogger
from .loop import SelfEvolvingSystem
from .sink import PredictionSink
from .types import TaskType


def _parse_task(s: str) -> TaskType:
    try:
        return TaskType(s)
    except ValueError:
        raise SystemExit(f"未知 task: {s}（可选 binary/multiclass/regression）")


def _run(cfg: SystemConfig, spec, stream, policy: str, run_name: str, quiet: bool = False,
         pred_dir: Optional[str] = None, bootstrap_data=None, checkpoint=None, resume=False,
         gate_data=None):
    if resume and (not checkpoint or not pred_dir):
        raise ValueError('--resume 需要 --checkpoint 和原来的 --pred-dir')
    sink = PredictionSink(pred_dir, spec.task.value, resume=resume) if pred_dir else None
    logger = ExperimentLogger(cfg.log, run_name=run_name)

    def emit(k, preds):
        if sink is not None:
            sink(k, preds)
        if not quiet:
            print(f"  [emit] batch {k}: {len(preds)} 条预测已输出")

    system = SelfEvolvingSystem(spec, cfg, logger=logger, policy_mode=policy, emit=emit)
    try:
        result = system.run(stream, bootstrap_data=bootstrap_data, gate_data=gate_data, checkpoint_path=checkpoint,
                            resume=resume, resume_validator=sink.validate_completed if sink else None,
                            checkpoint_outputs=sink.snapshot if sink else None)
    except Exception as exc:
        try:
            logger.abort(str(exc), system.initialization)
        except Exception:
            print('失败记录已保存本地，云端失败状态同步未完成', file=sys.stderr)
        raise
    if logger.cloud_url:
        print(f"\n云端实验: {logger.cloud_url}")
    return system, result, logger


def cmd_demo(args) -> int:
    cfg = SystemConfig.load(args.config) if args.config else SystemConfig()
    apply_env_overrides(cfg)
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.log.experiment = args.name

    s = SyntheticConfig(
        task=_parse_task(args.task),
        n_batches=args.batches,
        batch_size=args.batch_size,
        n_features=args.features,
        scenario=args.scenario,
        drift_start=args.drift_start,
        drift_strength=args.strength,
        seed=cfg.seed,
    )
    stream = SyntheticStream(s)
    spec = stream.spec
    spec.primary_metric = {
        TaskType.BINARY: "f1_macro",
        TaskType.MULTICLASS: "f1_macro",
        TaskType.REGRESSION: "mae",
    }[spec.task]

    print(f"\n=== 合成数据流：{spec.name} | {s.n_batches} 批 × {s.batch_size} 条 | 策略={args.policy} ===\n")
    system, result, logger = _run(cfg, spec, stream, args.policy, f"demo-{s.scenario}", quiet=args.quiet,
        pred_dir=args.pred_dir, checkpoint=args.checkpoint, resume=args.resume)

    _print_report(result.summary, system, quiet=args.quiet)
    print(f"\n记录目录: {logger.out_dir}")
    return 0


def cmd_run(args) -> int:
    import pandas as pd

    cfg = SystemConfig.load(args.config) if args.config else SystemConfig()
    apply_env_overrides(cfg)
    cfg.log.experiment = args.name

    data_dir = Path(args.data_dir)
    if args.unlabeled and not args.train_csv:
        raise ValueError("无标签预测必须通过 --train-csv 提供独立的带标签训练集")
    if args.gate_csv and not args.train_csv:
        raise ValueError("--gate-csv 需要 --train-csv 提供独立训练集")
    files = sorted(data_dir.glob("batch_*.csv"))
    probe = pd.read_csv(args.train_csv) if args.train_csv else (pd.read_csv(files[0]) if files else pd.read_csv(data_dir / "all.csv"))
    if args.target not in probe.columns:
        raise ValueError(f"训练集缺少目标列 {args.target}")
    spec = infer_spec(probe, name=data_dir.name, target_col=args.target)
    if args.task:
        spec.task = _parse_task(args.task)
    spec.primary_metric = {
        TaskType.BINARY: "f1_macro",
        TaskType.MULTICLASS: "f1_macro",
        TaskType.REGRESSION: "mae",
    }[spec.task]

    stream = CSVBatchStream(
        str(data_dir), spec, target_col=args.target, has_labels=not args.unlabeled
    )
    print(f"\n=== 真实数据流：{spec.name} | {len(spec.feature_names)} 维特征 | 策略={args.policy} ===\n")
    bootstrap_data = None
    if args.train_csv:
        bootstrap_data = (probe[spec.feature_names].to_numpy(dtype="float32"), probe[args.target].to_numpy())
    gate_data = None
    if args.gate_csv:
        gate_frame = pd.read_csv(args.gate_csv)
        required = [*spec.feature_names, args.target]
        missing = [col for col in required if col not in gate_frame.columns]
        if missing or gate_frame.empty:
            raise ValueError(f"独立闸门集缺列或为空: {missing}")
        if 'id' in probe.columns and 'id' in gate_frame.columns:
            train_ids = set(probe['id'])
            gate_ids = set(gate_frame['id'])
            if train_ids & gate_ids:
                raise ValueError("训练集与独立闸门集的设备编号重叠")
            stream_paths = files or [data_dir / 'all.csv']
            for batch_path in stream_paths:
                ids = pd.read_csv(batch_path, usecols=lambda col: col == 'id')
                if 'id' in ids and gate_ids & set(ids['id']):
                    raise ValueError(f"独立闸门集与 {batch_path.name} 的设备编号重叠")
        gate_data = (gate_frame[spec.feature_names].to_numpy(dtype="float32"),
                     gate_frame[args.target].to_numpy())
    system, result, logger = _run(cfg, spec, stream, args.policy, f"run-{data_dir.name}", quiet=args.quiet,
        pred_dir=args.pred_dir, bootstrap_data=bootstrap_data, checkpoint=args.checkpoint,
        resume=args.resume, gate_data=gate_data)

    _print_report(result.summary, system, quiet=args.quiet)
    print(f"\n记录目录: {logger.out_dir}")
    return 0


def _print_report(summary, system, quiet=False) -> None:
    print("\n" + "─" * 66)
    print(f"  批次数            : {summary['n_batches']}")
    print(f"  上线次数          : {summary['n_promotions']}")
    print(f"  系统均值({summary['task']}) : {summary['mean_system']:.4f}")
    print(f"  静态基线均值      : {summary['mean_static']:.4f}")
    print(f"  相对基线增益      : {summary['gain_vs_static']:+.4f}")
    b = summary["budget"]
    print(f"  算力/Token/标注    : {b['compute_s']:.1f}s / {b['tokens']} tok / {b['labels']} 条")
    print(f"  总成本 / 总增益    : {b['total_cost']:.2f} / {b['total_gain']:+.5f}")
    print(f"  单位成本增益      : {b['efficiency']:.6f}")
    print(f"  策略来源分布      : {summary['policy_sources']}")
    print("─" * 66)

    if not quiet and system.decisions:
        print("\n逐批决策：")
        print(f"{'批次':>4} {'漂移类型':<10} {'严重度':>6} {'动作':<12} {'晋升':>4} {'原因'}")
        for d in system.decisions:
            fp = d["fingerprint"]
            print(
                f"{d['batch']:>4} {fp['kind']:<10} {fp['severity']:>6.2f} "
                f"{d['plan']['action']:<12} {str(d['promoted']):>4}  {d['reason'][:30]}"
            )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="selftrain", description="智训引擎 —— 工业模型自进化系统")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="用合成漂移数据跑通全链路")
    d.add_argument("--scenario", default="mixed",
                   choices=["covariate", "concept", "label", "sudden", "mixed", "none"])
    d.add_argument("--task", default="binary", choices=["binary", "multiclass", "regression"])
    d.add_argument("--batches", type=int, default=16)
    d.add_argument("--batch-size", type=int, default=300)
    d.add_argument("--features", type=int, default=16)
    d.add_argument("--drift-start", type=int, default=3)
    d.add_argument("--strength", type=float, default=1.0)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--policy", default="rule", choices=["rule", "bandit", "llm"])
    d.add_argument("--config", default=None)
    d.add_argument("--name", default="demo")
    d.add_argument("--quiet", action="store_true")
    d.add_argument("--pred-dir", default=None, help="把每批预测写到这个目录（提交用）")
    d.add_argument('--checkpoint', help='每个完整批次后保存可信本地检查点')
    d.add_argument('--resume', action='store_true', help='从指定检查点恢复，输入必须包含原始完整数据前缀')
    d.set_defaults(func=cmd_demo)

    r = sub.add_parser("run", help="跑真实 CSV 分批数据")
    r.add_argument("--data-dir", required=True)
    r.add_argument("--target", default="y")
    r.add_argument("--train-csv", default=None, help="独立带标签训练集；提供后所有流批次均输出预测")
    r.add_argument("--gate-csv", default=None, help="独立候选验证集；不得与训练或流批次共享设备")
    r.add_argument("--task", default=None, choices=["binary", "multiclass", "regression"])
    r.add_argument("--unlabeled", action="store_true", help="隐藏流未给标签时使用")
    r.add_argument("--policy", default="rule", choices=["rule", "bandit", "llm"])
    r.add_argument("--config", default=None)
    r.add_argument("--name", default="run")
    r.add_argument("--quiet", action="store_true")
    r.add_argument("--pred-dir", default=None, help="把每批预测写到这个目录（提交用）")
    r.add_argument('--checkpoint', help='每个完整批次后保存可信本地检查点')
    r.add_argument('--resume', action='store_true', help='从指定检查点恢复，输入必须包含原始完整数据前缀')
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("cpu", help="CPU 工业模型续训（赛规的 10 轮协议）")
    c.add_argument("--package", default=None, help="单个模型包目录（含 baselines/、public/、checkpoint.npz）")
    c.add_argument("--packages-root", default=None,
                   help="包含多个模型包的目录；逐个跑完并汇总到 all-summary.json")
    c.add_argument("--out", required=True, help="输出目录；每轮一个子目录，最终写 final-checkpoint.npz")
    c.add_argument("--rounds", type=int, default=10)
    c.add_argument("--driver", default="table", choices=["table", "llm"],
                   help="table=固定轮换学习率；llm=PyTRiO 大模型逐轮决策")
    c.add_argument("--model", default="Qwen/Qwen3.8-27B")
    c.add_argument("--widen", default="auto",
                   help="第 1 轮前 Net2Net 加宽的宽度：auto=按 profile 用调优表，0=不改结构，或直接给整数")
    c.add_argument("--cloud", action="store_true", help="记录到 SwanLab")
    c.set_defaults(func=cmd_cpu)
    return p


def cmd_cpu(args) -> int:
    """赛规的 10 轮续训：从平台检查点开始，逐轮"读历史-做实验-依据结果决策"。"""
    from .cpu import LLMPolicy, TablePolicy, run_cpu_rounds

    if bool(args.package) == bool(args.packages_root):
        raise SystemExit("必须且只能提供 --package 或 --packages-root")

    def make_policy(_package):
        if args.driver != "llm":
            return TablePolicy()
        # 大模型不可用时保留可完成的本地策略，但逐轮记录实际决策来源。
        try:
            from .config import HarnessConfig
            from .pytrio_access import configure_credentials

            harness = HarnessConfig(credential_source="pytrio",
                                    base_url="https://pytrio.com/api/openai/v1",
                                    model=args.model, small_model=args.model,
                                    max_tokens=128, timeout_s=90, retries=0)
            configure_credentials(harness)
            return LLMPolicy(harness)
        except Exception as exc:
            print(f"[cpu] 大模型不可用（{type(exc).__name__}: {exc}），降级为轮换表策略", flush=True)
            return TablePolicy()

    def make_logger(package):
        if not args.cloud:
            return None
        from .config import LogConfig

        logger = ExperimentLogger(
            LogConfig(backend="swanlab", workspace="hequan", project="zhixunyinqing",
                      out_dir=str(Path(args.out) / "swanlab"), swanlab_mode="online"),
            run_name=f"cpu-{Path(package).name}")
        logger.start({"package": str(package), "rounds": args.rounds,
                      "driver": args.driver, "model": args.model},
                     {"protocol": "resume-from-accepted; gate decides accept/reject"})
        return logger

    widen_arg: Optional[object] = None
    if args.widen == "auto":
        from .cpu import DEFAULT_WIDEN

        widen_arg = DEFAULT_WIDEN
    elif str(args.widen).isdigit() and int(args.widen) > 0:
        widen_arg = int(args.widen)

    if args.packages_root:
        from .cpu import run_cpu_packages

        summary = run_cpu_packages(Path(args.packages_root), Path(args.out), args.rounds,
                                   policy_factory=make_policy,
                                   logger_factory=make_logger, widen_to=widen_arg)
        status = 1 if summary["failed"] else 0
    elif args.package:
        package = Path(args.package)
        logger = make_logger(package)
        try:
            summary = run_cpu_rounds(package, Path(args.out), args.rounds,
                                     make_policy(package), logger, widen_to=widen_arg)
        except Exception as exc:
            if logger is not None:
                logger.abort(f"{type(exc).__name__}: {exc}")
            raise
        status = 0
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return status


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

