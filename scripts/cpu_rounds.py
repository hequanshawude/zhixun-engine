"""按新赛规跑满 10 轮续训：每轮"读历史 → 做实验 → 依据验证结果决策"。

规则要点（2026-11-11 更新）：
    - 工业模型从平台检查点开始，在 CPU 环境续训；
    - 每个数据集进行 10 轮优化；
    - 评分对象是续训后的工业模型，不是 Harness 大模型；
    - 至少要有 3 次"读历史-做实验-依据结果决策"的完整记录。

因此这里不是简单地训 10 次，而是：
    每轮从**当前被采纳的检查点**继续 → 训练若干 epoch → 用公开验证集比较 →
    更好则采纳本轮、否则拒绝并保留上一轮。最终提交被采纳的检查点。
拒绝的轮次也保留在 runs/ 下，作为"实验过但未采纳"的证据。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:\智训引擎")
PY = ROOT / ".venv" / "Scripts" / "python.exe"
PKG = ROOT / "data" / "raw" / "cpu"
OUT = ROOT / "runs" / "cpu"
METRIC = {"A": ("macro_f1", True), "B": ("rmse", False), "C": ("rmse", False)}


def run_round(profile: str, tag: str, round_no: int, start_ckpt: Path,
              epochs: int, lr: float | None) -> dict:
    out = OUT / tag / f"r{round_no:02d}"
    if out.exists():
        raise SystemExit(f"{out} 已存在；请先清理再重跑")
    cmd = [str(PY), "-m", "baselines.train", "--profile", profile,
           "--train", "public/train.jsonl", "--validation", "public/validation.jsonl",
           "--resume" if round_no > 1 else "--from-baseline", str(start_ckpt),
           "--epochs", str(epochs), "--output", str(out), "--stage-id", f"round-{round_no:02d}"]
    if lr is not None:
        cmd += ["--learning-rate", str(lr)]
    proc = subprocess.run(cmd, cwd=PKG / tag, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{")), None)
    if proc.returncode != 0 or line is None:
        raise RuntimeError(f"{profile} round {round_no} 失败: {(proc.stderr or proc.stdout)[-300:]}")
    return json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=3, help="每轮训练的 epoch 数")
    ap.add_argument("--profiles", default="A,B,C")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lr-schedule", default="0.003,0.001,0.0005,0.002,0.0003",
                    help="逐轮轮换的学习率。确定性训练下，同参数重跑结果逐位相同，"
                         "必须换参数才能让每一轮成为真正的实验")
    ap.add_argument("--cloud", action="store_true", help="每轮记录到 SwanLab")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from selftrain.config import LogConfig
    from selftrain.experiment import ExperimentLogger

    profiles = [p.strip().upper() for p in args.profiles.split(",") if p.strip()]
    schedule = [float(x) for x in args.lr_schedule.split(",") if x.strip()] or [None]
    summary = {}

    for profile in profiles:
        tag = profile.lower()
        metric_name, higher = METRIC[profile]
        base_ckpt = PKG / tag / "checkpoint.npz"

        logger = None
        if args.cloud:
            logger = ExperimentLogger(
                LogConfig(backend="swanlab", workspace="hequan", project="zhixunyinqing",
                          out_dir=f"runs/cpu-{tag}-cloud", swanlab_mode="online"),
                run_name=f"cpu-{tag}-rounds")
            logger.start({"profile": profile, "rounds": args.rounds, "epochs": args.epochs,
                          "metric": metric_name, "higher_is_better": higher,
                          "protocol": "resume-from-accepted, accept only if validation improves"},
                         {"name": f"cpu-{profile}", "source": "platform baseline checkpoint"})

        ledger = []
        accepted = base_ckpt
        baseline_report = json.loads((PKG / tag / "training.json").read_text(encoding="utf-8"))
        best = baseline_report["validation_after"][metric_name]
        print(f"\n=== {profile}：基线 {metric_name}={best:.4f} ===", flush=True)

        for r in range(1, args.rounds + 1):
            lr = args.lr if args.lr is not None else schedule[(r - 1) % len(schedule)]
            rep = run_round(profile, tag, r, accepted, args.epochs, lr)
            value = rep["validation_after"][metric_name]
            improved = (value > best) if higher else (value < best)
            prev_best = best
            if improved:
                best = value
                accepted = OUT / tag / f"r{r:02d}" / "checkpoint.npz"
            record = {
                "profile": profile, "round": r, "metric": metric_name,
                "value": value, "previous_best": prev_best, "best_after": best,
                "decision": "accept" if improved else "reject",
                "learning_rate": lr,
                "rollback_to": str(accepted.relative_to(ROOT)).replace("\\", "/"),
                "epochs": f'{rep["initial_epochs"]}->{rep["final_epochs"]}',
                "steps": f'{rep["initial_steps"]}->{rep["final_steps"]}',
                "selected_epoch": rep.get("selected_epoch"),
                "validation": rep["validation_after"],
            }
            ledger.append(record)
            delta = (value - prev_best) if higher else (prev_best - value)
            print(f"  round {r:02d} (lr={lr}): {metric_name}={value:.4f}  "
                  f"{'采纳' if improved else '拒绝'}  (最优 {best:.4f}, 变化 {delta:+.4f})", flush=True)
            if logger:
                logger.log(r, {f"round_{metric_name}": value, "round_delta": delta,
                               f"best_{metric_name}": best, "accepted": int(improved)})

        target = OUT / tag / "rounds-ledger.jsonl"
        target.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ledger),
                          encoding="utf-8")
        accepted_round = next((x["round"] for x in reversed(ledger)
                               if x["decision"] == "accept"), 0)
        summary[profile] = {"baseline": baseline_report["validation_after"][metric_name],
                            "final": best, "accepted_round": accepted_round,
                            "accepted_checkpoint": str(accepted.relative_to(ROOT)).replace("\\", "/"),
                            "accepted": sum(1 for x in ledger if x["decision"] == "accept"),
                            "rejected": sum(1 for x in ledger if x["decision"] == "reject")}
        print(f"  → 最终采纳 round-{accepted_round:02d}，{metric_name}={best:.4f}", flush=True)
        if logger:
            logger.finish({"profile": profile, **summary[profile]})

    out = ROOT / "out" / "cpu-models" / "rounds-summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
