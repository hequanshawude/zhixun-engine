"""CPU 工业模型的 10 轮续训驱动（赛规核心任务的入口）。

赛规要求：
    工业模型从平台检查点开始，在 CPU 环境续训；每个数据集进行 10 轮优化；
    Harness 调用大模型读写代码、使用工具；评分对象是续训后的工业模型。

这个模块把"读历史 → 做实验 → 依据验证结果决策"固化成可执行入口，
镜像里通过 ``python -m selftrain cpu --package <目录> --out <目录>`` 调用。

设计约束：
    - 只读平台给的 public/train.jsonl 与 public/validation.jsonl，从不碰隐藏数据；
    - 每轮从"当前被采纳的检查点"继续，验证集更好才采纳，否则拒绝并回滚；
    - 大模型只决策超参（可选），是否采纳始终由验证集闸门说了算。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

PROFILE_METRIC = {"A": ("macro_f1", True), "B": ("rmse", False), "C": ("rmse", False)}
LR_TABLE = [0.003, 0.001, 0.0005, 0.002, 0.0003]
# 本地实验得到的每个数据集最优宽度：A 在 96/128 会过拟合，B/C 越宽越好（上限 128）。
# 平台包同为 A/B/C，可直接复用；未列出的 profile 不做加宽。
DEFAULT_WIDEN = {"A": 64, "B": 128, "C": 128}
EPOCH_CHOICES = [3, 5, 8]


@dataclass
class CpuPackage:
    """一个 CPU 模型包：代码、公开数据、起始检查点与基线报告。"""

    root: Path
    profile: str
    task: str
    metric: str
    higher_is_better: bool
    baseline: Dict[str, float]
    base_checkpoint: Path
    train: Path
    validation: Path
    baseline_learning_rate: float = 0.003

    @property
    def tag(self) -> str:
        return self.profile.lower()


def load_package(path: Path) -> CpuPackage:
    path = Path(path).resolve()
    required = ["baselines", "public/train.jsonl", "public/validation.jsonl",
                "checkpoint.npz", "training.json"]
    missing = [r for r in required if not (path / r).exists()]
    if missing:
        raise FileNotFoundError(f"{path} 缺少必需内容: {missing}")

    report = json.loads((path / "training.json").read_text(encoding="utf-8"))
    profile = str(report.get("profile", "")).upper()
    metric, higher = PROFILE_METRIC.get(profile, ("rmse", False))
    return CpuPackage(
        root=path, profile=profile, task=report.get("task", "regression"),
        metric=metric, higher_is_better=higher,
        baseline=report["validation_after"],
        base_checkpoint=path / "checkpoint.npz",
        train=path / "public/train.jsonl",
        validation=path / "public/validation.jsonl",
        baseline_learning_rate=float(report.get("learning_rate", 0.003)),
    )


# --------------------------------------------------------------------------
# 训练一轮
# --------------------------------------------------------------------------


def train_round(pkg: CpuPackage, start_ckpt: Path, out_dir: Path, epochs: int,
                lr: float, profile_json: Optional[Path] = None,
                timeout: int = 3600) -> Dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"输出目录已存在（每轮必须是新目录）: {out_dir}")
    cmd = [sys.executable, "-m", "baselines.train",
           "--profile", str(profile_json) if profile_json else pkg.profile,
           "--train", str(pkg.train), "--validation", str(pkg.validation),
           "--resume", str(start_ckpt),
           "--epochs", str(epochs), "--learning-rate", str(lr),
           "--output", str(out_dir), "--stage-id", out_dir.name]
    proc = subprocess.run(cmd, cwd=pkg.root, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{")), None)
    if proc.returncode != 0 or line is None:
        raise RuntimeError((proc.stderr or proc.stdout)[-400:])
    return json.loads(line)


# --------------------------------------------------------------------------
# 决策策略
# --------------------------------------------------------------------------


class TablePolicy:
    """没有大模型时的确定性策略：轮换学习率，epoch 固定。"""

    name = "table"

    def __init__(self) -> None:
        self.tokens = 0

    def choose(self, card: Dict[str, Any]) -> tuple[float, int, str]:
        r = card["round"]
        return LR_TABLE[(r - 1) % len(LR_TABLE)], 5, "固定轮换表"


class LLMPolicy:
    """让 PyTRiO 上的大模型根据历史账本选参数。调用失败自动降级为 TablePolicy。"""

    name = "llm"

    SYSTEM = """你是工业模型续训的策略控制器。每轮决定下一轮的学习率和 epoch 数。

训练是确定性的：用完全相同的参数重跑，结果逐位相同——所以每轮必须改变至少一个参数。

原则：
1. 上一轮被采纳 → 可沿相邻档位继续或小幅加大步长；
2. 上一轮被拒绝 → 不要重复同一组参数，换档位或降低学习率；
3. 学习率越小越保守，epoch 越多越容易过拟合；
4. 越接近轮次上限越保守；已收敛时继续训练只会更差。

只输出 JSON：{"learning_rate": <数值>, "epochs": <整数>, "reason": "<20字以内>"}"""

    def __init__(self, cfg, client=None, strict: bool = False) -> None:
        from .harness import PyTrioSamplingClient, LLMClient

        self.cfg = cfg
        self.tokens = 0
        self.last_source = "unavailable"
        self.strict = strict
        if client is not None:
            self.client = client
        elif cfg.credential_source == "pytrio":
            self.client = PyTrioSamplingClient(cfg, cfg.model)
        else:
            self.client = LLMClient(cfg, cfg.model)
        self.fallback = TablePolicy()

    def choose(self, card: Dict[str, Any]) -> tuple[float, int, str]:
        try:
            text, usage = self.client.chat(
                [{"role": "system", "content": self.SYSTEM},
                 {"role": "user", "content": json.dumps(card, ensure_ascii=False)}])
            self.tokens += usage.total
            match = re.search(r"\{.*\}", text, re.S)
            obj = json.loads(match.group(0)) if match else {}
            lr = float(obj.get("learning_rate"))
            ep = int(obj.get("epochs"))
            if lr not in LR_TABLE or ep not in EPOCH_CHOICES:
                raise ValueError("参数不在白名单")
            self.last_source = "llm"
            return lr, ep, str(obj.get("reason", ""))[:60]
        except Exception:
            if self.strict:
                raise
            self.last_source = "table_fallback"
            lr, ep, _ = self.fallback.choose(card)
            return lr, ep, "大模型不可用，回退轮换表"


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def _stability(ledger: List[Dict[str, Any]], pkg: CpuPackage,
               start_metric: float) -> Dict[str, Any]:
    """对应赛规客观分里的"最差阶段与恢复速度"。

    部署序列是单调不降的（只有更好才采纳），所以"最差阶段"指的是**起点基线**与
    **所有候选里最差的那次实验**；"恢复"指一串连续拒绝之后，第几轮重新拿到新的最优。
    """
    higher = pkg.higher_is_better
    values = [x["value"] for x in ledger]
    worst_candidate = min(values) if higher else max(values)

    streaks, current, recoveries = [], 0, []
    for x in ledger:
        if x["decision"] == "accept":
            if current:
                streaks.append(current)
                recoveries.append({"after_reject_streak": current, "recovered_at_round": x["round"]})
                current = 0
        else:
            current += 1
    if current:
        streaks.append(current)

    return {
        "deployed_worst": start_metric,
        "deployed_worst_note": "只采纳更好的候选，部署性能单调不降，最差点就是起点检查点",
        "worst_candidate": worst_candidate,
        "max_reject_streak": max(streaks) if streaks else 0,
        "recoveries": recoveries,
        "monotone_deployment": True,
    }


def run_cpu_packages(root: Path, out: Path, rounds: int = 10,
                     policy_factory=None, logger_factory=None,
                     widen_to: Optional[int | Dict[str, int]] = None) -> Dict[str, Any]:
    """一个目录下可能有多个模型包（A/B/C），逐个跑完并汇总。

    单个包失败**不能拖垮整轮**：平台只给有限次运行，一个包的结构异常不应让另外两个
    也没有成绩。失败的包在汇总里记成 error，其余照常完成。
    """
    root = Path(root).resolve()
    packages = sorted(p.parent for p in root.glob("*/training.json"))
    if not packages and (root / "training.json").exists():
        packages = [root]
    if not packages:
        raise FileNotFoundError(f"{root} 下没有找到任何模型包（缺 training.json）")

    combined: Dict[str, Any] = {}
    for pkg_path in packages:
        logger = None
        try:
            policy = policy_factory(pkg_path) if policy_factory else TablePolicy()
            logger = logger_factory(pkg_path) if logger_factory else None
            combined[pkg_path.name] = run_cpu_rounds(
                pkg_path, Path(out) / pkg_path.name, rounds, policy, logger, widen_to=widen_to)
        except Exception as exc:  # noqa: BLE001 - 单包失败要隔离
            if logger is not None:
                try:
                    logger.abort(f"{type(exc).__name__}: {exc}")
                except Exception:
                    pass
            combined[pkg_path.name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"[warn] {pkg_path.name} 失败，已隔离：{type(exc).__name__}: {exc}", flush=True)

    failed = [k for k, v in combined.items() if "error" in v]
    summary = {"packages": len(packages), "rounds_each": rounds,
               "failed": failed, "results": combined}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "all-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _prepare_package(pkg: CpuPackage, out: Path) -> CpuPackage:
    """把包复制到可写目录并打上"保留最优 epoch"补丁。

    平台下发的包是只读挂载，且 ``baselines/train.py`` 只存最后一个 epoch。
    不复制就打不了补丁，而补丁是本地验证过的收益来源之一。
    """
    from .patcher import ensure_best_epoch_patch

    work = out / "_package"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(pkg.root, work)
    status = ensure_best_epoch_patch(work / "baselines" / "train.py")
    print(f"[init] 工作副本 {work.name}，train.py 补丁: {status}", flush=True)
    return CpuPackage(
        root=work, profile=pkg.profile, task=pkg.task, metric=pkg.metric,
        higher_is_better=pkg.higher_is_better, baseline=pkg.baseline,
        base_checkpoint=work / "checkpoint.npz",
        train=work / "public" / "train.jsonl",
        validation=work / "public" / "validation.jsonl",
        baseline_learning_rate=pkg.baseline_learning_rate,
    )


def _widen(pkg: CpuPackage, out: Path, hidden: int) -> tuple[Path, Path]:
    """调用 ``selftrain.widen`` 在包内做 Net2Net 迁移，返回新检查点与配套 profile。"""
    target = out / "_widened"
    target.mkdir(parents=True, exist_ok=True)
    ckpt, prof = target / "checkpoint.npz", target / "profile.json"
    proc = subprocess.run(
        [sys.executable, "-m", "selftrain.widen", "--package", str(pkg.root),
         "--checkpoint", str(pkg.base_checkpoint), "--hidden", str(hidden),
         "--output", str(ckpt), "--profile-out", str(prof)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not ckpt.exists():
        raise RuntimeError(f"加宽失败: {(proc.stderr or proc.stdout)[-300:]}")
    return ckpt, prof


def run_cpu_rounds(package: Path, out: Path, rounds: int = 10,
                   policy: Any = None, logger: Any = None,
                   widen_to: Optional[int | Dict[str, int]] = None) -> Dict[str, Any]:
    """跑满 N 轮，返回最终采纳的检查点与逐轮账本。"""
    pkg = load_package(package)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    pkg = _prepare_package(pkg, out)
    policy = policy or TablePolicy()
    started = time.time()

    accepted = pkg.base_checkpoint
    best = pkg.baseline[pkg.metric]
    ledger: List[Dict[str, Any]] = []

    # 可选：第 1 轮之前先做 Net2Net 加宽（赛规允许，要求显式权重迁移）。
    # 本地实验证明加宽是这几个任务的主要收益来源，必须能在镜像内自动完成。
    profile_json: Optional[Path] = None
    width = widen_to.get(pkg.profile) if isinstance(widen_to, dict) else widen_to
    if width and width > 0:
        accepted, profile_json = _widen(pkg, out, width)
        print(f"[init] 已加宽到 hidden={width}，profile={profile_json.name}", flush=True)

    # 起点指标必须实测，不能直接信 training.json：
    # 如果传入的包已经带着本地调过的检查点，旧基线会让闸门把"比起点更差"误判成改进。
    # 用 lr=1e-8 跑一步，权重几乎不动，报告的 validation_after 就是起点的真实指标。
    probe_dir = out / "_probe"
    rep = train_round(pkg, accepted, probe_dir, 1, 1e-8, profile_json=profile_json)
    best = rep["validation_after"][pkg.metric]
    shutil.rmtree(probe_dir, ignore_errors=True)
    print(f"[init] 起点检查点实测 {pkg.metric}={best:.4f}（training.json 记录 "
          f"{pkg.baseline[pkg.metric]:.4f}）", flush=True)
    start_metric = best

    for r in range(1, rounds + 1):
        history = [{"r": x["round"], "lr": x["learning_rate"], "ep": x["epochs"],
                    "v": round(x["value"], 4), "ok": x["decision"] == "accept"}
                   for x in ledger[-5:]]
        card = {"dataset": pkg.profile, "metric": pkg.metric,
                "higher_is_better": pkg.higher_is_better,
                "baseline": round(pkg.baseline[pkg.metric], 4),
                "best_so_far": round(best, 4), "round": r, "rounds_left": rounds - r + 1,
                "history": history, "baseline_learning_rate": pkg.baseline_learning_rate,
                "allowed_learning_rate": LR_TABLE, "allowed_epochs": EPOCH_CHOICES}

        lr, epochs, reason = policy.choose(card)
        round_dir = out / "rounds" / f"r{r:02d}"
        rep = train_round(pkg, accepted, round_dir, epochs, lr, profile_json=profile_json)
        value = rep["validation_after"][pkg.metric]
        better = (value > best) if pkg.higher_is_better else (value < best)
        previous = best
        if better:
            best = value
            accepted = round_dir / "checkpoint.npz"

        record = {"round": r, "metric": pkg.metric, "value": value,
                  "previous_best": previous, "best_after": best,
                  "learning_rate": lr, "epochs": epochs,
                  "decision": "accept" if better else "reject",
                  "decided_by": getattr(policy, "last_source", getattr(policy, "name", "table")),
                  "reason": reason,
                  "checkpoint": str(accepted),
                  "selected_epoch": rep.get("selected_epoch"),
                  "validation": rep["validation_after"]}
        ledger.append(record)
        print(f"[round {r:02d}] lr={lr} ep={epochs} -> {pkg.metric}={value:.4f} "
              f"{'ACCEPT' if better else 'REJECT'} ({value - previous:+.4f})  {reason}",
              flush=True)
        if logger is not None:
            logger.log(r, {f"round_{pkg.metric}": value, "best": best,
                           "accepted": int(better), "tokens": getattr(policy, "tokens", 0)})
            logger.log_decision({"batch": r, "state": card, **record})

    final_path = out / "final-checkpoint.npz"
    final_path.write_bytes(Path(accepted).read_bytes())
    summary = {
        "profile": pkg.profile, "task": pkg.task, "metric": pkg.metric,
        "rounds": rounds, "accepted": sum(1 for x in ledger if x["decision"] == "accept"),
        "baseline": pkg.baseline[pkg.metric], "final": best,
        "delta": (best - pkg.baseline[pkg.metric]) if pkg.higher_is_better
                 else (pkg.baseline[pkg.metric] - best),
        "final_checkpoint": str(final_path),
        "decided_by": getattr(policy, "name", "table"),
        "llm_rounds": sum(x["decided_by"] == "llm" for x in ledger),
        "tokens": getattr(policy, "tokens", 0),
        "wall_clock_seconds": round(time.time() - started, 2),
        "start_metric": start_metric,
        "stability": _stability(ledger, pkg, start_metric),
    }
    if os.getenv("ZHIXUN_SCHEME_V2") == "1":
        fitting_script = Path(__file__).resolve().parents[1] / "scripts" / "scheme_v2.py"
        subprocess.run([sys.executable, str(fitting_script), "--profile", pkg.profile,
                        "--package", str(pkg.root), "--out", str(out)], check=True)
        selection = json.loads((out / "predictor" / "selection.json").read_text(encoding="utf-8"))
        summary["predictor"] = str(out / "predictor" / "predict_scheme.py")
        summary["predictor_selection"] = selection
        summary["wall_clock_seconds"] = round(time.time() - started, 2)
        # The official package remains mounted at /data. The writable copy is
        # only needed while fitting and would exceed the output size limit.
        shutil.rmtree(pkg.root / "public")
    (out / "rounds-ledger.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ledger), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    if logger is not None:
        logger.finish(summary)
    return summary
