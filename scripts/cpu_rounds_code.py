"""让 Qwen 真的改训练代码，而不只是调超参。

每轮流程：
    读当前 train.py → Qwen 给出 search/replace 补丁 + 超参 + 理由
    → 在沙箱副本里应用补丁 → 跑包自带的单元测试
    → 测试不过：回滚，本轮作废
    → 测试通过：用打补丁的代码训练 → 公开验证集比较 → 更好才同时采纳"代码 + 检查点"

安全边界：
    - 只允许改 baselines/train.py，且必须是"找得到、只出现一次"的精确替换；
    - 每轮最多 3 处修改；
    - 补丁后单元测试（含 Adam 状态、续训等价、数据边界）必须全绿；
    - 被拒绝的轮次代码回滚到上一次采纳的版本，不会累积未经验证的改动。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:\智训引擎")
PY = ROOT / ".venv" / "Scripts" / "python.exe"
PKG = ROOT / "data" / "raw" / "cpu"
SANDBOX = ROOT / "runs" / "cpu-code"
OUT = ROOT / "runs" / "cpu"

sys.path.insert(0, str(ROOT))

from selftrain.config import HarnessConfig, LogConfig  # noqa: E402
from selftrain.experiment import ExperimentLogger  # noqa: E402
from selftrain.harness import PyTrioSamplingClient  # noqa: E402
from selftrain.pytrio_access import configure_credentials  # noqa: E402

METRIC = {"A": ("macro_f1", True), "B": ("rmse", False), "C": ("rmse", False)}
MAX_EDITS = 3

SYSTEM = """你在维护一个 CPU 工业模型的训练代码。每轮你可以修改 baselines/train.py 的训练逻辑，
并给出下一轮的学习率和 epoch 数。

可用手段：
1. search/replace 补丁：给出原文片段和新片段，原文必须在文件里**只出现一次**。
2. 学习率（0.003/0.002/0.001/0.0005/0.0003 之一）与 epoch（3/5/8 之一）。

硬性约束：
- 不能破坏 checkpoint 的保存/加载格式、随机种子、标准化参数和数据边界；
- 改完必须能通过该包自带的单元测试（含"续训保持 Adam 状态且与连续训练等价"）；
- 每轮最多 3 处修改；不需要改代码时 edits 留空数组。

经验：训练是确定性的，同参数重跑结果逐位相同，所以每轮必须至少换一个参数或改一处逻辑。

可参考的代码改法（不限于此）：
- 在训练循环里加入学习率衰减（例如每 epoch 乘 0.7）；
- 调整梯度裁剪阈值或损失计算方式；
- 对输入特征做裁剪/归一化的稳健化处理；
- 早停：若本轮验证指标不再改善，提前结束剩余 epoch。

**前三轮至少尝试一次真实的代码修改**（edits 非空）。

只输出 JSON：
{"edits":[{"find":"<原文>","replace":"<新文>"}],"learning_rate":<数值>,"epochs":<整数>,"reason":"<30字以内>"}
"""


def apply_edits(text: str, edits: list) -> tuple[str, list]:
    applied = []
    for e in edits[:MAX_EDITS]:
        find, repl = e.get("find"), e.get("replace")
        if not isinstance(find, str) or not isinstance(repl, str) or not find:
            continue
        if text.count(find) != 1:
            return text, [{"error": f"find 出现 {text.count(find)} 次，跳过", "find": find[:60]}]
        text = text.replace(find, repl, 1)
        applied.append({"find": find[:120], "replace": repl[:120]})
    return text, applied


def run_tests(sandbox: Path) -> tuple[bool, str]:
    proc = subprocess.run([str(PY), "-m", "unittest", "discover", "-s", "baselines/tests", "-q"],
                          cwd=sandbox, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    return proc.returncode == 0, (proc.stderr or proc.stdout)[-300:]


def run_round(sandbox: Path, profile: str, round_no: int, start_ckpt: Path,
              epochs: int, lr: float, out_dir: Path) -> dict:
    cmd = [str(PY), "-m", "baselines.train", "--profile", profile,
           "--train", "public/train.jsonl", "--validation", "public/validation.jsonl",
           "--resume" if round_no > 1 else "--from-baseline", str(start_ckpt),
           "--epochs", str(epochs), "--learning-rate", str(lr),
           "--output", str(out_dir), "--stage-id", f"code-round-{round_no:02d}"]
    proc = subprocess.run(cmd, cwd=sandbox, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{")), None)
    if proc.returncode != 0 or line is None:
        raise RuntimeError((proc.stderr or proc.stdout)[-300:])
    return json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--profiles", default="A,B,C")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--cloud", action="store_true")
    args = ap.parse_args()

    cfg = HarnessConfig(credential_source="pytrio", base_url="https://pytrio.com/api/openai/v1",
                        model=args.model, small_model=args.model, max_tokens=700,
                        timeout_s=120, retries=0)
    configure_credentials(cfg)
    client = PyTrioSamplingClient(cfg, args.model)
    tokens_total = 0
    summary = {}

    for profile in [p.strip().upper() for p in args.profiles.split(",") if p.strip()]:
        tag = profile.lower()
        metric, higher = METRIC[profile]
        report = json.loads((PKG / tag / "training.json").read_text(encoding="utf-8"))
        baseline = report["validation_after"][metric]

        sandbox = SANDBOX / tag
        if sandbox.exists():
            shutil.rmtree(sandbox)
        sandbox.mkdir(parents=True)
        for item in (PKG / tag).iterdir():
            if item.name in ("baselines", "public", "checkpoint.npz"):
                (shutil.copytree(item, sandbox / item.name) if item.is_dir()
                 else shutil.copy2(item, sandbox / item.name))
        train_py = sandbox / "baselines" / "train.py"
        accepted_src = train_py.read_text(encoding="utf-8")
        accepted_ckpt = sandbox / "checkpoint.npz"
        best = baseline
        history, code_rejects = [], 0

        logger = None
        if args.cloud:
            logger = ExperimentLogger(
                LogConfig(backend="swanlab", workspace="hequan", project="zhixunyinqing",
                          out_dir=f"runs/cpu-{tag}-code", swanlab_mode="online"),
                run_name=f"cpu-{tag}-coderounds")
            logger.start({"profile": profile, "rounds": args.rounds, "driver": args.model,
                          "mode": "LLM 可修改 baselines/train.py（补丁 + 单元测试闸门）",
                          "baseline": baseline},
                         {"name": f"cpu-{profile}", "source": "platform baseline checkpoint"})

        print(f"\n=== {profile}：基线 {metric}={baseline:.4f}，Qwen 可改代码 ===", flush=True)
        for r in range(1, args.rounds + 1):
            source = train_py.read_text(encoding="utf-8")
            card = {
                "dataset": profile, "metric": metric, "higher_is_better": higher,
                "baseline": round(baseline, 4), "best_so_far": round(best, 4),
                "round": r, "rounds_left": args.rounds - r + 1,
                "baseline_learning_rate": report.get("learning_rate"),
                "history": [{"r": h["round"], "lr": h["learning_rate"], "ep": h["epochs"],
                             "v": round(h["value"], 4), "ok": h["decision"] == "accept",
                             "edited": bool(h.get("edits"))} for h in history[-5:]],
                "train_py": source,
            }
            text, usage = client.chat([{"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": json.dumps(card, ensure_ascii=False)}])
            tokens_total += usage.total
            m = re.search(r"\{.*\}", text, re.S)
            try:
                obj = json.loads(m.group(0)) if m else {}
            except Exception:
                obj = {}
            lr = obj.get("learning_rate")
            ep = obj.get("epochs")
            lr = float(lr) if isinstance(lr, (int, float)) else 0.001
            ep = int(ep) if ep in (3, 5, 8) else 5
            reason = str(obj.get("reason", ""))[:60]
            edits = obj.get("edits") if isinstance(obj.get("edits"), list) else []

            new_src, applied = apply_edits(accepted_src, edits)
            changed = new_src != accepted_src
            train_py.write_text(new_src if changed else accepted_src, encoding="utf-8")

            tests_ok, tests_msg = (True, "未改代码")
            if changed:
                tests_ok, tests_msg = run_tests(sandbox)
                if tests_ok:
                    # 单元测试覆盖不到调用链（例如改了 train_run 签名），补一次 1 epoch 冒烟
                    smoke_dir = ROOT / "runs" / "_smoke" / f"{tag}-r{r:02d}"
                    shutil.rmtree(smoke_dir, ignore_errors=True)
                    try:
                        run_round(sandbox, profile, 1, accepted_ckpt, 1, lr, smoke_dir)
                    except Exception as exc:
                        tests_ok, tests_msg = False, f"冒烟训练失败: {str(exc)[-140:]}"
                    finally:
                        shutil.rmtree(smoke_dir, ignore_errors=True)
            if changed and not tests_ok:
                code_rejects += 1
                train_py.write_text(accepted_src, encoding="utf-8")
                history.append({"round": r, "value": float("nan"), "decision": "code-reject",
                                "learning_rate": lr, "epochs": ep, "edits": applied,
                                "reason": f"单元测试失败: {tests_msg[-120:]}", "tokens": usage.total})
                print(f"  round {r:02d} 代码补丁被测试拒绝", flush=True)
                if logger:
                    logger.log(r, {"code_rejected": 1, "tokens": usage.total})
                continue

            out_dir = OUT / tag / f"c{r:02d}"
            try:
                rep = run_round(sandbox, profile, r, accepted_ckpt, ep, lr, out_dir)
            except Exception as exc:
                train_py.write_text(accepted_src, encoding="utf-8")
                history.append({"round": r, "value": float("nan"), "decision": "train-fail",
                                "learning_rate": lr, "epochs": ep, "edits": applied,
                                "reason": str(exc)[:120], "tokens": usage.total})
                print(f"  round {r:02d} 训练失败，已回滚", flush=True)
                continue

            value = rep["validation_after"][metric]
            improved = (value > best) if higher else (value < best)
            prev = best
            if improved:
                best = value
                accepted_ckpt = out_dir / "checkpoint.npz"
                if changed:
                    accepted_src = new_src
            else:
                train_py.write_text(accepted_src, encoding="utf-8")

            delta = (value - prev) if higher else (prev - value)
            record = {"round": r, "metric": metric, "value": value, "previous_best": prev,
                      "best_after": best, "learning_rate": lr, "epochs": ep,
                      "decision": "accept" if improved else "reject",
                      "edits": applied, "code_changed": changed, "tests": tests_ok,
                      "reason": reason, "tokens": usage.total,
                      "rollback_to": str(accepted_ckpt.relative_to(ROOT)).replace("\\", "/")}
            history.append(record)
            print(f"  round {r:02d} lr={lr} ep={ep}{' +代码' if changed else ''} → "
                  f"{metric}={value:.4f} {'采纳' if improved else '拒绝'} ({delta:+.4f})  「{reason}」",
                  flush=True)
            if logger:
                logger.log(r, {f"round_{metric}": value, "round_delta": delta,
                               "best": best, "accepted": int(improved),
                               "code_changed": int(changed), "tokens": usage.total})

        (OUT / tag / "rounds-ledger-code.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in history), encoding="utf-8")
        (OUT / tag / "accepted-train.py").write_text(accepted_src, encoding="utf-8")
        acc = [h for h in history if h["decision"] == "accept"]
        summary[profile] = {"baseline": baseline, "final": best,
                            "accepted": len(acc), "code_changes_accepted":
                                sum(1 for h in acc if h.get("code_changed")),
                            "code_rejects": code_rejects,
                            "checkpoint": str(accepted_ckpt.relative_to(ROOT)).replace("\\", "/")}
        print(f"  → 最终 {metric}={best:.4f}，采纳 {len(acc)} 轮，代码改动采纳 "
              f"{summary[profile]['code_changes_accepted']} 次", flush=True)
        if logger:
            logger.finish({"profile": profile, **summary[profile]})

    out = ROOT / "out" / "cpu-models" / "code-summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"driver": args.model, "tokens": tokens_total, "models": summary},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n总 Token: {tokens_total}\n汇总: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
