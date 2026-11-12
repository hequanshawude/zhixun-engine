"""让 Qwen 逐轮决定"下一轮怎么训"：读历史 → 选参数 → 执行 → 依据验证结果采纳/拒绝。

与 cpu_rounds.py 的区别：
    cpu_rounds.py 用固定学习率表驱动，大模型没有参与；
    这里每一轮的 (learning_rate, epochs) 都由 PyTRiO 上的 Qwen 根据历史账本决定，
    决策理由、Token 用量逐轮落盘，构成赛规要求的"读历史-做实验-依据结果决策"记录。

安全边界：
    模型只能从白名单里选参数（学习率 5 档、epoch 3/5/8），输出不合规就回退到规则；
    是否采纳仍由验证集闸门决定，模型说了不算。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"D:\智训引擎")))
sys.path.insert(0, str(Path(r"D:\智训引擎\scripts")))

from cpu_rounds import METRIC, OUT, PKG, ROOT, run_round  # noqa: E402
from selftrain.config import HarnessConfig, LogConfig  # noqa: E402
from selftrain.experiment import ExperimentLogger  # noqa: E402
from selftrain.harness import PyTrioSamplingClient  # noqa: E402
from selftrain.pytrio_access import configure_credentials  # noqa: E402

ALLOWED_LR = [0.003, 0.002, 0.001, 0.0005, 0.0003]
ALLOWED_EPOCHS = [3, 5, 8]

SYSTEM = """你是工业模型续训的策略控制器。每轮你要决定下一轮训练的学习率和 epoch 数。

背景：模型从平台检查点续训，公开验证集上越高/越低（看指标方向）越好。
训练是确定性的——**用完全相同的参数重跑，结果逐位相同**，所以每轮必须换参数才有信息量。

决策原则：
1. 上一轮被采纳（指标变好）→ 可以沿相邻档位继续，或加大一点步长。
2. 上一轮被拒绝（指标变差）→ 不要重复同一组参数，换档位或降低学习率。
3. 学习率越小越保守；epoch 越多越容易过拟合。
4. 越接近 10 轮上限，越应该保守（收敛后继续训只会更差）。
5. **每轮必须至少改变一个参数**；和上一轮相同的参数会得到逐位相同的结果，等于浪费一轮。
6. 前 3 轮建议探索不同档位（含较大学习率或较多 epoch），后段再收敛。

只输出 JSON：{"learning_rate": <数值>, "epochs": <整数>, "reason": "<20字以内>"}
"""


def state_card(profile, metric, higher, baseline, best, history, round_no, total, base_lr=None):
    recent = [{"r": h["round"], "lr": h["learning_rate"], "ep": h["epochs"],
               "v": round(h["value"], 4), "ok": h["decision"] == "accept"}
              for h in history[-5:]]
    return {
        "dataset": profile,
        "metric": metric,
        "higher_is_better": higher,
        "baseline": round(baseline, 4),
        "best_so_far": round(best, 4),
        "round": round_no,
        "rounds_left": total - round_no + 1,
        "history": recent,
        "baseline_learning_rate": base_lr,
        "allowed_learning_rate": ALLOWED_LR,
        "allowed_epochs": ALLOWED_EPOCHS,
    }


def ask(client, card):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(card, ensure_ascii=False)}]
    text, usage = client.chat(messages)
    # 不能用 parse_action：它要求 JSON 里必须有 action 字段，而这里输出的是参数对象
    match = re.search(r"\{.*\}", text, re.S)
    try:
        obj = json.loads(match.group(0)) if match else {}
    except Exception:
        obj = {}
    lr = obj.get("learning_rate", obj.get("params", {}).get("learning_rate"))
    ep = obj.get("epochs", obj.get("params", {}).get("epochs"))
    try:
        lr = float(lr)
    except (TypeError, ValueError):
        lr = None
    try:
        ep = int(ep)
    except (TypeError, ValueError):
        ep = None
    if lr is not None and not any(abs(lr - x) < 1e-9 for x in ALLOWED_LR):
        lr = None
    if ep not in ALLOWED_EPOCHS:
        ep = None
    return lr, ep, str(obj.get("reason", ""))[:60], usage.total, text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--profiles", default="A,B,C")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--base-model", default=None, help="trio:// 权重时的基模")
    ap.add_argument("--cloud", action="store_true")
    ap.add_argument("--out-tag", default="llm")
    ap.add_argument("--profile-pattern", default=None,
                    help="改用自定义 profile JSON（加宽后 hidden 不同，必须配套）")
    ap.add_argument("--from-pattern", default=None,
                    help="起始检查点模式，例如 out/cpu-models/wide-{tag}.npz；缺省用随包基线")
    args = ap.parse_args()

    cfg = HarnessConfig(credential_source="pytrio", base_url="https://pytrio.com/api/openai/v1",
                        model=args.model, small_model=args.model,
                        pytrio_base_model=args.base_model or "Qwen/Qwen3.5-4B",
                        max_tokens=128, timeout_s=90, retries=0)
    configure_credentials(cfg)
    client = PyTrioSamplingClient(cfg, args.model)

    profiles = [p.strip().upper() for p in args.profiles.split(",") if p.strip()]
    summary, total_tokens = {}, 0

    for profile in profiles:
        tag = profile.lower()
        metric, higher = METRIC[profile]
        base_ckpt = PKG / tag / "checkpoint.npz"
        if args.from_pattern:
            candidate = (ROOT / args.from_pattern.format(tag=tag)).resolve()
            if not candidate.exists():
                raise SystemExit(f"起始检查点不存在: {candidate}")
            base_ckpt = candidate
        report = json.loads((PKG / tag / "training.json").read_text(encoding="utf-8"))
        if args.from_pattern:
            report["learning_rate"] = 0.001
        baseline = report["validation_after"][metric]
        best = baseline
        accepted = base_ckpt
        history = []

        logger = None
        if args.cloud:
            logger = ExperimentLogger(
                LogConfig(backend="swanlab", workspace="hequan", project="zhixunyinqing",
                          out_dir=f"runs/cpu-{tag}-{args.out_tag}", swanlab_mode="online"),
                run_name=f"cpu-{tag}-{args.out_tag}")
            logger.start({"profile": profile, "rounds": args.rounds, "driver": args.model,
                          "metric": metric, "baseline": baseline,
                          "protocol": "LLM picks (lr, epochs) each round; gate decides accept"},
                         {"name": f"cpu-{profile}", "source": "platform baseline checkpoint"})

        print(f"\n=== {profile}：基线 {metric}={baseline:.4f}，驱动 {args.model} ===", flush=True)
        for r in range(1, args.rounds + 1):
            card = state_card(profile, metric, higher, baseline, best, history, r, args.rounds,
                              report.get("learning_rate"))
            lr, ep, reason, tokens, raw = ask(client, card)
            total_tokens += tokens
            source = "llm"
            if lr is None or ep is None:
                lr, ep, source = 0.001, 3, "fallback"
                reason = f"输出不合规，回退规则: {raw[:40]}"

            prof_arg = (str((ROOT / args.profile_pattern.format(tag=tag)).resolve())
                        if args.profile_pattern else profile)
            rep = run_round(prof_arg, tag, r, accepted, ep, lr)
            value = rep["validation_after"][metric]
            improved = (value > best) if higher else (value < best)
            prev = best
            if improved:
                best = value
                accepted = OUT / tag / f"r{r:02d}" / "checkpoint.npz"
            record = {"profile": profile, "round": r, "metric": metric, "value": value,
                      "previous_best": prev, "best_after": best,
                      "learning_rate": lr, "epochs": ep,
                      "decision": "accept" if improved else "reject",
                      "decided_by": source, "reason": reason, "tokens": tokens,
                      "rollback_to": str(accepted.relative_to(ROOT)).replace("\\", "/"),
                      "selected_epoch": rep.get("selected_epoch"),
                      "validation": rep["validation_after"]}
            history.append(record)
            delta = (value - prev) if higher else (prev - value)
            print(f"  round {r:02d} lr={lr} ep={ep} [{source}] → {metric}={value:.4f} "
                  f"{'采纳' if improved else '拒绝'} ({delta:+.4f})  「{reason}」", flush=True)
            if logger:
                logger.log(r, {f"round_{metric}": value, "round_delta": delta,
                               "best": best, "accepted": int(improved),
                               "tokens": tokens})

        (OUT / tag / f"rounds-ledger-{args.out_tag}.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in history), encoding="utf-8")
        acc = [h for h in history if h["decision"] == "accept"]
        summary[profile] = {"baseline": baseline, "final": best,
                            "accepted_round": acc[-1]["round"] if acc else 0,
                            "accepted": len(acc),
                            "checkpoint": str(accepted.relative_to(ROOT)).replace("\\", "/"),
                            "llm_decisions": sum(1 for h in history if h["decided_by"] == "llm"),
                            "tokens": sum(h["tokens"] for h in history)}
        print(f"  → 最终 {metric}={best:.4f}，采纳 {len(acc)} 轮", flush=True)
        if logger:
            logger.finish({"profile": profile, **summary[profile]})

    out = ROOT / "out" / "cpu-models" / f"llm-summary-{args.out_tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"driver": args.model, "total_tokens": total_tokens,
                               "models": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n总 Token: {total_tokens}\n汇总: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
