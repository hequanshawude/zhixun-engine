"""PyTRiO 上的 Qwen3.5-4B 策略 LoRA 训练。

两条数据来源：
    --dataset out/pytrio-training/train.jsonl   真实回放导出的状态卡（推荐）
    缺省                                        内置合成样例，仅用于链路自检

赛题规定驱动 Harness 的模型必须是选手在 PyTRiO 训练得到的，或官方 27B。
训练产出的 `trio://.../sampler_weights/<name>` 可以直接填进 harness.model。

用法：
    python scripts/train_pytrio_policy.py --dry-run                  # 只准备数据，不烧额度
    python scripts/train_pytrio_policy.py --train --steps 60 --name zhixun-policy-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from selftrain.config import LogConfig
from selftrain.experiment import ExperimentLogger
from selftrain.harness import RulePolicy, build_messages
from selftrain.types import DriftKind, Fingerprint

BASE_MODEL = "Qwen/Qwen3.5-4B"
OUTPUT = Path("out/pytrio-training")
MAX_BATCH_TOKENS = 8000


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{path} 是空的")
    return rows


def synthetic_examples() -> tuple[list[dict], list[dict]]:
    """无数据集时的兜底：规则生成的 50 个状态，40 训练 / 10 留出。"""
    rows = []
    for kind in (DriftKind.NONE, DriftKind.LABEL, DriftKind.COVARIATE,
                 DriftKind.CONCEPT, DriftKind.MIXED):
        for index in range(10):
            severity = round(0.12 + index * 0.075, 3)
            drop = round(index * 0.034, 3)
            fp = Fingerprint(kind, severity, drop, ["f001", "f002"],
                             prior_shift=round(index * 0.015, 3))
            state = {"batch": index + 1, "task": "multiclass", "drift": {
                "kind": kind.value, "severity": severity, "perf_drop": drop,
                "prior_shift": fp.prior_shift, "top_features": fp.top_features,
                "evidence": {}}, "anchor_metrics": {"f1_macro": round(.7 - drop, 3)},
                "recent_actions": [], "budget_left": {"compute_s": 600, "tokens": 6000},
                "best_available": False}
            action = RulePolicy().decide(fp, None, best_available=False).action
            rows.append({"messages": build_messages(state),
                         "answer": json.dumps({"action": action, "params": {}, "reason": "规则示范"},
                                              ensure_ascii=False)})
    return rows[:40], rows[40:]


def datum(row, tokenizer, trio):
    prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(row["answer"], add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        answer_ids.append(tokenizer.eos_token_id)
    tokens = prompt_ids + answer_ids
    weights = [0] * len(prompt_ids) + [1] * len(answer_ids)
    return trio.Datum(model_input=trio.ModelInput.from_ints(tokens=tokens[:-1]),
                      loss_fn_inputs={"weights": np.asarray(weights[1:], dtype=np.float32),
                                      "target_tokens": np.asarray(tokens[1:], dtype=np.int32)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="真正发起训练（会消耗 PyTRiO 额度）")
    ap.add_argument("--dry-run", action="store_true", help="只准备数据并统计 token，不调用接口")
    ap.add_argument("--dataset", default=None, help="训练集 jsonl；缺省用内置合成样例")
    ap.add_argument("--holdout", default=None, help="留出集 jsonl（仅记录用）")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--name", default="zhixun-policy-v1")
    ap.add_argument("--base-model", default=BASE_MODEL)
    args = ap.parse_args()

    if args.dataset:
        train = load_dataset(Path(args.dataset))
        holdout = load_dataset(Path(args.holdout)) if args.holdout else []
        source = str(args.dataset)
    else:
        train, holdout = synthetic_examples()
        source = "内置合成样例"

    OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"训练集 {len(train)} 条（{source}），留出 {len(holdout)} 条，"
          f"计划 {args.steps} 步 × batch {args.batch_size}")

    if args.dry_run or not args.train:
        print("未指定 --train，只做数据检查。")
        for row in train[:2]:
            print("  示例输入尾部:", row["messages"][-1]["content"][-160:].replace("\n", " "))
            print("  示例输出:", row["answer"])
        return 0

    import pytrio as trio

    client = trio.ServiceClient()
    caps = {m.model_name: m for m in client.get_server_capabilities().supported_models}
    if args.base_model not in caps or not caps[args.base_model].training.available:
        raise RuntimeError(f"{args.base_model} 当前不可训练（账号或服务未开放）")

    logger = ExperimentLogger(
        LogConfig(backend="swanlab", workspace="hequan", project="zhixunyinqing",
                  out_dir="runs/pytrio-training", swanlab_mode="online"),
        run_name=f"pytrio-policy-sft-{args.name}")
    logger.start({"base_model": args.base_model, "rank": args.rank, "steps": args.steps,
                  "batch_size": args.batch_size, "learning_rate": args.lr, "seed": 42,
                  "dataset": source, "train_examples": len(train),
                  "holdout_examples": len(holdout)},
                 {"name": args.name, "label_source": "rule-policy behaviour cloning",
                  "note": "状态卡来自真实回放（XJTU-SY / C-MAPSS / 合成流），未使用任何隐藏或官方标签"})

    try:
        print("创建训练客户端", flush=True)
        trainer = client.create_lora_training_client(base_model=args.base_model,
                                                     rank=args.rank, seed=42)
        tokenizer = trainer.get_tokenizer()

        for step in range(args.steps):
            start = (step * args.batch_size) % len(train)
            batch = [train[(start + i) % len(train)] for i in range(args.batch_size)]
            data = [datum(row, tokenizer, trio) for row in batch]
            token_count = sum(len(d.model_input) for d in data)
            if token_count > MAX_BATCH_TOKENS:
                raise ValueError(f"单步 {token_count} tokens 超过 {MAX_BATCH_TOKENS} 上限")

            result = trainer.forward_backward(data, "cross_entropy").result()
            trainer.optim_step(trio.AdamParams(learning_rate=args.lr)).result()
            loss = float(result.metrics.get("loss_mean", 0.0))
            logger.log(step + 1, {"loss_mean": loss, "input_tokens": token_count})
            if (step + 1) % 10 == 0 or step == 0:
                print(f"  step {step + 1}/{args.steps}: {token_count} tokens, loss={loss:.4f}",
                      flush=True)

        saved = trainer.save_weights_for_sampler(name=args.name).result()
        summary = {"status": "trained", "base_model": args.base_model,
                   "model_path": saved.path, "train_examples": len(train),
                   "holdout_examples": len(holdout), "steps": args.steps, "rank": args.rank}
        logger.finish(summary)
        (OUTPUT / "result.json").write_text(json.dumps(
            {**summary, "cloud_url": logger.cloud_url, "run_dir": str(logger.out_dir)},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n模型路径:", saved.path)
        print("云端实验:", logger.cloud_url)
        return 0
    except Exception as exc:
        logger.abort(type(exc).__name__)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
