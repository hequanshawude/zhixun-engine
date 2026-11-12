"""在留出集上评估训练出的策略模型：动作准确率 + 非法输出统计。

用途：
    训练 loss 降到 0.03 不代表可用——要看它输出的动作是否合法、是否与示范一致。
    这里的"非法"指动作不在该状态的动作掩码内，线上会被 validate_plan 打回并降级为规则。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selftrain.actions import ACTION_NAMES, plausible_actions
from selftrain.config import HarnessConfig
from selftrain.harness import PyTrioSamplingClient, parse_action


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="out/pytrio-training/holdout.jsonl")
    ap.add_argument("--model", default="trio://run_x6vhlyhhceca/sampler_weights/zhixun-policy-v1")
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--limit", type=int, default=0, help="只评估前 N 条，0 表示全部")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.holdout).open(encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    from selftrain.pytrio_access import configure_credentials

    cfg = HarnessConfig(credential_source="pytrio",
                        base_url="https://pytrio.com/api/openai/v1", model=args.model,
                        small_model=args.model, pytrio_base_model=args.base_model,
                        max_tokens=128, timeout_s=90, retries=0)
    configure_credentials(cfg)
    client = PyTrioSamplingClient(cfg, args.model)

    exact = 0
    illegal = 0
    unparsed = 0
    confusions: Counter = Counter()
    invalid_examples: list[str] = []
    tokens = 0

    for i, row in enumerate(rows, 1):
        state = json.loads(row["messages"][1]["content"].split("当前状态：")[1])
        kind = state["drift"]["kind"]
        allowed = set(plausible_actions(kind, bool(state.get("best_available"))))
        expected = json.loads(row["answer"])["action"]

        try:
            text, usage = client.chat(row["messages"])
            tokens += usage.total
        except Exception as exc:
            print(f"[{i}] 调用失败: {type(exc).__name__} {exc}")
            unparsed += 1
            continue

        obj = parse_action(text)
        if obj is None:
            unparsed += 1
            if len(invalid_examples) < 3:
                invalid_examples.append(f"非 JSON: {text[:120]!r}")
            continue

        action = str(obj.get("action", ""))
        if action not in ACTION_NAMES or action not in allowed:
            illegal += 1
            if len(invalid_examples) < 6:
                invalid_examples.append(
                    f"kind={kind} 期望={expected} 实际={action} 允许={sorted(allowed)}")
            continue
        if action == expected:
            exact += 1
        else:
            confusions[(expected, action)] += 1
        if i % 10 == 0:
            print(f"  ...{i}/{len(rows)}", flush=True)

    n = len(rows)
    print("\n" + "─" * 58)
    print(f"  样本数        : {n}")
    print(f"  合法且正确    : {exact} ({exact / n:.1%})")
    print(f"  合法但不同    : {n - exact - illegal - unparsed}")
    print(f"  动作非法/越权 : {illegal}")
    print(f"  无法解析      : {unparsed}")
    print(f"  推理 token    : {tokens}")
    print("─" * 58)
    if confusions:
        print("\n主要混淆（期望 → 实际）:")
        for (exp, got), cnt in confusions.most_common(5):
            print(f"  {exp} → {got}: {cnt}")
    if invalid_examples:
        print("\n问题样例:")
        for ex in invalid_examples:
            print("  " + ex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
