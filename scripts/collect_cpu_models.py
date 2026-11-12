"""汇总 10 轮续训产物：采纳的检查点、指标、逐轮决策与驱动方式。"""

import hashlib
import json
from pathlib import Path

ROOT = Path(r"D:\智训引擎")
OUT = ROOT / "out/cpu-models"
RUNS = ROOT / "runs/cpu"
LEDGER = "rounds-ledger-llm3.jsonl"

PROFILES = {"A": ("macro_f1", True), "B": ("rmse", False), "C": ("rmse", False)}
BASELINE = {"A": 0.7140297220788709, "B": 681.1598326043113, "C": 52.86857256611151}

OUT.mkdir(parents=True, exist_ok=True)
records = []
for profile, (metric, higher) in PROFILES.items():
    tag = profile.lower()
    ledger = [json.loads(l) for l in (RUNS / tag / LEDGER).open(encoding="utf-8") if l.strip()]
    accepted = [x for x in ledger if x["decision"] == "accept"]
    last = accepted[-1]
    ckpt = ROOT / last["rollback_to"]
    report = json.loads((ckpt.parent / "training.json").read_text(encoding="utf-8"))
    digest = "sha256:" + hashlib.sha256(ckpt.read_bytes()).hexdigest()
    after, base = last["value"], BASELINE[profile]
    llm_rounds = [x for x in ledger if x["decided_by"] == "llm"]
    records.append({
        "profile": profile,
        "task": report["task"],
        "rounds": len(ledger),
        "accepted_rounds": len(accepted),
        "accepted_round": last["round"],
        "driver": "Qwen/Qwen3.8-27B via PyTRiO",
        "llm_decided_rounds": len(llm_rounds),
        "fallback_rounds": len(ledger) - len(llm_rounds),
        "checkpoint": str(ckpt.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": digest,
        "parent_checkpoint_sha256": report["parent_checkpoint_sha256"],
        "selected_epoch": last["selected_epoch"],
        "metric": metric,
        "baseline": base,
        "final": after,
        "delta": (after - base) if higher else (base - after),
        "tokens": sum(x["tokens"] for x in ledger),
        "round_trace": [{"round": x["round"], "lr": x["learning_rate"], "epochs": x["epochs"],
                         "value": round(x["value"], 4), "decision": x["decision"],
                         "by": x["decided_by"], "reason": x["reason"]} for x in ledger],
    })

# 对照：允许 Qwen 改写 train.py 的那轮实验
code_variant = {}
for profile in PROFILES:
    path = RUNS / profile.lower() / "rounds-ledger-code.jsonl"
    if not path.exists():
        continue
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    acc = [x for x in rows if x["decision"] == "accept"]
    code_variant[profile] = {
        "accepted": len(acc),
        "code_edits_accepted": sum(1 for x in acc if x.get("code_changed")),
        "final": acc[-1]["value"] if acc else BASELINE[profile],
    }

manifest = {
    "source_packages": "cpu-{a,b,c}-*-20261110-v1.zip",
    "protocol": "每数据集 10 轮；Qwen 逐轮决定 (learning_rate, epochs)，验证集闸门决定采纳/拒绝",
    "trainer_patch": "按公开验证集保留最优 epoch（scripts/patch_cpu_trainer.py）",
    "variants": {
        "hyperparameter_only": "Qwen 只选 (lr, epochs)，checkpoints 在 runs/cpu/<tag>/r*",
        "code_editing": "Qwen 可改写 baselines/train.py，须过单元测试与 1-epoch 冒烟，checkpoints 在 runs/cpu/<tag>/c*",
    },
    "code_editing_comparison": code_variant,
    "models": records,
}
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
for r in records:
    print(f'{r["profile"]}: {r["accepted_rounds"]}/{r["rounds"]} 轮采纳（LLM 决策 {r["llm_decided_rounds"]}）'
          f'  {r["metric"]}={r["final"]:.4f} (基线 {r["baseline"]:.4f}, {r["delta"]:+.4f})')
print("\n->", OUT / "manifest.json")
