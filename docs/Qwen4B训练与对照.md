# 自训练 Qwen3.5-4B 策略模型（2026-11-11）

赛题原文：驱动 Harness 的大模型**必须**选用"选手通过 PyTRiO 平台训练得到的模型，或 PyTRiO 的
Qwen3.8-27B"。此前只有 8 条合成样本、2 步的链路验证，本文是第一次真正训练并上线评估。

## 1. 训练数据从哪来

不用凭空捏造的指纹，直接用主循环在真实回放里产生的**状态卡**：

| 来源 | 状态数 |
|---|---:|
| XJTU-SY（20 个随机切分 × 3 批） | 60 |
| C-MAPSS FD001（10 个切分 × 10 批） | 100 |
| 合成流（15 轮，覆盖弱/无/标签漂移） | 135 |
| 去重后 | **295**（251 训练 / 44 留出） |

标签是规则策略的动作（行为克隆）。动作分布：`finetune` 155、`rollback` 45、`reweight` 44、
`retrain` 32、`calibrate` 6、`noop` 13。

补合成流的原因：两个真实数据集上漂移都很强，规则几乎总是选 finetune/rollback，第一版数据里
**一个 noop 都没有**。"什么时候不该动"恰恰是最省成本的能力，必须补。

## 2. 训练

| 项 | 值 |
|---|---|
| 基模 | `Qwen/Qwen3.5-4B` |
| LoRA rank | 8 |
| 步数 × batch | 30 × 4（约 0.5 个 epoch） |
| 学习率 | 1e-4 |
| loss | 2.5134 → **0.0141** |
| 权重 | `trio://run_ifs2vac4k7l5/sampler_weights/zhixun-policy-v2` |
| 云端 | [0zsou8sd](https://swanlab.cn/@hequan/zhixunyinqing/runs/0zsou8sd) |

## 3. 关键修改：把可选动作写进状态卡

第一版（v1，60 步）在留出集上只有 **43.2%** 动作一致，还有 **9/44 条输出了非法动作**——
模型在 `mixed` 状态下输出了 `rollback`，而该状态的掩码里没有它。根因很简单：
**状态卡从来没告诉模型有哪些动作可选**，它只能凭"训练集里 rollback 很多"去赌。

修法是把 `allowed` 字段写进状态卡（`build_state_card`），并在系统提示里明确"只能从 allowed 里选"。
同时这也改善了 27B 的输入。

| 版本 | 步数 | 动作一致 | 非法动作 | 无法解析 |
|---|---:|---:|---:|---:|
| v1（无 allowed） | 60 | 43.2% | 9 | 0 |
| v2（含 allowed） | 30 | **54.5%** | **0** | 0 |

v2 剩余的混淆主要是 `finetune → reweight`（12 条）——两者都是"中等成本"动作，语义相近。

## 4. 上线结果

用 v2 权重驱动 Harness，跑两项真实数据（全部上云）：

| 策略 | XJTU-SY MAE | FD001 MAE | 推理 token（XJTU / FD001） | 决策来源 |
|---|---:|---:|---|---|
| 静态基线 | 341.4897 | 26.3993 | 0 / 0 | — |
| 固定规则 | 341.4897 | 26.5943 | 0 / 0 | 规则 |
| PyTRiO 27B | 341.4897 | 26.4337 | 1,993 / 6,870 | LLM 全成功 |
| **自训练 4B（v2）** | 341.4897 | **26.5108** | 2,081 / 7,342 | **LLM 全成功** |

- XJTU：3/3 由训练模型决策（v1 只有 1/3，其余因非法动作被降级）；[tq7txejx](https://swanlab.cn/@hequan/zhixunyinqing/runs/tq7txejx)
- FD001：10/10 全成功，MAE 26.5108（比规则好 0.08，比 27B 差 0.08，仍略差于静态）；[c39mhxg6](https://swanlab.cn/@hequan/zhixunyinqing/runs/c39mhxg6)

**没有一份结果优于静态基线。** 4B 只是"和规则差不多、比 27B 略差"，不能宣称性能优势。

## 5. 必须说清楚的局限

1. **不是干净的对照。** 27B 那组是在加 `allowed` 字段**之前**跑的，提示词不一样；要严格比较，
   27B 需要用新状态卡重跑一次（约 7k token）。
2. **留出集只有 44 条**，54.5% 的置信区间很宽，不足以判断模型好坏。
3. **训练 loss 0.014 是过拟合的信号**：251 条样本跑了 30 步，等于每条约 0.5 次。真正的瓶颈是
   数据量，不是步数。
4. **标签是规则的动作**，所以 4B 的上限大约就是规则的上限。要超过规则，得用"实际增益"做标签
   （只有 295 个状态里少数几个有明确胜负），或者用 27B 蒸馏。
5. **Token 反而更多**（7,342 vs 27B 的 6,870），因为状态卡多了 `allowed` 字段。4B 的省钱优势
   要等提示词稳定后再对比才成立。

## 6. 复现

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\build_policy_dataset.py --xjtu-seeds 20 --cmapss-seeds 10 --synth-runs 15
.\.venv\Scripts\python.exe scripts\train_pytrio_policy.py --train --dataset out/pytrio-training/train.jsonl --holdout out/pytrio-training/holdout.jsonl --steps 30 --name zhixun-policy-v2
.\.venv\Scripts\python.exe scripts\eval_policy_model.py --model trio://run_ifs2vac4k7l5/sampler_weights/zhixun-policy-v2
.\.venv\Scripts\python.exe -m selftrain run --data-dir data/processed/xjtu-sy/stream --train-csv data/processed/xjtu-sy/train.csv --target y --task regression --policy llm --config configs/xjtu-sy-4b-cloud.yaml
```

配置：`configs/xjtu-sy-4b-cloud.yaml`、`configs/cmapss-fd001-holdout-4b.yaml`。
权重只存在于 PyTRiO，通过 `trio://` 引用，不进镜像、不进交付包。
