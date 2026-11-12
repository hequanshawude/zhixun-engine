# PyTRiO Qwen3.8-27B Gas 对照（2026-11-10）

赛题允许直接使用 PyTRiO 的 `Qwen/Qwen3.8-27B` 驱动 Harness。20:59 的短请求恢复成功；随后一条留出策略提示在[官方采样 SDK 验证实验](https://swanlab.cn/@hequan/zhixunyinqing/runs/a1plvpru)中返回可解析 JSON。Linux 镜像里同样通过一条策略提示，[验证记录](https://swanlab.cn/@hequan/zhixunyinqing/runs/8c8ftk7q)。服务可用性会变化，运行前仍应检查。

## 接口修正

最初用 OpenAI 兼容接口、最多 512 输出 token 跑九批 Gas，只有 2 批得到合法决策；其余响应结构无效并触发本地规则回退。该次实验在[SwanLab](https://swanlab.cn/@hequan/zhixunyinqing/runs/dsxb8m5v)完整保留，平均 macro-F1 为 0.4944、消耗 8316 token，**不能代表 27B 的完整策略效果**。一条单独策略提示曾成功返回 JSON，因此不能把问题概括为服务不可用。

随后改用 PyTRiO 官方 `create_sampling_client`，通过 27B 分词器的对话模板设置 `enable_thinking=False`，去除输出末尾的结束标记后再解析 JSON。配置固定每次最多 128 输出 token，九批总预算 10,000 token；SDK 失败仍由 Harness 回退本地规则。

## 同数据结果

Gas 官方十批中的第一批用于初始化，后九批先预测、再根据本机标签模拟评估与更新。两种 LLM 策略使用同一批数据、相同 LightGBM 参数、随机种子 42 和 10,000 token 策略预算。

| 本机九批平均 macro-F1 | 分数 | 策略来源 | 推理 token |
|---|---:|---:|---:|
| 固定模型 | 0.3544 | 无 | 0 |
| PyTRiO 训练的 Qwen3.5-4B 权重 | 0.6113 | LLM 9/9 | 6109 |
| PyTRiO Qwen3.8-27B | **0.6241** | LLM 9/9 | 6156 |

27B 比本机固定模型高 0.2697，比训练权重方案高 0.0128；候选模型晋级 6 次。无推理回退、未知用量、训练执行失败或超时。完整记录见[SwanLab 九批实验](https://swanlab.cn/@hequan/zhixunyinqing/runs/gj1c62gs)和本地 `runs/gas-pytrio-27b/run-gas-20261110-210954-f5c615bb/`，配置见 `configs/gas-pytrio-27b.yaml`。

该结果只是本机带标签模拟，不是官方评分。训练权重方案仅做了 8 条合成样本、两步 LoRA 训练，这次差异也不足以证明 27B 在其他漂移流上必然更好。当前镜像依赖 PyTRiO 模型服务；正式统一离线环境是否允许该服务仍待赛方协议确认。

## 2026-11-11 复验（新代码 + 全部上云）

项目迁到 D 盘、加入 `gate.min_gain_rel` 与 `model.finetune_recency` 之后，重新验证 27B
两条链路。两次运行均走 PyTRiO 官方采样 SDK、`enable_thinking=False`，并**全部记录到 SwanLab**。

| 数据 | 批次数 | 决策来源 | 推理 token | 主指标 | 相对静态 | 云端记录 |
|---|---:|---|---:|---:|---:|---|
| XJTU-SY（轴承 RUL） | 3 | LLM 3/3 | 1,993 | MAE 341.4897 | ±0.0000 | [y5ft0tl2](https://swanlab.cn/@hequan/zhixunyinqing/runs/y5ft0tl2) |
| C-MAPSS FD001（独立闸门） | 10 | LLM 10/10 | 6,870 | MAE 26.4337 | −0.0343 | [tz5ro8u4](https://swanlab.cn/@hequan/zhixunyinqing/runs/tz5ro8u4) |

要点：

- **FD001 的 MAE 26.4337 与 11-10 记录逐位一致**，说明这轮闸门改动对旧配置没有影响
  （相对阈值默认 0，不启用）。
- 无解析回退、无执行超时、无未知用量。决策理由原文可在 `decisions.jsonl` 的
  `plan.rationale` 里查到，例如"性能无下降，漂移不显著"、"协变量漂移，加权重训"。
- 本轮共消耗 **8,863** 推理 token。
- 两条结论都仍是**本机带标签模拟**，不是官方评分；XJTU 只有 3 批，更只能算链路验证。

配置：`configs/xjtu-sy-27b-cloud.yaml`、`configs/cmapss-fd001-holdout-27b.yaml`。

## 镜像

当前准备镜像是 `zhixun-engine:preflight`，ID `sha256:fb4e2e04…`，1,070,592,958 字节
（2026-11-11 11:00 构建，含相对阈值与 `finetune_recency`）。镜像包含 PyTRiO SDK 与 4B、27B
分词器；断网时可以加载 27B 分词器，但调用 27B 模型仍需赛方许可的服务接口。
镜像里没有 API Key，也没有 27B 完整权重。

该轮项目自检有 80 项测试通过；后续加入 FD001 独立设备闸门后，最新自检为 81 项测试和六组本地演练通过，记录见 `out/preflight/report.json`。这验证通用流程与接入改动没有明显回归，不替代官方公开集评测。
