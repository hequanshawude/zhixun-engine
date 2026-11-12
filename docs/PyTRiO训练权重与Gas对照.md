# PyTRiO 训练权重与 Gas 本机对照（2026-11-10）

## 对照赛题

本机赛题 PDF 第 3 页要求：驱动 Harness 的大模型须是选手通过 PyTRiO 平台训练得到的模型，或 PyTRiO 的 Qwen3.8-27B。本次训练时 27B 接口返回 `no available server`，因此使用前一条路线：在 PyTRiO 上训练 Qwen3.5-4B 的 LoRA 权重，再将该权重用于 Harness 决策。当天稍后 27B 服务恢复，后续对照见[Qwen27B 真实数据对照](Qwen27B真实数据对照.md)。正式离线评测是否允许访问 PyTRiO 远程权重，仍以赛事平台公布的运行协议为准。

## 训练来源和权重

- 基模：`Qwen/Qwen3.5-4B`；LoRA rank 8，学习率 `1e-4`，随机种子 42。
- 训练资料：本项目规则策略生成的 50 条合成漂移状态，40 条训练、10 条留出；本次链路验证只用前 8 条做 2 步训练。没有使用官方隐藏标签。
- 两步平均损失：2.1641 → 1.3102。损失下降只证明这次训练正常执行，不代表泛化性能。
- PyTRiO 权重：`trio://run_wowuibjwrvmu/sampler_weights/zhixun-policy-sft-smoke`。
- [SwanLab 训练记录](https://swanlab.cn/@hequan/zhixunyinqing/runs/chh8583c)；本地元数据：`out/pytrio-training/result.json`。
- 留出样本经 PyTRiO 官方采样接口生成合法 JSON 决策，动作 `noop` 与规则标签一致；这仅验证 1 条样本的调用链。[SwanLab 验证记录](https://swanlab.cn/@hequan/zhixunyinqing/runs/fuofe7fr)。

PyTRiO 的 OpenAI 兼容接口对该权重返回的消息内容未能通过现有客户端的文本解析；本机运行使用官方 SDK 的 `create_sampling_client` 加载保存的权重，并明确关闭模型思考输出。调用、训练和保存方式参见[官方推理文档](https://docs.pytrio.com/docs/guide/sample)及[官方训练文档](https://docs.pytrio.com/docs/guide/train)。

## Gas 数据流本机结果

使用[UCI Gas Sensor Array Drift Dataset](https://archive.ics.uci.edu/dataset/224/gas) 的十批数据：第一批用于初始化，后九批按顺序预测和更新。参数见 `configs/gas-pytrio-trained.yaml`。九次策略决策全部来自训练权重，回退 0 次，未知用量 0 次。完整结果保存在 `runs/gas-pytrio-trained/run-gas-20261110-155828-a9a83d90/`，并已上传到[SwanLab 实验](https://swanlab.cn/@hequan/zhixunyinqing/runs/xcubyjd9)。

| 九批平均 macro-F1 | 分数 |
|---|---:|
| 固定模型 | 0.3544 |
| 先前规则策略 | 0.4894 |
| 本次 PyTRiO 训练权重策略 | 0.6113 |

本次相对固定模型高 0.2569，相对先前规则策略高 0.1219；训练权重策略有 5 次模型晋级，使用 6109 个推理 token、约 34.8 秒本机计算时间。训练本身的云端消耗不包含在这些本机成本数字中，应以 PyTRiO 账单为准。以上是本机带标签模拟分数，不是赛事官方得分。决策前没有把当前批标签送给策略模型，但本机模拟会在预测后使用标签评估和更新；正式平台的标签反馈时序及成本权重尚待确认。样本规模、数据集、标签可见性改变时，结果可能明显不同。

## 复现

在项目根目录运行（需已安装 `requirements-pytrio.txt` 并完成 PyTRiO、SwanLab 登录）：

```powershell
.\.venv\Scripts\python.exe scripts\train_pytrio_policy.py --train
.\.venv\Scripts\python.exe scripts\verify_trained_pytrio.py --sdk
.\.venv\Scripts\python.exe -m selftrain run --data-dir data/processed/gas --target y --task multiclass --policy llm --config configs/gas-pytrio-trained.yaml --name gas-trained --quiet --pred-dir out/preds/gas-trained
```

训练脚本每次运行会生成新权重；复现新的训练后，应把 `out/pytrio-training/result.json` 中的新路径写入配置的 `model` 和 `small_model`。当前配置固定指向本次已验证的权重。不要把 API Key 写入配置或上传到仓库。
