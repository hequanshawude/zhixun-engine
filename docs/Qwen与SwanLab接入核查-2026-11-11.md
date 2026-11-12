# CPU 续训入口的 Qwen 与 SwanLab 接入核查（2026-11-11）

本报告核查 `D:\智训引擎` 的实际执行路径。赛题新增的 A/B/C CPU 模型包是当前评分对象；早期 Gas/C-MAPSS 批流结果只作开发证据。所有本节数字来自本机公开验证集，**不是隐藏测试或正式平台成绩**。

## 修复的执行缺口

1. `selftrain cpu --packages-root ... --cloud` 原先把空的 `--package` 传给 `Path`，会在训练前失败；并且三包共用一条 SwanLab 记录，第一包结束后后续两包无法独立归档。现在按包创建独立策略实例和云端实验，任一包失败在汇总中列出并返回非零状态。
2. 原云端只写每轮数值，缺少完整的“读历史、选参数、训练、采纳/拒绝”决策。现在每轮将状态卡及决策写入本地 `decisions.jsonl`，同步为 SwanLab Text，并在账本记录实际来源 `llm` 或 `table_fallback`。
3. 稳定性摘要的 `worst_candidate` 对分类取了最高分、回归取了最低误差，方向相反；现已修正。历史摘要不自动改写。

## 实测证据

- 代码测试：**94 项通过**；新增多包独立云端记录、真实决策来源、最差候选方向的检查。
- A 包单轮 Qwen＋SwanLab 联调：Qwen 选学习率 0.001、5 epoch，使用 338 token，Macro-F1 0.7140→0.7187；[云端实验](https://swanlab.cn/@hequan/zhixunyinqing/runs/w2xjikcs)，本地 `out/cpu-qwen-swanlab-smoke-20261111/`。
- 三包各一轮表策略联调：A/B/C 各自建立并结束一条云端实验，分别见 [A](https://swanlab.cn/@hequan/zhixunyinqing/runs/x3wzbvbn)、[B](https://swanlab.cn/@hequan/zhixunyinqing/runs/nim8jqsj)、[C](https://swanlab.cn/@hequan/zhixunyinqing/runs/5ar5dpc2)。用于验证多包日志生命周期，不计作 Qwen 决策。
- **完整三包 ×10 轮**：A/B/C 共 30 轮，`llm_rounds=10/10` 各自成立，回退 0 次；每包有独立云端实验和本地逐轮账本。本次实测：

| 包 | 本次公开验证结果 | 原包基线 | Qwen token | SwanLab |
|---|---:|---:|---:|---|
| A 气体分类 Macro-F1 ↑ | 0.7279 | 0.7140 | 4,782 | [A 十轮](https://swanlab.cn/@hequan/zhixunyinqing/runs/o1pb08zs) |
| B 轴承 RUL RMSE ↓ | 625.67 | 681.16 | 4,876 | [B 十轮](https://swanlab.cn/@hequan/zhixunyinqing/runs/q5oq2zc6) |
| C 发动机 RUL RMSE ↓ | 51.56 | 52.87 | 4,831 | [C 十轮](https://swanlab.cn/@hequan/zhixunyinqing/runs/umurs5p5) |

逐轮文件在 `out/cpu-qwen-swanlab-full-20261111/{a,b,c}/rounds-ledger.jsonl`，汇总在同目录 `all-summary.json`，云端链接保存在 `swanlab/cpu-*/cloud.json`。实验记录中有完整状态卡、参数、结果、是否采纳、模型来源及累计 token。

**本次结果没有超过先前冻结的最佳检查点**：A 0.7292、B 602.55、C 50.98（见 `out/cpu-models/final.json`）。本次联调文件单独放在 `out/cpu-qwen-swanlab-full-20261111/`，未覆盖 `out/cpu-models/best/`。Qwen 部分轮次重复了已尝试参数，后续应在新验证集或官方协议下评估策略改进，不能反复调同一公开验证集后宣称稳健增益。

## 容器联调

- 当前镜像 `zhixun-engine:preflight` 挂载 A 包、输出目录及本机 PyTRiO/SwanLab 凭据（只读），限制 4 核/2 GiB，显式使用 `--driver llm --cloud`。`out/docker-cpu-online-audit/summary.json` 显示 `decided_by=llm`、`llm_rounds=1`、338 token、Macro-F1 0.7140→0.7187；[容器云端实验](https://swanlab.cn/@hequan/zhixunyinqing/runs/3xd1p444) 已生成，并有逐轮决策记录。这验证了本机有网络和凭据时的容器调用链。
- 同一镜像在 `--network none` 下单轮执行成功，但 `llm_rounds=0`、`decided_by=table`；证据在 `out/docker-cpu-cloud-audit/summary.json`。断网运行不能作为大模型 Harness 已接通的证据。

## 运行方式与边界

本机网络可用且已完成 PyTRiO、SwanLab 登录时：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe -m selftrain cpu --packages-root data/raw/cpu --out out/cpu-new-run --rounds 10 --driver llm --model Qwen/Qwen3.8-27B --widen auto --cloud
```

每次运行必须使用新的输出目录；命令会创建三条云端记录并消耗模型额度。本机登录状态不在交付包中。当前 Docker 默认命令含 `--driver llm`，但**没有** `--cloud`；运行于有网络且已安全注入凭据的环境时，须显式加 `--cloud`。断网容器会回退表策略，本机云端结果不能证明正式离线容器可调用 Qwen 或上传 SwanLab。正式提交前必须从赛方获取容器网络、凭据注入、云端归档/回传以及隐藏测试调用协议，按官方接口复验。演示视频也尚未录制。
