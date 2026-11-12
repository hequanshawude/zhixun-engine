# 智训引擎 · 平台开放前准备版

本项目依据本地《智讯引擎赛题.pdf》开发，用于工业模型漂移流上的检测、归因、实验、晋级与回滚。当前是可运行的本地准备版，尚未完成官方平台验收。

2026-11-11 最新准备版：镜像默认入口已改为赛规的 **CPU 模型 10 轮续训**（`python -m selftrain cpu`），94 项测试通过。移交给其他 AI 时先读 [近期工作日志与交接](docs/项目工作日志与AI交接-2026-11-10.md)；归档内容和使用方法见 [交付包说明](docs/交付包说明.md)。

**最新状态（2026-11-11）**：赛题要求已更新，核心任务变为"从平台检查点续训三个 CPU 工业模型、每个 10 轮"。
当前三个数据集在**本机公开验证集**上均优于原包基线（A Macro-F1 +2.1%、B RMSE −11.5%、C RMSE −3.6%）；隐藏测试尚未验证。镜像内可全自动跑满 10 轮，断网时 Qwen 会退到表策略。
先读 [赛题要求更新](docs/赛题要求更新-2026-11-11.md) → [合规自查](docs/合规自查-2026-11-11.md) → [CPU 模型续训](docs/CPU模型续训.md) → [技术报告](docs/技术报告.md)。

**镜像怎么用**：

```bash
docker run --rm --network none -v <模型包目录>:/data:ro -v <输出目录>:/out zhixun-engine:preflight
# 自动：识别包 → Net2Net 加宽 → 实测起点 → 10 轮决策 → final-checkpoint.npz
```

**先读 [赛题要求更新](docs/赛题要求更新-2026-11-11.md)。** `run-cloud.cmd` 只运行旧的合成数据流演练；CPU 三包的 Qwen＋SwanLab 实测命令见 [接入核查](docs/Qwen与SwanLab接入核查-2026-11-11.md)。规则以官方通知和 SDK 为准，本机验证数字不是正式平台成绩。

CPU 续训的 Qwen 与 SwanLab 联调见 [11 月 11 日接入核查](docs/Qwen与SwanLab接入核查-2026-11-11.md)。本机运行需显式加 `--driver llm --cloud`；断网容器会降级为表策略，正式离线环境的模型及云端记录接口仍待赛方确认。

已通过 PyTRiO 实际训练 Qwen3.5-4B 权重，并用它完成 Gas 真实数据流本机对照，详见 [PyTRiO 训练权重与 Gas 对照](docs/PyTRiO训练权重与Gas对照.md)。正式离线评测如何访问该权重仍待官方平台协议确认。

赛题指定的 PyTRiO Qwen3.8-27B 服务现已恢复，并通过官方采样 SDK 完成 Gas 九批本机对照，详见 [Qwen27B 真实数据对照](docs/Qwen27B真实数据对照.md)。

C-MAPSS FD001 的设备分组回归回放与独立设备闸门复验已完成；27B 和规则策略的批均 MAE 仍略差于各自同切分的本地静态模型，详见 [FD001 真实数据对照](docs/C-MAPSS-FD001真实数据对照.md)。该结果已完整记录，不能作为官方基线达标证据。

## 本地运行（Windows）

项目已有 `.venv`，请使用这个环境，系统默认 Python 版本不符合要求。

```powershell
.\.venv\Scripts\python.exe -m selftrain demo --scenario mixed --batches 14 --pred-dir out/preds
.\.venv\Scripts\python.exe scripts/preflight.py
```

也可以双击根目录 `run-demo.cmd` 运行演示，或 `check-project.cmd` 执行自检。两者会自动选择项目环境。`requirements-windows.lock.txt` 保存本机已安装依赖版本。

新环境要求 Python 3.10 及以上：创建虚拟环境后安装 `requirements.txt`。检查脚本会运行测试、二分类/多分类/回归 × 规则/策略记忆共六组演练，并在 `out/preflight` 保存报告、测试输出、环境版本、代码哈希和逐批实验日志。

## 数据接入

临时 CSV 协议：流目录包含按名称排序的 `batch_000.csv`、`batch_001.csv` 等（必须补零），或者含 `batch` 列的 `all.csv`。数值特征列必须与训练集一致。`id/time/batch` 不作为特征。分类标签当前按数值编码处理，正式编码与样本 ID 输出规则需适配官方 SDK。

```powershell
.\.venv\Scripts\python.exe -m selftrain run --train-csv data/train.csv --data-dir data/stream --target y --task binary --pred-dir out/preds
# 无标签流：训练集必须独立提供
.\.venv\Scripts\python.exe -m selftrain run --train-csv data/train.csv --data-dir data/hidden --target y --task binary --unlabeled --pred-dir out/preds
```

提供 `--train-csv` 后，流中每批均输出预测；不提供时首批用于初始化，不输出该批预测。无标签模式只检测与预测，不训练、不评估；尚未实现延迟标签回传。不要把隐藏标签放入本地评估入口。

可额外提供 `--gate-csv`，用独立带标签设备集检查候选晋级；该集合不进入训练历史。若训练集与闸门集带 `id`，入口会拒绝设备编号重叠，并核对闸门集与流批次不重叠。此模式目前不支持检查点恢复。

当前处理顺序：旧模型预测并输出 → 检测漂移 → 在排除验证窗口与隔离间隔的历史数据上训练候选 → 在验证窗口评估 → 决定下一批模型 → 将本批加入历史。默认 `val_batches=1/purge_batches=0` 仍为本批验证；窗口包含当前批与最近的有标签历史批次，指标按合并样本计算，历史不足时暂不更新。回滚仍使用本批已验证更好的静态基线。细节见 [配置生效与交付验证](docs/配置生效与交付验证.md)。

## 框架

- `data.py`：合成流、CSV 适配；`drift.py/attribution.py`：分布漂移与规则归因。
- `actions.py/models.py`：校准、重加权、近期加权重训、全量重训、模型切换、回滚、保持。`finetune` 当前是加权重训，并非神经网络增量微调。
- `loop.py/evaluation.py`：先预测后学习、候选闸门、版本管理、静态基线。
- `harness.py/playbook.py`：LLM 决策、失败降级、带规则先验的策略记忆。PyTRiO 训练权重已通过官方采样 SDK 在本机验证。
- `budget.py`：预算预估和成本账本；`experiment.py`：本地日志与可选 SwanLab；`sink.py`：逐批 CSV 输出。

## 离线镜像演练

```powershell
docker build -t zhixun-engine:preflight .
docker run --rm --network none --cpus 4 --memory 8g -v "${PWD}/data:/data:ro" -v "${PWD}/out:/out" zhixun-engine:preflight run --data-dir /data/stream --train-csv /data/train.csv --pred-dir /out/preds --config /app/configs/offline.yaml
```

上面的 `run` 是旧批流入口；当前 CPU 三包赛题使用镜像默认的 `cpu` 入口。平台资源与交付约束以新赛规及最终 SDK 为准，见 [赛题要求更新](docs/赛题要求更新-2026-11-11.md)。本机现已安装 Docker Desktop；容器核查见 [接入核查](docs/Qwen与SwanLab接入核查-2026-11-11.md)。依赖在构建时安装；离线构建需同 Linux/Python/架构的 wheels。

## 一键运行

```bat
run-all.cmd                                   :: 完整跑一遍，全部上云
run-all.cmd --seeds 10 --skip-preflight       :: 少跑几个切分、跳过自检
run-all.cmd --local                           :: 只在本地记录，不联网
run-all.cmd --with-llm                        :: 额外跑 PyTRiO Qwen3.8-27B（消耗额度）
run-all.cmd --dry-run                         :: 只打印要执行的命令
```

等价于 `python scripts/run_all.py`（Linux/容器里直接用它）。默认按顺序执行：

| 步骤 | 内容 | 约耗时 |
|---|---|---:|
| ① 自检 | 91 项测试 + 六组合成流演练 | 2 min |
| ② XJTU-SY 基线 | 规则策略 3 批回放 | 10 s |
| ③ C-MAPSS FD001 基线 | 规则策略 10 批 + 独立闸门 | 20 s |
| ④ XJTU-SY 配对验证 | 30 个随机切分 | 3.5 min |
| ⑤ FD001 配对验证 | 30 个随机切分 | 9 min |

②③④⑤ 都会写入 SwanLab，结束打印汇总表（步骤 / 状态 / 用时 / 实验链接），
并落一份 `out/run-all/summary-<时间戳>.json`。任何一步失败都不中断后续，最终返回非零退出码。

27B 默认**不跑**：每次约 7k token，容易被误触消耗额度，必须显式加 `--with-llm`。

## 实验记录与限制

已支持规则/探索策略的按批检查点与中断恢复，使用 `--checkpoint` 保存、`--resume` 恢复，详见 [中断恢复说明](docs/中断恢复说明.md)。63 项测试通过，独立进程恢复后的 5 批预测与连续运行一致。检查点仅用于可信本地文件，正式平台恢复协议仍待 SDK 接入。

模型回复解析、费用记录与故障降级已补齐，57 项检查通过；本机和断网容器各通过 6 项模拟接口验证，见 [模型服务容错验证](docs/模型服务容错验证.md)。LLM 失败时降级为固定规则，探索策略仅用于显式实验；正式模型服务尚未接入。

训练执行已增加独立进程超时保护，初始化耗时计入总成本。31 项检查和三类任务、两类故障的云端验收结果见 [运行保护验证](docs/运行保护验证.md)。默认首次训练限时 120 秒、候选限时 60 秒，需按官方要求调整；这不等于整批处理的硬时限。

首轮 36 次对照见 [多场景测试结论](docs/多场景测试结论.md)。随后接入策略探索，又完成新种子的 36 次云端对照和读回核验，详见 [策略探索验证](docs/策略探索验证.md)：探索相对规则 4 胜、1 平、13 负，默认仍使用规则，探索版作为实验选项。双击 `run-benchmark.cmd` 可重新运行原种子矩阵，新种子复现命令见验证报告。

本地输出包含 `meta.json`、`records.jsonl`、`decisions.jsonl`、`summary.json`、`playbook.json`。配置中的 API 密钥会脱敏，运行目录带随机后缀防止同秒冲突。

赛题要求全部训练、调整、评估实验按规范在云端完整记录；本地日志只能作为保底，不能替代云端链接。SwanLab 当前上传数值指标、脱敏配置、逐批决策与结果摘要；正式实验归档字段仍需按官方规范验收。默认离线配置不访问云端；开发期启用云端前需配置账号。

本地静态基线是本项目初始化模型，不等于官方基线。预算包含执行前估算与候选训练进程限时，尚未限制整个批次的全部处理时间；本地耗时与累计候选增益也不是官方成本/收益口径。正式平台开放后优先按准备清单完成这些项目。

