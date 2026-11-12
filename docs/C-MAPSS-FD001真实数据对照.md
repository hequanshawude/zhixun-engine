# C-MAPSS FD001 发动机寿命回归对照（2026-11-10）

数据来自 [NASA C-MAPSS 公共集目录](https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data)。本机原始 ZIP 的 SHA-256、完整设备分组与批次行数见 `data/processed/cmapss-fd001/manifest.json`；复现脚本是 `scripts/prepare_cmapss.py`。这是赛前公开数据回放，不是官方平台评分。

## 数据边界

仅使用官方 `train_FD001.txt` 的 100 台发动机、20,631 条记录。固定种子 42 按发动机分组：70 台、14,407 条用于初始化；30 台、6,224 条分为十批，每批三台完整轨迹。初始化、各回放批次之间的发动机编号互不重叠。官方 `test_FD001.txt` 与 `RUL_FD001.txt` 均未参与训练、调参或本次评价。

目标为每台训练发动机的末周期减去当前周期，**不截断 RUL**。模型输入为已观测的周期、3 个工况设置及 21 个传感器，共 25 列；设备编号、末周期和目标值不作为输入。每批先预测，再在本机模拟器中解锁整批标签，用来评价和供以后批次更新。只有在完整轨迹结束后才能知道这些 RUL 标签，因此这是一种回顾性、按设备分组的回放，不能证明实时故障预测时可以及时得到标签。正式比赛须按赛方的反馈时点重做适配。

## 同切分结果

两次运行使用相同的 LightGBM 参数、种子 42、回放批次和云端实验记录。指标是十批**批均 MAE**，数值越低越好；本地静态模型是初始化后固定不更新的 LightGBM，不等于官方基线。

| 策略 | 十批批均 MAE | 相对本地静态模型 | 决策来源 | 候选晋级 |
|---|---:|---:|---:|---:|
| 本地静态模型 | **26.2437** | — | — | 0 |
| 固定规则 | 26.4924 | 差 0.2487 | 规则 10/10 | 2 |
| PyTRiO Qwen3.8-27B | 26.6328 | 差 0.3891 | LLM 10/10 | 5 |

27B 消耗 6,854 推理 token，无解析回退、未知用量、训练失败或超时；[27B 云端记录](https://swanlab.cn/@hequan/zhixunyinqing/runs/jsq0ix3y)。规则策略消耗 0 推理 token；[规则云端记录](https://swanlab.cn/@hequan/zhixunyinqing/runs/d66lmm8k)。本地摘要位于 `runs/cmapss-fd001-27b/` 下两次运行目录。

两种动态策略都没有越过本地静态模型。候选在当前设备批次通过晋级闸门，不能保证对下一批新设备有效；本机账本记录的累计候选增益也不等于十批实际预测增益。FD001 目前应以静态模型为保底，后续从训练设备内做更稳健的按设备验证，再用未参与选择的数据检验。不能用这组结果声称满足赛题的“综合性能不低于官方静态基线”要求。

## 独立设备闸门复验

针对首次回放暴露的候选跨设备失效问题，从原来的 70 台初始化发动机中，按固定种子 2026 留出 10 台、2,116 条记录，另 60 台、12,291 条用于训练。原先的 30 台十批回放设备保持不变。候选必须同时通过当前回放批次与这 10 台独立设备的闸门；闸门样本始终不加入模型训练历史。切分见 `scripts/prepare_cmapss_holdout.py` 与 `data/processed/cmapss-fd001-holdout/manifest.json`，执行配置见 `configs/cmapss-fd001-holdout-27b.yaml`。

| 新切分下的策略 | 十批批均 MAE | 相对同切分静态模型 | 候选晋级 | 推理 token |
|---|---:|---:|---:|---:|
| 本地静态模型（60 台训练） | **26.3993** | — | 0 | 0 |
| 固定规则 | 26.5943 | 差 0.1949 | 1 | 0 |
| PyTRiO Qwen3.8-27B | 26.4337 | 差 0.0343 | 2 | 6,833 |

[27B 复验记录](https://swanlab.cn/@hequan/zhixunyinqing/runs/s2lfvsw4)与[规则复验记录](https://swanlab.cn/@hequan/zhixunyinqing/runs/6d2szeqn)均已上传。27B 十次决策均有效，无解析回退或训练失败。编号 2 的批次预测比静态模型略好，编号 3 的批次变差后回滚；其余批次与静态模型一致。独立闸门减少了不稳定晋级，但整体仍未超过静态模型。新旧实验的初始化训练设备数不同，**不能直接比较新旧 MAE 的绝对值来证明方案改进**；只能比较各自同切分下相对静态模型的差值。固定的 10 台闸门也会受反复选择影响，正式方案需要新的未参与选择的数据或官方验证集继续检验。

## 重复切分配对检验（2026-11-11）

前面两节的结论都建立在**单次切分**上（60/10/30 台固定分配）。单次切分只能给出一个数字，
无法回答"这个差值是策略效果还是抽样噪声"。补 `scripts/cmapss_validate.py`：每个种子从
100 台发动机里重新抽 60 训练 / 10 独立闸门 / 30 回放（10 批 × 3 台），在同一切分上
同时跑系统与静态基线，跨 30 个种子汇总。

| 指标 | 值 |
|---|---:|
| 静态 / 系统 MAE | 26.63 / 26.56 |
| 配对差值均值 | **+0.07 MAE**（正 = 系统更好） |
| 95% 置信区间 | [−0.02, +0.17] |
| 胜 / 负 / 平 | 43 / 30 / **227** |
| 动手时胜率 | 58.9% |
| 符号检验 p | **0.1597** |
| 各种子均值范围 | [−0.49, +1.50] |

三条结论：

1. **单次切分的 −0.19 是抽样噪声。** 30 个切分的平均是 **+0.07**（方向为正），
   说明那一次恰好抽到了不利的分配。这也说明"用一次回放论证方案好坏"不可靠。
2. **仍不显著。** p=0.16，CI 下界 −0.02 几乎贴住 0。按 5% 水平不能宣称优于静态，
   但比 XJTU-SY（p=0.46）有说服力得多——样本结构不同（10 批 × 3 台 vs 3 批 × 2 轴承）。
3. **76% 的批次选择不动（227/300 平局）。** 系统大部分时间保持静态模型，
   只在证据足够时更新；动手的 73 次里 43 胜 30 负。这是"不伤害基线"该有的形态。

云端记录：[i2iqgbga](https://swanlab.cn/@hequan/zhixunyinqing/runs/i2iqgbga)，明细见
`out/cmapss-validation/validation-rule.json`。与 XJTU-SY 的结果对照：

| 数据集 | 配对差值 | 95% CI | 胜率 | p |
|---|---:|---|---:|---:|
| XJTU-SY | +16.00 | [−1.43, +37.94] | 58.6% | 0.46 |
| C-MAPSS FD001 | +0.07 | [−0.02, +0.17] | 58.9% | 0.16 |

两个数据集都指向同一个结论：**当前策略稳定地"略好或持平"，但没有统计显著的增益**。
对赛题那道"不低于静态基线"的硬门槛是安全的，但不能宣称性能优势。

## 复现

```powershell
.\.venv\Scripts\python.exe scripts/prepare_cmapss.py
.\.venv\Scripts\python.exe -m selftrain run --data-dir data/processed/cmapss-fd001/stream --train-csv data/processed/cmapss-fd001/train.csv --target y --task regression --policy llm --config configs/cmapss-fd001-27b.yaml --pred-dir out/preds/cmapss-fd001-27b --quiet
```

复现需要已登录 PyTRiO 和 SwanLab；密钥不在配置文件和交付包里。规则对照将 `--policy llm` 改为 `--policy rule`，预测目录另设为空目录。公开原始数据和处理后 CSV 均不包含在交付 ZIP 中。

独立闸门复验先运行 `scripts/prepare_cmapss_holdout.py`，然后把训练集改为 `data/processed/cmapss-fd001-holdout/train.csv`，增加 `--gate-csv data/processed/cmapss-fd001-holdout/gate.csv`，配置改为 `configs/cmapss-fd001-holdout-27b.yaml`。该模式当前不支持检查点恢复。

重复切分配对检验（约 9 分钟，30 个种子 × 10 批）：

```powershell
.\.venv\Scripts\python.exe scripts\cmapss_validate.py --seeds 30 --policy rule --cloud
```

它只读 `data/processed/cmapss-fd001/` 下已处理好的 CSV，不重新解析原始包；
输出 `out/cmapss-validation/validation-rule.json` 并同步到 SwanLab。
