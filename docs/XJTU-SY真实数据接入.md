# XJTU-SY 轴承数据接入（2026-11-11）

处理《项目工作日志与 AI 交接》"接手后优先处理"第 4 条：XJTU-SY 的下载、校验、解压、
特征化与按轴承分组切分。结论先说：**数据已完整接入并能跑通全链路，但回放规模太小，
不构成任何性能结论。**

## 1. 下载与校验

六个 RAR 分卷来自作者仓库链接的 MediaFire 镜像。

| 项 | 值 |
|---|---|
| 分卷数 | 6（5 × 744,488,960 + 1 × 722,155,640 字节） |
| 总字节 | 4,444,600,440（4.44 GB） |
| 校验 | 逐卷 SHA-256 与镜像元数据一致，6/6 通过 |
| 清单 | `data/raw/xjtu-sy/download.json` |

过程中遇到一次 **MediaFire 全站 403 限流**（下载约 3.5 GB 后，六个文件页同时返回 403）。
等待约 45 秒后恢复 200，续传成功。判断依据是：403 只作用于文件页、且重启会话无效，
不是文件被删除。这条经验值得记下来——**遇到批量 403 先等，不要反复重试**。

## 2. 解压

系统没有装任何 RAR 工具，Windows 自带的 `tar`（bsdtar 3.8.9）能读第一卷，但
到分卷边界就报 `Too small block encountered`，无法跨卷。

改用 `winget install 7zip.7zip`（26.06）后一次解压成功：

```
9,217 个文件，11.38 GB，44 秒
```

## 3. 数据结构

```
XJTU-SY_Bearing_Datasets/
  35Hz12kN/   Bearing1_1 .. Bearing1_5
  37.5Hz11kN/ Bearing2_1 .. Bearing2_5
  40Hz10kN/   Bearing3_1 .. Bearing3_5
```

每个 CSV 是一次采样：32,768 点 × 2 通道（水平/垂直振动），无表头以外的元数据。

| 轴承 | 快照数 | 轴承 | 快照数 | 轴承 | 快照数 |
|---|---:|---|---:|---|---:|
| Bearing1_1 | 123 | Bearing2_1 | 491 | Bearing3_1 | 2538 |
| Bearing1_2 | 161 | Bearing2_2 | 161 | Bearing3_2 | 2496 |
| Bearing1_3 | 158 | Bearing2_3 | 533 | Bearing3_3 | 371 |
| Bearing1_4 | 122 | Bearing2_4 | 42 | Bearing3_4 | 1515 |
| Bearing1_5 | 52 | Bearing2_5 | 339 | Bearing3_5 | 114 |

注意寿命跨度差 **60 倍**（42 到 2538 个快照），任何按轴承平均的指标都会被长寿命轴承主导。

## 4. 特征化

原始 32,768 点无法直接进批流，按快照提 38 维特征（每通道 19 维）：

- **时域 11 维**：mean / std / rms / peak / p2p / skew / kurt / crest / shape / impulse / clearance
- **频域 8 维**：谱质心、谱展宽、主频、谱熵、四个等带宽能量占比

外加 `condition_id`（工况整数编码）与 `cycle`（快照序号）参与建模；`id` 与 `y` 不参与。
目标为**不截断 RUL** = 该轴承最后快照序号 − 当前快照序号，与 C-MAPSS 口径一致。

脚本 `scripts/prepare_xjtu_sy.py`，耗时 70 秒。

> 采样率按 25.6 kHz 计算（XJTU-SY 公开说明的采样频率）。若赛方给出不同定义，
> 需同步修改 `_freq_features` 的 `fs`。

## 5. 切分（按轴承分组，杜绝相邻窗口泄漏）

| 集合 | 轴承 |
|---|---|
| 训练（9） | 1_1, 1_2, 1_4, 2_1, 2_3, 2_5, 3_1, 3_2, 3_3 |
| 回放批 1 | Bearing1_3, Bearing1_5（35Hz12kN） |
| 回放批 2 | Bearing2_2, Bearing2_4（37.5Hz11kN） |
| 回放批 3 | Bearing3_4, Bearing3_5（40Hz10kN） |

同一轴承的全部快照只出现在一个集合里；批次按工况 1→2→3 推进，天然构成一次
"工况漂移"。产物：`data/processed/xjtu-sy/{train.csv, stream/batch_00{1..3}.csv, manifest.json}`。
训练 7,174 行，回放 2,042 行。

## 6. 基线结果（本机，规则策略）

| 策略 | 三批批均 MAE | 相对静态 |
|---|---:|---:|
| 静态基线（9 轴承训练后不再更新） | 341.4897 | — |
| 规则策略（绝对阈值 min_gain=0.002） | 243.5166 | **好 97.97** |
| 规则策略（相对阈值 min_gain_rel=2%） | 341.4897 | 持平 |

**这两行都不能当作性能结论。** 逐批看很清楚：

| 批 | 系统 | 静态 | 候选 | 决策 |
|---|---:|---:|---:|---|
| 0 | 85.72 | 85.72 | 583.45 | 拒绝（退化） |
| 1 | 241.28 | 241.28 | 238.44 | 绝对阈值下晋级（+2.84） |
| 2 | 697.47 / 403.55 | 697.47 | 458.96 | — |

批 1 那次 "+2.84" 的晋级，让批 2 的预测从 697 变成 404，于是整体均值好了 98。
换句话说：**全部差异来自一次晋级，以及它在下一批上的运气**。三批 × 每批 2 个轴承，
批间 MAE 波动本身就有几百，这个样本量不足以区分策略好坏。

## 6.1 改用重复切分做配对检验

单次切分的结论不可用，所以补了 `scripts/xjtu_validate.py`：对每个随机种子重新切分
（9 训练 / 6 回放 / 3 批），**在同一切分上同时跑系统与静态基线**，得到逐批配对差值，
再跨种子汇总。

30 个种子（90 个批次）的结果：

| 指标 | 值 |
|---|---:|
| 静态 / 系统 MAE | 565.27 / 549.27 |
| 配对差值均值 | **+16.00 MAE**（正 = 系统更好） |
| 95% 置信区间 | [−1.43, +37.94] |
| 胜 / 负 / 平 | 17 / 12 / **61** |
| 动手时的胜率 | 58.6%（17/29） |
| 符号检验 p | 0.4583 |
| 各种子均值范围 | [−48.43, +228.58] |

三条结论：

1. **不显著。** 置信区间跨 0，符号检验 p=0.46。方向偏正（17 胜 12 负），但样本量
   撑不起"动态优于静态"的说法。
2. **系统 68% 的批次选择不动（61/90 平局）。** 这说明闸门确实是保守的——在证据不足时
   它不冒险，这正是"不伤害静态基线"该有的行为。
3. **单次切分能给出 +228，也能给出 −48。** 之前那次 +98 完全落在噪声范围内，
   现在有了量化依据，不用再靠"感觉不可信"。

**因此：XJTU-SY 只能用来验证"不显著变差"，不能用来宣称性能增益。** 要证明增益，
需要新的未参与选择的设备组或官方验证集。

## 7. 顺带修掉的闸门缺陷

原闸门只有绝对阈值（`min_gain=0.002`、`max_drop=0.02`）。在 F1（0~1）上合理，
但在 RUL 的 MAE（数十到数千）上等于**任何变化都算显著**——批 1 的 +2.84 就是这么
蒙混过关的。

新增 `gate.min_gain_rel` / `gate.max_drop_rel`，按锚点指标量级缩放，0 表示不启用
（保持既有结果不变）。阈值计算改为：

```
min_gain = max(min_gain, min_gain_rel × |anchor|)
max_drop = max(max_drop, max_drop_rel × |anchor|)
```

`configs/xjtu-sy-rule.yaml` 启用 `min_gain_rel: 0.02`、`max_drop_rel: 0.10`。
81 项测试仍全部通过（默认值为 0，行为不变）。

## 8. 待办

1. **样本量（已部分解决）**：单次切分不可用，已改为 30 个随机切分的配对检验
   （`scripts/xjtu_validate.py`）。但 15 个轴承仍是硬上限——若要有把握地分辨 5% 量级的
   差异，需要更多设备。下一步可考虑"逐快照滑动验证"或按寿命分层，把评估单元从
   "轴承"细化到"寿命区间"。
2. **采样率与特征口径**：25.6 kHz 是公开说明值，赛方若有指定需同步。
3. **目标定义**：当前是不截断 RUL。C-MAPSS 那边也未截断；若赛方要求分段（如 RUL 截断在 125），
   两个数据集的 `y` 都要改。
4. **尚未上云**：本轮全部本地 jsonl，没有消耗 PyTRiO/SwanLab 额度。

## 9. 复现

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\download_xjtu_sy.py     # 断点续传 + 逐卷校验
& 'C:\Program Files\7-Zip\7z.exe' x data\raw\xjtu-sy\XJTU-SY_Bearing_Datasets.part01.rar -odata\raw\xjtu-sy\extract -y
.\.venv\Scripts\python.exe scripts\prepare_xjtu_sy.py
.\.venv\Scripts\python.exe -m selftrain run --data-dir data/processed/xjtu-sy/stream --train-csv data/processed/xjtu-sy/train.csv --target y --task regression --policy rule --config configs/xjtu-sy-rule.yaml --pred-dir out/preds/xjtu-sy
.\.venv\Scripts\python.exe scripts\xjtu_validate.py --seeds 30 --policy rule   # 约 200 秒
```

`xjtu_validate.py` 只读 `data/processed/xjtu-sy/` 下已算好的 CSV，不重新提特征；
输出 `out/xjtu-validation/validation-rule.json`，含逐批明细与逐种子均值。
