"""Prepare a bearing-disjoint XJTU-SY RUL replay from the authors' RAR volumes.

每个 CSV 是一次采样（32,768 点 × 2 通道）。原始振动不能直接喂给批流，所以先按快照
提特征，再按轴承分组切分：同一轴承的所有快照只会出现在训练集或某一个回放批次里，
绝不会跨集合——这是避免"相邻窗口泄漏"的关键。

目标 y = 该轴承最后一个快照序号 - 当前快照序号（不截断），与 C-MAPSS 的口径一致。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path("data/raw/xjtu-sy/extract/XJTU-SY_Bearing_Datasets")
TARGET = Path("data/processed/xjtu-sy")
SEED = 42

# 每个工况固定留 2 个轴承做回放，其余训练；顺序即批次顺序（工况 1 → 2 → 3 漂移）。
TRAIN_BEARINGS = {
    "35Hz12kN": ["Bearing1_1", "Bearing1_2", "Bearing1_4"],
    "37.5Hz11kN": ["Bearing2_1", "Bearing2_3", "Bearing2_5"],
    "40Hz10kN": ["Bearing3_1", "Bearing3_2", "Bearing3_3"],
}
STREAM_BEARINGS = {
    "35Hz12kN": ["Bearing1_3", "Bearing1_5"],
    "37.5Hz11kN": ["Bearing2_2", "Bearing2_4"],
    "40Hz10kN": ["Bearing3_4", "Bearing3_5"],
}
BEARINGS_PER_BATCH = 2


# --------------------------------------------------------------------------
# 特征
# --------------------------------------------------------------------------


def _time_features(x: np.ndarray) -> dict:
    eps = 1e-12
    absx = np.abs(x)
    mean = float(x.mean())
    std = float(x.std())
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(absx.max())
    p2p = float(x.max() - x.min())
    # 有量纲与无量纲指标并用：量纲指标反映退化程度，无量纲指标对工况变化更稳健
    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "peak": peak,
        "p2p": p2p,
        "skew": float(np.mean((x - mean) ** 3) / (std**3 + eps)),
        "kurt": float(np.mean((x - mean) ** 4) / (std**4 + eps)),
        "crest": peak / (rms + eps),
        "shape": rms / (absx.mean() + eps),
        "impulse": peak / (absx.mean() + eps),
        "clearance": peak / (np.mean(np.sqrt(absx)) ** 2 + eps),
    }


def _freq_features(x: np.ndarray, fs: float = 25600.0) -> dict:
    n = len(x)
    spec = np.abs(np.fft.rfft(x - x.mean())) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = spec**2
    total = power.sum() + 1e-12
    p = power / total

    centroid = float((freqs * p).sum())
    spread = float(np.sqrt(((freqs - centroid) ** 2 * p).sum()))
    dom = float(freqs[int(np.argmax(power[1:]) + 1)]) if n > 2 else 0.0
    entropy = float(-(p * np.log(p + 1e-12)).sum())

    # 四个等宽带能量占比：低频段主要反映工况，高频段对早期故障更敏感
    edges = np.linspace(0, fs / 2, 5)
    bands = {}
    for i in range(4):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        bands[f"band{i}"] = float(power[mask].sum() / total)
    return {"spec_centroid": centroid, "spec_spread": spread,
            "spec_dom": dom, "spec_entropy": entropy, **bands}


def snapshot_features(path: Path) -> dict:
    frame = pd.read_csv(path)
    if frame.shape[1] != 2:
        raise ValueError(f"{path} 列数异常: {frame.shape}")
    out = {}
    for name, series in zip(("h", "v"), frame.columns):
        x = frame[series].to_numpy(dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError(f"{path} 存在非有限值")
        out.update({f"{name}_{k}": v for k, v in _time_features(x).items()})
        out.update({f"{name}_{k}": v for k, v in _freq_features(x).items()})
    return out


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def prepare(source: Path = SOURCE, target: Path = TARGET) -> dict:
    if not source.exists():
        raise FileNotFoundError(f"找不到解压后的数据: {source}")

    rows = []
    bearing_meta = {}
    # 工况用整数编码参与特征：字符串列会让特征矩阵无法转成 float
    cond_index = {name: i for i, name in enumerate(sorted(TRAIN_BEARINGS))}
    for condition, bearings in TRAIN_BEARINGS.items():
        for bearing in bearings + STREAM_BEARINGS[condition]:
            folder = source / condition / bearing
            files = sorted(folder.glob("*.csv"), key=lambda p: int(p.stem))
            if not files:
                raise FileNotFoundError(f"{folder} 没有快照")
            order = [int(p.stem) for p in files]
            if order != list(range(1, len(files) + 1)):
                raise ValueError(f"{bearing} 快照序号不连续: {order[:5]}...")

            last = len(files) - 1
            for idx, path in enumerate(files):
                feats = snapshot_features(path)
                feats.update({"id": bearing, "condition_id": cond_index[condition],
                              "cycle": idx + 1, "y": float(last - idx)})
                rows.append(feats)
            bearing_meta[bearing] = {"condition": condition, "condition_id": cond_index[condition],
                                     "snapshots": len(files), "lifespan": last}

    frame = pd.DataFrame(rows)
    feature_cols = [c for c in frame.columns if c not in ("id", "cycle", "y")]
    if not np.isfinite(frame[feature_cols + ["y"]].to_numpy()).all():
        raise ValueError("特征或目标存在非有限值")

    # 切分校验：同一轴承只能属于一侧
    train_ids = [b for bs in TRAIN_BEARINGS.values() for b in bs]
    stream_ids = [b for bs in STREAM_BEARINGS.values() for b in bs]
    if set(train_ids) & set(stream_ids):
        raise ValueError("训练轴承与回放轴承重叠")
    if set(train_ids) | set(stream_ids) != set(bearing_meta):
        raise ValueError("轴承覆盖不完整")

    target.mkdir(parents=True, exist_ok=True)
    stream_dir = target / "stream"
    stream_dir.mkdir(exist_ok=True)
    for old in stream_dir.glob("batch_*.csv"):
        old.unlink()

    bootstrap = frame[frame.id.isin(train_ids)].sort_values(["condition_id", "id", "cycle"])
    bootstrap.to_csv(target / "train.csv", index=False, float_format="%.9g")

    batches = []
    for k, offset in enumerate(range(0, len(stream_ids), BEARINGS_PER_BATCH), 1):
        ids = stream_ids[offset:offset + BEARINGS_PER_BATCH]
        batch = frame[frame.id.isin(ids)].sort_values(["id", "cycle"])
        filename = f"batch_{k:03d}.csv"
        batch.to_csv(stream_dir / filename, index=False, float_format="%.9g")
        batches.append({"file": filename, "bearing_ids": ids,
                        "conditions": [bearing_meta[b]["condition"] for b in ids],
                        "rows": int(len(batch))})

    if len(bootstrap) + sum(b["rows"] for b in batches) != len(frame):
        raise ValueError("切分未覆盖每一行恰好一次")

    manifest = {
        "source": "https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets",
        "source_note": "六个 RAR 分卷由 MediaFire 镜像取得并逐卷 SHA-256 校验，见 data/raw/xjtu-sy/download.json",
        "split_seed": SEED, "split_unit": "bearing",
        "target": "不截断 RUL = 该轴承最后快照序号 - 当前快照序号",
        "feedback": "回顾性；每批完整轴承轨迹结束后才向模拟器揭示标签",
        "sampling": "每个 CSV 为一次采样，32,768 点 × 2 通道；特征按快照提取",
        "feature_columns": feature_cols,
        "excluded_from_features": ["id", "y", "工况名（仅以 condition 编码参与）"],
        "bootstrap": {"file": "train.csv", "bearing_ids": train_ids, "rows": int(len(bootstrap))},
        "batches": batches,
        "bearings": bearing_meta,
        "caveat": "每个回放批次仅 2 个轴承，批间方差很大；本切分用于打通管线，不用于宣称性能",
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = prepare()
    print(f"XJTU-SY ready: {result['bootstrap']['rows']} bootstrap rows, "
          f"{sum(b['rows'] for b in result['batches'])} replay rows, "
          f"{len(result['batches'])} bearing-disjoint batches, "
          f"{len(result['feature_columns'])} features")
