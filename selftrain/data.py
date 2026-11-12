"""数据层：统一 Batch 流接口。

两个实现：
    SyntheticStream —— 合成带漂移的数据流，用于开发期自测（无 A 包也能跑通全链路）
    CSVBatchStream  —— 读 A 包/隐藏测试流下发的分批 CSV

A 包真实数据到位后，只需要实现一个 ``__iter__ -> Batch`` 的类，其它模块无需改动。
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd

from .types import Batch, DatasetSpec, TaskType


# --------------------------------------------------------------------------
# 合成数据流
# --------------------------------------------------------------------------


@dataclass
class SyntheticConfig:
    task: TaskType = TaskType.BINARY
    n_batches: int = 14
    batch_size: int = 300
    n_features: int = 16
    scenario: str = "mixed"          # covariate / concept / label / sudden / mixed / none
    drift_start: int = 3             # 从第几个 batch 开始漂
    drift_strength: float = 1.0
    noise: float = 0.35
    seed: int = 0


class SyntheticStream:
    """合成漂移流。

    每个 batch 单独生成 X，按 scenario 施加三种漂移之一：
        covariate : P(X) 平移（传感器零点漂移 / 工况变化）
        concept   : P(y|X) 旋转（设备磨损后同样的输入含义变了）
        label     : P(y) 变化（产线良率/故障率变化）
        sudden    : 中途阶跃，模拟更换原料批次
    """

    def __init__(self, cfg: SyntheticConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        d = cfg.n_features
        self.feature_names = [f"f{i:02d}" for i in range(d)]

        # 真实概念
        self._w0 = self.rng.standard_normal(d) / np.sqrt(d)
        self._b0 = 0.0 if cfg.task.is_classification else 0.0
        # 用于 concept drift 的正交方向
        u = self.rng.standard_normal(d)
        u -= (u @ self._w0) * self._w0 / (self._w0 @ self._w0)
        self._u = u / (np.linalg.norm(u) + 1e-9)
        # 受漂移影响的特征子集
        self._drift_feats = self.rng.choice(d, size=max(2, d // 4), replace=False)
        # 多分类的额外头
        self._heads = self.rng.standard_normal((max(2, 3), d)) / np.sqrt(d)

    @property
    def spec(self) -> DatasetSpec:
        n_classes = 2 if self.cfg.task == TaskType.BINARY else 3
        return DatasetSpec(
            name=f"synthetic-{self.cfg.scenario}-{self.cfg.task.value}",
            task=self.cfg.task,
            feature_names=self.feature_names,
            n_classes=n_classes,
        )

    def _progress(self, k: int) -> float:
        if k < self.cfg.drift_start:
            return 0.0
        span = max(1, self.cfg.n_batches - self.cfg.drift_start - 1)
        return min(1.0, (k - self.cfg.drift_start) / span)

    def _make_batch(self, k: int) -> Batch:
        c = self.cfg
        t = self._progress(k)
        s = c.drift_strength * t
        X = self.rng.standard_normal((c.batch_size, c.n_features))

        w = self._w0.copy()
        b = self._b0
        meta = {"progress": t, "scenario": c.scenario, "gt_drift": 0.0}

        if c.scenario in ("covariate", "mixed"):
            shift = np.zeros(c.n_features)
            shift[self._drift_feats] = 1.2 * s
            X += shift
            meta["gt_drift"] = float(np.abs(shift).mean())

        if c.scenario in ("concept", "mixed"):
            angle = 1.1 * s
            w = np.cos(angle) * self._w0 + np.sin(angle) * self._u * np.linalg.norm(self._w0) * 2.0
            meta["gt_drift"] = max(meta["gt_drift"], float(angle))

        if c.scenario == "sudden" and k >= c.n_batches // 2:
            X[:, self._drift_feats] += 1.5
            w = np.cos(0.9) * self._w0 + np.sin(0.9) * self._u * np.linalg.norm(self._w0) * 2.0
            meta["gt_drift"] = 1.5

        if c.scenario == "label":
            # 通过截距改变先验，P(X) 与 P(y|X) 形态不变
            b = b + 1.6 * s
            meta["gt_drift"] = float(s)

        if c.task == TaskType.BINARY:
            logit = X @ w + b
            p = 1.0 / (1.0 + np.exp(-logit))
            y = (self.rng.random(c.batch_size) < p).astype(int)
        elif c.task == TaskType.MULTICLASS:
            logits = np.stack([X @ hh + b for hh in self._heads], axis=1)
            logits += np.log(np.array([0.6, 0.3, 0.1]))  # 不均衡先验
            p = np.exp(logits - logits.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            y = np.array([self.rng.choice(3, p=pi) for pi in p])
        else:
            y = X @ w * 2.0 + b + self.rng.normal(0, c.noise, c.batch_size)

        meta["batch_prior"] = _prior(y, c.task)
        return Batch(index=k, X=X.astype(np.float32), y=y, feature_names=self.feature_names, meta=meta)

    def __iter__(self) -> Iterator[Batch]:
        for k in range(self.cfg.n_batches):
            yield self._make_batch(k)


def _prior(y: np.ndarray, task: TaskType) -> float:
    if task.is_classification:
        return float(np.mean(y))
    return float(np.mean(y))


# --------------------------------------------------------------------------
# CSV 批流（A 包 / 评测下发）
# --------------------------------------------------------------------------


class CSVBatchStream:
    """读取目录下按序排列的批次文件。

    支持两种布局：
        1. data_dir/batch_000.csv, batch_001.csv, ...
        2. data_dir/all.csv，含 ``batch`` 列
    """

    def __init__(
        self,
        data_dir: str,
        spec: DatasetSpec,
        target_col: str = "y",
        pattern: str = "batch_*.csv",
        has_labels: bool = True,
        max_batches: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.spec = spec
        self.target_col = target_col
        self.pattern = pattern
        self.has_labels = has_labels
        self.max_batches = max_batches
        self._files = sorted(glob.glob(str(self.data_dir / pattern)))

    def __iter__(self) -> Iterator[Batch]:
        if self._files:
            for k, path in enumerate(self._files):
                if self.max_batches is not None and k >= self.max_batches:
                    break
                df = pd.read_csv(path)
                yield self._to_batch(k, df, name=os.path.basename(path))
        else:
            all_path = self.data_dir / "all.csv"
            if not all_path.exists():
                raise FileNotFoundError(
                    f"{self.data_dir} 下既没有 {self.pattern} 也没有 all.csv"
                )
            df = pd.read_csv(all_path)
            col = "batch" if "batch" in df.columns else None
            groups = df.groupby(col) if col else [(0, df)]
            for k, (_, sub) in enumerate(groups):
                if self.max_batches is not None and k >= self.max_batches:
                    break
                yield self._to_batch(k, sub, name=f"batch_{k}")

    def _to_batch(self, k: int, df: pd.DataFrame, name: str) -> Batch:
        feats = self.spec.feature_names
        missing = [c for c in feats if c not in df.columns]
        if missing:
            raise ValueError(f"{name} 缺少训练时的特征列: {missing}")
        if df.empty:
            raise ValueError(f"{name} 是空批次")
        X = df[feats].to_numpy(dtype=np.float32)
        y = df[self.target_col].to_numpy() if (self.has_labels and self.target_col in df.columns) else None
        return Batch(index=k, X=X, y=y, feature_names=list(feats), meta={"file": name})


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------


def stack_batches(batches: Sequence[Batch]) -> tuple[np.ndarray, Optional[np.ndarray]]:
    Xs = [b.X for b in batches if b.X is not None and len(b.X)]
    X = np.vstack(Xs) if Xs else np.empty((0, 0))
    if all(b.y is not None for b in batches):
        y = np.concatenate([b.y for b in batches])
    else:
        y = None
    return X, y


def infer_spec(df: pd.DataFrame, name: str, target_col: str = "y") -> DatasetSpec:
    """从一张表推断任务类型（用于临时接入未知 CSV）。"""
    feats = [c for c in df.columns if c not in (target_col, "batch", "time", "id")]
    if target_col not in df.columns:
        return DatasetSpec(name=name, task=TaskType.REGRESSION, feature_names=feats)
    y = df[target_col]
    if y.dtype.kind in "if" and y.nunique() > 12:
        return DatasetSpec(name=name, task=TaskType.REGRESSION, feature_names=feats, primary_metric="mae")
    n_unique = int(y.nunique())
    if n_unique == 2:
        return DatasetSpec(name=name, task=TaskType.BINARY, feature_names=feats, n_classes=2)
    return DatasetSpec(name=name, task=TaskType.MULTICLASS, feature_names=feats, n_classes=n_unique)


def train_val_split_time(
    X: np.ndarray, y: np.ndarray, val_frac: float = 0.2, purge: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """时序切分（带 purge），用于离线评估候选模型，避免泄漏。"""
    n = len(X)
    n_val = max(1, int(n * val_frac))
    cut = n - n_val
    left = max(0, cut - purge)
    return X[:left], y[:left], X[cut:], y[cut:]
