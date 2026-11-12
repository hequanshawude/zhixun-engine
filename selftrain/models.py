"""模型层：统一封装，CPU 友好，支持样本权重（重加权动作要用）。

模型注册表（可扩展）：
    hgb      HistGradientBoosting，默认主力，快且抗 NaN
    logreg   逻辑回归，极快，作为"便宜"替身
    ridge    岭回归（回归任务）
    lightgbm LightGBM（装了才可用）
    dummy    多数类/均值，用于兜底与对照
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .types import TaskType

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:  # pragma: no cover
    _HAS_LGB = False


# --------------------------------------------------------------------------


@dataclass
class ModelBundle:
    """一个已训练模型 + 元信息。version 用于 registry / 审计。"""

    version: str
    kind: str
    task: TaskType
    estimator: Any
    params: Dict[str, Any] = field(default_factory=dict)
    trained_on: int = 0
    created_at: float = field(default_factory=time.time)
    parent: Optional[str] = None
    note: str = ""
    scaler: Any = None

    # -- 推断 --
    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self._scale(X)
        return self.estimator.predict(Xs)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        Xs = self._scale(X)
        if hasattr(self.estimator, "predict_proba"):
            try:
                return self.estimator.predict_proba(Xs)
            except Exception:
                return None
        return None

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            return self.scaler.transform(X)
        return X

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "kind": self.kind,
            "task": self.task.value,
            "params": self.params,
            "trained_on": self.trained_on,
            "parent": self.parent,
            "note": self.note,
        }


# --------------------------------------------------------------------------
# 构造
# --------------------------------------------------------------------------


def build_estimator(kind: str, task: TaskType, params: Optional[Dict[str, Any]] = None):
    params = dict(params or {})
    rs = params.pop("random_state", 42)

    if not _HAS_SKLEARN:
        raise RuntimeError("缺少 scikit-learn，请先安装依赖")

    if kind == "dummy":
        if task.is_classification:
            return DummyClassifier(strategy="most_frequent")
        return DummyRegressor(strategy="mean")

    if kind == "hgb":
        if task.is_classification:
            return HistGradientBoostingClassifier(
                max_iter=params.pop("max_iter", 120),
                learning_rate=params.pop("learning_rate", 0.1),
                max_leaf_nodes=params.pop("max_leaf_nodes", 31),
                min_samples_leaf=params.pop("min_samples_leaf", 20),
                l2_regularization=params.pop("l2_regularization", 1.0),
                random_state=rs,
                **params,
            )
        return HistGradientBoostingRegressor(
            max_iter=params.pop("max_iter", 160),
            learning_rate=params.pop("learning_rate", 0.1),
            max_leaf_nodes=params.pop("max_leaf_nodes", 31),
            min_samples_leaf=params.pop("min_samples_leaf", 20),
            l2_regularization=params.pop("l2_regularization", 1.0),
            random_state=rs,
            **params,
        )

    if kind == "lightgbm":
        if not _HAS_LGB:
            raise RuntimeError("未安装 lightgbm")
        base = dict(
            n_estimators=params.pop("n_estimators", 200),
            learning_rate=params.pop("learning_rate", 0.08),
            num_leaves=params.pop("num_leaves", 31),
            subsample=params.pop("subsample", 0.9),
            colsample_bytree=params.pop("colsample_bytree", 0.9),
            reg_lambda=params.pop("reg_lambda", 1.0),
            random_state=rs,
            n_jobs=params.pop("n_jobs", 2),
            verbose=-1,
        )
        base.update(params)
        if task.is_classification:
            return lgb.LGBMClassifier(**base)
        return lgb.LGBMRegressor(**base)

    if kind == "logreg":
        if not task.is_classification:
            return Ridge(alpha=params.pop("alpha", 1.0), random_state=rs)
        return LogisticRegression(
            C=params.pop("C", 1.0),
            max_iter=params.pop("max_iter", 300),
            class_weight=params.pop("class_weight", None),
            random_state=rs,
            n_jobs=None,
        )

    if kind == "ridge":
        if task.is_classification:
            return LogisticRegression(C=params.pop("C", 1.0), max_iter=400, random_state=rs)
        return Ridge(alpha=params.pop("alpha", 1.0), random_state=rs)

    raise ValueError(f"未知模型类型: {kind}")


_NEEDS_SCALE = {"logreg", "ridge"}


def fit_model(
    kind: str,
    task: TaskType,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    version: str = "",
    parent: Optional[str] = None,
    note: str = "",
) -> ModelBundle:
    """训练并打包。返回的 bundle 自带计时信息在 note 外，由调用方统计成本。"""
    est = build_estimator(kind, task, params)
    scaler = None
    Xf = X
    if kind in _NEEDS_SCALE:
        scaler = StandardScaler()
        Xf = scaler.fit_transform(X)

    fit_kwargs: Dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    if kind == "lightgbm" and task.is_classification and len(np.unique(y)) == 2:
        # sklearn 接口会自动处理二分类
        pass

    try:
        est.fit(Xf, y, **fit_kwargs)
    except TypeError:
        # 某些估计器不支持 sample_weight：退化为无权重
        est.fit(Xf, y)

    return ModelBundle(
        version=version or f"{kind}-{int(time.time()*1000)}",
        kind=kind,
        task=task,
        estimator=est,
        params=dict(params or {}),
        trained_on=int(len(X)),
        parent=parent,
        note=note,
        scaler=scaler,
    )


# --------------------------------------------------------------------------
# 包装器
# --------------------------------------------------------------------------


class ThresholdedModel:
    """二分类决策阈值包装：`calibrate` 动作的产物，零训练成本。"""

    def __init__(self, base: ModelBundle, threshold: float):
        self.base = base
        self.threshold = float(threshold)
        self.version = f"{base.version}-thr{threshold:.3f}"

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        if proba is None:
            return (self.base.predict(X) > self.threshold).astype(int)
        return (proba[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        return self.base.predict_proba(X)


class BiasedRegressor:
    """回归的偏移校正：`calibrate` 在回归任务上的等价物。"""

    def __init__(self, base: ModelBundle, bias: float):
        self.base = base
        self.bias = float(bias)
        self.version = f"{base.version}+b{bias:+.3f}"

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.base.predict(X), dtype=float) + self.bias

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        return None
