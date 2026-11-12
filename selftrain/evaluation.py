"""评估与上线闸门。

两个原则：
    1. 时序切分 + purge，绝不随机切（工业时序随机切 = 分数虚高）
    2. 晋级必须是"显著更好"，退化必须"立刻回滚"，中间地带一律不动
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .config import GateConfig
from .types import TaskType

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False


PRIMARY = {
    TaskType.BINARY: "f1_macro",
    TaskType.MULTICLASS: "f1_macro",
    TaskType.REGRESSION: "mae",
}

LOWER_IS_BETTER = {"mae", "rmse", "mse", "loss"}


def higher_is_better(metric: str) -> bool:
    return metric not in LOWER_IS_BETTER


def primary_metric(task: TaskType) -> str:
    return PRIMARY[task]


def compute_metrics(
    task: TaskType,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if not _HAS_SKLEARN:
        if task.is_classification:
            return {"accuracy": float(np.mean(y_true == y_pred))}
        return {"mae": float(np.mean(np.abs(y_true - y_pred)))}

    if task.is_classification:
        out: Dict[str, float] = {}
        try:
            out["accuracy"] = float(accuracy_score(y_true, y_pred))
        except Exception:
            out["accuracy"] = float("nan")
        try:
            out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        except Exception:
            out["f1_macro"] = float("nan")
        if proba is not None and task == TaskType.BINARY:
            try:
                out["auc"] = float(roc_auc_score(y_true, proba[:, 1]))
            except Exception:
                pass
        return out

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def score_model(model, task: TaskType, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    pred = model.predict(X)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
    return compute_metrics(task, y, pred, proba)


def primary_value(metrics: Dict[str, float], task: TaskType) -> float:
    key = primary_metric(task)
    v = metrics.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        # 退化到 accuracy / 任意一个
        for k in ("accuracy", "mae", "rmse"):
            if k in metrics and not np.isnan(metrics[k]):
                return float(metrics[k])
        return float("nan")
    return float(v)


# --------------------------------------------------------------------------


@dataclass
class GateResult:
    accept: bool
    rollback: bool
    reason: str
    delta: float          # 候选相对锚点的主指标变化（正 = 变好）
    vs_static: float      # 候选相对静态基线的差值（正 = 更好）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accept": self.accept,
            "rollback": self.rollback,
            "reason": self.reason,
            "delta": round(self.delta, 5),
            "vs_static": round(self.vs_static if np.isfinite(self.vs_static) else float("nan"), 5),
        }


class PromotionGate:
    """晋级闸门。默认保守：宁可不动，也不要拿一个来路不明的模型去赌。"""

    def __init__(self, cfg: GateConfig):
        self.cfg = cfg

    def check(
        self,
        task: TaskType,
        anchor_metrics: Dict[str, float],
        cand_metrics: Dict[str, float],
        static_metrics: Optional[Dict[str, float]] = None,
        anchor_primary: Optional[float] = None,
        cand_primary: Optional[float] = None,
        static_primary: Optional[float] = None,
    ) -> GateResult:
        sign = 1.0 if higher_is_better(primary_metric(task)) else -1.0
        a = primary_value(anchor_metrics, task) if anchor_primary is None else anchor_primary
        c = primary_value(cand_metrics, task) if cand_primary is None else cand_primary
        s = primary_value(static_metrics or {}, task) if static_primary is None else static_primary

        if not np.isfinite(c):
            return GateResult(False, False, "候选指标无效(NaN)", 0.0, float("nan"))

        delta = sign * (c - a)
        vs_static = sign * (c - s) if (s is not None and np.isfinite(s)) else float("nan")

        # 阈值必须随指标量纲缩放：F1 在 0~1 之间时 0.002 是合理门限，
        # 但 RUL 的 MAE 在几百量级时，0.002 等于"任何变化都算显著"。
        scale = abs(a) if (np.isfinite(a) and abs(a) > 1e-9) else 1.0
        min_gain = max(self.cfg.min_gain, self.cfg.min_gain_rel * scale)
        max_drop = max(self.cfg.max_drop, self.cfg.max_drop_rel * scale)

        if not np.isfinite(a):
            return GateResult(True, False, "无锚点基准，直接上线", delta, vs_static)

        # 1) 明显变差 → 拒绝，并考虑回滚
        if delta <= -max_drop:
            rb = self.cfg.rollback_on_breach
            return GateResult(False, rb, f"候选退化 {delta:+.4f} <= -{max_drop:.4f}", delta, vs_static)

        # 2) 低于静态基线 → 拒绝（门槛③）
        if np.isfinite(vs_static) and vs_static < -self.cfg.static_guard:
            return GateResult(False, False, f"低于静态基线 {vs_static:+.4f}", delta, vs_static)

        # 3) 增益不足 → 不动
        if delta < min_gain:
            return GateResult(False, False, f"增益 {delta:+.4f} 未达阈值 {min_gain:.4f}", delta, vs_static)

        return GateResult(True, False, f"晋级：增益 {delta:+.4f}", delta, vs_static)


def normalized_drop(now: float, ref: float, task: TaskType) -> float:
    """把指标变化归一化成 "变差了多少"（正 = 变差），用于归因。"""
    if not (np.isfinite(now) and np.isfinite(ref)):
        return 0.0
    sign = 1.0 if higher_is_better(primary_metric(task)) else -1.0
    scale = max(abs(ref), 1e-6)
    d = sign * (ref - now) / scale
    return float(np.clip(d, -1.0, 1.0))
