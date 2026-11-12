"""动作菜单：把"怎么救模型"离散化成 7 个可计价、可审计的动作。

为什么要离散化：
    - LLM 只能在有限选项里选，才不会被幻觉带偏
    - 每个动作有确定的成本区间，预算才能在执行前卡住
    - 才能做 bandit / 消融，才能回答"为什么选这个"

成本从低到高排列，策略层的默认偏好也按这个顺序（除非证据很强）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .budget import Estimate
from .config import ModelConfig
from .models import BiasedRegressor, ModelBundle, ThresholdedModel, fit_model
from .types import DatasetSpec, TaskType

try:
    from sklearn.linear_model import LogisticRegression
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False


# --------------------------------------------------------------------------
# 动作目录
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionSpec:
    name: str
    label: str
    risk: str            # low / medium / high
    description: str
    order: int           # 成本序（越小越便宜）


CATALOG: Dict[str, ActionSpec] = {
    "noop": ActionSpec("noop", "不动作", "low", "漂移不显著或不影响指标，保持锚点不动。", 0),
    "rollback": ActionSpec("rollback", "回滚", "low", "回退到历史最佳版本，用于止损。", 1),
    "calibrate": ActionSpec("calibrate", "校准", "low", "不重训，只调二分类阈值/回归偏置，修复标签漂移。", 2),
    "reweight": ActionSpec("reweight", "重加权", "medium", "按密度比给样本加权后重训，针对协变量漂移。", 3),
    "finetune": ActionSpec("finetune", "增量微调", "medium", "以近期数据为主、旧数据为辅做加权训练。", 4),
    "retrain": ActionSpec("retrain", "全量重训", "high", "用全部历史数据重新训练，针对概念漂移。", 5),
    "switch_model": ActionSpec("switch_model", "换模型族", "high", "换用另一族模型，应对结构性失效。", 6),
}

ACTION_NAMES: List[str] = [k for k, _ in sorted(CATALOG.items(), key=lambda kv: kv[1].order)]


# 动作掩码：不同漂移类型下"讲得通"的动作集合。
# 它同时是安全机制（防止策略层在无关方向上乱花预算）和 bandit 的先验。
_PLAUSIBLE: Dict[str, List[str]] = {
    "none": ["noop", "calibrate"],
    "label": ["noop", "calibrate", "reweight"],
    "covariate": ["noop", "calibrate", "reweight", "finetune"],
    "concept": ["noop", "finetune", "retrain", "switch_model"],
    "mixed": ["noop", "calibrate", "reweight", "finetune", "retrain"],
    "unknown": ["noop", "calibrate", "reweight", "finetune"],
}


def plausible_actions(kind: str, has_best: bool = False) -> List[str]:
    acts = list(_PLAUSIBLE.get(kind, _PLAUSIBLE["unknown"]))
    if has_best:
        acts.append("rollback")
    return sorted(set(acts), key=lambda a: CATALOG[a].order)


@dataclass
class ActionContext:
    """动作执行上下文。

    重要：``train_*`` 与 ``recent_*`` 都只含 **≤ k-1** 的历史数据，
    当前批 k 永远不参与训练——它只用于验证候选。否则候选在验证集上
    会"背答案"，闸门就形同虚设。
    """

    spec: DatasetSpec
    batch_index: int
    train_X: np.ndarray
    train_y: np.ndarray
    recent_X: np.ndarray          # 历史中最近一批（k-1），用于校准/密度比
    recent_y: np.ndarray
    anchor: ModelBundle
    registry: Dict[str, ModelBundle]
    best_version: str
    cfg: ModelConfig
    rng_seed: int = 42
    # 按"批龄"给出的逐行权重（可选）。行序在真实数据里未必等于时间序
    # （例如按设备分组排列的 CSV），所以优先用批次边界算衰减。
    batch_weights: Optional[np.ndarray] = None

    @property
    def n_train(self) -> int:
        return int(len(self.train_X))


@dataclass
class ExecutionResult:
    model: ModelBundle
    note: str
    seconds: float
    extra: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# 成本预估
# --------------------------------------------------------------------------


def estimate_cost(name: str, ctx: ActionContext) -> Estimate:
    """执行前的成本预估。系数按 4 核 CPU + HistGB 实测校准。"""
    n = max(ctx.n_train, 100)
    per_row = 6e-4 if ctx.cfg.default_kind == "hgb" else 3e-4
    table = {
        "noop": 0.0,
        "rollback": 0.0,
        "calibrate": 0.02,
        "reweight": per_row * n * 1.4,     # 多一次密度比分类器
        "finetune": per_row * n * 0.7,
        "retrain": per_row * n,
        "switch_model": per_row * n * 1.3,
    }
    return Estimate(compute_seconds=float(table.get(name, per_row * n)))


# --------------------------------------------------------------------------
# 执行器
# --------------------------------------------------------------------------


class ActionExecutor:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def execute(self, name: str, ctx: ActionContext) -> ExecutionResult:
        if name not in CATALOG:
            raise ValueError(f"非法动作: {name}")
        t0 = time.perf_counter()
        method = getattr(self, f"_do_{name}")
        result = method(ctx)
        result.seconds = time.perf_counter() - t0
        return result

    # ---------------- 各动作 ----------------

    def _do_noop(self, ctx: ActionContext) -> ExecutionResult:
        return ExecutionResult(ctx.anchor, "保持当前锚点", 0.0)

    def _do_rollback(self, ctx: ActionContext) -> ExecutionResult:
        best = ctx.registry.get(ctx.best_version, ctx.anchor)
        return ExecutionResult(best, f"回滚到 {ctx.best_version}", 0.0)

    def _do_calibrate(self, ctx: ActionContext) -> ExecutionResult:
        """零重训：只动决策边界。二分类扫阈值，回归算偏置。"""
        X, y = ctx.recent_X, ctx.recent_y
        if ctx.spec.task == TaskType.MULTICLASS:
            return ExecutionResult(ctx.anchor, "多分类不使用二分类阈值或回归偏置校准", 0.0)
        if len(X) == 0:
            return ExecutionResult(ctx.anchor, "无近期标注，校准跳过", 0.0)

        if ctx.spec.task == TaskType.BINARY:
            proba = ctx.anchor.predict_proba(X)
            if proba is None:
                return ExecutionResult(ctx.anchor, "模型无概率输出，无法校准", 0.0)
            p = proba[:, 1]
            best_thr, best_f1 = 0.5, _f1_at(p, y, 0.5)
            base_f1 = best_f1
            for thr in np.linspace(0.30, 0.70, 21):
                f1 = _f1_at(p, y, float(thr))
                if f1 > best_f1:
                    best_f1, best_thr = f1, float(thr)

            # 校准只在"确实明显更好"时才采用——在几百条样本上扫阈值很容易过拟合，
            # 盲目校准会把一个还行的模型调坏（实测掉 10 个点）。
            if best_f1 < base_f1 + 0.01:
                return ExecutionResult(
                    ctx.anchor, f"校准无效(最优{best_f1:.3f} vs 基线{base_f1:.3f})，保持原模型", 0.0
                )
            wrapped = ThresholdedModel(ctx.anchor, best_thr)
            bundle = ModelBundle(
                version=f"{ctx.anchor.version}#cal{ctx.batch_index}",
                kind="calibrated",
                task=ctx.spec.task,
                estimator=wrapped,
                trained_on=0,
                parent=ctx.anchor.version,
                note=f"阈值校准 thr={best_thr:.2f}",
            )
            return ExecutionResult(bundle, f"阈值 {best_thr:.2f}，近期 F1={best_f1:.3f}", 0.0,
                                   {"threshold": best_thr})

        pred = np.asarray(ctx.anchor.predict(X), dtype=float)
        bias = float(np.mean(y - pred))
        wrapped = BiasedRegressor(ctx.anchor, bias)
        bundle = ModelBundle(
            version=f"{ctx.anchor.version}#bia{ctx.batch_index}",
            kind="calibrated",
            task=ctx.spec.task,
            estimator=wrapped,
            trained_on=0,
            parent=ctx.anchor.version,
            note=f"偏置校正 {bias:+.4f}",
        )
        return ExecutionResult(bundle, f"偏置 {bias:+.4f}", 0.0, {"bias": bias})

    def _do_reweight(self, ctx: ActionContext) -> ExecutionResult:
        """密度比加权：让训练分布靠近当前分布，但不动标签关系。"""
        w = density_ratio_weights(ctx.train_X, ctx.recent_X)
        bundle = fit_model(
            self.cfg.default_kind,
            ctx.spec.task,
            ctx.train_X,
            ctx.train_y,
            sample_weight=w,
            params=self.cfg.training_params(),
            version=f"rw-{ctx.batch_index}",
            parent=ctx.anchor.version,
            note="密度比重加权",
        )
        return ExecutionResult(bundle, f"重加权训练 {ctx.n_train} 样本", 0.0,
                               {"weight_max": float(np.max(w))})

    def _do_finetune(self, ctx: ActionContext) -> ExecutionResult:
        """只吃历史数据，按"越新越重要"加权。绝不含当前验证批。

        权重来源有两种：``batch`` 模式按批次到达顺序衰减（正确的时间语义），
        ``row`` 模式退回旧的行序近似（仅适用于追加写入的单一时间序列）。
        """
        X, y = ctx.train_X, ctx.train_y
        n = len(X)
        if ctx.batch_weights is not None and len(ctx.batch_weights) == n:
            w = np.asarray(ctx.batch_weights, dtype=float)
            w = w / w.mean()
            mode = "按批次"
        else:
            # 行序近似时间序：越靠后的样本权重越高
            pos = np.linspace(0.0, 1.0, max(n, 1))
            w = 0.15 + 0.85 * pos**2
            w = w / w.mean()
            mode = "按行序"

        bundle = fit_model(
            self.cfg.default_kind,
            ctx.spec.task,
            X,
            y,
            sample_weight=w,
            params=self.cfg.training_params(),
            version=f"ft-{ctx.batch_index}",
            parent=ctx.anchor.version,
            note=f"增量微调({mode}加权)",
        )
        return ExecutionResult(bundle, f"增量微调 {n} 样本（{mode}加权）", 0.0)

    def _do_retrain(self, ctx: ActionContext) -> ExecutionResult:
        X, y = ctx.train_X, ctx.train_y
        bundle = fit_model(
            self.cfg.default_kind,
            ctx.spec.task,
            X,
            y,
            params=self.cfg.training_params(),
            version=f"rt-{ctx.batch_index}",
            parent=ctx.anchor.version,
            note="全量重训",
        )
        return ExecutionResult(bundle, f"全量重训 {len(X)} 样本", 0.0)

    def _do_switch_model(self, ctx: ActionContext) -> ExecutionResult:
        X, y = ctx.train_X, ctx.train_y
        bundle = fit_model(
            self.cfg.alt_kind,
            ctx.spec.task,
            X,
            y,
            params=self.cfg.training_params(alternate=True),
            version=f"sw-{ctx.batch_index}",
            parent=ctx.anchor.version,
            note=f"换模型族 {self.cfg.alt_kind}",
        )
        return ExecutionResult(bundle, f"切换为 {self.cfg.alt_kind}，{len(X)} 样本", 0.0)


# --------------------------------------------------------------------------


def _f1_at(p: np.ndarray, y: np.ndarray, thr: float) -> float:
    """给定概率与阈值，算正类 F1（校准动作内部用）。"""
    pred = (p >= thr).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def density_ratio_weights(ref_X: np.ndarray, target_X: np.ndarray) -> np.ndarray:
    """用 logistic 分类器估计 p(target|x)/p(ref|x)，再换算成重要性权重。"""
    n_ref, n_tar = len(ref_X), len(target_X)
    if n_ref < 10 or n_tar < 10 or not _HAS_SKLEARN:
        return np.ones(n_ref)

    X = np.vstack([ref_X, target_X])
    s = np.concatenate([np.zeros(n_ref), np.ones(n_tar)])
    try:
        clf = LogisticRegression(max_iter=300, class_weight="balanced")
        clf.fit(X, s)
        p = clf.predict_proba(ref_X)[:, 1]
        p = np.clip(p, 1e-3, 1 - 1e-3)
        w = p / (1.0 - p)
        w = np.clip(w, 0.05, 20.0)
        return w / w.mean()
    except Exception:
        return np.ones(n_ref)


def catalog_for_prompt() -> List[Dict[str, Any]]:
    """给 LLM 的动作卡（紧凑版，控制 token）。"""
    return [
        {
            "name": a.name,
            "desc": a.description,
            "risk": a.risk,
            "order": a.order,
        }
        for a in sorted(CATALOG.values(), key=lambda x: x.order)
    ]
