"""核心数据类型：全系统只在这里定义"契约"，其它模块只依赖它。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class TaskType(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"

    @property
    def is_classification(self) -> bool:
        return self in (TaskType.BINARY, TaskType.MULTICLASS)


class DriftKind(str, Enum):
    NONE = "none"
    COVARIATE = "covariate"   # P(X) 变，P(y|X) 大致不变
    CONCEPT = "concept"       # P(y|X) 变
    LABEL = "label"           # P(y) 变
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class DatasetSpec:
    """一个数据集（A 包里的某个任务）的静态描述。"""

    name: str
    task: TaskType
    feature_names: List[str]
    target_name: str = "y"
    n_classes: int = 2
    higher_is_better: bool = True
    primary_metric: str = "accuracy"   # 由 metrics 模块自动填充

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task.value,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "primary_metric": self.primary_metric,
        }


@dataclass
class Batch:
    """一批下发数据。``y`` 可能为 None（尚未拿到标签）。"""

    index: int
    X: np.ndarray
    y: Optional[np.ndarray] = None
    feature_names: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def labeled(self) -> bool:
        return self.y is not None


@dataclass
class DriftSignal:
    """无标签侧漂移统计。"""

    severity: float                       # 0..1，归一化后的综合严重度
    per_feature_psi: Dict[str, float] = field(default_factory=dict)
    per_feature_ks: Dict[str, float] = field(default_factory=dict)   # KS p-value
    top_features: List[str] = field(default_factory=list)
    frac_drifted: float = 0.0
    mmd: float = 0.0
    reference_size: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "severity": round(self.severity, 4),
            "frac_drifted": round(self.frac_drifted, 4),
            "mmd": round(self.mmd, 5),
            "top_features": self.top_features[:5],
        }


@dataclass
class Fingerprint:
    """归因结果：告诉策略层"漂的是什么、有多严重、值不值得动"。"""

    kind: DriftKind
    severity: float
    performance_drop: float      # 归一化，>0 表示变差
    top_features: List[str] = field(default_factory=list)
    prior_shift: float = 0.0     # 标签分布偏移（分类为 TV 距离）
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "severity": round(self.severity, 3),
            "performance_drop": round(self.performance_drop, 4),
            "prior_shift": round(self.prior_shift, 4),
            "top_features": self.top_features[:5],
        }


@dataclass
class ActionPlan:
    """策略层输出的一个动作。"""

    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    source: str = "rule"          # llm / bandit / rule / validator

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "params": self.params,
            "rationale": self.rationale[:200],
            "source": self.source,
        }


@dataclass
class CostRecord:
    """一次动作的账目。客观分 = 增益 / 成本，所以这里必须诚实。"""

    batch_index: int
    action: str
    compute_seconds: float = 0.0
    llm_tokens: int = 0
    label_queries: int = 0
    # 归一化成本，用于排序/预算
    cost_units: float = 0.0
    gain: float = 0.0            # 本次动作带来的指标增益（可为负）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch": self.batch_index,
            "action": self.action,
            "compute_s": round(self.compute_seconds, 3),
            "tokens": self.llm_tokens,
            "labels": self.label_queries,
            "cost": round(self.cost_units, 3),
            "gain": round(self.gain, 5),
        }


@dataclass
class Decision:
    """主循环对单个 batch 的完整决策记录（也是审计日志）。"""

    batch_index: int
    fingerprint: Dict[str, Any]
    plan: Dict[str, Any]
    promoted: bool
    reason: str
    anchor_before: str
    anchor_after: str
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_candidate: Dict[str, float] = field(default_factory=dict)
    static_metrics: Dict[str, float] = field(default_factory=dict)
    gain: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch": self.batch_index,
            "fingerprint": self.fingerprint,
            "plan": self.plan,
            "promoted": self.promoted,
            "reason": self.reason,
            "anchor": f"{self.anchor_before} -> {self.anchor_after}",
            "metrics_before": {k: round(v, 4) for k, v in self.metrics_before.items()},
            "metrics_candidate": {k: round(v, 4) for k, v in self.metrics_candidate.items()},
            "static": {k: round(v, 4) for k, v in self.static_metrics.items()},
            "gain": round(self.gain, 6),
        }
