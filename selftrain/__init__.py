"""智训引擎 —— 工业模型自进化系统（Industrial Model Self-Evolution Harness）.

分层：
    model 层：模型自迭代（训练 / 校准 / 重训 / 回滚）
    policy 层：策略自改进（playbook bandit / LLM harness）

入口见 ``selftrain.cli``。
"""

from .types import (  # noqa: F401
    ActionPlan,
    Batch,
    CostRecord,
    DatasetSpec,
    Decision,
    DriftSignal,
    Fingerprint,
    TaskType,
)

__all__ = [
    "ActionPlan",
    "Batch",
    "CostRecord",
    "DatasetSpec",
    "Decision",
    "DriftSignal",
    "Fingerprint",
    "TaskType",
]

__version__ = "0.1.0"
