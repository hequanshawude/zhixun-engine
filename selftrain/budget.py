"""预算账本。

客观分 ≈ 单位成本换来的增益，所以"记账"本身就是评分项，不是附属功能。
每个动作都必须先申请预算（``afford``），再执行，再 ``spend`` 记账。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import BudgetConfig
from .types import CostRecord


@dataclass
class Estimate:
    """执行前的成本预估。"""

    compute_seconds: float = 0.0
    llm_tokens: int = 0
    label_queries: int = 0


class Ledger:
    def __init__(self, cfg: BudgetConfig):
        self.cfg = cfg
        self.records: List[CostRecord] = []
        self.compute_seconds = 0.0
        self.initialization_seconds = 0.0
        self.llm_tokens = 0
        self.label_queries = 0
        self.iterations = 0
        self.total_gain = 0.0
        self._blocked: List[str] = []

    # ---------- 成本换算 ----------
    def units(self, e: Estimate) -> float:
        return (
            e.compute_seconds
            + e.llm_tokens * self.cfg.token_cost
            + e.label_queries * self.cfg.label_cost
        )

    # ---------- 预算检查 ----------
    def afford(self, e: Estimate) -> tuple[bool, str]:
        if self.iterations >= self.cfg.max_iterations:
            return False, "迭代次数已达上限"
        if self.compute_seconds + e.compute_seconds > self.cfg.max_compute_seconds:
            return False, "算力预算不足"
        if self.llm_tokens + e.llm_tokens > self.cfg.max_llm_tokens:
            return False, "Token 预算不足"
        if self.label_queries + e.label_queries > self.cfg.max_label_queries:
            return False, "标注配额不足"
        return True, ""

    # ---------- 记账 ----------
    def record_initialization(self, seconds: float) -> None:
        self.initialization_seconds += seconds
        self.compute_seconds += seconds

    def spend(
        self,
        batch_index: int,
        action: str,
        e: Estimate,
        gain: float = 0.0,
        compute_seconds: Optional[float] = None,
    ) -> CostRecord:
        c = self.cfg
        compute = e.compute_seconds if compute_seconds is None else compute_seconds
        rec = CostRecord(
            batch_index=batch_index,
            action=action,
            compute_seconds=compute,
            llm_tokens=e.llm_tokens,
            label_queries=e.label_queries,
            cost_units=compute + e.llm_tokens * c.token_cost + e.label_queries * c.label_cost,
            gain=float(gain),
        )
        self.records.append(rec)
        self.compute_seconds += compute
        self.llm_tokens += e.llm_tokens
        self.label_queries += e.label_queries
        self.total_gain += float(gain)
        self.iterations += 1
        return rec

    def block(self, reason: str) -> None:
        self._blocked.append(reason)

    # ---------- 观察 ----------
    @property
    def total_cost(self) -> float:
        return (
            self.compute_seconds
            + self.llm_tokens * self.cfg.token_cost
            + self.label_queries * self.cfg.label_cost
        )

    @property
    def efficiency(self) -> float:
        """单位成本增益——这就是客观分最核心的那个比值。"""
        tc = self.total_cost
        return float(self.total_gain / tc) if tc > 1e-9 else 0.0

    def remaining(self) -> Dict[str, float]:
        return {
            "compute_s": max(0.0, self.cfg.max_compute_seconds - self.compute_seconds),
            "tokens": max(0, self.cfg.max_llm_tokens - self.llm_tokens),
            "labels": max(0, self.cfg.max_label_queries - self.label_queries),
            "iterations": max(0, self.cfg.max_iterations - self.iterations),
        }

    def used_ratio(self) -> float:
        parts = [
            self.compute_seconds / max(self.cfg.max_compute_seconds, 1e-9),
            self.llm_tokens / max(self.cfg.max_llm_tokens, 1),
            self.label_queries / max(self.cfg.max_label_queries, 1),
        ]
        return float(sum(parts) / len(parts))

    def summary(self) -> Dict[str, Any]:
        return {
            "compute_s": round(self.compute_seconds, 2),
            "initialization_s": round(self.initialization_seconds, 2),
            "tokens": self.llm_tokens,
            "labels": self.label_queries,
            "iterations": self.iterations,
            "total_cost": round(self.total_cost, 2),
            "total_gain": round(self.total_gain, 5),
            "efficiency": round(self.efficiency, 6),
            "blocked": self._blocked[-5:],
        }
